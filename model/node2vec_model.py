import queue
import threading
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import Node2Vec


class _ProducerFailure:
    def __init__(self, exception: BaseException) -> None:
        self.exception = exception


class _PrefetchIterator:
    _END = object()

    def __init__(self, source, depth: int, pin_memory: bool) -> None:
        self._source = iter(source)
        self._queue = queue.Queue(maxsize=max(1, int(depth)))
        self._stop = threading.Event()
        self._pin_memory = bool(pin_memory)
        self._closed = False
        self._thread = threading.Thread(target=self._produce, name="node2vec-walk-prefetch", daemon=True)
        self._thread.start()

    def _put(self, item) -> bool:
        while not self._stop.is_set():
            try:
                self._queue.put(item, timeout=0.1)
                return True
            except queue.Full:
                continue
        return False

    def _produce(self) -> None:
        try:
            for pos_rw, neg_rw in self._source:
                if self._stop.is_set():
                    break
                if self._pin_memory:
                    pos_rw = pos_rw.pin_memory()
                    neg_rw = neg_rw.pin_memory()
                if not self._put((pos_rw, neg_rw)):
                    return
        except BaseException as exc:
            self._put(_ProducerFailure(exc))
        finally:
            self._put(self._END)

    def __iter__(self):
        return self

    def __next__(self):
        item = self._queue.get()
        if item is self._END:
            self.close()
            raise StopIteration
        if isinstance(item, _ProducerFailure):
            self.close()
            raise item.exception
        return item

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._stop.set()
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        if self._thread is not threading.current_thread():
            self._thread.join(timeout=2.0)

    def __del__(self):
        self.close()


class _PrefetchLoader:
    def __init__(self, source, depth: int, pin_memory: bool) -> None:
        self._source = source
        self._depth = max(1, int(depth))
        self._pin_memory = bool(pin_memory)

    def __len__(self) -> int:
        return len(self._source)

    def __iter__(self):
        return _PrefetchIterator(self._source, depth=self._depth, pin_memory=self._pin_memory)


class Node2VecEncoder(nn.Module):
    def __init__(self, edge_index: torch.Tensor | None, num_nodes: int, emb_dim: int, walk_length: int = 20, context_size: int = 10, walks_per_node: int = 10, num_negative_samples: int = 1, p: float = 1.0, q: float = 1.0, sparse: bool = True, sampler_state: tuple[torch.Tensor, torch.Tensor] | None = None) -> None:
        super().__init__()
        if sampler_state is not None:
            construction_edge_index = torch.empty((2, 0), dtype=torch.long)
        elif edge_index is None:
            raise ValueError("edge_index is required when sampler_state is absent")
        else:
            construction_edge_index = edge_index
        self.node2vec = Node2Vec(
            edge_index=construction_edge_index,
            embedding_dim=emb_dim,
            walk_length=walk_length,
            context_size=context_size,
            walks_per_node=walks_per_node,
            num_negative_samples=num_negative_samples,
            p=p,
            q=q,
            num_nodes=num_nodes,
            sparse=sparse,
        )
        if sampler_state is not None:
            (rowptr, col) = sampler_state
            if rowptr.device.type != "cpu" or col.device.type != "cpu":
                raise ValueError("Node2Vec sampler_state must remain on CPU")
            if int(rowptr.numel()) != int(num_nodes) + 1:
                raise ValueError("Node2Vec sampler_state rowptr does not match num_nodes")
            self.node2vec.rowptr = rowptr
            self.node2vec.col = col

    def sampler_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.node2vec.rowptr, self.node2vec.col)

    def loader(self, batch_size: int, shuffle: bool = True, *, prefetch_batches: int = 0, pin_memory: bool = False):
        loader = self.node2vec.loader(batch_size=batch_size, shuffle=shuffle)
        if int(prefetch_batches) <= 0:
            return loader
        return _PrefetchLoader(loader, depth=int(prefetch_batches), pin_memory=pin_memory)

    def rw_loss(self, pos_rw: torch.Tensor, neg_rw: torch.Tensor) -> torch.Tensor:
        return self.node2vec.loss(pos_rw, neg_rw)

    @torch.no_grad()
    def forward(self) -> torch.Tensor:
        return self.node2vec()


class DotProductPredictor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decode_is_symmetric = True

    def forward(self, z: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index
        return (z[src] * z[dst]).sum(dim=-1)


class _ReferenceMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        if int(num_layers) < 1:
            raise ValueError("num_layers must be positive")
        dims = [in_channels] + [hidden_channels] * (int(num_layers) - 1) + [out_channels]
        self.lins = nn.ModuleList(nn.Linear(left, right) for left, right in zip(dims, dims[1:]))
        self.dropout = float(dropout)

    def reset_parameters(self) -> None:
        for lin in self.lins:
            lin.reset_parameters()

    def _forward(self, x):
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return self.lins[-1](x)


class ReferenceNodeMLP(_ReferenceMLP):
    def __init__(self, in_channels: int, hidden_channels: int = 128, out_channels: int = 128, num_layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__(in_channels, hidden_channels, hidden_channels if int(num_layers) == 1 else out_channels, num_layers, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._forward(x)


class ReferenceMLPScore(_ReferenceMLP):
    def __init__(self, in_channels: int = 128, hidden_channels: int = 128, out_channels: int = 1, num_layers: int = 3, dropout: float = 0.0) -> None:
        super().__init__(in_channels, hidden_channels, out_channels, num_layers, dropout)

    def forward(self, x_i: torch.Tensor, x_j: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self._forward(x_i * x_j))


class ReferenceN2VLink(nn.Module):
    def __init__(self, input_channels: int, hidden_channels: int = 128, num_layers: int = 3, predictor_layers: int = 3, dropout: float = 0.0, node_encode_batch_size: int = 262144) -> None:
        super().__init__()
        self.decode_is_symmetric = True
        self.decode_is_dedup_safe = True
        self.encoder = ReferenceNodeMLP(input_channels, hidden_channels, hidden_channels, num_layers, dropout)
        self.predictor = ReferenceMLPScore(hidden_channels, hidden_channels, 1, predictor_layers, dropout)
        self.dropout = float(dropout)
        self.output_channels = int(hidden_channels)
        self.node_encode_batch_size = max(1, int(node_encode_batch_size))
        self.register_buffer("_node_features", None, persistent=False)

    def reset_parameters(self) -> None:
        self.encoder.reset_parameters()
        self.predictor.reset_parameters()

    def set_node_features(self, features: torch.Tensor) -> None:
        self._node_features = features

    @torch.no_grad()
    def embed(self, data=None) -> torch.Tensor:
        del data
        if self._node_features is None:
            raise RuntimeError("ReferenceN2VLink node features were not initialized")
        if self.training:
            return self._node_features
        num_nodes = int(self._node_features.size(0))
        encoded = self._node_features.new_empty((num_nodes, self.output_channels))
        for start in range(0, num_nodes, self.node_encode_batch_size):
            end = min(start + self.node_encode_batch_size, num_nodes)
            encoded[start:end] = self.encoder(self._node_features[start:end])
        return encoded

    def decode(self, features: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if self.training and self.dropout != 0.0:
            raise RuntimeError("Endpoint-only reference Node2Vec decoding is not training-exact with dropout; use full-table training.")
        (src, dst) = edge_index
        if self.training:
            z_src = self.encoder(features[src])
            z_dst = self.encoder(features[dst])
        else:
            z_src = features[src]
            z_dst = features[dst]
        return self.predictor(z_src, z_dst).view(-1)
