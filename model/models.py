from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GATConv
from torch_sparse import SparseTensor, matmul as sparse_matmul

FUSED_GAT_MAX_EDGES_PER_SPMM = 1 << 25
FUSED_GAT_INITIAL_ROWS_PER_CHUNK = 1 << 16
FUSED_GAT_IMPLEMENTATION = "reddit-row-chunked-fused-sparse-gat-v1"
FUSED_GAT_EXECUTION_PATH = "destination-row-chunked-sparse-spmm"


def _destination_row_chunks(rowptr, *, max_edges=FUSED_GAT_MAX_EDGES_PER_SPMM, initial_rows=FUSED_GAT_INITIAL_ROWS_PER_CHUNK):
    if not torch.is_tensor(rowptr) or rowptr.ndim != 1:
        raise ValueError("GAT rowptr must be a rank-1 tensor.")
    if rowptr.numel() < 2:
        raise ValueError("GAT rowptr must contain at least two entries.")
    max_edges = int(max_edges)
    initial_rows = int(initial_rows)
    if max_edges <= 0 or initial_rows <= 0:
        raise ValueError("GAT chunk limits must be positive.")
    num_rows = int(rowptr.numel()) - 1
    chunks = []

    def append_bounded(row_start, row_end):
        edge_count = int((rowptr[row_end] - rowptr[row_start]).item())
        if edge_count <= max_edges:
            chunks.append((row_start, row_end))
            return
        if row_end - row_start <= 1:
            raise RuntimeError(
                f"One fused GAT destination row contains {edge_count} edges, exceeding the safe per-SpMM cap of {max_edges}."
            )
        row_middle = row_start + (row_end - row_start) // 2
        append_bounded(row_start, row_middle)
        append_bounded(row_middle, row_end)

    for row_start in range(0, num_rows, initial_rows):
        append_bounded(row_start, min(num_rows, row_start + initial_rows))
    if num_rows and (
        not chunks
        or chunks[0][0] != 0
        or chunks[-1][1] != num_rows
        or any((left != previous_right for ((_, previous_right), (left, _)) in zip(chunks, chunks[1:])))
    ):
        raise RuntimeError("GAT row chunking did not cover every row exactly.")
    return tuple(chunks)


def _overflow_safe_sparse_matmul(original, sparse, dense, *args, max_edges=FUSED_GAT_MAX_EDGES_PER_SPMM, initial_rows=FUSED_GAT_INITIAL_ROWS_PER_CHUNK, **kwargs):
    if not isinstance(sparse, SparseTensor) or int(sparse.nnz()) <= max_edges:
        return original(sparse, dense, *args, **kwargs)
    chunks = _destination_row_chunks(sparse.storage.rowptr(), max_edges=max_edges, initial_rows=initial_rows)
    parts = []
    for row_start, row_end in chunks:
        sparse_chunk = sparse.narrow(0, row_start, row_end - row_start)
        if int(sparse_chunk.nnz()) > int(max_edges):
            raise RuntimeError("Fused GAT produced an oversized SpMM chunk.")
        parts.append(original(sparse_chunk, dense, *args, **kwargs))
    return torch.cat(parts, dim=0)


class MFEncoder(nn.Module):
    def __init__(self, num_nodes: int, emb_dim: int, *, hidden_channels: Optional[int] = None, out_channels: Optional[int] = None, num_layers: int = 0, dropout: float = 0.0, reference_style: bool = False, embedding_init_std: Optional[float] = None):
        super().__init__()
        self.reference_style = bool(reference_style)
        self.embedding_init_std = None if embedding_init_std is None else float(embedding_init_std)
        self.num_layers = max(0, int(num_layers))
        hidden_channels = int(emb_dim if hidden_channels is None else hidden_channels)
        out_channels = int(hidden_channels if out_channels is None else out_channels)
        table_dim = out_channels if self.num_layers == 0 else int(emb_dim)
        if self.reference_style:
            self.lins = nn.ModuleList()
        self.emb = nn.Embedding(int(num_nodes), table_dim)
        if self.embedding_init_std is not None:
            nn.init.normal_(self.emb.weight, std=self.embedding_init_std)
        elif not self.reference_style:
            nn.init.xavier_uniform_(self.emb.weight)
        if not self.reference_style:
            self.lins = nn.ModuleList()
        if self.num_layers == 1:
            self.lins.append(nn.Linear(table_dim, out_channels))
        elif self.num_layers > 1:
            self.lins.append(nn.Linear(table_dim, hidden_channels))
            for _ in range(self.num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = float(dropout)
        self.out_channels = table_dim if self.num_layers == 0 else out_channels

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        if self.embedding_init_std is not None:
            nn.init.normal_(self.emb.weight, std=self.embedding_init_std)
        elif self.reference_style:
            self.emb.reset_parameters()
        else:
            nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, adj_t=None):
        del x, adj_t
        x = self.emb.weight
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        if self.lins:
            x = self.lins[-1](x)
        return x


class NodeEmbeddingEncoder(nn.Module):
    def __init__(self, num_nodes, emb_dim, encoder):
        super().__init__()
        self.emb = nn.Embedding(num_nodes, emb_dim)
        self.encoder = encoder
        self.out_channels = encoder.out_channels
        nn.init.xavier_uniform_(self.emb.weight)

    def forward(self, x, adj_t=None):
        return self.encoder(self.emb.weight, adj_t)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.emb.weight)
        reset = getattr(self.encoder, "reset_parameters", None)
        if callable(reset):
            reset()


class _GraphStack(nn.Module):
    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, graph):
        for conv in self.convs[:-1]:
            x = F.dropout(F.relu(conv(x, graph)), p=self.dropout, training=self.training)
        return self.convs[-1](x, graph)


class GCNEncoder(_GraphStack):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.out_channels = out_channels
        if num_layers == 1:
            self.convs.append(GCNConv(in_channels, out_channels, cached=True))
        elif num_layers > 1:
            self.convs.append(GCNConv(in_channels, hidden_channels, cached=True))
            for _ in range(num_layers - 2):
                self.convs.append(GCNConv(hidden_channels, hidden_channels, cached=True))
            self.convs.append(GCNConv(hidden_channels, out_channels, cached=True))
        self.dropout = dropout


class FusedSparseGATConv(GATConv):
    def message_and_aggregate(self, adj_t, x, alpha):
        source = x[0] if isinstance(x, (tuple, list)) else x
        if source.dim() != 3:
            raise RuntimeError(f"Fused GAT propagation expects transformed features [N,H,C], got {tuple(source.shape)}")
        if alpha.dim() != 2 or alpha.size(1) != self.heads:
            raise RuntimeError(f"Fused GAT propagation expects attention [E,H], got {tuple(alpha.shape)} for {self.heads} heads")
        outputs = []
        for head in range(self.heads):
            weighted = adj_t.set_value(alpha[:, head].contiguous(), layout="coo")
            outputs.append(_overflow_safe_sparse_matmul(sparse_matmul, weighted, source[:, head, :], reduce=self.aggr))
        return torch.stack(outputs, dim=1)


class GATEncoder(_GraphStack):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.5, head=4, fused_sparse_propagation=False):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.fused_sparse_propagation = bool(fused_sparse_propagation)
        conv_type = FusedSparseGATConv if self.fused_sparse_propagation else GATConv
        out_c = int(self.out_channels // head)
        hidden_c = int(self.hidden_channels // head)
        if num_layers == 1:
            self.convs.append(conv_type(in_channels, out_c, heads=head))
        elif num_layers > 1:
            self.convs.append(conv_type(in_channels, hidden_c, heads=head))
            for _ in range(num_layers - 2):
                self.convs.append(conv_type(hidden_channels, hidden_c, heads=head))
            self.convs.append(conv_type(hidden_channels, out_c, heads=head))
        self.dropout = dropout


class SAGEEncoder(_GraphStack):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=1, dropout=0.5):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.out_channels = out_channels
        if num_layers == 1:
            self.convs.append(SAGEConv(in_channels, out_channels, normalize=False))
        else:
            self.convs.append(SAGEConv(in_channels, hidden_channels, normalize=False))
            for _ in range(num_layers - 2):
                self.convs.append(SAGEConv(hidden_channels, hidden_channels, normalize=False))
            self.convs.append(SAGEConv(hidden_channels, out_channels, normalize=False))
        self.dropout = dropout


class DotProductDecoder(nn.Module):
    def forward(self, z, edge_label_index):
        (src, dst) = edge_label_index
        return (z[src] * z[dst]).sum(dim=-1)


class MLPEncoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.out_channels = out_channels
        layers = []
        dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i != len(dims) - 2:
                layers.extend((nn.ReLU(), nn.Dropout(dropout)))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x, adj_t=None):
        return self.mlp(x)

    def reset_parameters(self):
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                module.reset_parameters()


class MLPDecoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout):
        super().__init__()
        self.lins = nn.ModuleList()
        if num_layers == 1:
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()

    def forward(self, z, edge_label_index):
        (src, dst) = edge_label_index
        x = z[src] * z[dst]
        for lin in self.lins[:-1]:
            x = F.dropout(F.relu(lin(x)), p=self.dropout, training=self.training)
        return self.lins[-1](x).view(-1)


class LinkPredictor(nn.Module):
    def __init__(self, encoder: nn.Module, pred_layers: Optional[int] = None, dropout: float = 0.0, dot: bool = False):
        super().__init__()
        self.decode_is_symmetric = True
        self.decode_is_dedup_safe = True
        self.encoder = encoder
        self.pred_layers = int(pred_layers) if pred_layers is not None else 1
        self.out_channels = encoder.out_channels
        self.decoder = (
            DotProductDecoder()
            if dot
            else MLPDecoder(
                in_channels=self.out_channels,
                hidden_channels=self.out_channels,
                out_channels=1,
                num_layers=self.pred_layers,
                dropout=dropout,
            )
        )

    def reset_parameters(self):
        reset_encoder = getattr(self.encoder, "reset_parameters", None)
        if callable(reset_encoder):
            reset_encoder()
        reset_decoder = getattr(self.decoder, "reset_parameters", None)
        if callable(reset_decoder):
            reset_decoder()

    def embed(self, data) -> torch.Tensor:
        if hasattr(data, "adj_t") and data.adj_t is not None:
            return self.encoder(data.x, data.adj_t)
        return self.encoder(data.x, data.edge_index)

    def decode(self, z: torch.Tensor, edge_label_index: torch.Tensor) -> torch.Tensor:
        return self.decoder(z, edge_label_index)

    def forward(self, data) -> torch.Tensor:
        z = self.embed(data)
        return self.decode(z, data.edge_label_index)
