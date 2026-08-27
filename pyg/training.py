import math
import weakref
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch_geometric.utils import negative_sampling
from torch_sparse import SparseTensor
from types import SimpleNamespace
from model.decoder_training import train_cached_decoder_minibatches


def _is_base_link_predictor(model):
    return model.__class__.__name__ == "LinkPredictor"


def _clip_grad_norm_legacy(parameters, max_norm, norm_type=2.0):
    parameters = list(parameters)
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return torch.tensor(0.0)
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    first_device = parameters[0].grad.device
    if norm_type == float("inf"):
        total_norm = max((parameter.grad.detach().abs().max().to(first_device) for parameter in parameters))
    else:
        total_norm = torch.norm(
            torch.stack([torch.norm(parameter.grad.detach(), norm_type).to(first_device) for parameter in parameters]), norm_type
        )
    clip_coefficient = max_norm / (total_norm + 1e-06)
    if clip_coefficient < 1:
        for parameter in parameters:
            parameter.grad.detach().mul_(clip_coefficient.to(parameter.grad.device))
    return total_norm


def _clip_reference_link_predictor_gradients(model, max_norm=1.0):
    encoder = getattr(model, "encoder", None)
    decoder = getattr(model, "decoder", None)
    if encoder is not None:
        _clip_grad_norm_legacy(encoder.parameters(), float(max_norm))
    if decoder is not None:
        _clip_grad_norm_legacy(decoder.parameters(), float(max_norm))


_SHARED_TRAIN_EDGE_IDS = {}


def _cached_train_edge_ids(model, train_pos, num_nodes, device):
    key = (int(train_pos.data_ptr()), int(train_pos.size(0)), int(num_nodes), device.type, device.index)
    if getattr(model, "_fast_train_edge_key", None) == key:
        return model._fast_train_edge_ids
    shared = _SHARED_TRAIN_EDGE_IDS.get(key)
    if shared is not None and shared[0]() is train_pos:
        model._fast_train_edge_ids = shared[1]
        model._fast_train_edge_key = key
    else:
        edges = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
        lo = torch.minimum(edges[:, 0], edges[:, 1])
        hi = torch.maximum(edges[:, 0], edges[:, 1])
        ids = torch.sort(lo * int(num_nodes) + hi).values
        model._fast_train_edge_ids = torch.unique_consecutive(ids)
        model._fast_train_edge_key = key
        _SHARED_TRAIN_EDGE_IDS[key] = (
            weakref.ref(train_pos, lambda _ref, cache_key=key: _SHARED_TRAIN_EDGE_IDS.pop(cache_key, None)),
            model._fast_train_edge_ids,
        )
    return model._fast_train_edge_ids


def _sample_strict_negative_edges(model, train_pos, num_nodes, num_samples, device):
    edge_ids = _cached_train_edge_ids(model, train_pos, num_nodes, device)
    parts = []
    remaining = int(num_samples)
    nodes = max(1, int(num_nodes))
    pair_space = float(nodes) * float(nodes)
    invalid_probability = min(0.95, (float(nodes) + 2.0 * float(edge_ids.numel())) / pair_space)
    valid_probability = max(0.05, 1.0 - invalid_probability)
    while remaining > 0:
        expected_draw = int(math.ceil(remaining / valid_probability))
        expected_invalid = max(1.0, float(expected_draw) * invalid_probability)
        safety = int(math.ceil(8.0 * math.sqrt(expected_invalid))) + 64
        draw = max(4096, min(4 * remaining + 64, expected_draw + safety))
        src = torch.randint(0, int(num_nodes), (draw,), device=device)
        dst = torch.randint(0, int(num_nodes), (draw,), device=device)
        lo = torch.minimum(src, dst)
        hi = torch.maximum(src, dst)
        ids = lo * int(num_nodes) + hi
        valid = lo != hi
        idx = torch.searchsorted(edge_ids, ids)
        present = torch.zeros(draw, dtype=torch.bool, device=device)
        inside = idx < edge_ids.numel()
        present[inside] = edge_ids[idx[inside]] == ids[inside]
        valid &= ~present
        if bool(valid.any()):
            take = min(remaining, int(valid.sum().item()))
            keep = torch.nonzero(valid, as_tuple=False).view(-1)[:take]
            parts.append(torch.stack([lo[keep], hi[keep]], dim=0))
            remaining -= take
    return torch.cat(parts, dim=1)


def _mf_minibatch_train(model, train_pos, x, optimizer, batch_size, adj_t):
    device = x.device
    num_nodes = int(x.size(0))
    num_edges = int(train_pos.size(0))
    if num_edges == 0:
        return 0.0
    train_pos_device = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
    neg_edge = _sample_strict_negative_edges(model, train_pos_device, num_nodes, num_edges, device)
    batch_size = max(1, int(batch_size))
    if device.type == "cuda" and num_edges >= 1000000:
        batch_size = max(batch_size, 131072)
    order = torch.randperm(num_edges, device=device)
    z = model.embed(SimpleNamespace(x=x, adj_t=adj_t, edge_index=None))
    total_loss = 0.0
    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        index = order[start:end]
        optimizer.zero_grad(set_to_none=True)
        pos_edge = train_pos_device[index].t().contiguous()
        neg_batch = neg_edge[:, start:end].contiguous()
        pos_logits = model.decode(z, pos_edge).view(-1)
        neg_logits = model.decode(z, neg_batch).view(-1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = end - start
        total_loss += float(loss.detach().item()) * count
    return total_loss / float(num_edges)


def _reference_mf_minibatch_train(model, train_pos, x, optimizer, batch_size, adj_t):
    device = x.device
    num_nodes = int(x.size(0))
    num_edges = int(train_pos.size(0))
    if num_edges == 0:
        return 0.0
    train_pos_device = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
    batch_size = max(1, int(batch_size))
    total_loss = 0.0
    graph_data = _make_graph_data(x, adj_t)
    loader = DataLoader(range(num_edges), batch_size=batch_size, shuffle=True)
    for index_cpu in loader:
        index = index_cpu.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        z = model.embed(graph_data)
        pos_edge = train_pos_device[index].t().contiguous()
        pos_logits = model.decode(z, pos_edge).view(-1)
        neg_edge = torch.randint(0, num_nodes, pos_edge.shape, dtype=torch.long, device=device)
        neg_logits = model.decode(z, neg_edge).view(-1)
        pos_probability = torch.sigmoid(pos_logits)
        neg_probability = torch.sigmoid(neg_logits)
        loss = -torch.log(pos_probability + 1e-15).mean() - torch.log(1.0 - neg_probability + 1e-15).mean()
        loss.backward()
        _clip_reference_link_predictor_gradients(model)
        optimizer.step()
        count = int(index.numel())
        total_loss += float(loss.detach().item()) * count
    return total_loss / float(num_edges)


def _sample_epoch_positive_edges(model, train_pos, device):
    num_edges = int(train_pos.size(0))
    cap = max(0, int(getattr(model, "train_samples_per_epoch", 0)))
    sample_count = num_edges if cap <= 0 else min(cap, num_edges)
    if sample_count == num_edges:
        return train_pos.to(device=device, dtype=torch.long, non_blocking=True)
    if sample_count * 10 <= num_edges:
        index = torch.empty(0, dtype=torch.long, device=train_pos.device)
        while int(index.numel()) < sample_count:
            remaining = sample_count - int(index.numel())
            draw = max(4096, int(remaining * 1.08) + 64)
            candidate = torch.randint(0, num_edges, (draw,), device=train_pos.device)
            index = torch.unique(torch.cat([index, candidate]), sorted=False)
        if int(index.numel()) > sample_count:
            order = torch.randperm(int(index.numel()), device=train_pos.device)[:sample_count]
            index = index[order]
    else:
        index = torch.randperm(num_edges, device=train_pos.device)[:sample_count]
    return train_pos[index].to(device=device, dtype=torch.long, non_blocking=True)


def _sample_loose_negative_edges(num_nodes, num_samples, device):
    edge = torch.randint(0, int(num_nodes), (2, int(num_samples)), device=device, dtype=torch.long)
    bad = edge[0] == edge[1]
    while bool(bad.any()):
        edge[1, bad] = torch.randint(0, int(num_nodes), (int(bad.sum().item()),), device=device)
        bad = edge[0] == edge[1]
    return edge


def _make_graph_data(x, adj_t, csr_rowptr=None, csr_col=None):
    return SimpleNamespace(x=x, adj_t=adj_t, edge_index=None, num_nodes=int(x.size(0)), csr_rowptr=csr_rowptr, csr_col=csr_col)


def _clear_model_decode_cache(model):
    clear = getattr(model, "clear_decode_cache", None)
    if callable(clear):
        clear()


def _clear_model_graph_cache(model):
    clear = getattr(model, "clear_graph_cache", None)
    if callable(clear):
        clear()
    else:
        _clear_model_decode_cache(model)


def _clear_gcnconv_cache(model):
    for module in model.modules():
        if hasattr(module, "_cached_edge_index"):
            module._cached_edge_index = None
        if hasattr(module, "_cached_adj_t"):
            module._cached_adj_t = None


def _undirected_pair_ids(edges, num_nodes):
    edges = edges.to(dtype=torch.long)
    lo = torch.minimum(edges[:, 0], edges[:, 1])
    hi = torch.maximum(edges[:, 0], edges[:, 1])
    return lo * int(num_nodes) + hi


def _target_edge_masked_adjacency(train_pos, target_pos, num_nodes, device, *, train_pair_ids=None):
    train_pos = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
    target_pos = target_pos.to(device=device, dtype=torch.long, non_blocking=True)
    if train_pair_ids is None:
        train_pair_ids = _undirected_pair_ids(train_pos, num_nodes)
    target_pair_ids = torch.unique(_undirected_pair_ids(target_pos, num_nodes), sorted=True)
    remove = torch.zeros(train_pos.size(0), dtype=torch.bool, device=device)
    if target_pair_ids.numel() and train_pair_ids.numel():
        positions = torch.searchsorted(target_pair_ids, train_pair_ids)
        inside = positions < target_pair_ids.numel()
        remove[inside] = target_pair_ids[positions[inside]] == train_pair_ids[inside]
    retained = train_pos[~remove].t().contiguous()
    masked_edges = torch.cat([retained, retained.flip(0)], dim=1)
    values = torch.ones(masked_edges.size(1), dtype=torch.float32, device=device)
    return SparseTensor.from_edge_index(masked_edges, values, (int(num_nodes), int(num_nodes)))


def _planetoid_advanced_minibatch_train(model, train_pos, x, optimizer, batch_size):
    device = x.device
    num_nodes = int(x.size(0))
    full_train = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
    pos_sample = _sample_epoch_positive_edges(model, full_train, device)
    num_edges = int(pos_sample.size(0))
    if num_edges == 0:
        return 0.0
    negative_protocol = str(getattr(model, "train_negative_protocol", ""))
    if negative_protocol == "uniform-random-endpoints":
        neg_epoch = torch.randint(0, num_nodes, (2, num_edges), dtype=torch.long, device=device)
    else:
        neg_epoch = _sample_strict_negative_edges(model, full_train, num_nodes, num_edges, device)
    order = torch.randperm(num_edges, device=device)
    train_pair_ids = _undirected_pair_ids(full_train, num_nodes)
    batch_size = max(1, int(batch_size))
    total_loss = 0.0
    for start in range(0, num_edges, batch_size):
        end = min(start + batch_size, num_edges)
        index = order[start:end]
        target_pos = pos_sample[index]
        masked_adj = _target_edge_masked_adjacency(full_train, target_pos, num_nodes, device, train_pair_ids=train_pair_ids)
        optimizer.zero_grad(set_to_none=True)
        _clear_gcnconv_cache(model)
        _clear_model_graph_cache(model)
        z = model.embed(_make_graph_data(x, masked_adj))
        pos_edge = target_pos.t().contiguous()
        neg_edge = neg_epoch[:, index].contiguous()
        pos_logits = model.decode(z, pos_edge).view(-1)
        neg_logits = model.decode(z, neg_edge).view(-1)
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        )
        loss.backward()
        trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        count = end - start
        total_loss += float(loss.detach().item()) * count
    _clear_gcnconv_cache(model)
    _clear_model_graph_cache(model)
    return total_loss / float(num_edges)


def _reference_minibatch_compatible(model, train_pos, num_nodes):
    if bool(getattr(model, "ranked_selector_training_contract", False)):
        return False
    if not _is_base_link_predictor(model):
        return False
    if model.encoder.__class__.__name__ == "MFEncoder":
        return False
    if getattr(model, "decoder", None).__class__.__name__ == "DotProductDecoder":
        return False
    return int(num_nodes) <= 25000 and int(train_pos.size(0)) <= 500000


def _reference_minibatch_train(model, train_pos, x, optimizer, batch_size):
    device = x.device
    train_pos = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
    num_edges = int(train_pos.size(0))
    num_nodes = int(x.size(0))
    if num_edges == 0:
        return 0.0
    batch_size = max(1, int(batch_size))
    total_loss = 0.0
    total_examples = 0
    uses_message_passing = model.encoder.__class__.__name__ != "MLPEncoder"
    feature_only_graph = None if uses_message_passing else _make_graph_data(x, None)
    loader = DataLoader(range(num_edges), batch_size=batch_size, shuffle=True)
    for index_cpu in loader:
        index = index_cpu.to(device=device, non_blocking=True)
        if uses_message_passing:
            keep = torch.ones(num_edges, dtype=torch.bool, device=device)
            keep[index] = False
            masked_pos = train_pos[keep].t().contiguous()
            masked_edges = torch.cat([masked_pos, masked_pos.flip(0)], dim=1)
            masked_adj = SparseTensor.from_edge_index(
                masked_edges, torch.ones(masked_edges.size(1), dtype=torch.float32, device=device), (num_nodes, num_nodes)
            )
            graph_data = _make_graph_data(x, masked_adj)
        else:
            graph_data = feature_only_graph
        optimizer.zero_grad(set_to_none=True)
        _clear_gcnconv_cache(model)
        _clear_model_decode_cache(model)
        z = model.embed(graph_data)
        pos_edge = train_pos[index].t().contiguous()
        pos_logits = model.decode(z, pos_edge).view(-1)
        neg_edge = torch.randint(0, num_nodes, pos_edge.shape, dtype=torch.long, device=device)
        neg_logits = model.decode(z, neg_edge).view(-1)
        reference_optimizer = bool(getattr(model, "reference_optimizer", False))
        if reference_optimizer:
            pos_probability = torch.sigmoid(pos_logits)
            neg_probability = torch.sigmoid(neg_logits)
            loss = -torch.log(pos_probability + 1e-15).mean() - torch.log(1.0 - neg_probability + 1e-15).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits)) + F.binary_cross_entropy_with_logits(
                neg_logits, torch.zeros_like(neg_logits)
            )
        loss.backward()
        if reference_optimizer:
            _clip_reference_link_predictor_gradients(model)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(pos_edge.size(1))
        total_loss += float(loss.detach().item()) * count
        total_examples += count
    _clear_gcnconv_cache(model)
    return total_loss / max(total_examples, 1)


def _reference_gae_dense_loss_and_embedding_grad(embeddings, positive_row, positive_col, *, block_size=2048):
    num_nodes = int(embeddings.size(0))
    if num_nodes <= 0:
        return (0.0, torch.zeros_like(embeddings))
    del block_size
    device = embeddings.device
    positive_row = positive_row.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
    positive_col = positive_col.to(device=device, dtype=torch.long, non_blocking=True).view(-1)
    z = embeddings.detach().requires_grad_(True)
    target = torch.zeros((num_nodes, num_nodes), dtype=z.dtype, device=device)
    if positive_row.numel():
        target[positive_row, positive_col] = 1.0
    probability = torch.sigmoid(torch.mm(z, z.t()))
    loss = F.binary_cross_entropy(probability.view(-1), target.view(-1))
    loss.backward()
    gradient = torch.zeros_like(embeddings) if z.grad is None else z.grad.detach()
    return (float(loss.detach().item()), gradient)


def _reference_gae_train(model, x, optimizer, adj_t, block_size=2048):
    optimizer.zero_grad(set_to_none=True)
    graph_data = _make_graph_data(x, adj_t)
    z_graph = model.embed(graph_data)
    (positive_row, positive_col, _) = adj_t.coo()
    (loss_value, embedding_grad) = _reference_gae_dense_loss_and_embedding_grad(z_graph, positive_row, positive_col, block_size=block_size)
    z_graph.backward(embedding_grad)
    _clip_reference_link_predictor_gradients(model)
    optimizer.step()
    return loss_value


def _full_graph_train(model, train_pos, x, optimizer, batch_size, adj_t, csr_rowptr=None, csr_col=None):
    device = x.device
    num_nodes = int(x.size(0))
    pos_sample = _sample_epoch_positive_edges(model, train_pos, device)
    num_edges = int(pos_sample.size(0))
    if num_edges == 0:
        return 0.0
    if bool(getattr(model, "strict_train_negatives", True)):
        neg_edge = _sample_strict_negative_edges(model, train_pos, num_nodes, num_edges, device)
    else:
        neg_edge = _sample_loose_negative_edges(num_nodes, num_edges, device)
    graph_data = _make_graph_data(x, adj_t, csr_rowptr, csr_col)
    optimizer.zero_grad(set_to_none=True)
    _clear_model_decode_cache(model)
    z_graph = model.embed(graph_data)
    pos_edge = pos_sample.t().contiguous()
    auxiliary = model.auxiliary_loss(pos_edge) if hasattr(model, "auxiliary_loss") else None
    return train_cached_decoder_minibatches(
        model,
        optimizer,
        z_graph,
        pos_edge,
        neg_edge,
        auxiliary_loss=auxiliary,
        update_batch_size=max(1, int(batch_size)),
        positive_weight=0.5,
        negative_weight=0.5,
        clear_decode_cache=_clear_model_decode_cache,
    )


def all_train(model, train_pos, x, optimizer, batch_size, adj_t=None, csr_rowptr=None, csr_col=None):
    model.train()
    if bool(getattr(model, "reference_dense_gae", False)):
        if getattr(model, "_pyg_reported_training_path", None) != "reference-dense-gae":
            print("effective_training_path=reference-dense-gae", flush=True)
            model._pyg_reported_training_path = "reference-dense-gae"
        return _reference_gae_train(model, x, optimizer, adj_t, block_size=int(getattr(model, "gae_reconstruction_block_size", 2048)))
    if _is_base_link_predictor(model) and model.encoder.__class__.__name__ == "MFEncoder":
        if bool(getattr(model, "reference_planetoid_mf", False)):
            if getattr(model, "_pyg_reported_training_path", None) != "reference-mf":
                print("effective_training_path=reference-mf", flush=True)
                model._pyg_reported_training_path = "reference-mf"
            return _reference_mf_minibatch_train(model, train_pos, x, optimizer, batch_size, adj_t)
        return _mf_minibatch_train(model, train_pos, x, optimizer, batch_size, adj_t)
    if bool(getattr(model, "planetoid_target_edge_masked", False)):
        path = "planetoid-advanced-target-edge-masked"
        if getattr(model, "_pyg_reported_training_path", None) != path:
            print(f"effective_training_path={path}", flush=True)
            print(f"training_protocol={getattr(model, 'training_protocol', path)}", flush=True)
            print(f"protocol_fidelity={getattr(model, 'protocol_fidelity', 'scalable-implementation')}", flush=True)
            print(f"train_negative_protocol={getattr(model, 'train_negative_protocol', 'strict-unobserved-nonself-edge')}", flush=True)
            model._pyg_reported_training_path = path
        return _planetoid_advanced_minibatch_train(model, train_pos, x, optimizer, batch_size)
    if _reference_minibatch_compatible(model, train_pos, x.size(0)):
        if getattr(model, "_pyg_reported_training_path", None) != "reference":
            print("effective_training_path=reference", flush=True)
            model._pyg_reported_training_path = "reference"
        return _reference_minibatch_train(model, train_pos, x, optimizer, batch_size)
    if adj_t is not None:
        if getattr(model, "_pyg_reported_training_path", None) != "full-graph":
            print("effective_training_path=full-graph", flush=True)
            model._pyg_reported_training_path = "full-graph"
        return _full_graph_train(model, train_pos, x, optimizer, batch_size, adj_t, csr_rowptr, csr_col)
    total_loss = 0.0
    total_examples = 0
    num_nodes = x.size(0)
    full_edge_index = train_pos.t().contiguous()
    full_edge_index = torch.cat((full_edge_index, full_edge_index[[1, 0]]), dim=1)
    perm_all = torch.randperm(train_pos.size(0), device=train_pos.device)
    for start in range(0, train_pos.size(0), batch_size):
        perm = perm_all[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        mask = torch.ones(train_pos.size(0), dtype=torch.bool, device=train_pos.device)
        mask[perm] = False
        train_edge_mask = train_pos[mask].t().contiguous()
        train_edge_mask = torch.cat((train_edge_mask, train_edge_mask[[1, 0]]), dim=1)
        edge_weight_mask = torch.ones(train_edge_mask.size(1), dtype=torch.float32, device=train_pos.device)
        adj_batch = SparseTensor.from_edge_index(train_edge_mask, edge_weight_mask, (num_nodes, num_nodes)).to(train_pos.device)
        z = model.embed(_make_graph_data(x, adj_batch))
        pos_edge = train_pos[perm].t().contiguous()
        pos_logits = model.decode(z, pos_edge).view(-1)
        neg_edge = negative_sampling(full_edge_index, num_nodes=num_nodes, num_neg_samples=pos_edge.size(1), method="sparse").to(
            pos_logits.device
        )
        neg_logits = model.decode(z, neg_edge).view(-1)
        logits = torch.cat([pos_logits, neg_logits], dim=0)
        labels = torch.cat([torch.ones_like(pos_logits), torch.zeros_like(neg_logits)], dim=0)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        if hasattr(model, "auxiliary_loss"):
            loss = loss + model.auxiliary_loss(pos_edge)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        batch_n = pos_edge.size(1)
        total_loss += float(loss.item()) * batch_n
        total_examples += batch_n
    return total_loss / max(total_examples, 1)
