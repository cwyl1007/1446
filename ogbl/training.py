import math
import weakref
import torch
import torch.nn.functional as F
from torch_sparse import SparseTensor
from model.decoder_training import train_cached_decoder_minibatches
from .fast_negatives import (
    EndpointCorruptionGroupedNegativeEdges,
    is_streaming_negative_edges,
    make_forbidden_edge_ids,
    sample_global_negative_edges,
)
from .mf_protocol import reference_mf_batch_count, reference_mf_cpu_shuffle_batches, reference_mf_decode_batch

_PPA_TRAIN_SAMPLES = {
    "mf": 0,
    "gae": 0,
    "mlp": 0,
    "mlpip": 0,
    "ppr": 0,
    "concat": 0,
    "concatip": 0,
    "gcn": 0,
    "gat": 0,
    "sage": 0,
    "peg": 524288,
    "neo-gnn": 262144,
    "neo_gnn": 262144,
    "neognn": 262144,
    "buddy": 131072,
    "ncn": 131072,
    "nbfnet": 131072,
    "seal": 65536,
    "ncnc": 65536,
    "lpformer": 32768,
    "lp-former": 32768,
    "lp_former": 32768,
    "lpf": 32768,
}
_PPA_DECODE_BATCH = {
    "mf": 1048576,
    "gae": 1048576,
    "mlp": 524288,
    "mlpip": 524288,
    "ppr": 524288,
    "concat": 524288,
    "concatip": 524288,
    "gcn": 524288,
    "gat": 524288,
    "sage": 524288,
    "peg": 262144,
    "neo-gnn": 131072,
    "neo_gnn": 131072,
    "neognn": 131072,
    "buddy": 131072,
    "ncn": 131072,
    "nbfnet": 131072,
    "seal": 131072,
    "ncnc": 65536,
    "lpformer": 16384,
    "lp-former": 16384,
    "lp_former": 16384,
    "lpf": 16384,
}


def make_ogbl_optimizer(model, lr, weight_decay, device):
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if bool(getattr(model, "reference_optimizer", False)):
        kwargs["foreach"] = False
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if str(device).startswith("cuda") and (not bool(getattr(model, "reference_optimizer", False))):
        try:
            return torch.optim.Adam(parameters, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(parameters, **kwargs)


def recommended_train_samples_per_epoch(dataset_name, model_name):
    dataset_key = str(dataset_name).strip().lower()
    if dataset_key != "ogbl-ppa":
        return 0
    return int(_PPA_TRAIN_SAMPLES.get(str(model_name).strip().lower(), 1048576))


def recommended_decode_batch_size(dataset_name, model_name, default=65536):
    if str(dataset_name).strip().lower() != "ogbl-ppa":
        return int(default)
    return int(_PPA_DECODE_BATCH.get(str(model_name).strip().lower(), default))


def find_result_key(results, metric_name):
    want = str(metric_name).strip().lower()
    if not want:
        return None
    aliases = {want}
    if want.startswith("hits@"):
        suffix = want.split("@", 1)[1]
        if suffix.isdigit():
            aliases.add(f"mrr_hit{int(suffix)}")
    elif want.startswith("mrr_hit"):
        suffix = want[len("mrr_hit") :]
        if suffix.isdigit():
            aliases.add(f"hits@{int(suffix)}")
    for key in results.keys():
        if str(key).strip().lower() in aliases:
            return str(key)
    return None


def _clear_gcnconv_cache(model):
    for module in model.modules():
        if hasattr(module, "_cached_edge_index"):
            module._cached_edge_index = None
        if hasattr(module, "_cached_adj_t"):
            module._cached_adj_t = None


def move_graph_data_to_device(data, device):
    device = torch.device(device)
    if getattr(data, "_lp_graph_device", None) == str(device):
        return data
    x = getattr(data, "x", None)
    if x is not None and x.device != device:
        data.x = x.to(device, non_blocking=True)
    adj_t = getattr(data, "adj_t", None)
    if adj_t is not None:
        try:
            data.adj_t = adj_t.to(device)
        except TypeError:
            data.adj_t = adj_t.to(str(device))
    data._lp_graph_device = str(device)
    return data


def _tensor_nbytes(tensor):
    return int(tensor.numel()) * int(tensor.element_size())


_AUTO_CACHED_EVAL_KEYS = frozenset({"pos_train_edge", "train_val_edge", "pos_valid_edge", "neg_valid_edge"})


def cache_eval_edges_on_device(eval_edges, device, option="auto", max_bytes=8 * 1024**3):
    option = str(option or "auto").strip().lower()
    device = torch.device(device)
    if option in ("0", "false", "no", "off") or device.type != "cuda":
        return (dict(eval_edges), False, 0)
    all_cacheable = {
        key: value
        for (key, value) in eval_edges.items()
        if torch.is_tensor(value) and (not is_streaming_negative_edges(value)) or isinstance(value, EndpointCorruptionGroupedNegativeEdges)
    }
    force = option in ("1", "true", "yes", "on")
    tensors = all_cacheable if force else {key: value for (key, value) in all_cacheable.items() if key in _AUTO_CACHED_EVAL_KEYS}
    total_bytes = sum(
        (
            _tensor_nbytes(value) if torch.is_tensor(value) else int(value.storage_nbytes)
            for value in tensors.values()
            if value.device != device
        )
    )
    if total_bytes == 0:
        return (dict(eval_edges), True, 0)
    budget = int(max_bytes)
    try:
        (free_bytes, _) = torch.cuda.mem_get_info(device)
        budget = min(budget, int(free_bytes * 0.2))
    except Exception:
        pass
    if not force and total_bytes > budget:
        return (dict(eval_edges), False, total_bytes)
    cached = dict(eval_edges)
    try:
        for key, value in tensors.items():
            if value.device != device:
                if isinstance(value, EndpointCorruptionGroupedNegativeEdges):
                    cached[key] = value.cache_on_device(device=device, non_blocking=True)
                else:
                    cached[key] = value.to(device=device, non_blocking=True)
        return (cached, True, total_bytes)
    except RuntimeError as exc:
        if "out of memory" not in str(exc).lower() or force:
            raise
        del cached
        torch.cuda.empty_cache()
        return (dict(eval_edges), False, total_bytes)


def _model_is_mf(model):
    encoder = getattr(model, "encoder", None)
    return encoder is not None and encoder.__class__.__name__ == "MFEncoder"


_SHARED_FORBIDDEN_TRAIN_IDS = {}


def _cached_forbidden_train_ids(model, pos_train_edge, num_nodes, device):
    directed = bool(getattr(model, "directed", False))
    key = (int(pos_train_edge.data_ptr()), int(pos_train_edge.size(1)), int(num_nodes), device.type, device.index, directed)
    model_key = getattr(model, "_ogbl_forbidden_key", None)
    if model_key == key:
        return model._ogbl_forbidden_ids
    shared = _SHARED_FORBIDDEN_TRAIN_IDS.get(key)
    if shared is not None and shared[0]() is pos_train_edge:
        model._ogbl_forbidden_ids = shared[1]
        model._ogbl_forbidden_key = key
    else:
        (src, dst) = pos_train_edge
        if directed:
            ids = torch.sort(src * int(num_nodes) + dst).values
        else:
            lo = torch.minimum(src, dst)
            hi = torch.maximum(src, dst)
            ids = torch.sort(lo * int(num_nodes) + hi).values
        model._ogbl_forbidden_ids = torch.unique_consecutive(ids)
        model._ogbl_forbidden_key = key
        _SHARED_FORBIDDEN_TRAIN_IDS[key] = (
            weakref.ref(pos_train_edge, lambda _ref, cache_key=key: _SHARED_FORBIDDEN_TRAIN_IDS.pop(cache_key, None)),
            model._ogbl_forbidden_ids,
        )
    return model._ogbl_forbidden_ids


def _sample_exact_gpu_negatives(model, pos_train_edge, num_nodes, num_samples, device, seed):
    forbidden = _cached_forbidden_train_ids(model, pos_train_edge, num_nodes, device)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    target = int(num_samples)
    parts = []
    filled = 0
    nodes = max(1, int(num_nodes))
    pair_space = float(nodes) * float(nodes)
    invalid_probability = min(0.95, (float(nodes) + 2.0 * float(forbidden.numel())) / pair_space)
    valid_probability = max(0.05, 1.0 - invalid_probability)
    while filled < target:
        remaining = target - filled
        expected_draw = int(math.ceil(remaining / valid_probability))
        expected_invalid = max(1.0, float(expected_draw) * invalid_probability)
        safety = int(math.ceil(8.0 * math.sqrt(expected_invalid))) + 64
        draw = max(4096, min(4 * remaining + 64, expected_draw + safety))
        src = torch.randint(0, int(num_nodes), (draw,), device=device, generator=generator)
        dst = torch.randint(0, int(num_nodes), (draw,), device=device, generator=generator)
        directed = bool(getattr(model, "directed", False))
        if directed:
            ids = src * int(num_nodes) + dst
            valid = src != dst
        else:
            lo = torch.minimum(src, dst)
            hi = torch.maximum(src, dst)
            ids = lo * int(num_nodes) + hi
            valid = lo != hi
        pos = torch.searchsorted(forbidden, ids)
        inside = pos < forbidden.numel()
        if forbidden.numel() > 0:
            safe_pos = pos.clamp_max(forbidden.numel() - 1)
            valid &= ~(inside & (forbidden[safe_pos] == ids))
        keep = torch.nonzero(valid, as_tuple=False).view(-1)
        if keep.numel() > 0:
            take = min(remaining, int(keep.numel()))
            keep = keep[:take]
            parts.append(torch.stack([src[keep], dst[keep]], dim=0))
            filled += take
    return torch.cat(parts, dim=1)[:, :target].contiguous()


def _sample_training_negative_edges(model, data, pos_train_edge, num_nodes, num_samples, device, sampler, seed):
    sampler = str(sampler or "auto").strip().lower()
    if bool(getattr(model, "shared_ip_training_contract", False)) or bool(
        getattr(model, "ranked_selector_training_contract", False)
    ):
        return _sample_exact_gpu_negatives(model, pos_train_edge, num_nodes, num_samples, device, seed)
    if bool(getattr(model, "reference_random_endpoint_negatives", False)):
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed))
        return torch.randint(0, int(num_nodes), (2, int(num_samples)), dtype=torch.long, device=device, generator=generator)
    if sampler == "auto":
        sampler = "fast" if int(num_nodes) >= 100000 else "pyg"
    if sampler in ("random", "simple", "uniform"):
        return torch.randint(0, int(num_nodes), (2, int(num_samples)), dtype=torch.long, device=device)
    if sampler == "fast":
        if device.type == "cuda" or str(getattr(model, "dataset_name", "")).lower() == "ogbl-citation2":
            return _sample_exact_gpu_negatives(model, pos_train_edge, num_nodes, num_samples, device, seed)
        forbidden = getattr(model, "_ogbl_forbidden_cpu", None)
        if forbidden is None:
            edge_index = getattr(data, "edge_index", None)
            if edge_index is None:
                (row, col, _) = data.adj_t.coo()
                edge_index = torch.stack([row, col], dim=0).cpu()
            forbidden = make_forbidden_edge_ids(edge_index.cpu(), int(num_nodes))
            model._ogbl_forbidden_cpu = forbidden
        neg = sample_global_negative_edges(int(num_nodes), int(num_samples), forbidden, int(seed), strict=True)
        return neg.t().contiguous().to(device, non_blocking=True)
    from torch_geometric.utils import negative_sampling

    method = "dense" if int(num_nodes) <= 6000 else "sparse"
    edge_index_dev = getattr(model, "_ogbl_edge_index_dev", None)
    if edge_index_dev is None or edge_index_dev.device != device:
        edge_index = getattr(data, "edge_index", None)
        if edge_index is None:
            (row, col, _) = data.adj_t.coo()
            edge_index = torch.stack([row, col], dim=0)
        edge_index_dev = edge_index.to(device, non_blocking=True)
        model._ogbl_edge_index_dev = edge_index_dev
    try:
        return negative_sampling(edge_index=edge_index_dev, num_nodes=int(num_nodes), num_neg_samples=int(num_samples), method=method)
    except Exception:
        edge_index_cpu = getattr(model, "_ogbl_edge_index_cpu", None)
        if edge_index_cpu is None:
            edge_index_cpu = edge_index_dev.cpu()
            model._ogbl_edge_index_cpu = edge_index_cpu
        return negative_sampling(edge_index=edge_index_cpu, num_nodes=int(num_nodes), num_neg_samples=int(num_samples), method=method).to(
            device, non_blocking=True
        )


def _backward_bce_edge_loss(model, z, pos_edge, neg_edge, decode_batch_size, auxiliary_loss=None):
    npos = int(pos_edge.size(1))
    nneg = int(neg_edge.size(1))
    if npos == 0 or nneg == 0:
        return 0.0
    decode_batch_size = int(decode_batch_size) if decode_batch_size else max(npos, nneg)
    decode_batch_size = max(1, decode_batch_size)
    pos_chunks = int(math.ceil(npos / decode_batch_size))
    neg_chunks = int(math.ceil(nneg / decode_batch_size))
    total_chunks = pos_chunks + neg_chunks
    chunk_id = 0
    loss_value = 0.0
    auxiliary_pending = auxiliary_loss
    for start in range(0, npos, decode_batch_size):
        end = min(start + decode_batch_size, npos)
        logits = model.decode(z, pos_edge[:, start:end]).view(-1)
        if bool(getattr(model, "reference_probability_loss", False)):
            probability = torch.sigmoid(logits.float())
            loss = -torch.log(probability + 1e-15).sum() / float(npos)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, torch.ones_like(logits), reduction="sum") / float(npos)
        if auxiliary_pending is not None:
            loss = loss + auxiliary_pending
            auxiliary_pending = None
        loss_value += float(loss.detach().item())
        chunk_id += 1
        loss.backward(retain_graph=chunk_id < total_chunks)
    for start in range(0, nneg, decode_batch_size):
        end = min(start + decode_batch_size, nneg)
        logits = model.decode(z, neg_edge[:, start:end]).view(-1)
        if bool(getattr(model, "reference_probability_loss", False)):
            probability = torch.sigmoid(logits.float())
            loss = -torch.log(1.0 - probability + 1e-15).sum() / float(nneg)
        else:
            loss = F.binary_cross_entropy_with_logits(logits, torch.zeros_like(logits), reduction="sum") / float(nneg)
        loss_value += float(loss.detach().item())
        chunk_id += 1
        loss.backward(retain_graph=chunk_id < total_chunks)
    return loss_value


_SHARED_POS_TRAIN_EDGES = {}


def _cached_pos_train_edges(model, pos_train_edge, device):
    expected = int(pos_train_edge.size(0)) if pos_train_edge.dim() == 2 and pos_train_edge.size(-1) == 2 else int(pos_train_edge.size(1))
    cached = getattr(model, "_pos_train_edge_cache", None)
    key = (int(pos_train_edge.data_ptr()), expected, device.type, device.index)
    if cached is not None and getattr(model, "_pos_train_edge_key", None) == key:
        return cached
    shared = _SHARED_POS_TRAIN_EDGES.get(key)
    if shared is not None and shared[0]() is pos_train_edge:
        model._pos_train_edge_cache = shared[1]
        model._pos_train_edge_key = key
    else:
        edges = pos_train_edge.to(device=device, dtype=torch.long, non_blocking=True)
        if edges.dim() == 2 and edges.size(1) == 2:
            edges = edges.t().contiguous()
        model._pos_train_edge_cache = edges
        model._pos_train_edge_key = key
        _SHARED_POS_TRAIN_EDGES[key] = (
            weakref.ref(pos_train_edge, lambda _ref, cache_key=key: _SHARED_POS_TRAIN_EDGES.pop(cache_key, None)),
            edges,
        )
    return model._pos_train_edge_cache


def _sample_epoch_positive_edges(model, pos_train_edge, batch_size, max_batches, seed):
    n = int(pos_train_edge.size(1))
    cap = max(0, int(getattr(model, "train_samples_per_epoch", 0)))
    if max_batches is not None and int(max_batches) > 0:
        max_batch_examples = int(max_batches) * max(1, int(batch_size))
        cap = max_batch_examples if cap <= 0 else min(cap, max_batch_examples)
    sample_count = n if cap <= 0 else min(n, cap)
    if sample_count == n:
        return pos_train_edge
    generator = torch.Generator(device=pos_train_edge.device)
    generator.manual_seed(int(seed))
    if sample_count * 10 <= n:
        selected = torch.empty(0, dtype=torch.long, device=pos_train_edge.device)
        while selected.numel() < sample_count:
            need = sample_count - int(selected.numel())
            draw = max(4096, int(math.ceil(need * 1.08)) + 64)
            candidate = torch.randint(0, n, (draw,), device=pos_train_edge.device, generator=generator)
            selected = torch.unique(torch.cat([selected, candidate]), sorted=False)
        if selected.numel() > sample_count:
            order = torch.randperm(selected.numel(), device=pos_train_edge.device, generator=generator)[:sample_count]
            index = selected[order]
        else:
            index = selected
    else:
        index = torch.randperm(n, device=pos_train_edge.device, generator=generator)[:sample_count]
    return pos_train_edge[:, index].contiguous()


def _clear_model_decode_cache(model):
    clear = getattr(model, "clear_decode_cache", None)
    if callable(clear):
        clear()


def _reference_minibatch_compatible(model, data, pos_train_edge):
    if _model_is_mf(model):
        return False
    if getattr(model, "encoder", None).__class__.__name__ == "MLPEncoder":
        return False
    if getattr(model, "decoder", None).__class__.__name__ == "DotProductDecoder":
        return False
    if model.__class__.__name__ != "LinkPredictor":
        return False
    return int(data.num_nodes) <= 10000 and int(pos_train_edge.size(1)) <= 2000000


def _reference_minibatch_train_ogbl(
    model, optimizer, data, pos_train_edge, device, batch_size, decode_bs, show_batch_progress, seed, max_batches
):
    from torch_geometric.utils import negative_sampling

    nedge = int(pos_train_edge.size(1))
    if nedge == 0:
        return 0.0
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    order = torch.randperm(nedge, device=device, generator=generator)
    starts = range(0, nedge, max(1, int(batch_size)))
    if max_batches is not None and int(max_batches) > 0:
        starts = list(starts)[: int(max_batches)]
    if show_batch_progress:
        from tqdm import tqdm

        starts = tqdm(
            starts,
            total=len(starts) if isinstance(starts, list) else int(math.ceil(nedge / batch_size)),
            desc="reference train batches",
            leave=False,
        )
    total_loss = 0.0
    total = 0
    for start in starts:
        end = min(start + int(batch_size), nedge)
        batch_index = order[start:end]
        keep_index = torch.cat([order[:start], order[end:]], dim=0)
        masked_pos = pos_train_edge[:, keep_index]
        masked_edge_index = torch.cat([masked_pos, masked_pos.flip(0)], dim=1)
        masked_adj = SparseTensor.from_edge_index(
            masked_edge_index,
            torch.ones(masked_edge_index.size(1), dtype=torch.float32, device=device),
            (int(data.num_nodes), int(data.num_nodes)),
        )
        batch_pos = pos_train_edge[:, batch_index]
        method = "dense" if int(data.num_nodes) <= 6000 else "sparse"
        if bool(getattr(model, "reference_random_endpoint_negatives", False)):
            batch_neg = torch.randint(0, int(data.num_nodes), batch_pos.size(), dtype=torch.long, device=device, generator=generator)
        else:
            batch_neg = negative_sampling(
                edge_index=masked_edge_index, num_nodes=int(data.num_nodes), num_neg_samples=int(batch_pos.size(1)), method=method
            )
        graph = type("MaskedTrainGraph", (), {})()
        graph.x = data.x
        graph.num_nodes = int(data.num_nodes)
        graph.edge_index = masked_edge_index
        graph.adj_t = masked_adj
        optimizer.zero_grad(set_to_none=True)
        _clear_gcnconv_cache(model)
        _clear_model_decode_cache(model)
        z = model.embed(graph)
        auxiliary = model.auxiliary_loss(batch_pos) if hasattr(model, "auxiliary_loss") else None
        loss_value = _backward_bce_edge_loss(model, z, batch_pos, batch_neg, decode_bs, auxiliary_loss=auxiliary)
        trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        _clear_model_decode_cache(model)
        count = int(batch_pos.size(1))
        total_loss += float(loss_value) * count
        total += count
    _clear_gcnconv_cache(model)
    return total_loss / max(total, 1)


def _mf_minibatch_train_ogbl(
    model, optimizer, data, pos_train_edge, device, batch_size, negative_sampler, seed, max_batches, show_batch_progress
):
    reference_mf = bool(getattr(model, "reference_ogbl_mf", False))
    num_nodes = int(data.num_nodes)
    pos_epoch = _sample_epoch_positive_edges(model, pos_train_edge, batch_size, max_batches, seed)
    n = int(pos_epoch.size(1))
    if n == 0:
        return 0.0
    if reference_mf:
        batches = reference_mf_cpu_shuffle_batches(n, batch_size, max_batches=max_batches)
        batch_count = reference_mf_batch_count(n, batch_size, max_batches=max_batches)
        model.reference_mf_rng_protocol = "continuous-cpu-dataloader-shuffle+continuous-device-negative-rng"
    else:
        generator = torch.Generator(device=device)
        generator.manual_seed(int(seed) + 29)
        order = torch.randperm(n, device=device, generator=generator)
        ranges = range(0, n, max(1, int(batch_size)))
        batches = (order[start : min(start + max(1, int(batch_size)), n)] for start in ranges)
        batch_count = int(math.ceil(n / max(1, int(batch_size))))
    if show_batch_progress:
        from tqdm import tqdm

        batches = tqdm(batches, total=batch_count, desc="train batches", leave=False)
    z = model.embed(data)
    total_loss = 0.0
    total = 0
    for index in batches:
        if index.device != device:
            index = index.to(device=device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        pos_edge = pos_epoch[:, index]
        if reference_mf:
            (pos_logits, neg_logits) = reference_mf_decode_batch(model, z, pos_edge, num_nodes, device)
        else:
            neg_edge = _sample_training_negative_edges(
                model, data, pos_train_edge, num_nodes, int(pos_edge.size(1)), device, negative_sampler, seed + 17 + total
            )
            pos_logits = model.decode(z, pos_edge).view(-1)
            neg_logits = model.decode(z, neg_edge).view(-1)
        if reference_mf:
            pos_probability = torch.sigmoid(pos_logits.float())
            neg_probability = torch.sigmoid(neg_logits.float())
            loss = -torch.log(pos_probability + 1e-15).mean()
            loss = loss - torch.log(1.0 - neg_probability + 1e-15).mean()
        else:
            loss = F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
            loss = loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
        loss.backward()
        if not reference_mf:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        count = int(pos_edge.size(1))
        total_loss += float(loss.detach().item()) * count
        total += count
    return total_loss / max(total, 1)


def _dense_gae_train_ogbl(model, optimizer, data):
    optimizer.zero_grad(set_to_none=True)
    _clear_model_decode_cache(model)
    z = model.embed(data)
    logits = torch.mm(z, z.t())
    adj_t = getattr(data, "adj_t", None)
    if adj_t is not None:
        target = adj_t.to_dense()
    else:
        edge_index = data.edge_index
        target = logits.new_zeros(logits.shape)
        target[edge_index[0], edge_index[1]] = 1.0
    target = (target != 0).to(dtype=logits.dtype, device=logits.device)
    loss = F.binary_cross_entropy_with_logits(logits, target)
    loss.backward()
    trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
    if trainable:
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
    optimizer.step()
    _clear_model_decode_cache(model)
    return float(loss.detach().item())


def _legacy_minibatch_train_ogbl(
    model, optimizer, data, pos_train_edge, device, batch_size, negative_sampler, decode_bs, show_batch_progress, seed, max_batches
):
    n = int(pos_train_edge.size(1))
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    perm = torch.randperm(n, device=device, generator=generator)
    ranges = range(0, n, batch_size)
    if show_batch_progress:
        from tqdm import tqdm

        ranges = tqdm(ranges, total=int(math.ceil(n / batch_size)), desc="train batches", leave=False)
    total_loss = 0.0
    total = 0
    batch_id = 0
    max_batches = None if max_batches is None or int(max_batches) <= 0 else int(max_batches)
    for start in ranges:
        if max_batches is not None and batch_id >= max_batches:
            break
        idx = perm[start : start + batch_size]
        optimizer.zero_grad(set_to_none=True)
        _clear_model_decode_cache(model)
        z = model.embed(data)
        pos_edge = pos_train_edge[:, idx]
        neg_edge = _sample_training_negative_edges(
            model,
            data,
            pos_train_edge,
            int(data.num_nodes),
            pos_edge.size(1),
            device,
            negative_sampler,
            int(seed) + int(start) + batch_id * 1000003,
        )
        auxiliary_loss = model.auxiliary_loss(pos_edge) if hasattr(model, "auxiliary_loss") else None
        loss_value = _backward_bce_edge_loss(model, z, pos_edge, neg_edge, decode_bs, auxiliary_loss=auxiliary_loss)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        _clear_model_decode_cache(model)
        batch_examples = int(pos_edge.size(1))
        total_loss += float(loss_value) * batch_examples
        total += batch_examples
        batch_id += 1
    return total_loss / total if total else 0.0


def train_one_epoch_ogbl(
    model,
    optimizer,
    data,
    pos_train_edge,
    device,
    batch_size=65536,
    negative_sampler="auto",
    train_decode_batch_size=None,
    show_batch_progress=False,
    seed=0,
    max_batches=None,
    training_path="auto",
):
    model.train()
    device = torch.device(device)
    data = move_graph_data_to_device(data, device)
    num_nodes = int(data.num_nodes)
    batch_size = max(1, int(batch_size))
    decode_bs = max(1, int(train_decode_batch_size or batch_size))
    full_pos = _cached_pos_train_edges(model, pos_train_edge, device)
    if bool(getattr(model, "ogbl_dense_gae", False)):
        path = "dense-full-adjacency"
        if getattr(model, "_ogbl_reported_training_path", None) != path:
            print(f"effective_training_path={path}", flush=True)
            model._ogbl_reported_training_path = path
        model.effective_training_path = path
        return _dense_gae_train_ogbl(model, optimizer, data)
    path = str(training_path or "auto").strip().lower().replace("_", "-")
    if path == "auto":
        path = "reference" if _reference_minibatch_compatible(model, data, full_pos) else "full-graph"
        if getattr(model, "_ogbl_reported_training_path", None) != path:
            print(f"effective_training_path={path}", flush=True)
            model._ogbl_reported_training_path = path
    model.effective_training_path = path
    if path in ("reference", "masked"):
        return _reference_minibatch_train_ogbl(
            model, optimizer, data, full_pos, device, batch_size, decode_bs, show_batch_progress, seed, max_batches
        )
    if path in ("legacy", "minibatch", "mini-batch"):
        return _legacy_minibatch_train_ogbl(
            model, optimizer, data, full_pos, device, batch_size, negative_sampler, decode_bs, show_batch_progress, seed, max_batches
        )
    if _model_is_mf(model):
        return _mf_minibatch_train_ogbl(
            model, optimizer, data, full_pos, device, batch_size, negative_sampler, seed, max_batches, show_batch_progress
        )
    pos_epoch = _sample_epoch_positive_edges(model, full_pos, batch_size, max_batches, seed)
    n = int(pos_epoch.size(1))
    if n == 0:
        return 0.0
    neg_epoch = _sample_training_negative_edges(model, data, full_pos, num_nodes, n, device, negative_sampler, seed + 17)
    optimizer.zero_grad(set_to_none=True)
    _clear_model_decode_cache(model)
    z_graph = model.embed(data)
    auxiliary = model.auxiliary_loss(pos_epoch) if hasattr(model, "auxiliary_loss") else None
    decode_micro_batch_size = None
    model_mode = str(getattr(model, "mode", "")).strip().lower().replace("-", "").replace("_", "")
    if model_mode == "lpformer":
        decode_micro_batch_size = min(decode_bs, max(1, int(getattr(model, "decode_batch_size", decode_bs))))
        batch_plan = (decode_bs, decode_micro_batch_size)
        if getattr(model, "_ogbl_reported_decoder_batch_plan", None) != batch_plan:
            print(f"effective_decoder_optimizer_batch_size={decode_bs}", flush=True)
            print(f"effective_decoder_micro_batch_size={decode_micro_batch_size}", flush=True)
            print("deferred_neighbor_sum_backward=true", flush=True)
            model._ogbl_reported_decoder_batch_plan = batch_plan
    loss_value = train_cached_decoder_minibatches(
        model,
        optimizer,
        z_graph,
        pos_epoch,
        neg_epoch,
        auxiliary_loss=auxiliary,
        update_batch_size=decode_bs,
        decode_micro_batch_size=decode_micro_batch_size,
        positive_weight=1.0,
        negative_weight=1.0,
        clear_decode_cache=_clear_model_decode_cache,
        seed=seed + 29,
        show_progress=show_batch_progress,
        progress_desc="cached decoder batches",
    )
    _clear_model_decode_cache(model)
    return float(loss_value)
