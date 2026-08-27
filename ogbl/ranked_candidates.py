from contextlib import contextmanager
import os
import uuid
import torch
from torch_sparse import SparseTensor
from .data_core import _heart_ppr_eps
from .fast_negatives import RaggedGroupedNegativeEdges
from .heart_sparse import _ppr_push_scores_for_node, _ra_sparse_scores_for_node, _rank_positive_scores

_RAGGED_CACHE_VERSION = 2
_PPR_ALPHA = 0.15
_DEFAULT_PPR_ITERS = 25


def _rank_valid_endpoint_candidates(
    node,
    score_rowptr,
    score_col,
    degree,
    inv_degree,
    invalid_rowptr,
    invalid_col,
    max_candidates,
    generator,
    ppr_eps,
    max_pushes,
    filter_existing,
):
    node = int(node)
    (ra_ids, ra_values) = _ra_sparse_scores_for_node(node, score_rowptr, score_col, inv_degree)
    (ppr_ids, ppr_values) = _ppr_push_scores_for_node(node, score_rowptr, score_col, degree, eps=float(ppr_eps), max_pushes=max_pushes)
    invalid = torch.tensor([node], dtype=torch.long)
    if filter_existing:
        (start, end) = (int(invalid_rowptr[node]), int(invalid_rowptr[node + 1]))
        if end > start:
            invalid = torch.cat([invalid, invalid_col[start:end].to(torch.long)])
    invalid = invalid.unique()
    if ra_ids.numel():
        keep = ~torch.isin(ra_ids, invalid)
        (ra_ids, ra_values) = (ra_ids[keep], ra_values[keep])
    if ppr_ids.numel():
        keep = ~torch.isin(ppr_ids, invalid)
        (ppr_ids, ppr_values) = (ppr_ids[keep], ppr_values[keep])
    (ra_ids, ra_ranks) = _rank_positive_scores(ra_ids, ra_values)
    (ppr_ids, ppr_ranks) = _rank_positive_scores(ppr_ids, ppr_values)
    if ra_ids.numel() and ppr_ids.numel():
        candidates = torch.unique(torch.cat([ra_ids, ppr_ids]))
    elif ra_ids.numel():
        candidates = ra_ids.unique()
    elif ppr_ids.numel():
        candidates = ppr_ids.unique()
    else:
        return torch.empty(0, dtype=torch.long)
    ra_default = int(ra_ranks.max().item()) + 1 if ra_ranks.numel() else 1
    ppr_default = int(ppr_ranks.max().item()) + 1 if ppr_ranks.numel() else 1
    combined = torch.full((candidates.numel(),), min(ra_default, ppr_default), dtype=torch.long)
    for ids, ranks in ((ra_ids, ra_ranks), (ppr_ids, ppr_ranks)):
        if not ids.numel():
            continue
        order = torch.argsort(ids)
        positions = torch.searchsorted(ids[order], candidates)
        inside = positions < ids.numel()
        matched = torch.zeros_like(inside)
        matched[inside] = ids[order][positions[inside]] == candidates[inside]
        combined[matched] = torch.minimum(combined[matched], ranks[order][positions[matched]])
    permutation = torch.randperm(candidates.numel(), generator=generator)
    candidates = candidates[permutation]
    combined = combined[permutation]
    order = torch.argsort(combined, stable=True)
    return candidates[order[: min(int(max_candidates), int(order.numel()))]]


def _build_ranked_ragged_split(
    pos, score_rowptr, score_col, invalid_rowptr, invalid_col, max_per_side, seed, ppr_eps, max_pushes, filter_existing
):
    pos = pos.to(torch.long).cpu().contiguous()
    score_rowptr = score_rowptr.to(torch.long).cpu()
    score_col = score_col.to(torch.long).cpu()
    invalid_rowptr = invalid_rowptr.to(torch.long).cpu()
    invalid_col = invalid_col.to(torch.long).cpu()
    generator = torch.Generator().manual_seed(int(seed))
    degree = (score_rowptr[1:] - score_rowptr[:-1]).to(torch.long)
    degree_float = degree.to(torch.float32)
    inv_degree = torch.zeros_like(degree_float)
    inv_degree[degree > 0] = 1.0 / degree_float[degree > 0]
    endpoint_cache = {}
    for index, node in enumerate(torch.unique(pos.reshape(-1)).tolist(), 1):
        endpoint_cache[int(node)] = _rank_valid_endpoint_candidates(
            node,
            score_rowptr,
            score_col,
            degree,
            inv_degree,
            invalid_rowptr,
            invalid_col,
            int(max_per_side) + 1,
            generator,
            ppr_eps,
            max_pushes,
            filter_existing,
        )
        if index % 100 == 0:
            print(f"  ranked {index} endpoint nodes", flush=True)
    return _assemble_ranked_ragged_split(pos, endpoint_cache, max_per_side)


def _assemble_ranked_ragged_split(pos, endpoint_cache, max_per_side):
    pos = pos.to(torch.long).cpu().contiguous()
    flat_parts = []
    offsets = [0]
    side_counts = torch.zeros((pos.size(0), 2), dtype=torch.long)
    for row, (left_endpoint, right_endpoint) in enumerate(pos.tolist()):
        left = endpoint_cache[int(left_endpoint)]
        right = endpoint_cache[int(right_endpoint)]
        left = left[left != int(right_endpoint)][: int(max_per_side)]
        right = right[right != int(left_endpoint)][: int(max_per_side)]
        side_counts[row, 0] = left.numel()
        side_counts[row, 1] = right.numel()
        if left.numel():
            flat_parts.append(torch.stack([torch.full_like(left, int(left_endpoint)), left], dim=1))
        if right.numel():
            flat_parts.append(torch.stack([right, torch.full_like(right, int(right_endpoint))], dim=1))
        offsets.append(offsets[-1] + int(left.numel() + right.numel()))
    flat = torch.cat(flat_parts, dim=0) if flat_parts else torch.empty((0, 2), dtype=torch.long)
    if pos.size(0) and (not flat.numel()):
        raise RuntimeError("No positive-RA/PPR candidates exist for this split.")
    return RaggedGroupedNegativeEdges(
        flat, torch.tensor(offsets, dtype=torch.long), max_per_side=int(max_per_side), side_counts=side_counts
    )


def _require_cuda_device(device):
    return torch.device(device or "cuda")


def _recommended_gpu_batch_size(device, num_nodes):
    try:
        (free_bytes, total_bytes) = torch.cuda.mem_get_info(device)
    except Exception:
        return 32
    bytes_per_endpoint = max(1, int(num_nodes)) * 64
    reserve = min(8 * 1024**3, max(2 * 1024**3, int(total_bytes) // 4))
    budget = max(256 * 1024**2, int(free_bytes) - reserve)
    memory_bound = max(1, budget // bytes_per_endpoint)
    hard_cap = 512 if int(num_nodes) <= 50000 else 128
    return max(1, min(hard_cap, int(memory_bound)))


def _dense_positive_min_ranks(scores):
    (batch_size, num_nodes) = scores.shape
    (sorted_values, order) = torch.sort(scores, dim=1, descending=True)
    positive = sorted_values > 0
    positions = torch.arange(1, num_nodes + 1, dtype=torch.int32, device=scores.device).view(1, -1)
    starts = positive.clone()
    if num_nodes > 1:
        starts[:, 1:] &= sorted_values[:, 1:] != sorted_values[:, :-1]
    first_positions = torch.where(starts, positions, torch.zeros_like(positions))
    rank_sorted = torch.cummax(first_positions, dim=1).values
    ranks = torch.empty((batch_size, num_nodes), dtype=torch.int32, device=scores.device)
    ranks.scatter_(1, order, rank_sorted)
    default_rank = first_positions.amax(dim=1) + 1
    nonpositive = scores <= 0
    ranks[nonpositive] = default_rank.view(-1, 1).expand_as(ranks)[nonpositive]
    return ranks


def _rank_positive_union_gpu(ra, ppr, invalid_nodes_by_column, max_candidates, generator, ppr_eps):
    (num_nodes, batch_size) = ra.shape
    for column, invalid in enumerate(invalid_nodes_by_column):
        invalid = invalid.to(device=ra.device, dtype=torch.long, non_blocking=True)
        invalid = invalid[(invalid >= 0) & (invalid < num_nodes)].unique()
        if invalid.numel():
            ra[invalid, column] = -1.0
            ppr[invalid, column] = -1.0
    ppr[(ppr > 0) & (ppr < float(ppr_eps))] = 0.0
    ra_rows = ra.t().contiguous()
    ppr_rows = ppr.t().contiguous()
    positive_union = (ra_rows > 0) | (ppr_rows > 0)
    combined = torch.minimum(_dense_positive_min_ranks(ra_rows), _dense_positive_min_ranks(ppr_rows))
    outputs = []
    for row in range(batch_size):
        candidates = torch.nonzero(positive_union[row], as_tuple=False).view(-1)
        if not candidates.numel():
            outputs.append(torch.empty(0, dtype=torch.long))
            continue
        permutation = torch.randperm(candidates.numel(), generator=generator, device=ra.device)
        candidates = candidates[permutation]
        candidate_ranks = combined[row, candidates]
        order = torch.argsort(candidate_ranks, stable=True)
        selected = candidates[order[: min(int(max_candidates), int(order.numel()))]]
        outputs.append(selected.to(device="cpu", dtype=torch.long))
    return outputs


def _build_ranked_ragged_split_gpu(
    pos,
    score_rowptr,
    score_col,
    invalid_rowptr,
    invalid_col,
    max_per_side,
    seed,
    ppr_eps,
    filter_existing,
    device,
    endpoint_batch_size,
    ppr_iters,
    split_name,
):
    device = _require_cuda_device(device)
    pos = pos.to(torch.long).cpu().contiguous()
    score_rowptr = score_rowptr.to(torch.long).cpu().contiguous()
    score_col = score_col.to(torch.long).cpu().contiguous()
    invalid_rowptr = invalid_rowptr.to(torch.long).cpu().contiguous()
    invalid_col = invalid_col.to(torch.long).cpu().contiguous()
    num_nodes = int(score_rowptr.numel() - 1)
    ppr_iters = max(1, int(ppr_iters or _DEFAULT_PPR_ITERS))
    recommended = _recommended_gpu_batch_size(device, num_nodes)
    requested = max(1, int(endpoint_batch_size or recommended))
    effective_batch_size = min(requested, recommended)
    if effective_batch_size != requested:
        print(f"GPU endpoint batch reduced from {requested} to {effective_batch_size} for available memory.", flush=True)
    score_rowptr_device = score_rowptr.to(device=device, non_blocking=True)
    score_col_device = score_col.to(device=device, non_blocking=True)
    values = torch.ones(score_col_device.numel(), dtype=torch.float32, device=device)
    adj = SparseTensor(
        rowptr=score_rowptr_device, col=score_col_device, value=values, sparse_sizes=(num_nodes, num_nodes), is_sorted=True, trust_data=True
    )
    degree = (score_rowptr_device[1:] - score_rowptr_device[:-1]).to(torch.float32)
    inv_degree = torch.zeros_like(degree)
    inv_degree[degree > 0] = 1.0 / degree[degree > 0]
    inv_degree_column = inv_degree.view(-1, 1)
    generator = torch.Generator(device=device)
    generator.manual_seed(int(seed))
    unique_nodes = torch.unique(pos.reshape(-1)).to(torch.long).tolist()
    endpoint_cache = {}
    print(
        f"Building GPU ranked candidates for {split_name}: endpoints={len(unique_nodes)} batch={effective_batch_size} ppr_iters={ppr_iters} device={device}",
        flush=True,
    )
    with torch.no_grad():
        for start in range(0, len(unique_nodes), effective_batch_size):
            nodes = [int(node) for node in unique_nodes[start : start + effective_batch_size]]
            batch_size = len(nodes)
            evec = torch.zeros((num_nodes, batch_size), dtype=torch.float32, device=device)
            weights = torch.zeros_like(evec)
            node_tensor = torch.tensor(nodes, dtype=torch.long, device=device)
            column_tensor = torch.arange(batch_size, dtype=torch.long, device=device)
            evec[node_tensor, column_tensor] = 1.0
            invalid_batch = []
            for column, node in enumerate(nodes):
                score_start = int(score_rowptr[node])
                score_end = int(score_rowptr[node + 1])
                if score_end > score_start:
                    neighbors = score_col_device[score_start:score_end]
                    weights[neighbors, column] = inv_degree[neighbors]
                invalid = torch.tensor([node], dtype=torch.long)
                if filter_existing:
                    invalid_start = int(invalid_rowptr[node])
                    invalid_end = int(invalid_rowptr[node + 1])
                    if invalid_end > invalid_start:
                        invalid = torch.cat([invalid, invalid_col[invalid_start:invalid_end]])
                invalid_batch.append(invalid.unique())
            ra = adj.matmul(weights)
            ppr = evec.clone()
            for _ in range(ppr_iters):
                ppr = _PPR_ALPHA * evec + (1.0 - _PPR_ALPHA) * adj.matmul(ppr * inv_degree_column)
            ranked = _rank_positive_union_gpu(ra, ppr, invalid_batch, int(max_per_side) + 1, generator, ppr_eps)
            for node, candidates in zip(nodes, ranked):
                endpoint_cache[node] = candidates
            done = min(start + batch_size, len(unique_nodes))
            print(f"  GPU ranked {done}/{len(unique_nodes)} {split_name} endpoint nodes", flush=True)
            del evec, weights, node_tensor, column_tensor, ra, ppr
    del adj, values, degree, inv_degree, inv_degree_column
    del score_rowptr_device, score_col_device
    torch.cuda.empty_cache()
    return _assemble_ranked_ragged_split(pos, endpoint_cache, max_per_side)


def _cache_path(cache_dir, data_name, seed, eval_cap, max_per_side, backend, nvalid, ntest):
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(
        cache_dir,
        f"{data_name}_ranked_valid_ragged_v{_RAGGED_CACHE_VERSION}_seed{int(seed)}_cap{int(eval_cap or 0)}_max{int(max_per_side)}_{backend}_v{int(nvalid)}_t{int(ntest)}.pt",
    )


def _serialize(pool):
    return {"flat_edges": pool.flat_edges, "rowptr": pool.rowptr, "side_counts": pool.side_counts, "max_per_side": int(pool.max_per_side)}


def _deserialize(payload, expected_rows, max_per_side):
    pool = RaggedGroupedNegativeEdges(
        payload["flat_edges"], payload["rowptr"], int(payload.get("max_per_side", max_per_side)), payload.get("side_counts")
    )
    if pool.num_pos != int(expected_rows) or pool.max_per_side > int(max_per_side):
        raise ValueError("Cached ragged candidate dimensions do not match.")
    return pool


def _try_load_ranked_cache(path, out, max_per_side):
    if not path or not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        valid = _deserialize(payload["valid"], out["valid_pos"].size(0), max_per_side)
        test = _deserialize(payload["test"], out["test_pos"].size(0), max_per_side)
        print(f"Loaded ranked valid candidates from {path}", flush=True)
        return (valid, test)
    except Exception as exc:
        print(f"Ignoring unreadable ranked candidate cache {path}: {exc}", flush=True)
        return None


@contextmanager
def _exclusive_ranked_cache_build(path):
    if not path:
        yield
        return
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, "a+b")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError) as exc:
            print(f"WARNING: cache locking unavailable for {lock_path}: {exc}", flush=True)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        lock_file.close()


def _atomic_save_ranked_cache(path, valid, test):
    temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        torch.save({"valid": _serialize(valid), "test": _serialize(test)}, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass


def _build_ranked_valid_pool_uncached(out, data_name, max_per_side, seed, backend_label, max_pushes, device, batch_size, ppr_iters):
    filter_existing = data_name != "ogbl-collab"
    (train_rowptr, train_col) = (out["csr_train_rowptr"], out["csr_train_col"])
    test_score_rowptr = out["csr_tv_rowptr"] if data_name == "ogbl-collab" else train_rowptr
    test_score_col = out["csr_tv_col"] if data_name == "ogbl-collab" else train_col
    ppr_eps = max(_heart_ppr_eps(data_name), 5e-05)
    print(
        f"Building unpadded ranked-valid candidates max_per_side={max_per_side} backend={backend_label}"
        + (f" device={device}" if backend_label == "gpu" else ""),
        flush=True,
    )
    if backend_label == "gpu":
        valid = _build_ranked_ragged_split_gpu(
            out["valid_pos"],
            train_rowptr,
            train_col,
            train_rowptr,
            train_col,
            max_per_side,
            seed + 1000,
            ppr_eps,
            filter_existing,
            device,
            batch_size,
            ppr_iters,
            "valid",
        )
        test = _build_ranked_ragged_split_gpu(
            out["test_pos"],
            test_score_rowptr,
            test_score_col,
            out["csr_tv_rowptr"],
            out["csr_tv_col"],
            max_per_side,
            seed + 1001,
            ppr_eps,
            filter_existing,
            device,
            batch_size,
            ppr_iters,
            "test",
        )
    else:
        valid = _build_ranked_ragged_split(
            out["valid_pos"],
            train_rowptr,
            train_col,
            train_rowptr,
            train_col,
            max_per_side,
            seed + 1000,
            ppr_eps,
            max_pushes,
            filter_existing,
        )
        test = _build_ranked_ragged_split(
            out["test_pos"],
            test_score_rowptr,
            test_score_col,
            out["csr_tv_rowptr"],
            out["csr_tv_col"],
            max_per_side,
            seed + 1001,
            ppr_eps,
            max_pushes,
            filter_existing,
        )
    return (valid, test)


def load_or_build_ranked_valid_pool(
    out, data_name, max_per_side, seed, eval_cap, backend, cache_dir, cache_enabled, *, device=None, batch_size=None, ppr_iters=None
):
    backend = str(backend or "auto").strip().lower()
    if backend in {"gpu", "cuda", "batched", "exact-batched"}:
        backend_label = "gpu"
        max_pushes = None
        device = _require_cuda_device(device)
        ppr_iters = max(1, int(ppr_iters or _DEFAULT_PPR_ITERS))
        train_nnz = int(out["csr_train_col"].numel())
        train_valid_nnz = int(out["csr_tv_col"].numel())
        cache_backend_label = f"gpu-power{ppr_iters}-tr{train_nnz}-tv{train_valid_nnz}"
    elif backend in {"fast", "approx", "approximate"}:
        backend_label = "fast"
        max_pushes = 200000
        cache_backend_label = backend_label
    elif backend in {"auto", "official", "reference", "dense"}:
        backend_label = "official"
        max_pushes = None
        cache_backend_label = backend_label
    else:
        raise ValueError(f"Unknown ranked-candidate backend {backend!r}; expected gpu, official, or fast.")
    path = _cache_path(
        cache_dir, data_name, seed, eval_cap, max_per_side, cache_backend_label, out["valid_pos"].size(0), out["test_pos"].size(0)
    )
    if cache_enabled:
        cached = _try_load_ranked_cache(path, out, max_per_side)
        if cached is not None:
            return (cached[0], cached[1], backend_label, path)
    with _exclusive_ranked_cache_build(path if cache_enabled else None):
        if cache_enabled:
            cached = _try_load_ranked_cache(path, out, max_per_side)
            if cached is not None:
                return (cached[0], cached[1], backend_label, path)
        (valid, test) = _build_ranked_valid_pool_uncached(
            out, data_name, max_per_side, seed, backend_label, max_pushes, device, batch_size, ppr_iters
        )
        if cache_enabled and path:
            _atomic_save_ranked_cache(path, valid, test)
            print(f"Saved ranked valid candidates to {path}", flush=True)
        return (valid, test, backend_label, path)
