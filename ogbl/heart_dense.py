from contextlib import contextmanager, suppress
from functools import lru_cache
import hashlib
import os
import time
import uuid
import numba
import numpy as np
import torch
from torch_sparse import SparseTensor

try:
    from numba import cuda as numba_cuda
except (ImportError, AttributeError):
    numba_cuda = None
_ANDERSEN_BACKEND_DECISIONS = {}
_HEART_RESUME_SHARD_VERSION = 1
from .data_core import _ensure_heart_eligibility_filters, _fsync_directory, _fsync_file, _heart_ppr_eps
from utils.heart_protocol import GENERATED_HEART_TIE_SEED


@numba.jit(nopython=True, parallel=True, cache=True)
def _andersen_ppr_for_selected_nodes_numba(indptr, indices, out_degree, source_nodes, alpha, eps):
    alpha_eps = alpha * eps
    neighbors = [[0]] * len(source_nodes)
    values = [[0.0]] * len(source_nodes)
    for source_index in numba.prange(len(source_nodes)):
        source = np.int64(source_nodes[source_index])
        p = {source: 0.0}
        residual = {source: alpha}
        queue = [source]
        queued = {source: True}
        while len(queue) > 0:
            node = queue.pop()
            queued[node] = False
            value = residual[node] if node in residual else 0.0
            if node in p:
                p[node] += value
            else:
                p[node] = value
            residual[node] = 0.0
            for neighbor in indices[indptr[node] : indptr[node + 1]]:
                increment = (1.0 - alpha) * value / out_degree[node]
                if neighbor in residual:
                    residual[neighbor] += increment
                else:
                    residual[neighbor] = increment
                neighbor_value = residual[neighbor] if neighbor in residual else 0.0
                if neighbor_value >= alpha_eps * out_degree[neighbor]:
                    if not queued.get(neighbor, False):
                        queue.append(neighbor)
                        queued[neighbor] = True
        neighbors[source_index] = list(p.keys())
        values[source_index] = list(p.values())
    return (neighbors, values)


_andersen_ppr_for_selected_nodes_cuda_kernel = None
if numba_cuda is not None:

    @numba_cuda.jit
    def _andersen_ppr_for_selected_nodes_cuda_kernel(
        indptr, indices, out_degree, source_nodes, alpha, eps, estimates, residuals, queued, stacks, status, num_nodes
    ):
        source_index = numba_cuda.grid(1)
        if source_index >= source_nodes.size:
            return
        source = source_nodes[source_index]
        offset = source_index * num_nodes
        alpha_eps = alpha * eps
        residuals[offset + source] = alpha
        queued[offset + source] = 1
        stacks[offset] = source
        top = 1
        while top > 0:
            top -= 1
            node = stacks[offset + top]
            queued[offset + node] = 0
            value = residuals[offset + node]
            estimates[offset + node] += value
            residuals[offset + node] = 0.0
            begin = indptr[node]
            end = indptr[node + 1]
            for edge_index in range(begin, end):
                neighbor = indices[edge_index]
                increment = (1.0 - alpha) * value / out_degree[node]
                neighbor_offset = offset + neighbor
                neighbor_value = residuals[neighbor_offset] + increment
                residuals[neighbor_offset] = neighbor_value
                if neighbor_value >= alpha_eps * out_degree[neighbor]:
                    if queued[neighbor_offset] == 0:
                        if top >= num_nodes:
                            status[source_index] = 1
                            return
                        stacks[offset + top] = neighbor
                        top += 1
                        queued[neighbor_offset] = 1


def _recommended_endpoint_batch_size(device, num_nodes):
    device = torch.device(device)
    if device.type != "cuda":
        return 16
    try:
        (free_bytes, _) = torch.cuda.mem_get_info(device)
    except Exception:
        return 64
    per_endpoint = max(1, int(num_nodes)) * 24
    estimate = max(1, max(0, int(free_bytes) - 8 * 1024**3) // per_endpoint)
    if free_bytes >= 60 * 1024**3:
        return max(16, min(128, estimate))
    if free_bytes >= 30 * 1024**3:
        return max(16, min(64, estimate))
    return max(8, min(32, estimate))


def _andersen_ppr_for_selected_nodes_cuda(nodes, rowptr, col, degree, *, alpha, eps, device):
    device = torch.device(device)
    source_nodes = torch.as_tensor(sorted({int(node) for node in nodes}), dtype=torch.int64, device=device)
    source_count = int(source_nodes.numel())
    if source_count == 0:
        return {}
    num_nodes = int(degree.numel())
    rowptr_device = rowptr.to(device=device, dtype=torch.int64, non_blocking=True)
    col_device = col.to(device=device, dtype=torch.int64, non_blocking=True)
    degree_device = degree.to(device=device, dtype=torch.int64, non_blocking=True)
    scratch_size = source_count * num_nodes
    estimates = torch.zeros(scratch_size, dtype=torch.float64, device=device)
    residuals = torch.zeros_like(estimates)
    queued = torch.zeros(scratch_size, dtype=torch.uint8, device=device)
    stacks = torch.empty(scratch_size, dtype=torch.int32, device=device)
    status = torch.zeros(source_count, dtype=torch.int32, device=device)
    torch.cuda.synchronize(device)
    threads = 128
    blocks = (source_count + threads - 1) // threads
    device_index = device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    with numba_cuda.gpus[int(device_index)]:
        _andersen_ppr_for_selected_nodes_cuda_kernel[blocks, threads](
            rowptr_device,
            col_device,
            degree_device,
            source_nodes,
            float(alpha),
            float(eps),
            estimates,
            residuals,
            queued,
            stacks,
            status,
            int(num_nodes),
        )
        numba_cuda.synchronize()
    if bool(status.any().item()):
        raise RuntimeError("Exact CUDA Andersen LIFO scratch stack overflowed.")
    estimates_fp32 = estimates.view(source_count, num_nodes).to(torch.float32)
    positive_locations = torch.nonzero(estimates_fp32 > 0, as_tuple=False)
    result = {}
    if positive_locations.numel():
        order = torch.argsort(positive_locations[:, 0], stable=True)
        positive_locations = positive_locations[order]
        support_rows = positive_locations[:, 0]
        support_nodes = positive_locations[:, 1]
        support_scores = estimates_fp32[support_rows, support_nodes]
        support_nodes_cpu = support_nodes.cpu()
        support_scores_cpu = support_scores.cpu()
        support_counts = torch.bincount(support_rows, minlength=source_count).cpu().tolist()
    else:
        support_nodes_cpu = torch.empty(0, dtype=torch.long)
        support_scores_cpu = torch.empty(0, dtype=torch.float32)
        support_counts = [0] * source_count
    offset = 0
    for source_index, node in enumerate(source_nodes.cpu().tolist()):
        count = int(support_counts[source_index])
        result[int(node)] = (
            support_nodes_cpu[offset : offset + count].contiguous(),
            support_scores_cpu[offset : offset + count].contiguous(),
        )
        offset += count
    return result


def _andersen_ppr_for_selected_nodes(nodes, rowptr, col, degree, *, alpha, eps, chunk_size=512, device=None):
    source_nodes = np.asarray(sorted({int(node) for node in nodes}), dtype=np.int64)
    requested_device = torch.device(device or "cpu")
    rowptr_np = rowptr.detach().cpu().numpy().astype(np.int64, copy=False)
    col_np = col.detach().cpu().numpy().astype(np.int64, copy=False)
    degree_np = degree.detach().cpu().numpy().astype(np.int64, copy=False)
    chunk_size = max(1, int(chunk_size))

    def run_cpu(selected_nodes):
        result = {}
        for start in range(0, len(selected_nodes), chunk_size):
            chunk = selected_nodes[start : start + chunk_size]
            (neighbors, values) = _andersen_ppr_for_selected_nodes_numba(rowptr_np, col_np, degree_np, chunk, float(alpha), float(eps))
            for index, node in enumerate(chunk.tolist()):
                ids = torch.as_tensor(neighbors[index], dtype=torch.long)
                scores = torch.as_tensor(values[index], dtype=torch.float32)
                keep = scores > 0
                result[int(node)] = (ids[keep].contiguous(), scores[keep].contiguous())
        return result

    if not source_nodes.size:
        return {}
    cuda_eligible = int(degree.numel()) >= 50000 and requested_device.type == "cuda" and numba_cuda is not None
    if not cuda_eligible:
        return run_cpu(source_nodes)
    graph_key = (int(rowptr.data_ptr()), int(col.data_ptr()), int(degree.numel()), float(alpha), float(eps), str(requested_device))
    decision = _ANDERSEN_BACKEND_DECISIONS.get(graph_key)
    required = int(degree.numel()) * int(source_nodes.size) * (8 + 8 + 1 + 4)
    try:
        (free_bytes, _) = torch.cuda.mem_get_info(requested_device)
    except Exception:
        free_bytes = 0
    reserve = 8 * 1024**3
    cuda_fits = required <= max(0, int(free_bytes) - reserve)
    if decision == "cpu" or not cuda_fits:
        return run_cpu(source_nodes)
    try:
        if decision is None:
            warm_nodes = source_nodes[:1]
            run_cpu(warm_nodes)
            _andersen_ppr_for_selected_nodes_cuda(warm_nodes, rowptr, col, degree, alpha=alpha, eps=eps, device=requested_device)
            sample_nodes = source_nodes[: min(32, len(source_nodes))]
            started = time.perf_counter()
            cpu_sample = run_cpu(sample_nodes)
            cpu_seconds = time.perf_counter() - started
            started = time.perf_counter()
            cuda_sample = _andersen_ppr_for_selected_nodes_cuda(
                sample_nodes, rowptr, col, degree, alpha=alpha, eps=eps, device=requested_device
            )
            cuda_seconds = time.perf_counter() - started
            parity = True
            for node in sample_nodes.tolist():
                (cpu_ids, cpu_scores) = cpu_sample[int(node)]
                (cuda_ids, cuda_scores) = cuda_sample[int(node)]
                cpu_order = torch.argsort(cpu_ids)
                parity &= torch.equal(cpu_ids[cpu_order], cuda_ids)
                parity &= torch.equal(cpu_scores[cpu_order], cuda_scores)
                if not parity:
                    break
            decision = "cuda" if parity and cuda_seconds < cpu_seconds else "cpu"
            _ANDERSEN_BACKEND_DECISIONS[graph_key] = decision
            print(
                f"Exact Andersen backend autotune: cpu={cpu_seconds:.4f}s cuda={cuda_seconds:.4f}s sample={len(sample_nodes)} parity={parity} selected={decision}",
                flush=True,
            )
            if len(sample_nodes) == len(source_nodes):
                return cuda_sample if decision == "cuda" else cpu_sample
        if decision == "cuda":
            return _andersen_ppr_for_selected_nodes_cuda(source_nodes, rowptr, col, degree, alpha=alpha, eps=eps, device=requested_device)
    except Exception as exc:
        _ANDERSEN_BACKEND_DECISIONS[graph_key] = "cpu"
        print(
            f"WARNING: exact CUDA Andersen was unavailable or failed its parity guard ({exc}); using exact sparse CPU Andersen.", flush=True
        )
        with suppress(Exception):
            torch.cuda.empty_cache()
    return run_cpu(source_nodes)


@contextmanager
def _temporary_cuda_matmul_tf32(value):
    if value is None:
        yield
        return
    previous = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = bool(value)
    try:
        yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@contextmanager
def _temporary_torch_num_threads(value):
    if value is None:
        yield
        return
    previous = torch.get_num_threads()
    torch.set_num_threads(max(1, int(value)))
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@lru_cache(maxsize=16)
def _heart_seeded_tie_order(num_nodes):
    generator = torch.Generator(device="cpu").manual_seed(int(GENERATED_HEART_TIE_SEED))
    return torch.randperm(int(num_nodes), generator=generator)


@lru_cache(maxsize=16)
def _heart_seeded_tie_priority(num_nodes):
    permutation = _heart_seeded_tie_order(int(num_nodes))
    priority = torch.empty(int(num_nodes), dtype=torch.long)
    priority[permutation] = torch.arange(int(num_nodes), dtype=torch.long)
    return priority


def _prepare_endpoint_topk_state(combined_rank, invalid_mask, k):
    combined_rank = combined_rank.detach().cpu().to(torch.long).view(-1)
    invalid_mask = invalid_mask.detach().cpu().to(torch.bool).view(-1)
    k = int(k)
    legal_nodes = torch.nonzero(~invalid_mask, as_tuple=False).view(-1)
    legal_count = int(legal_nodes.numel())
    if legal_count < k:
        raise RuntimeError(f"Only {legal_count} legal candidates remain; {k} are required.")
    legal_ranks = combined_rank[legal_nodes]
    cutoff = torch.topk(legal_ranks, k, largest=False, sorted=False).values.max()
    relevant_mask = legal_ranks <= cutoff
    relevant = legal_nodes[relevant_mask]
    relevant_ranks = legal_ranks[relevant_mask]
    rank_groups = [relevant[relevant_ranks == int(rank)].contiguous() for rank in torch.unique(relevant_ranks, sorted=True).tolist()]
    return {"num_nodes": int(combined_rank.numel()), "k": k, "rank_groups": rank_groups}


def _select_prepared_topk(prepared):
    num_nodes = int(prepared["num_nodes"])
    k = int(prepared["k"])
    base_priority = _heart_seeded_tie_priority(num_nodes)
    parts = []
    remaining = k
    for group in prepared["rank_groups"]:
        if remaining <= 0:
            break
        take = min(remaining, int(group.numel()))
        group_priority = base_priority[group]
        order = (
            torch.topk(group_priority, take, largest=False, sorted=True).indices
            if take < int(group.numel())
            else torch.argsort(group_priority)
        )
        parts.append(group[order[:take]])
        remaining -= take
    return torch.cat(parts, dim=0).contiguous()


def _sorted_membership(sorted_values, queries):
    sorted_values = sorted_values.to(torch.long).view(-1)
    queries = queries.to(torch.long).view(-1)
    if sorted_values.numel() == 0 or queries.numel() == 0:
        return torch.zeros(queries.numel(), dtype=torch.bool)
    positions = torch.searchsorted(sorted_values, queries)
    in_range = positions < int(sorted_values.numel())
    result = torch.zeros(queries.numel(), dtype=torch.bool)
    if in_range.any():
        result[in_range] = sorted_values[positions[in_range]] == queries[in_range]
    return result


def _sparse_positive_min_ranks(node_ids, scores, *, assume_unique=False):
    node_ids = node_ids.detach().cpu().to(torch.long).view(-1)
    scores = scores.detach().cpu().to(torch.float32).view(-1)
    positive = scores > 0
    node_ids = node_ids[positive]
    scores = scores[positive]
    if node_ids.numel() == 0:
        return (node_ids, torch.empty(0, dtype=torch.long), 0)
    order = torch.argsort(scores, descending=False, stable=True)
    sorted_scores = scores[order]
    positions = torch.arange(1, scores.numel() + 1, dtype=torch.long)
    starts = torch.ones(scores.numel(), dtype=torch.bool)
    if scores.numel() > 1:
        starts[1:] = sorted_scores[1:] != sorted_scores[:-1]
    first_positions = torch.where(starts, positions, torch.zeros_like(positions))
    ascending = torch.cummax(first_positions, dim=0).values
    max_positive_rank = int(first_positions.max().item())
    sorted_ranks = max_positive_rank - ascending + 1
    ranks = torch.empty_like(sorted_ranks)
    ranks[order] = sorted_ranks
    return (node_ids, ranks, max_positive_rank)


def _take_seeded_tie_nodes(num_nodes, disallowed, count):
    count = int(count)
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    disallowed = torch.unique(disallowed.detach().cpu().to(torch.long).view(-1), sorted=True)
    tie_order = _heart_seeded_tie_order(int(num_nodes))
    parts = []
    remaining = count
    start = 0
    while remaining > 0 and start < int(num_nodes):
        span = max(1024, 4 * remaining)
        end = min(int(num_nodes), start + span)
        candidates = tie_order[start:end]
        keep = ~_sorted_membership(disallowed, candidates)
        selected = candidates[keep][:remaining]
        if selected.numel():
            parts.append(selected)
            remaining -= int(selected.numel())
        start = end
    return torch.cat(parts, dim=0).contiguous()


@numba.njit(cache=True)
def _partial_sort_push_heap_numba(heap_ranks, heap_nodes, hole_index, top_index, value_rank, value_node):
    parent = (hole_index - 1) // 2
    while hole_index > top_index and heap_ranks[parent] < value_rank:
        heap_ranks[hole_index] = heap_ranks[parent]
        heap_nodes[hole_index] = heap_nodes[parent]
        hole_index = parent
        parent = (hole_index - 1) // 2
    heap_ranks[hole_index] = value_rank
    heap_nodes[hole_index] = value_node


@numba.njit(cache=True)
def _partial_sort_adjust_heap_numba(heap_ranks, heap_nodes, hole_index, heap_length, value_rank, value_node):
    top_index = hole_index
    second_child = hole_index
    while second_child < (heap_length - 1) // 2:
        second_child = 2 * (second_child + 1)
        if heap_ranks[second_child] < heap_ranks[second_child - 1]:
            second_child -= 1
        heap_ranks[hole_index] = heap_ranks[second_child]
        heap_nodes[hole_index] = heap_nodes[second_child]
        hole_index = second_child
    if heap_length % 2 == 0 and second_child == (heap_length - 2) // 2:
        second_child = 2 * (second_child + 1)
        heap_ranks[hole_index] = heap_ranks[second_child - 1]
        heap_nodes[hole_index] = heap_nodes[second_child - 1]
        hole_index = second_child - 1
    _partial_sort_push_heap_numba(heap_ranks, heap_nodes, hole_index, top_index, value_rank, value_node)


@numba.njit(cache=True)
def _partial_sort_sparse_topk_numba(num_nodes, k, zero_rank, explicit_nodes, explicit_ranks):
    heap_ranks = np.full(k, zero_rank, dtype=np.int64)
    heap_nodes = np.arange(k, dtype=np.int64)
    explicit_count = explicit_nodes.shape[0]
    explicit_index = 0
    while explicit_index < explicit_count and explicit_nodes[explicit_index] < k:
        heap_ranks[explicit_nodes[explicit_index]] = explicit_ranks[explicit_index]
        explicit_index += 1
    if k > 1:
        parent = (k - 2) // 2
        while True:
            value_rank = heap_ranks[parent]
            value_node = heap_nodes[parent]
            _partial_sort_adjust_heap_numba(heap_ranks, heap_nodes, parent, k, value_rank, value_node)
            if parent == 0:
                break
            parent -= 1
    cursor = k
    while explicit_index < explicit_count:
        node = explicit_nodes[explicit_index]
        while cursor < node and zero_rank < heap_ranks[0]:
            _partial_sort_adjust_heap_numba(heap_ranks, heap_nodes, 0, k, zero_rank, cursor)
            cursor += 1
        rank = explicit_ranks[explicit_index]
        if rank < heap_ranks[0]:
            _partial_sort_adjust_heap_numba(heap_ranks, heap_nodes, 0, k, rank, node)
        cursor = node + 1
        explicit_index += 1
    while cursor < num_nodes and zero_rank < heap_ranks[0]:
        _partial_sort_adjust_heap_numba(heap_ranks, heap_nodes, 0, k, zero_rank, cursor)
        cursor += 1
    last = k - 1
    while last > 0:
        value_rank = heap_ranks[last]
        value_node = heap_nodes[last]
        heap_ranks[last] = heap_ranks[0]
        heap_nodes[last] = heap_nodes[0]
        _partial_sort_adjust_heap_numba(heap_ranks, heap_nodes, 0, last, value_rank, value_node)
        last -= 1
    return heap_nodes


def _partial_sort_sparse_topk(num_nodes, k, zero_rank, explicit_nodes, explicit_ranks):
    num_nodes = int(num_nodes)
    k = int(k)
    explicit_nodes = explicit_nodes.detach().cpu().to(torch.long).contiguous().view(-1)
    explicit_ranks = explicit_ranks.detach().cpu().to(torch.long).contiguous().view(-1)
    if not 0 < k <= num_nodes:
        raise ValueError("partial-sort k must lie in [1, num_nodes]")
    valid = explicit_nodes.numel() == explicit_ranks.numel()
    if explicit_nodes.numel():
        valid &= int(explicit_nodes.min()) >= 0 and int(explicit_nodes.max()) < num_nodes
        valid &= explicit_nodes.numel() < 2 or not bool((explicit_nodes[1:] <= explicit_nodes[:-1]).any())
    if not valid:
        raise ValueError("partial-sort nodes and ranks must be aligned, in range, and strictly ordered")
    selected = _partial_sort_sparse_topk_numba(num_nodes, k, int(zero_rank), explicit_nodes.numpy(), explicit_ranks.numpy())
    return torch.from_numpy(selected).to(torch.long).contiguous()


def _prepare_sparse_metric_support(num_nodes, node_ids, scores, invalid_nodes, *, assume_unique=False):
    num_nodes = int(num_nodes)
    invalid_nodes = invalid_nodes.detach().cpu().to(torch.long).view(-1)
    invalid_nodes = torch.unique(invalid_nodes[(invalid_nodes >= 0) & (invalid_nodes < int(num_nodes))], sorted=True)
    node_ids = node_ids.detach().cpu().to(torch.long).view(-1)
    scores = scores.detach().cpu().to(torch.float32).view(-1)
    keep = (node_ids >= 0) & (node_ids < num_nodes) & (scores > 0)
    (node_ids, scores) = (node_ids[keep], scores[keep])
    if invalid_nodes.numel() and node_ids.numel():
        keep = ~_sorted_membership(invalid_nodes, node_ids)
        (node_ids, scores) = (node_ids[keep], scores[keep])
    (node_ids, ranks, max_rank) = _sparse_positive_min_ranks(node_ids, scores, assume_unique=assume_unique)
    if node_ids.numel():
        id_order = torch.argsort(node_ids)
        node_ids = node_ids[id_order].contiguous()
        scores = scores[id_order].contiguous()
        ranks = ranks[id_order].contiguous()
        maximum = scores.max()
        maximum_count = int((scores == maximum).sum().item())
        if maximum_count == 1:
            below_maximum = scores < maximum
            if below_maximum.any():
                previous_maximum = scores[below_maximum].max()
                max_rank_without_unique_max = int((scores < previous_maximum).sum().item()) + 1
            else:
                max_rank_without_unique_max = 0
        else:
            max_rank_without_unique_max = int(max_rank)
    else:
        maximum = torch.tensor(0.0, dtype=torch.float32)
        maximum_count = 0
        max_rank_without_unique_max = 0
    return {
        "node_ids": node_ids,
        "scores": scores,
        "ranks": ranks,
        "max_rank": int(max_rank),
        "maximum": maximum,
        "maximum_count": int(maximum_count),
        "max_rank_without_unique_max": int(max_rank_without_unique_max),
    }


def _adjust_sparse_metric_for_counterpart(prepared, counterpart):
    node_ids = prepared["node_ids"]
    scores = prepared["scores"]
    ranks = prepared["ranks"]
    max_rank = int(prepared["max_rank"])
    if counterpart is None or node_ids.numel() == 0:
        return (node_ids, ranks, max_rank)
    counterpart = int(counterpart)
    location = int(torch.searchsorted(node_ids, counterpart).item())
    if location >= int(node_ids.numel()) or int(node_ids[location]) != counterpart:
        return (node_ids, ranks, max_rank)
    counterpart_score = scores[location]
    if bool(counterpart_score == prepared["maximum"]):
        if int(prepared["maximum_count"]) == 1:
            adjusted_max_rank = int(prepared["max_rank_without_unique_max"])
            delta = max_rank - adjusted_max_rank
            adjusted_ranks = ranks - int(delta)
        else:
            adjusted_max_rank = max_rank
            adjusted_ranks = ranks
    else:
        adjusted_max_rank = max_rank - 1
        adjusted_ranks = ranks - (scores <= counterpart_score).to(torch.long)
    keep = torch.ones(node_ids.numel(), dtype=torch.bool)
    keep[location] = False
    return (node_ids[keep].contiguous(), adjusted_ranks[keep].contiguous(), int(adjusted_max_rank))


def _select_prepared_sparse_fused_topk(
    num_nodes, ra_prepared, ppr_prepared, invalid_nodes, k, *, counterpart=None, released_fallback_plan=False
):
    num_nodes = int(num_nodes)
    k = int(k)
    invalid_nodes = invalid_nodes.detach().cpu().to(torch.long).view(-1)
    if counterpart is not None:
        invalid_nodes = torch.cat([invalid_nodes, torch.tensor([int(counterpart)], dtype=torch.long)])
    invalid_nodes = torch.unique(invalid_nodes[(invalid_nodes >= 0) & (invalid_nodes < int(num_nodes))], sorted=True)
    legal_count = num_nodes - int(invalid_nodes.numel())
    if not released_fallback_plan and legal_count < k:
        raise RuntimeError(f"Only {legal_count} legal candidates remain; {k} are required.")
    (ra_ids, ra_ranks, ra_max_rank) = _adjust_sparse_metric_for_counterpart(ra_prepared, counterpart)
    (ppr_ids, ppr_ranks, ppr_max_rank) = _adjust_sparse_metric_for_counterpart(ppr_prepared, counterpart)
    zero_rank = min(int(ra_max_rank) + 1, int(ppr_max_rank) + 1)
    supports = [ids for ids in (ra_ids, ppr_ids) if ids.numel()]
    union = torch.unique(torch.cat(supports), sorted=True) if supports else torch.empty(0, dtype=torch.long)
    fused_ranks = torch.full((union.numel(),), int(zero_rank), dtype=torch.long)
    if ra_ids.numel():
        locations = torch.searchsorted(union, ra_ids)
        fused_ranks[locations] = torch.minimum(fused_ranks[locations], ra_ranks)
    if ppr_ids.numel():
        locations = torch.searchsorted(union, ppr_ids)
        fused_ranks[locations] = torch.minimum(fused_ranks[locations], ppr_ranks)
    select_count = min(k, int(union.numel())) if released_fallback_plan else k
    fallback_count = k - select_count
    if released_fallback_plan:
        zero_pool_count = int(legal_count) - int(union.numel())
        if fallback_count and zero_pool_count <= 0:
            raise RuntimeError("HeaRT source-style zero-evidence fill requested from an empty pool.")

    def finish(selected):
        return (selected, union.contiguous(), int(fallback_count)) if released_fallback_plan else selected
    if int(select_count) == 0:
        return finish(torch.empty(0, dtype=torch.long))
    use_partial_sort = int(select_count) * 64 <= int(num_nodes)
    if not use_partial_sort:
        dense_rank = torch.full((num_nodes,), int(zero_rank), dtype=torch.long)
        if union.numel():
            dense_rank[union] = fused_ranks
        if invalid_nodes.numel():
            dense_rank[invalid_nodes] = int(zero_rank) + 1
        selected = torch.topk(-dense_rank.to(torch.float32), int(select_count)).indices.to(torch.long).contiguous()
        return finish(selected)
    explicit_nodes = torch.unique(torch.cat([union, invalid_nodes]), sorted=True)
    explicit_ranks = torch.full((explicit_nodes.numel(),), int(zero_rank), dtype=torch.long)
    if union.numel():
        locations = torch.searchsorted(explicit_nodes, union)
        explicit_ranks[locations] = fused_ranks
    if invalid_nodes.numel():
        locations = torch.searchsorted(explicit_nodes, invalid_nodes)
        explicit_ranks[locations] = int(zero_rank) + 1
    selected = _partial_sort_sparse_topk(num_nodes, int(select_count), int(zero_rank), explicit_nodes, explicit_ranks)
    return finish(selected)


def _complement_ordinals_to_nodes(num_nodes, disallowed, ordinals):
    num_nodes = int(num_nodes)
    disallowed_np = torch.unique(disallowed.detach().cpu().to(torch.long).view(-1), sorted=True).numpy().astype(np.int64, copy=False)
    ordinal_np = np.asarray(ordinals, dtype=np.int64).reshape(-1)
    if ordinal_np.size == 0:
        return torch.empty(0, dtype=torch.long)
    lower = ordinal_np.copy()
    upper = ordinal_np + int(disallowed_np.size)
    target = ordinal_np + 1
    while np.any(lower < upper):
        middle = (lower + upper) // 2
        blocked_through = np.searchsorted(disallowed_np, middle, side="right")
        allowed_through = middle + 1 - blocked_through
        move_left = allowed_through >= target
        upper = np.where(move_left, middle, upper)
        lower = np.where(move_left, lower, middle + 1)
    return torch.from_numpy(lower.astype(np.int64, copy=False)).to(torch.long)


def _sample_zero_evidence_with_replacement(rng, num_nodes, disallowed, count):
    count = int(count)
    if count <= 0:
        return torch.empty(0, dtype=torch.long)
    disallowed = torch.unique(disallowed.detach().cpu().to(torch.long).view(-1), sorted=True)
    available = int(num_nodes) - int(disallowed.numel())
    ordinals = rng.choice(available, size=count, replace=True)
    return _complement_ordinals_to_nodes(int(num_nodes), disallowed, ordinals)


def _source_exact_selected_row_ra(nonweighted_adjacency, weighted_adjacency, endpoint_nodes):
    endpoint_nodes = torch.as_tensor(endpoint_nodes, dtype=torch.long, device="cpu").view(-1)
    if endpoint_nodes.numel() == 0:
        return (torch.zeros(1, dtype=torch.long), torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float32))
    selected_rows = nonweighted_adjacency[endpoint_nodes]
    ra_rows = selected_rows @ weighted_adjacency
    (rowptr, col, values) = ra_rows.csr()
    if values is None:
        values = torch.ones(col.numel(), dtype=torch.float32)
    return (rowptr.cpu().to(torch.long), col.cpu().to(torch.long), values.cpu().to(torch.float32))


def _exact_score_graph(out, edge_uv, graph_name, device):
    num_nodes = int(out["num_nodes"])
    directed = bool(out.get("heart_directed", False))
    source_ra_edge_index = out.get(f"heart_{graph_name}_ra_edge_index")
    if source_ra_edge_index is not None:
        ppr_edge_index = edge_uv if directed else torch.cat([edge_uv, edge_uv.flip(0)], dim=1)
        ra_edge_index = source_ra_edge_index
    elif directed:
        ppr_edge_index = edge_uv
        ra_edge_index = edge_uv.flip(0)
    else:
        ppr_edge_index = torch.cat([edge_uv, edge_uv.flip(0)], dim=1)
        ra_edge_index = ppr_edge_index
    shared_score_graph = int(ra_edge_index.data_ptr()) == int(ppr_edge_index.data_ptr()) and tuple(ra_edge_index.shape) == tuple(
        ppr_edge_index.shape
    )
    ra_adj = SparseTensor.from_edge_index(
        ra_edge_index.cpu(), torch.ones(ra_edge_index.size(1), dtype=torch.float32), (num_nodes, num_nodes)
    )
    if shared_score_graph:
        ra_adj = ra_adj.coalesce()
    (ra_rowptr, ra_col, _) = ra_adj.csr()
    candidate_degree = out.get(f"heart_{graph_name}_ra_degree")
    if candidate_degree is None:
        candidate_degree = ra_adj.sum(dim=0).to_dense()
    candidate_degree = candidate_degree.cpu().to(torch.float32)
    inv_candidate_degree = torch.zeros_like(candidate_degree)
    positive = candidate_degree > 0
    inv_candidate_degree[positive] = 1.0 / candidate_degree[positive]
    weighted_ra_adj = ra_adj * inv_candidate_degree.view(1, -1)
    ppr_adj = (
        ra_adj
        if shared_score_graph
        else SparseTensor.from_edge_index(
            ppr_edge_index.cpu(), torch.ones(ppr_edge_index.size(1), dtype=torch.float32), (num_nodes, num_nodes)
        ).coalesce()
    )
    (ppr_rowptr, ppr_col, _) = ppr_adj.csr()
    ppr_degree = (ppr_rowptr[1:] - ppr_rowptr[:-1]).to(torch.long)
    ppr_rowptr = ppr_rowptr.cpu()
    ppr_col = ppr_col.cpu()
    ppr_degree = ppr_degree.cpu()
    selected_row_sparse_sparse = source_ra_edge_index is not None
    return {
        "ra_selected_row_sparse_sparse": bool(selected_row_sparse_sparse),
        "ra_left": ra_adj if selected_row_sparse_sparse else None,
        "ra_right": weighted_ra_adj if selected_row_sparse_sparse else None,
        "ra_projector": None if selected_row_sparse_sparse else weighted_ra_adj.to(device).t(),
        "ra_rowptr": ra_rowptr.cpu(),
        "ra_col": ra_col.cpu(),
        "ppr_rowptr": ppr_rowptr,
        "ppr_col": ppr_col,
        "ppr_degree": ppr_degree,
    }


def _resume_tensor_digest(tensors):
    digest = hashlib.sha256()
    digest.update(b"ogb-heart-resume-shard-tensors-v1;")
    for name, value in tensors:
        tensor = value.detach().cpu().contiguous()
        digest.update(f"{name};shape={tuple(tensor.shape)};dtype={tensor.dtype};".encode("utf-8"))
        digest.update(tensor.view(torch.uint8).reshape(-1).numpy().tobytes(order="C"))
        digest.update(b";")
    return digest.hexdigest()


def _resume_shard_path(resume_state, graph_label, batch_index):
    if not isinstance(resume_state, dict):
        return None
    directory = resume_state.get("directory")
    identity = resume_state.get("identity")
    if not directory or not identity:
        return None
    safe_label = "".join((character if character.isalnum() else "-" for character in str(graph_label).lower())).strip("-")
    return os.path.join(str(directory), f"{safe_label or 'graph'}_batch{int(batch_index):08d}.pt")


def _save_resume_shard(resume_state, graph_label, batch_index, descriptors, split_names, outputs, fallback_counts, fallback_supports, k):
    path = _resume_shard_path(resume_state, graph_label, batch_index)
    if path is None:
        return
    descriptors = descriptors.detach().cpu().to(torch.int64).contiguous()
    occurrence_count = int(descriptors.size(0))
    hard_nodes = torch.full((occurrence_count, int(k)), -1, dtype=torch.int32)
    shard_fallback_counts = torch.empty(occurrence_count, dtype=torch.int16)
    support_rowptr = torch.zeros(occurrence_count + 1, dtype=torch.int64)
    support_parts = []
    support_total = 0
    for occurrence_index, descriptor in enumerate(descriptors.tolist()):
        split_index, row, side, _endpoint, _counterpart = map(int, descriptor)
        split_name = split_names[split_index]
        fallback_count = int(fallback_counts[split_name][row, side])
        hard_count = int(k) - fallback_count
        offset = 0 if side == 0 else int(k)
        if hard_count:
            hard_nodes[occurrence_index, :hard_count] = outputs[split_name][row, offset : offset + hard_count]
        shard_fallback_counts[occurrence_index] = fallback_count
        if fallback_count:
            support = fallback_supports[split_name, row, side].detach().cpu().to(torch.int32).contiguous()
            support_parts.append(support)
            support_total += int(support.numel())
        support_rowptr[occurrence_index + 1] = support_total
    support_nodes = torch.cat(support_parts).contiguous() if support_parts else torch.empty(0, dtype=torch.int32)
    tensors = (
        ("descriptors", descriptors),
        ("hard_nodes", hard_nodes),
        ("fallback_counts", shard_fallback_counts),
        ("support_rowptr", support_rowptr),
        ("support_nodes", support_nodes),
    )
    payload = {
        "metadata": {
            "version": int(_HEART_RESUME_SHARD_VERSION),
            "identity": str(resume_state["identity"]),
            "graph_label": str(graph_label),
            "batch_index": int(batch_index),
            "candidates_per_side": int(k),
            "tensor_sha256": _resume_tensor_digest(tensors),
        },
        **dict(tensors),
    }
    os.makedirs(os.path.dirname(path), exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        torch.save(payload, temporary_path)
        _fsync_file(temporary_path)
        os.replace(temporary_path, path)
        _fsync_directory(os.path.dirname(path))
    except Exception as exc:
        print(f"WARNING: failed to save HeaRT resume shard {path}: {exc}", flush=True)
    finally:
        with suppress(OSError):
            os.remove(temporary_path)


def _load_resume_shard(
    resume_state, graph_label, batch_index, expected_descriptors, split_names, outputs, fallback_counts, fallback_supports, k, num_nodes
):
    path = _resume_shard_path(resume_state, graph_label, batch_index)
    if path is None or not os.path.isfile(path):
        return False
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        metadata = payload.get("metadata")
        expected_metadata = {
            "version": int(_HEART_RESUME_SHARD_VERSION),
            "identity": str(resume_state["identity"]),
            "graph_label": str(graph_label),
            "batch_index": int(batch_index),
            "candidates_per_side": int(k),
        }
        if not isinstance(metadata, dict) or any((metadata.get(key) != value for (key, value) in expected_metadata.items())):
            raise ValueError("resume metadata mismatch")
        names = ("descriptors", "hard_nodes", "fallback_counts", "support_rowptr", "support_nodes")
        tensors = tuple((name, payload[name]) for name in names)
        descriptors, hard_nodes, shard_fallback_counts, support_rowptr, support_nodes = (value for _, value in tensors)
        if metadata.get("tensor_sha256") != _resume_tensor_digest(tensors):
            raise ValueError("resume tensor digest mismatch")
        expected_descriptors = expected_descriptors.detach().cpu().to(torch.int64).contiguous()
        if not torch.equal(descriptors, expected_descriptors):
            raise ValueError("resume occurrence schedule mismatch")
        occurrence_count = int(descriptors.size(0))
        contracts = (
            (descriptors, torch.int64, (occurrence_count, 5)),
            (hard_nodes, torch.int32, (occurrence_count, int(k))),
            (shard_fallback_counts, torch.int16, (occurrence_count,)),
            (support_rowptr, torch.int64, (occurrence_count + 1,)),
            (support_nodes, torch.int32, (int(support_nodes.numel()),)),
        )
        if any(value.dtype != dtype or tuple(value.shape) != shape for value, dtype, shape in contracts):
            raise ValueError("resume tensor contract mismatch")
        if int(support_rowptr[0]) != 0 or int(support_rowptr[-1]) != int(support_nodes.numel()) or bool((support_rowptr[1:] < support_rowptr[:-1]).any()):
            raise ValueError("resume support row pointers are invalid")

        def invalid_nodes(values, endpoint, counterpart, *, unique=True):
            return values.numel() and (
                int(values.min()) < 0 or int(values.max()) >= int(num_nodes)
                or (unique and int(torch.unique(values).numel()) != int(values.numel()))
                or bool((values == endpoint).any()) or bool((values == counterpart).any())
            )

        for occurrence_index, descriptor in enumerate(descriptors.tolist()):
            split_index, row, side, endpoint, counterpart = map(int, descriptor)
            if not 0 <= split_index < len(split_names):
                raise ValueError("resume split index is invalid")
            fallback_count = int(shard_fallback_counts[occurrence_index])
            if not 0 <= fallback_count <= int(k):
                raise ValueError("resume fallback count is invalid")
            hard_count = int(k) - fallback_count
            hard = hard_nodes[occurrence_index, :hard_count]
            suffix = hard_nodes[occurrence_index, hard_count:]
            if (suffix.numel() and bool((suffix != -1).any())) or invalid_nodes(hard, endpoint, counterpart):
                raise ValueError("resume hard-prefix contract failed")
            support_begin = int(support_rowptr[occurrence_index])
            support_end = int(support_rowptr[occurrence_index + 1])
            support = support_nodes[support_begin:support_end]
            expected_support_count = hard_count if fallback_count else 0
            if int(support.numel()) != expected_support_count or invalid_nodes(support, endpoint, counterpart):
                raise ValueError("resume fallback support contract failed")
            split_name = split_names[split_index]
            offset = 0 if side == 0 else int(k)
            if hard_count:
                outputs[split_name][row, offset : offset + hard_count] = hard
            fallback_counts[split_name][row, side] = fallback_count
            if fallback_count:
                fallback_supports[split_name, row, side] = support.contiguous()
        return True
    except Exception as exc:
        print(f"Ignoring invalid HeaRT resume shard {path}: {exc}", flush=True)
        return False


def _build_ranked_double_sided_neg_exact(out, data_name, k2, seed, device, *, resume_state=None):
    _ensure_heart_eligibility_filters(out, data_name)
    num_nodes = int(out["num_nodes"])
    device = torch.device(device)
    batch_size = _recommended_endpoint_batch_size(device, num_nodes)
    ppr_eps = _heart_ppr_eps(data_name)
    fallback_rng = np.random.RandomState(int(seed))
    train_graph = _exact_score_graph(out, out["train_uv"], "train", device)
    test_graph = _exact_score_graph(out, out["tv_uv"], "tv", device) if str(data_name).lower() == "ogbl-collab" else train_graph

    def filters(split_name, side):
        direction = "out" if int(side) == 0 else "in"
        return (out[f"heart_{split_name}_{direction}_rowptr"], out[f"heart_{split_name}_{direction}_col"])

    def rank_splits(split_items, graph, graph_label):
        split_names = [name for (name, _positives) in split_items]
        split_indices = {name: index for (index, name) in enumerate(split_names)}
        work_filters = {}
        work_occurrences = {}
        outputs = {split_name: torch.empty((positives.size(0), 2 * int(k2)), dtype=torch.int32) for (split_name, positives) in split_items}
        fallback_counts = {split_name: torch.zeros((positives.size(0), 2), dtype=torch.int16) for (split_name, positives) in split_items}
        fallback_supports = {}
        for split_name, positives in split_items:
            for row, edge in enumerate(positives.tolist()):
                for side, endpoint, counterpart in ((0, int(edge[0]), int(edge[1])), (1, int(edge[1]), int(edge[0]))):
                    (rowptr, col) = filters(split_name, side)
                    work_key = (endpoint, (int(rowptr.data_ptr()), int(col.data_ptr())))
                    if work_key not in work_filters:
                        work_filters[work_key] = (rowptr, col)
                    work_occurrences.setdefault(work_key, []).append((split_name, int(row), int(side), int(endpoint), int(counterpart)))
        keys_by_endpoint = {}
        for key in work_filters:
            keys_by_endpoint.setdefault(int(key[0]), []).append(key)
        endpoint_nodes = sorted(keys_by_endpoint)
        print(
            f"Building exact Andersen OGB HeaRT scores: graph={graph_label} endpoints={len(endpoint_nodes)} eligibility-views={len(work_filters)} batch={batch_size} device={device}",
            flush=True,
        )
        completed = 0
        with _temporary_cuda_matmul_tf32(False if device.type == "cuda" else None), _temporary_torch_num_threads(1 if device.type == "cuda" else None), torch.no_grad():
            for batch_index, start in enumerate(range(0, len(endpoint_nodes), batch_size)):
                batch_nodes = endpoint_nodes[start : start + batch_size]
                keys = [key for node in batch_nodes for key in keys_by_endpoint[node]]
                descriptor_values = [
                    (int(split_indices[split_name]), int(row), int(side), int(endpoint), int(counterpart))
                    for key in keys
                    for split_name, row, side, endpoint, counterpart in work_occurrences[key]
                ]
                descriptors = torch.as_tensor(descriptor_values, dtype=torch.int64).reshape(-1, 5).contiguous()
                if _load_resume_shard(
                    resume_state,
                    graph_label,
                    batch_index,
                    descriptors,
                    split_names,
                    outputs,
                    fallback_counts,
                    fallback_supports,
                    int(k2),
                    num_nodes,
                ):
                    completed += len(keys)
                    print(f"  resumed exact-ranked {completed}/{len(work_filters)} endpoint/filter views", flush=True)
                    continue
                width = len(batch_nodes)
                exact_ppr = _andersen_ppr_for_selected_nodes(
                    batch_nodes,
                    graph["ppr_rowptr"],
                    graph["ppr_col"],
                    graph["ppr_degree"],
                    alpha=0.15,
                    eps=ppr_eps,
                    chunk_size=max(1, len(batch_nodes)),
                    device=device,
                )
                if bool(graph["ra_selected_row_sparse_sparse"]):
                    (support_rowptr, support_nodes_cpu, support_scores_cpu) = _source_exact_selected_row_ra(
                        graph["ra_left"], graph["ra_right"], batch_nodes
                    )
                    support_counts = (support_rowptr[1:] - support_rowptr[:-1]).tolist()
                else:
                    weights = torch.zeros((num_nodes, width), dtype=torch.float32, device=device)
                    for column, node in enumerate(batch_nodes):
                        begin = int(graph["ra_rowptr"][node])
                        end = int(graph["ra_rowptr"][node + 1])
                        if end > begin:
                            neighbors = graph["ra_col"][begin:end].to(device)
                            weights[:, column].index_add_(0, neighbors, torch.ones(neighbors.numel(), dtype=torch.float32, device=device))
                    ra = graph["ra_projector"].matmul(weights)
                    positive_locations = torch.nonzero(ra > 0, as_tuple=False)
                    if positive_locations.numel():
                        support_order = torch.argsort(positive_locations[:, 1], stable=True)
                        positive_locations = positive_locations[support_order]
                        support_scores = ra[positive_locations[:, 0], positive_locations[:, 1]]
                        support_nodes_cpu = positive_locations[:, 0].cpu()
                        support_columns_cpu = positive_locations[:, 1].cpu()
                        support_scores_cpu = support_scores.cpu()
                        support_counts = torch.bincount(support_columns_cpu, minlength=width).tolist()
                    else:
                        support_nodes_cpu = torch.empty(0, dtype=torch.long)
                        support_scores_cpu = torch.empty(0, dtype=torch.float32)
                        support_counts = [0] * width
                ra_support = {}
                offset = 0
                for column, node in enumerate(batch_nodes):
                    count = int(support_counts[column])
                    ra_support[int(node)] = (support_nodes_cpu[offset : offset + count], support_scores_cpu[offset : offset + count])
                    offset += count
                for key in keys:
                    node = int(key[0])
                    (rowptr, col) = work_filters[key]
                    (begin, end) = (int(rowptr[node]), int(rowptr[node + 1]))
                    base_invalid = torch.cat([col[begin:end].cpu().to(torch.long), torch.tensor([node], dtype=torch.long)])
                    (ra_ids, ra_scores) = ra_support[node]
                    (ppr_ids, ppr_scores) = exact_ppr[node]
                    ra_prepared = _prepare_sparse_metric_support(num_nodes, ra_ids, ra_scores, base_invalid, assume_unique=True)
                    ppr_prepared = _prepare_sparse_metric_support(num_nodes, ppr_ids, ppr_scores, base_invalid, assume_unique=True)
                    chosen_by_counterpart = {}
                    for split_name, row, side, endpoint, counterpart in work_occurrences[key]:
                        chosen = chosen_by_counterpart.get(counterpart)
                        if chosen is None:
                            chosen = _select_prepared_sparse_fused_topk(
                                num_nodes,
                                ra_prepared,
                                ppr_prepared,
                                base_invalid,
                                int(k2),
                                counterpart=counterpart,
                                released_fallback_plan=True,
                            )
                            chosen_by_counterpart[counterpart] = chosen
                        (hard_nodes, zero_support, fallback_count) = chosen
                        hard_count = int(k2) - int(fallback_count)
                        if int(hard_nodes.numel()) != hard_count or int(torch.unique(hard_nodes).numel()) != hard_count:
                            raise RuntimeError("HeaRT selector failed the exact hard-prefix contract.")
                        negative = outputs[split_name]
                        offset = 0 if side == 0 else int(k2)
                        if hard_count:
                            negative[row, offset : offset + hard_count] = hard_nodes.to(torch.int32)
                        fallback_counts[split_name][row, side] = int(fallback_count)
                        if fallback_count:
                            fallback_supports[split_name, int(row), int(side)] = zero_support.to(torch.int32).contiguous()
                _save_resume_shard(
                    resume_state, graph_label, batch_index, descriptors, split_names, outputs, fallback_counts, fallback_supports, int(k2)
                )
                completed += len(keys)
                print(f"  exact-ranked {completed}/{len(work_filters)} endpoint/filter views", flush=True)
                del exact_ppr, ra_support, support_nodes_cpu, support_scores_cpu
                if bool(graph["ra_selected_row_sparse_sparse"]):
                    del support_rowptr
                else:
                    del weights, ra, positive_locations
        for split_name, positives in split_items:
            negative = outputs[split_name]
            split_fallback = fallback_counts[split_name]
            fallback_flat = torch.nonzero(split_fallback.reshape(-1) > 0, as_tuple=False).view(-1).tolist()
            for flat_index in fallback_flat:
                (row, side) = divmod(int(flat_index), 2)
                edge = positives[row]
                endpoint = int(edge[side])
                counterpart = int(edge[1 - side])
                fallback_count = int(split_fallback[row, side])
                support = fallback_supports.pop((split_name, int(row), int(side))).to(torch.long)
                (rowptr, col) = filters(split_name, side)
                (begin, end) = (int(rowptr[endpoint]), int(rowptr[endpoint + 1]))
                zero_disallowed = torch.cat(
                    [col[begin:end].cpu().to(torch.long), support, torch.tensor([endpoint, counterpart], dtype=torch.long)]
                )
                fill = _sample_zero_evidence_with_replacement(fallback_rng, num_nodes, zero_disallowed, fallback_count)
                hard_count = int(k2) - fallback_count
                offset = 0 if side == 0 else int(k2)
                negative[row, offset + hard_count : offset + int(k2)] = fill.to(torch.int32)
        if fallback_supports:
            raise RuntimeError("Unconsumed HeaRT zero-evidence fallback plans remain.")
        return (
            {name: value.contiguous() for (name, value) in outputs.items()},
            {name: value.contiguous() for (name, value) in fallback_counts.items()},
        )

    if str(data_name).lower() == "ogbl-collab":
        (valid_outputs, valid_fallbacks) = rank_splits([("valid", out["valid_pos"])], train_graph, "train")
        (test_outputs, test_fallbacks) = rank_splits([("test", out["test_pos"])], test_graph, "train+valid")
        valid_output = valid_outputs["valid"]
        test_output = test_outputs["test"]
        valid_fallback = valid_fallbacks["valid"]
        test_fallback = test_fallbacks["test"]
    else:
        (outputs, fallback_counts) = rank_splits([("valid", out["valid_pos"]), ("test", out["test_pos"])], train_graph, "train")
        (valid_output, test_output) = (outputs["valid"], outputs["test"])
        valid_fallback = fallback_counts["valid"]
        test_fallback = fallback_counts["test"]
    return (valid_output, test_output, valid_fallback, test_fallback)


def _ranked_backend_for(out, requested):
    del out
    requested = str(requested or "auto").strip().lower()
    cpu_requested = requested in {"cpu", "dense", "exact-cpu"}
    supported = {
        "auto", "gpu", "exact", "exact-gpu", "batched", "exact-batched", "safe", "safe-fast", "official", "reference",
        "andersen", "fast", "approx", "approximate", "cpu", "dense", "exact-cpu",
    }
    if requested not in supported:
        raise ValueError(f"Unknown ranked-negative backend: {requested!r}")
    if requested in {"fast", "approx", "approximate"}:
        print("Ignoring approximate OGB HeaRT backend request: generated v12 always uses exact selected-endpoint Andersen PPR.", flush=True)
    use_gpu = torch.cuda.is_available() and (not cpu_requested)
    return "exact-andersen-gpu-ra" if use_gpu else "exact-andersen-cpu-ra"


def _generate_ranked_double_sided_neg(out, data_name, k2, seed, backend, *, resume_state=None):
    device = torch.device("cuda" if str(backend) == "exact-andersen-gpu-ra" else "cpu")
    return _build_ranked_double_sided_neg_exact(out, data_name, k2, seed, device, resume_state=resume_state)
