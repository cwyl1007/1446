from contextlib import contextmanager
import hashlib
import inspect
import json
import os
import tempfile
import numpy as np
import numba
import torch
from torch_sparse import SparseTensor

from .grouped_negatives import (
    ENDPOINT_GROUPED_FORMAT,
    IndependentSideBufferedEndpointGroupedNegativeWriter,
    _file_sha256,
    load_streamed_grouped_negatives,
)
from .planetoid_inputs import load_fixed_planetoid_features

from .data_core import (
    _HEART_POOL_CACHE_VERSION,
    _HEART_TOPK_TIE_SEED,
    _andersen_ppr_for_node,
    _atomic_save,
    _bounded_raw_graph_identity,
    _canonical_known_graph_sha256,
    _exclusive_cache_build,
    _full_integer_tensor_sha256,
    _heart_pool_cache_file,
    _heart_ppr_eps,
    _heart_seeded_tie_priority,
    _load_dataset,
    _load_or_create_split,
    _make_adj,
    _sample_rows,
    _split_tensor_sha256,
    _torch_load,
)

_HEART_CACHE_METADATA_VERSION = 11
_HEART_SELECTOR_RECIPE_VERSION = 19
_HEART_CANDIDATE_SEED = 42
_HEART_PPR_ALPHA = 0.15
_HEART_FINGERPRINT_SAMPLES = 1021
_HEART_SCORE_BATCH_SIZE = 256
_HEART_SCORE_DTYPE = torch.float32
_HEART_SCORE_MATH = "strict-fp32-tf32-disabled"
_HEART_SELECTION_PROTOCOL = "full-legal-released-max-rank-min-fusion-hard-top250-compact-stream-v23"
_HEART_ENDPOINT_SCHEDULE = "validation-endpoints-then-test-only-batched-v2"
_HEART_CACHE_VALIDATION_CONTRACT = (
    "atomic-pair-manifest+artifact-sha256+shape+orientation+range+self+query+"
    "released-train-trainvalid-filter+released-max-plus-one-zero-rank+hard-prefix+"
    "fallback-multiset-v7"
)
_HEART_DENSE_CACHE_MAX_BYTES = 512 * 1024 * 1024
_HEART_ENDPOINT_GROUP_MAX_ROWS = 8192
_HEART_RELEASE_COSINE_DATASETS = frozenset({"cora", "citeseer", "pubmed"})
_HEART_HEURISTICS = ("ra", "ppr", "cosine")
_FIXED_INPUT_CACHE_METADATA_KEYS = {
    "released_positive_split": "fixed_positive_split",
}


@contextmanager
def _strict_heart_math():
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    previous_threads = torch.get_num_threads()
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.set_num_threads(1)
        yield
    finally:
        try:
            torch.set_num_threads(previous_threads)
        finally:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32


def _uses_release_cosine(data_name):
    return str(data_name).strip().lower() in _HEART_RELEASE_COSINE_DATASETS


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


def _andersen_ppr_for_eval_nodes(nodes, rowptr, col, degree, alpha, eps):
    source_nodes = np.asarray([int(node) for node in nodes], dtype=np.int64)
    (neighbors, values) = _andersen_ppr_for_selected_nodes_numba(
        rowptr.detach().cpu().numpy().astype(np.int64, copy=False), col.detach().cpu().numpy().astype(np.int64, copy=False),
        degree.detach().cpu().numpy().astype(np.int64, copy=False), source_nodes, float(alpha), float(eps),
    )
    out = {}
    for index, node in enumerate(source_nodes.tolist()):
        ids = torch.as_tensor(neighbors[index], dtype=torch.long)
        scores = torch.as_tensor(values[index], dtype=torch.float32)
        keep = scores > 0
        out[int(node)] = (ids[keep].contiguous(), scores[keep].contiguous())
    return out


def _sample_tensor_values(tensor, sample_count=_HEART_FINGERPRINT_SAMPLES):
    tensor = tensor.detach()
    total = int(tensor.numel())
    if total == 0:
        return torch.empty(0, dtype=tensor.dtype)
    count = min(max(1, int(sample_count)), total)
    if count == 1:
        flat_indices = torch.zeros(1, dtype=torch.long)
    else:
        flat_indices = torch.arange(count, dtype=torch.long) * (total - 1) // (count - 1)
    if tensor.dim() == 0:
        sampled = tensor.reshape(1)
    else:
        remaining = flat_indices
        reversed_coordinates = []
        for size in reversed(tuple(int(value) for value in tensor.shape)):
            reversed_coordinates.append(remaining.remainder(size))
            remaining = torch.div(remaining, size, rounding_mode="floor")
        coordinates = tuple((coordinate.to(tensor.device) for coordinate in reversed(reversed_coordinates)))
        sampled = tensor[coordinates]
    return sampled.to(device="cpu").contiguous().reshape(-1)


def _sampled_tensor_sha256(tensor, sample_count=_HEART_FINGERPRINT_SAMPLES):
    digest = hashlib.sha256()
    digest.update(f"shape={tuple(tensor.shape)};dtype={tensor.dtype};numel={int(tensor.numel())};".encode("utf-8"))
    sampled = _sample_tensor_values(tensor, sample_count)
    if sampled.numel():
        byte_values = sampled.clone().view(torch.uint8)
        digest.update(bytes(byte_values.tolist()))
    return digest.hexdigest()


def _heart_candidate_tensor_sha256(valid_neg, test_neg):
    digest = hashlib.sha256()
    digest.update(b"pyg-heart-candidate-tensors-v1;")
    for name, value in (("valid", valid_neg), ("test", test_neg)):
        digest.update(f"{name}:".encode("ascii"))
        digest.update(_full_integer_tensor_sha256(value).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def _heart_candidate_storage(base, draw_per_side):
    logical_bytes = ((int(base["valid_pos"].size(0)) + int(base["test_pos"].size(0))) * 4 * int(draw_per_side)
                     * torch.tensor([], dtype=torch.long).element_size())
    storage = ENDPOINT_GROUPED_FORMAT if logical_bytes > int(_HEART_DENSE_CACHE_MAX_BYTES) else "dense-int64-edge-tensors"
    return (storage, int(logical_bytes))


def _heart_candidate_recipe_sha256(metadata):
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(b"pyg-heart-v23-recipe;" + encoded).hexdigest()


def _normalize_heart_cache_metadata(metadata):
    normalized = {
        _FIXED_INPUT_CACHE_METADATA_KEYS.get(key, key): value
        for key, value in metadata.items()
    }
    if normalized.get("positive_split_source") == "heart-released-positive-txt":
        normalized["positive_split_source"] = "fixed-planetoid-positive-txt"
    if normalized.get("feature_source") == "released-heart-gnn-feature":
        normalized["feature_source"] = "fixed-planetoid-gnn-feature"
    selector_digest = normalized.get("selector_implementation_sha256")
    if selector_digest:
        from utils.cache_compat import relocated_pyg_heart_selector_fingerprint

        normalized["selector_implementation_sha256"] = relocated_pyg_heart_selector_fingerprint(normalized.get("backend"), selector_digest)
    return normalized


def _heart_endpoint_artifact_sha256(valid_manifest, test_manifest):
    digest = hashlib.sha256()
    digest.update(b"pyg-heart-endpoint-artifact-v1;")
    for split_name, path in (("valid", valid_manifest), ("test", test_manifest)):
        digest.update(f"{split_name}:".encode("ascii"))
        digest.update(_file_sha256(path).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def _heart_graph_sample_sha256(base):
    digest = hashlib.sha256()
    for key in (
        "csr_train_rowptr",
        "csr_train_col",
        "train_pos",
        "all_valid_pos",
        "all_test_pos",
        "valid_pos",
        "test_pos",
    ):
        value = base.get(key)
        digest.update(f"{key}:".encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(b"missing;")
        else:
            digest.update(_sampled_tensor_sha256(value).encode("ascii"))
            digest.update(b";")
    for key in ("split_tensor_sha256", "known_graph_sha256", "known_graph_sha256_method", "raw_graph_identity"):
        digest.update(f"{key}:{base.get(key, 'missing')};".encode("utf-8"))
    return digest.hexdigest()


def _raw_heart_selector_implementation_sha256(backend):
    functions = [
        _source_compatible_ra,
        _uses_release_cosine,
        _heart_seeded_tie_priority,
        _dense_min_ranks_by_column,
        _prepare_masked_metric_rank_matrices,
        _stable_score_priority_keys,
        _compact_metric_shortlists_by_column,
        _compact_metric_shortlist_column,
        _compact_endpoint_rank_state,
        _compact_occurrence_candidates,
        _counterpart_max_rank_delta,
        _source_prepare_filter_evidence_state,
        _correct_metric_ranks_for_counterpart,
        _source_occurrence_candidates_from_metric_ranks,
        _source_occurrence_candidates_with_mutating_cosine,
        _source_complement_ordinals_to_nodes,
        _source_sample_zero_pool_with_replacement,
        _source_occurrence_schedule,
        _endpoint_invalid_mask,
        _endpoint_side_occurrence_schedule,
        _compact_state_to_cpu,
        _append_endpoint_candidate_rows,
        _finalize_deferred_fallback_records,
        _source_score_endpoint_groups,
        _source_select_endpoint_occurrences,
        _source_assemble_selected_occurrences,
        _source_faithful_grouped_negatives,
        _stream_compact_endpoint_filter_batch,
        _ensure_test_positive_filter,
        _heart_candidate_storage,
        _heart_candidate_recipe_sha256,
        _heart_endpoint_artifact_sha256,
        _heart_candidate_tensor_sha256,
        IndependentSideBufferedEndpointGroupedNegativeWriter,
        _validate_generated_heart_negatives,
        _strict_heart_math,
        _build_heart_negatives_gpu if str(backend) == "gpu" else _build_heart_negatives_dense,
    ]
    if str(backend) != "gpu":
        functions.append(_andersen_ppr_for_node)
    else:
        functions.extend([_andersen_ppr_for_eval_nodes, _andersen_ppr_for_selected_nodes_numba])
    digest = hashlib.sha256()
    for function in functions:
        digest.update(f"{function.__module__}.{function.__qualname__}:".encode("utf-8"))
        try:
            source = inspect.getsource(function)
        except (OSError, TypeError):
            code = getattr(function, "__code__", None)
            source = repr(
                (
                    getattr(code, "co_code", b""),
                    getattr(code, "co_consts", ()),
                    getattr(code, "co_names", ()),
                )
            )
        digest.update(source.encode("utf-8") if isinstance(source, str) else source)
        digest.update(b";")
    return digest.hexdigest()


def _heart_selector_implementation_sha256(backend):
    from utils.cache_compat import relocated_pyg_heart_selector_fingerprint

    return relocated_pyg_heart_selector_fingerprint(backend, _raw_heart_selector_implementation_sha256(backend))


def _heart_cache_metadata(base, data_name, seed, draw_per_side, backend, heart_ppr_iters, ppr_eps):
    features = base.get("x")
    feature_metadata = {
        "feature_path": base.get("heart_feature_path"),
        "feature_used_for_selection": True,
        "feature_present": torch.is_tensor(features),
        "feature_shape": tuple(features.shape) if torch.is_tensor(features) else None,
        "feature_dtype": str(features.dtype) if torch.is_tensor(features) else None,
        "feature_sample_sha256": _sampled_tensor_sha256(features) if torch.is_tensor(features) else None,
        "feature_fingerprint_method": "shape-dtype-even-sample-sha256-v1",
    }
    validation_filter_count = int(base["train_pos"].size(0))
    test_filter_count = validation_filter_count + int(base["all_valid_pos"].size(0))
    validation_filter_fingerprint = hashlib.sha256(
        f"pyg-heart-validation-filter-v1;split={base['split_tensor_sha256']};edges={validation_filter_count}".encode("utf-8")
    ).hexdigest()
    test_filter_fingerprint = hashlib.sha256(
        f"pyg-heart-test-filter-v1;split={base['split_tensor_sha256']};edges={test_filter_count}".encode("utf-8")
    ).hexdigest()
    (candidate_storage, logical_candidate_bytes) = _heart_candidate_storage(base, draw_per_side)
    release_cosine_compat = _uses_release_cosine(data_name)
    return {
        "metadata_version": int(_HEART_CACHE_METADATA_VERSION),
        "candidate_cache_version": int(_HEART_POOL_CACHE_VERSION),
        "selector_recipe_version": int(_HEART_SELECTOR_RECIPE_VERSION),
        "dataset": str(data_name).lower(),
        "positive_split_source": str(base.get("heart_positive_split_source", "unspecified")),
        "fixed_positive_split": bool(base.get("heart_fixed_positive_split", False)),
        "feature_source": str(base.get("heart_feature_source", "unspecified")),
        "split_seed": int(seed),
        "edge_split": [float(value) for value in base["split"]],
        "eval_cap": int(base.get("effective_eval_cap") or 0),
        "num_nodes": int(base["num_nodes"]),
        "valid_positive_count": int(base["valid_pos"].size(0)),
        "test_positive_count": int(base["test_pos"].size(0)),
        "train_filter_nnz": int(base["csr_train_col"].numel()),
        "validation_filter_nnz": int(2 * validation_filter_count),
        "test_filter_nnz": int(2 * test_filter_count),
        "candidate_filter_scope": "released:validation=train;test=train+uncapped-valid;query-counterpart-and-self-always-excluded",
        "validation_filter_scope": "train",
        "test_filter_scope": "train+uncapped-valid",
        "validation_filter_fingerprint_method": "split-bound-train-sha256-v1",
        "validation_filter_fingerprint": validation_filter_fingerprint,
        "test_filter_fingerprint_method": "split-bound-train-valid-sha256-v1",
        "test_filter_fingerprint": test_filter_fingerprint,
        "complete_positive_graph_undirected_count": int(base["known_positive_count"]),
        "complete_positive_graph_fingerprint_method": str(base["known_graph_sha256_method"]),
        "complete_positive_graph_fingerprint": str(base["known_graph_sha256"]),
        "split_tensor_sha256": str(base["split_tensor_sha256"]),
        "split_cache_digest_version": int(base["split_cache_digest_version"]),
        "raw_graph_identity": str(base["raw_graph_identity"]),
        "raw_graph_identity_method": str(base["raw_graph_identity_method"]),
        "cache_validation_contract": _HEART_CACHE_VALIDATION_CONTRACT,
        "draw_per_side": int(draw_per_side),
        "grouped_negative_count": int(2 * draw_per_side),
        "backend": str(backend),
        "heuristic_set": list(_HEART_HEURISTICS),
        "cosine_state_policy": (
            "released-validation-then-test-inplace-row-reranking"
            if release_cosine_compat
            else "fresh-query-mask-no-cross-occurrence-state"
        ),
        "released_cosine_compat_exact": bool(release_cosine_compat),
        "rank_aggregation": "min",
        "selection_protocol": _HEART_SELECTION_PROTOCOL,
        "selection_calibration": "none-fixed-heart-k",
        "selection_hard_topk_per_side": int(draw_per_side),
        "selection_selected_per_side": int(draw_per_side),
        "candidate_slot_capacity_policy": "fixed250-with-replacement-fallback-no-unique-minimum",
        "selection_policy": "released-smallest-min-fused-ranks-with-source-fallback",
        "metric_zero_rank_policy": "per-heuristic-max-rank-plus-one",
        "metric_invalid_rank_policy": "per-heuristic-max-rank-plus-two",
        "selection_tie_break": "cpu-torch-topk-source-api-dense;seed42-total-order-streamed-equal-rank",
        "selection_tie_break_seed": None,
        "selection_zero_evidence_fill": "numpy-randomstate42-uniform-with-replacement",
        "query_counterpart_mask_stage": "pre-rank-exact-vector-correction",
        "query_counterpart_rank_correction": "positive-scipy-min-reversal-per-metric-max-sentinel-v3",
        "endpoint_score_schedule": _HEART_ENDPOINT_SCHEDULE,
        "candidate_storage_policy": "dense-under-512mib-otherwise-endpoint-grouped-v2",
        "candidate_storage_format": candidate_storage,
        "candidate_logical_dense_bytes": logical_candidate_bytes,
        "candidate_dense_logical_byte_limit": int(_HEART_DENSE_CACHE_MAX_BYTES),
        "candidate_endpoint_group_max_rows": int(_HEART_ENDPOINT_GROUP_MAX_ROWS),
        "score_batch_size": int(_HEART_SCORE_BATCH_SIZE) if str(backend) == "gpu" else 1,
        "score_dtype": str(_HEART_SCORE_DTYPE),
        "score_math": _HEART_SCORE_MATH,
        "ra_semantics": "released-weight-before-matmul-fp32",
        "cuda_tf32": False if str(backend) == "gpu" else None,
        "ppr_method": "andersen-local-push",
        "ppr_alpha": float(_HEART_PPR_ALPHA),
        "ppr_eps": float(ppr_eps),
        "ppr_iters": int(heart_ppr_iters),
        "fallback_rng_seed": int(_HEART_TOPK_TIE_SEED),
        "fallback_rng": "numpy-randomstate-mt19937",
        "fallback_sampling": "released-zero-score-uniform-with-replacement-in-positive-row-order",
        "fallback_pool_representation": "shared-csr-invalid-ref+bounded-evidence+ordinal-complement-v1",
        "unique_candidates_per_positive_side": False,
        "duplicate_candidates_scope": "released-zero-evidence-fallback-only",
        "semi_hard_rank_band": False,
        "reference_candidate_count_exact": True,
        "reference_rank_fusion_exact": True,
        "reference_rank_fusion_deviation": None,
        "reference_hard_topk_exact": True,
        "reference_hard_topk_core_exact": True,
        "reference_fallback_sampling_exact": True,
        "reference_fallback_sampling_deviation": None,
        "released_artifact_exact": False,
        "released_artifact_exact_reason": (
            "generated candidates follow the released construction, but torch.topk does not specify identities inside equal-rank"
            " ties; dense CPU identities can vary with the PyTorch/libstdc++ version and the scalable streamed representation"
            " resolves them by a fixed seed-42 total order"
        ),
        "eligibility_policy": "released-train-trainvalid-filter",
        "reference_source_filter_exact": True,
        "reference_eligibility_deviation": "none: validation masks train and test masks train+validation; both also mask the query counterpart and self",
        "graph_sample_sha256": _heart_graph_sample_sha256(base),
        "selector_implementation_sha256": _heart_selector_implementation_sha256(backend),
        **feature_metadata,
    }


def _validate_generated_heart_negatives(split_name, negatives, positives, k, num_nodes, known_rowptr=None, known_col=None):
    if not torch.is_tensor(negatives):
        raise TypeError(f"cached {split_name} negatives are not a tensor")
    negatives = negatives.detach().to(device="cpu", dtype=torch.long)
    positives = positives.detach().to(device="cpu", dtype=torch.long)
    expected_shape = (int(positives.size(0)) * 2 * int(k), 2)
    if tuple(negatives.shape) != expected_shape:
        raise ValueError(f"cached {split_name}-negative shape {tuple(negatives.shape)} does not match {expected_shape}")
    if negatives.numel() and (int(negatives.min()) < 0 or int(negatives.max()) >= int(num_nodes)):
        raise ValueError(f"cached {split_name} negatives contain invalid nodes")
    grouped = negatives.view(int(positives.size(0)), 2 * int(k), 2)
    left = grouped[:, : int(k)]
    right = grouped[:, int(k) :]
    source = positives[:, 0].view(-1, 1)
    target = positives[:, 1].view(-1, 1)
    for side, actual, expected in (("left", left[:, :, 0], source), ("right", right[:, :, 1], target)):
        if not torch.equal(actual, expected.expand_as(actual)):
            raise ValueError(f"cached {split_name} {side}-side grouping is invalid")
    left_candidates = left[:, :, 1]
    right_candidates = right[:, :, 0]
    if bool(left_candidates.eq(source).any() or right_candidates.eq(target).any()):
        raise ValueError(f"cached {split_name} negatives contain self-loops")
    if bool(left_candidates.eq(target).any() or right_candidates.eq(source).any()):
        raise ValueError(f"cached {split_name} negatives contain their positive query edge")
    if known_rowptr is not None and known_col is not None:
        endpoints = torch.cat([source.expand_as(left_candidates), target.expand_as(right_candidates)], dim=1).reshape(-1)
        candidates = torch.cat([left_candidates, right_candidates], dim=1).reshape(-1)
        order = torch.argsort(endpoints, stable=True)
        endpoints, candidates = endpoints[order], candidates[order]
        unique_endpoints, counts = torch.unique_consecutive(endpoints, return_counts=True)
        offset = 0
        for endpoint, count in zip(unique_endpoints.tolist(), counts.tolist()):
            stop = offset + int(count)
            neighbors = known_col[int(known_rowptr[endpoint]) : int(known_rowptr[endpoint + 1])].to(torch.long)
            if neighbors.numel():
                positions = torch.searchsorted(neighbors, candidates[offset:stop])
                inside = positions < neighbors.numel()
                if bool(inside.any()) and bool((neighbors[positions[inside]] == candidates[offset:stop][inside]).any()):
                    raise ValueError(f"cached {split_name} negatives include a known positive edge")
            offset = stop
    return negatives.contiguous()


def _source_compatible_ra(ra_projector, neighbor_indicator):
    return ra_projector.matmul(neighbor_indicator)


def _dense_min_ranks_by_column(scores):
    num_nodes = int(scores.size(0))
    sortable = torch.where(scores > 0, scores, torch.full_like(scores, torch.inf))
    (sorted_values, order) = torch.sort(sortable, dim=0, descending=False)
    positive = (sorted_values > 0) & torch.isfinite(sorted_values)
    positions = torch.arange(1, num_nodes + 1, dtype=torch.int32, device=scores.device).view(-1, 1)
    starts = positive.clone()
    if num_nodes > 1:
        starts[1:] &= sorted_values[1:] != sorted_values[:-1]
    first_positions = torch.where(starts, positions, torch.zeros_like(positions))
    ascending_rank_sorted = torch.cummax(first_positions, dim=0).values
    ranks = torch.empty_like(ascending_rank_sorted)
    ranks.scatter_(0, order, ascending_rank_sorted)
    max_positive_rank = first_positions.amax(dim=0, keepdim=True)
    max_expanded = max_positive_rank.expand_as(ranks)
    positive_mask = scores > 0
    ranks[positive_mask] = max_expanded[positive_mask] - ranks[positive_mask] + 1
    del max_expanded
    zero_rank = max_positive_rank + 1
    invalid_rank = max_positive_rank + 2
    ranks[scores == 0] = zero_rank.expand_as(ranks)[scores == 0]
    ranks[scores < 0] = invalid_rank.expand_as(ranks)[scores < 0]
    return ranks


def _prepare_masked_metric_rank_matrices(score_matrices, invalid_mask):
    states = []
    for values in score_matrices:
        masked = values.clone()
        masked[invalid_mask] = -1.0
        ranks = _dense_min_ranks_by_column(masked)
        positive = masked > 0
        max_rank = torch.where(positive, ranks, torch.zeros_like(ranks)).amax(dim=0)
        neg_inf = torch.full_like(masked, -torch.inf)
        max_score = torch.where(positive, masked, neg_inf).amax(dim=0)
        max_count = (positive & masked.eq(max_score.view(1, -1))).sum(dim=0)
        below_max = positive & masked.lt(max_score.view(1, -1))
        second_score = torch.where(below_max, masked, neg_inf).amax(dim=0)
        has_second = below_max.any(dim=0)
        below_second_count = (positive & masked.lt(second_score.view(1, -1))).sum(dim=0)
        unique_max_removed_rank = torch.where(
            has_second,
            below_second_count.to(max_rank.dtype) + 1,
            torch.zeros_like(max_rank),
        )
        states.append(
            {
                "scores": masked,
                "ranks": ranks,
                "max_rank": max_rank,
                "max_score": max_score,
                "max_count": max_count,
                "unique_max_removed_rank": unique_max_removed_rank,
            }
        )
    return states


def _stable_score_priority_keys(scores, legal_mask, tie_priority):
    if scores.dtype != torch.float32:
        raise TypeError("Stable HeaRT score keys require exact FP32 scores.")
    if not scores.is_contiguous():
        scores = scores.contiguous()
    tie_priority = tie_priority.to(device=scores.device, dtype=torch.long)
    priority = tie_priority.view(-1, *[1] * (scores.dim() - 1)).expand_as(scores)
    base = int(scores.size(0)) + 1
    keys = priority.clone()
    positive = legal_mask & scores.gt(0) & torch.isfinite(scores)
    negative_or_nonfinite = legal_mask & ~positive & ~scores.eq(0)
    score_bits = scores.view(torch.int32).to(torch.int64)
    keys[positive] = -score_bits[positive] * base + priority[positive]
    keys[negative_or_nonfinite] = base + priority[negative_or_nonfinite]
    keys[~legal_mask] = torch.iinfo(torch.int64).max
    return keys


def _compact_metric_shortlists_by_column(scores, invalid_mask, k, tie_priority):
    scores = torch.as_tensor(scores)
    invalid_mask = torch.as_tensor(invalid_mask, device=scores.device, dtype=torch.bool)
    columns = int(scores.size(1))
    k = int(k)
    legal = ~invalid_mask
    legal_counts = legal.sum(dim=0)
    if bool(legal_counts.lt(k + 1).any()):
        raise RuntimeError("Compact counterpart correction requires at least K+1 base-legal nodes for every endpoint.")
    keep = k + 1
    keys = _stable_score_priority_keys(scores, legal, tie_priority)
    selected = torch.topk(keys, keep, dim=0, largest=False, sorted=True).indices.contiguous()
    del keys
    selected_scores = torch.gather(scores, 0, selected)
    selected_positive = selected_scores.gt(0) & torch.isfinite(selected_scores)
    positive = legal & scores.gt(0) & torch.isfinite(scores)
    positive_counts = positive.sum(dim=0).to(torch.long)
    positions = torch.arange(1, keep + 1, dtype=torch.long, device=scores.device).view(-1, 1).expand(keep, columns)
    group_end = selected_positive.clone()
    if keep > 1:
        group_end[:-1] &= ~selected_positive[1:] | selected_scores[:-1].ne(selected_scores[1:])
    end_marks = torch.where(group_end, positions, torch.full_like(positions, keep + 1))
    full_group_end = torch.flip(torch.cummin(torch.flip(end_marks, dims=(0,)), dim=0).values, dims=(0,))
    truncated = positive_counts.gt(keep)
    threshold = selected_scores[-1]
    full_boundary_count = (positive & scores.eq(threshold.view(1, -1))).sum(dim=0).to(torch.long)
    selected_boundary_count = (selected_positive & selected_scores.eq(threshold.view(1, -1))).sum(dim=0).to(torch.long)
    boundary_extra = torch.where(truncated, full_boundary_count - selected_boundary_count, torch.zeros_like(full_boundary_count))
    boundary_members = selected_positive & selected_scores.eq(threshold.view(1, -1)) & truncated.view(1, -1)
    full_group_end = full_group_end + boundary_members.to(torch.long) * boundary_extra.view(1, -1)
    has_positive = positive_counts.gt(0)
    max_count = torch.where(has_positive, full_group_end[0], torch.zeros_like(positive_counts))
    max_rank = torch.where(has_positive, positive_counts - max_count + 1, torch.zeros_like(positive_counts))
    selected_ranks = (full_group_end - max_count.view(1, -1) + 1).to(torch.int32)
    max_score = torch.where(has_positive, selected_scores[0], torch.full((columns,), -torch.inf, dtype=torch.float32, device=scores.device))
    below_max = selected_positive & selected_scores.lt(max_score.view(1, -1))
    second_index = below_max.to(torch.int64).argmax(dim=0)
    second_score = torch.gather(selected_scores, 0, second_index.view(1, -1)).view(-1)
    has_second = below_max.any(dim=0)
    selected_second_count = (selected_positive & selected_scores.eq(second_score.view(1, -1))).sum(dim=0).to(torch.long)
    second_count = selected_second_count + torch.where(truncated & has_second & threshold.eq(second_score), boundary_extra, torch.zeros_like(boundary_extra))
    unique_max_removed_rank = torch.where(max_count.eq(1) & has_second, positive_counts - second_count, torch.zeros_like(positive_counts))
    zero_rank = (max_rank + 1).view(1, -1).expand_as(selected_ranks)
    invalid_rank = (max_rank + 2).view(1, -1).expand_as(selected_ranks)
    selected_zero = ~selected_positive & selected_scores.eq(0)
    selected_ranks[selected_zero] = zero_rank[selected_zero].to(torch.int32)
    selected_other = ~selected_positive & ~selected_scores.eq(0)
    selected_ranks[selected_other] = invalid_rank[selected_other].to(torch.int32)
    return {
        "ids": selected,
        "scores": selected_scores,
        "ranks": selected_ranks,
        "max_score": max_score,
        "max_count": max_count,
        "max_rank": max_rank,
        "unique_max_removed_rank": unique_max_removed_rank,
    }


def _compact_metric_shortlist_column(batch_state, column):
    column = int(column)
    summary_cpu = batch_state["summary_cpu"]
    return {
        "ids": batch_state["ids"][:, column].contiguous(),
        "scores": batch_state["scores"][:, column].contiguous(),
        "ranks": batch_state["ranks"][:, column].contiguous(),
        "max_score": batch_state["max_score"][column],
        "max_count": int(summary_cpu["max_count"][column]),
        "max_rank": int(summary_cpu["max_rank"][column]),
        "unique_max_removed_rank": int(summary_cpu["unique_max_removed_rank"][column]),
    }


def _compact_endpoint_rank_state(metric_shortlists, evidence_count, tie_priority):
    num_nodes = int(tie_priority.numel())
    k = int(metric_shortlists[0]["ids"].numel()) - 1
    device = metric_shortlists[0]["ids"].device
    union = torch.unique(torch.cat([metric["ids"] for metric in metric_shortlists]), sorted=True)
    metric_count = len(metric_shortlists)
    union_size = int(union.numel())
    ranks = torch.empty((metric_count, union_size), dtype=torch.int32, device=device)
    scores = torch.zeros((metric_count, union_size), dtype=torch.float32, device=device)
    member = torch.zeros((metric_count, union_size), dtype=torch.bool, device=device)
    for metric_index, metric in enumerate(metric_shortlists):
        ranks[metric_index].fill_(int(metric["max_rank"]) + 1)
        positions = torch.searchsorted(union, metric["ids"])
        ranks[metric_index, positions] = metric["ranks"]
        scores[metric_index, positions] = metric["scores"]
        member[metric_index, positions] = True
    priority = torch.as_tensor(tie_priority, device=device, dtype=torch.long)[union]
    return {
        "num_nodes": num_nodes,
        "k": k,
        "union": union,
        "ranks": ranks,
        "scores": scores,
        "member": member,
        "evidence_count": int(evidence_count),
        "priority": priority,
        "metrics": [
            {
                "max_score": metric["max_score"],
                "max_count": int(metric["max_count"]),
                "max_rank": int(metric["max_rank"]),
                "unique_max_removed_rank": int(metric["unique_max_removed_rank"]),
            }
            for metric in metric_shortlists
        ],
    }


def _counterpart_max_rank_delta(scores, maximum, maximum_count, max_rank, unique_max_removed_rank):
    positive = scores.gt(0) & torch.isfinite(scores)
    max_rank = torch.as_tensor(max_rank, device=scores.device, dtype=torch.int32)
    removed_rank = torch.as_tensor(unique_max_removed_rank, device=scores.device, dtype=torch.int32)
    zeros = torch.zeros_like(scores, dtype=torch.int32)
    return torch.where(positive & scores.lt(maximum), -torch.ones_like(zeros),
                       torch.where(positive & (maximum_count == 1), removed_rank - max_rank, zeros))


def _compact_occurrence_candidates(prepared, counterparts, counterpart_scores):
    union = prepared["union"]
    device = union.device
    counterparts = torch.as_tensor(counterparts, device=device, dtype=torch.long).view(-1)
    rows = int(counterparts.numel())
    counterpart_scores = torch.as_tensor(counterpart_scores, device=device, dtype=torch.float32)
    combined = None
    for metric_index, summary in enumerate(prepared["metrics"]):
        base_ranks = prepared["ranks"][metric_index]
        base_scores = prepared["scores"][metric_index]
        nominated = prepared["member"][metric_index]
        cp_score = counterpart_scores[metric_index]
        cp_positive = cp_score.gt(0) & torch.isfinite(cp_score)
        maximum = torch.as_tensor(summary["max_score"], device=device, dtype=torch.float32)
        max_rank = torch.tensor(int(summary["max_rank"]), device=device, dtype=torch.int32)
        delta = _counterpart_max_rank_delta(cp_score, maximum, summary["max_count"], max_rank, summary["unique_max_removed_rank"])
        corrected = base_ranks.view(1, -1).expand(rows, -1).clone()
        positive_candidates = nominated & base_scores.gt(0)
        corrected[:, positive_candidates] = (base_ranks[positive_candidates].view(1, -1) + delta.view(-1, 1)
            + (cp_positive.view(-1, 1) & base_scores[positive_candidates].view(1, -1).gt(cp_score.view(-1, 1))).to(torch.int32))
        corrected_max_rank = max_rank + delta
        corrected[:, ~positive_candidates] = corrected_max_rank.view(-1, 1) + 1
        combined = corrected if combined is None else torch.minimum(combined, corrected)
    query_mask = union.view(1, -1).eq(counterparts.view(-1, 1))
    base = int(prepared["num_nodes"]) + 1
    keys = combined.to(torch.int64) * base + prepared["priority"].view(1, -1)
    keys = keys.masked_fill(query_mask, torch.iinfo(torch.int64).max)
    order = torch.topk(keys, int(prepared["k"]), largest=False, sorted=True, dim=1).indices
    counterpart_evidence = (counterpart_scores.gt(0) & torch.isfinite(counterpart_scores)).any(dim=0)
    hard_counts = torch.clamp(torch.full((rows,), int(prepared["evidence_count"]), dtype=torch.long, device=device)
                              - counterpart_evidence.to(torch.long), max=int(prepared["k"]))
    return (order, hard_counts)


def _source_prepare_filter_evidence_state(metric_states, column, invalid_mask, k, *, filter_col, filter_bounds, endpoint):
    column = int(column)
    k = int(k)
    invalid = invalid_mask[:, column]
    positive_evidence = torch.zeros_like(invalid, dtype=torch.bool)
    for state in metric_states:
        positive_evidence |= state["scores"][:, column].gt(0)
    evidence = ~invalid & positive_evidence
    evidence_count = int(evidence.sum().item())
    evidence_ids = None
    if evidence_count <= k:
        evidence_ids = torch.nonzero(evidence, as_tuple=False).view(-1).to(device="cpu", dtype=torch.long).contiguous()
    (filter_start, filter_end) = (int(value) for value in filter_bounds)
    return {
        "num_nodes": int(invalid.numel()),
        "evidence_count": evidence_count,
        "evidence_ids": evidence_ids,
        "invalid_ref": {
            "col": filter_col,
            "start": filter_start,
            "end": filter_end,
            "endpoint": int(endpoint),
        },
    }


def _correct_metric_ranks_for_counterpart(state, column, counterpart):
    column = int(column)
    counterpart = int(counterpart)
    scores = state["scores"][:, column]
    ranks = state["ranks"][:, column]
    counterpart_score = scores[counterpart]
    maximum = state["max_score"][column]
    maximum_count = state["max_count"][column]
    max_rank = state["max_rank"][column]
    unique_removed = state["unique_max_removed_rank"][column]
    counterpart_positive = counterpart_score > 0
    delta_max_rank = _counterpart_max_rank_delta(
        counterpart_score,
        maximum,
        maximum_count,
        max_rank,
        unique_removed,
    )
    positive = scores > 0
    corrected = ranks.clone()
    corrected[positive] = (
        ranks[positive]
        + delta_max_rank
        + (counterpart_positive & scores[positive].gt(counterpart_score)).to(
            corrected.dtype
        )
    )
    corrected_max_rank = max_rank + delta_max_rank
    corrected[scores == 0] = corrected_max_rank + 1
    corrected[scores < 0] = corrected_max_rank + 2
    corrected[counterpart] = corrected_max_rank + 2
    return (corrected, corrected_max_rank)


def _source_occurrence_candidates_from_metric_ranks(metric_states, column, invalid_mask, counterpart, k, evidence_state):
    column = int(column)
    counterpart = int(counterpart)
    k = int(k)
    combined = None
    for state in metric_states:
        (corrected, _) = _correct_metric_ranks_for_counterpart(state, column, counterpart)
        combined = corrected if combined is None else torch.minimum(combined, corrected)
    base_evidence_count = int(evidence_state["evidence_count"])
    compact_evidence = evidence_state["evidence_ids"]
    counterpart_is_evidence = False
    if base_evidence_count <= k:
        location = int(torch.searchsorted(compact_evidence, counterpart).item())
        counterpart_is_evidence = location < int(compact_evidence.numel()) and int(compact_evidence[location]) == counterpart
    nonzero_evidence_count = base_evidence_count - int(counterpart_is_evidence)
    take = min(k, nonzero_evidence_count)
    invalid = invalid_mask[:, column].clone()
    invalid[counterpart] = True
    combined[invalid] = torch.iinfo(combined.dtype).max
    chosen = torch.topk(-combined.to(device="cpu", dtype=torch.float32), take).indices if take > 0 else torch.empty(0, dtype=torch.long)
    chosen = chosen.to(device="cpu", dtype=torch.long).contiguous()
    if nonzero_evidence_count >= k:
        return (chosen, None)
    return (chosen, (evidence_state, counterpart))


def _source_occurrence_candidates_with_mutating_cosine(metric_states, column, invalid_mask, counterpart, k, cosine_state, *, filter_col, filter_bounds, endpoint):
    column = int(column)
    counterpart = int(counterpart)
    k = int(k)
    cosine_state = cosine_state.view(-1)
    num_nodes = int(cosine_state.numel())
    combined = None
    zero_evidence = torch.ones(num_nodes, dtype=torch.bool, device=cosine_state.device)
    for state in metric_states:
        scores = state["scores"][:, column]
        metric_zero = scores.eq(0)
        metric_zero[counterpart] = False
        zero_evidence &= metric_zero
        (corrected, _) = _correct_metric_ranks_for_counterpart(state, column, counterpart)
        combined = corrected if combined is None else torch.minimum(combined, corrected)
    invalid = invalid_mask[:, column].clone()
    invalid[counterpart] = True
    cosine_state[invalid] = -1.0
    zero_evidence &= cosine_state.eq(0)
    cosine_ranks = _dense_min_ranks_by_column(cosine_state.view(-1, 1)).view(-1)
    cosine_state.copy_(cosine_ranks.to(dtype=cosine_state.dtype))
    combined = torch.minimum(combined, cosine_ranks)
    combined[invalid] = torch.iinfo(combined.dtype).max
    legal_nonzero_count = num_nodes - int(zero_evidence.sum().item()) - int(invalid.sum().item())
    take = min(k, max(0, int(legal_nonzero_count)))
    chosen = torch.topk(-combined.to(device="cpu", dtype=torch.float32), take).indices if take > 0 else torch.empty(0, dtype=torch.long)
    chosen = chosen.to(device="cpu", dtype=torch.long).contiguous()
    if legal_nonzero_count >= k:
        return (chosen, None)
    legal_nonzero = ~invalid & ~zero_evidence
    evidence_ids = torch.nonzero(legal_nonzero, as_tuple=False).view(-1).to(device="cpu", dtype=torch.long).contiguous()
    evidence_state = {
        "num_nodes": num_nodes,
        "evidence_ids": evidence_ids,
        "invalid_ref": {
            "col": filter_col,
            "start": int(filter_bounds[0]),
            "end": int(filter_bounds[1]),
            "endpoint": int(endpoint),
        },
    }
    return (chosen, (evidence_state, counterpart))


def _source_complement_ordinals_to_nodes(num_nodes, sorted_base_disallowed, sorted_extra_disallowed, ordinals):
    num_nodes = int(num_nodes)
    base = np.asarray(sorted_base_disallowed, dtype=np.int64).reshape(-1)
    extra = np.asarray(sorted_extra_disallowed, dtype=np.int64).reshape(-1)
    ordinal_values = np.asarray(ordinals, dtype=np.int64).reshape(-1)
    if ordinal_values.size == 0:
        return torch.empty(0, dtype=torch.long)
    disallowed_count = int(base.size) + int(extra.size)
    available = num_nodes - disallowed_count
    if available <= 0:
        raise RuntimeError("Released HeaRT fallback has an empty zero pool.")
    lower = ordinal_values.copy()
    upper = ordinal_values + disallowed_count
    target = ordinal_values + 1
    while np.any(lower < upper):
        middle = (lower + upper) // 2
        blocked_through = np.searchsorted(base, middle, side="right")
        blocked_through += np.searchsorted(extra, middle, side="right")
        allowed_through = middle + 1 - blocked_through
        move_left = allowed_through >= target
        upper = np.where(move_left, middle, upper)
        lower = np.where(move_left, lower, middle + 1)
    return torch.from_numpy(lower.astype(np.int64, copy=False)).to(torch.long)


def _source_sample_zero_pool_with_replacement(rng, zero_pool, count):
    (evidence_state, counterpart) = zero_pool
    num_nodes = int(evidence_state["num_nodes"])
    evidence_ids = evidence_state["evidence_ids"]
    invalid_ref = evidence_state["invalid_ref"]
    base_tensor = invalid_ref["col"][int(invalid_ref["start"]) : int(invalid_ref["end"])]
    base = base_tensor.numpy().astype(np.int64, copy=False)
    bounded_parts = [evidence_ids.numpy().astype(np.int64, copy=False)]
    bounded_parts.append(np.asarray([int(counterpart), int(invalid_ref["endpoint"])], dtype=np.int64))
    extra = np.unique(np.concatenate(bounded_parts))
    extra = extra[(extra >= 0) & (extra < num_nodes)]
    if base.size and extra.size:
        positions = np.searchsorted(base, extra)
        inside = positions < int(base.size)
        present = np.zeros(extra.size, dtype=np.bool_)
        present[inside] = base[positions[inside]] == extra[inside]
        extra = extra[~present]
    zero_count = num_nodes - int(base.size) - int(extra.size)
    if zero_count <= 0:
        raise RuntimeError("Released HeaRT fallback has an empty zero pool.")
    count = int(count)
    ordinals = rng.choice(zero_count, size=count, replace=True)
    return _source_complement_ordinals_to_nodes(num_nodes, base, extra, ordinals)


def _source_occurrence_schedule(valid_pos, test_pos):
    occurrences = {}
    order = 0
    for split_name, positives in (("valid", valid_pos), ("test", test_pos)):
        for edge in positives.tolist():
            (source, target) = (int(edge[0]), int(edge[1]))
            for endpoint, counterpart in ((source, target), (target, source)):
                occurrences.setdefault(endpoint, []).append((order, split_name, int(counterpart)))
                order += 1
    return occurrences


def _endpoint_invalid_mask(endpoints, rowptr, col, num_nodes, device):
    mask = torch.zeros((int(num_nodes), len(endpoints)), dtype=torch.bool, device=device)
    bounds = []
    for column, endpoint in enumerate(endpoints):
        endpoint = int(endpoint)
        start = int(rowptr[endpoint])
        end = int(rowptr[endpoint + 1])
        bounds.append((start, end))
        if end > start:
            mask[col[start:end].to(device=device, dtype=torch.long), column] = True
        mask[endpoint, column] = True
    return (mask, bounds)


def _endpoint_side_occurrence_schedule(positives, num_nodes, side):
    positives = torch.as_tensor(positives, dtype=torch.long, device="cpu").contiguous()
    side = int(side)
    endpoints = positives[:, side]
    counts = torch.bincount(endpoints, minlength=int(num_nodes)).to(torch.long)
    rowptr = torch.empty(int(num_nodes) + 1, dtype=torch.long)
    rowptr[0] = 0
    torch.cumsum(counts, dim=0, out=rowptr[1:])
    rows = torch.argsort(endpoints, stable=True).to(torch.int32).contiguous()
    return {"rowptr": rowptr, "rows": rows}


def _compact_state_to_cpu(prepared):
    state = {key: prepared[key].detach().cpu().contiguous() for key in ("union", "ranks", "scores", "member", "priority")}
    state.update(
        num_nodes=int(prepared["num_nodes"]),
        k=int(prepared["k"]),
        evidence_count=int(prepared["evidence_count"]),
        metrics=[
            {
                "max_score": metric["max_score"].detach().cpu(),
                "max_count": int(metric["max_count"]),
                "max_rank": int(metric["max_rank"]),
                "unique_max_removed_rank": int(metric["unique_max_removed_rank"]),
            }
            for metric in prepared["metrics"]
        ],
    )
    return state


def _append_endpoint_candidate_rows(writer, *, side, endpoint, row_ids, candidate_nodes, hard_prefix_count, stats):
    row_ids = torch.as_tensor(row_ids, dtype=torch.long).cpu().view(-1)
    candidate_nodes = torch.as_tensor(candidate_nodes, dtype=torch.long).cpu()
    hard_prefix_count = torch.as_tensor(hard_prefix_count, dtype=torch.long).cpu().view(-1)

    def append_range(start, end):
        rows = row_ids[start:end]
        candidates = candidate_nodes[start:end]
        prefix = hard_prefix_count[start:end]
        union = torch.unique(candidates.reshape(-1), sorted=True)
        if int(union.numel()) > 65535:
            if end - start <= 1:
                raise RuntimeError("One HeaRT occurrence exceeds the uint16 endpoint union.")
            middle = start + (end - start) // 2
            append_range(start, middle)
            append_range(middle, end)
            return
        local = torch.searchsorted(union, candidates)
        writer.append_endpoint_group(int(side), int(endpoint), rows, union, local, prefix)
        stats["groups"] += 1
        stats["occurrences"] += int(rows.numel())
        stats["union_nodes"] += int(union.numel())
        stats["max_group_rows"] = max(int(stats["max_group_rows"]), int(rows.numel()))
        stats["max_union_nodes"] = max(int(stats["max_union_nodes"]), int(union.numel()))

    for start in range(0, int(row_ids.numel()), _HEART_ENDPOINT_GROUP_MAX_ROWS):
        append_range(start, min(start + _HEART_ENDPOINT_GROUP_MAX_ROWS, int(row_ids.numel())))


def _finalize_deferred_fallback_records(records, writers, positives_by_split, seed, cache_path, stats_by_split):
    if not records:
        return
    total_rows = 0
    locations = {
        split_name: np.zeros((int(positives.size(0)), 2), dtype=np.uint64)
        for split_name, positives in positives_by_split.items()
    }
    for record_id, record in enumerate(records):
        count = int(record["rows"].numel())
        record["storage_start"] = total_rows
        total_rows += count
        rows = record["rows"].numpy().astype(np.int64, copy=False)
        side = int(record["side"])
        split_name = record["split"]
        locations[split_name][rows, side] = (np.uint64(record_id + 1) << np.uint64(32)) | np.arange(count, dtype=np.uint64)
    cache_dir = os.path.dirname(os.path.abspath(cache_path))
    (descriptor, spool_path) = tempfile.mkstemp(prefix=".heart-v23-fallback-", suffix=".u32", dir=cache_dir)
    os.close(descriptor)
    fallback_rng = np.random.RandomState(int(seed))
    k = int(records[0]["state"]["k"])
    spool = None
    try:
        spool = np.memmap(spool_path, mode="w+", dtype=np.uint32, shape=(total_rows, k))
        replay_rows_per_block = 65536
        for split_name in ("valid", "test"):
            location_map = locations[split_name]
            for row_start in range(0, int(location_map.shape[0]), replay_rows_per_block):
                row_end = min(row_start + replay_rows_per_block, int(location_map.shape[0]))
                coordinates = np.argwhere(location_map[row_start:row_end] != 0)
                for local_row, side in coordinates:
                    row = row_start + int(local_row)
                    side = int(side)
                    location = int(location_map[row, side])
                    record_id = (location >> 32) - 1
                    local_id = location & 0xFFFFFFFF
                    record = records[record_id]
                    counterpart_scores = record["counterpart_scores"][:, local_id]
                    counterpart_evidence = bool((counterpart_scores > 0).any().item())
                    hard_count = int(record["state"]["evidence_count"]) - int(counterpart_evidence)
                    hard_count = min(k, hard_count)
                    need = k - hard_count
                    if need <= 0:
                        continue
                    zero_pool = (record["evidence_state"], int(record["counterparts"][local_id]))
                    sampled = _source_sample_zero_pool_with_replacement(fallback_rng, zero_pool, need)
                    storage_row = int(record["storage_start"]) + local_id
                    spool[storage_row, hard_count:k] = sampled.numpy().astype(np.uint32, copy=False)
        spool.flush()
        del locations
        for record in records:
            count = int(record["rows"].numel())
            start = int(record["storage_start"])
            stop = start + count
            candidates = torch.from_numpy(np.asarray(spool[start:stop]).astype(np.int64, copy=True))
            (order, hard_counts) = _compact_occurrence_candidates(record["state"], record["counterparts"], record["counterpart_scores"])
            for row in range(count):
                hard_count = int(hard_counts[row])
                if hard_count:
                    candidates[row, :hard_count] = record["state"]["union"][order[row, :hard_count]]
            _append_endpoint_candidate_rows(writers[record["split"]], side=record["side"], endpoint=record["endpoint"],
                                            row_ids=record["rows"], candidate_nodes=candidates,
                                            hard_prefix_count=hard_counts, stats=stats_by_split[record["split"]])
    finally:
        if spool is not None:
            del spool
        try:
            os.remove(spool_path)
        except FileNotFoundError:
            pass


def _source_score_endpoint_groups(valid_pos, test_pos):
    valid_nodes = torch.unique(valid_pos.reshape(-1)).tolist()
    valid_set = set(valid_nodes)
    test_only_nodes = [node for node in torch.unique(test_pos.reshape(-1)).tolist() if node not in valid_set]
    return [("validation", valid_nodes), ("test-only", test_only_nodes)]


def _source_select_endpoint_occurrences(endpoint, endpoint_occurrences, scores, filters, k, num_nodes, tie_namespace="unspecified"):
    selected_by_order = {}
    release_mutating_cosine = _uses_release_cosine(tie_namespace)
    (ra, ppr, cosine) = scores
    device = ra.device
    score_vectors = (ra, ppr) if release_mutating_cosine else scores
    cosine_state = cosine.clone() if release_mutating_cosine else None
    for split_name, (rowptr, col) in filters.items():
        split_occurrences = [occurrence for occurrence in endpoint_occurrences if occurrence[1] == split_name]
        if not split_occurrences:
            continue
        (invalid, bounds) = _endpoint_invalid_mask([endpoint], rowptr, col, num_nodes, device)
        filter_bounds = bounds[0]
        metric_states = _prepare_masked_metric_rank_matrices([values.view(-1, 1) for values in score_vectors], invalid)
        evidence_state = None
        if not release_mutating_cosine:
            evidence_state = _source_prepare_filter_evidence_state(metric_states, 0, invalid, k, filter_col=col,
                                                                    filter_bounds=filter_bounds, endpoint=endpoint)
        for occurrence_order, _, counterpart in split_occurrences:
            if release_mutating_cosine:
                (chosen, zero_pool) = _source_occurrence_candidates_with_mutating_cosine(
                    metric_states, 0, invalid, counterpart, k, cosine_state, filter_col=col,
                    filter_bounds=filter_bounds, endpoint=endpoint)
            else:
                (chosen, zero_pool) = _source_occurrence_candidates_from_metric_ranks(
                    metric_states, 0, invalid, counterpart, k, evidence_state
                )
            selected_by_order[int(occurrence_order)] = (endpoint, chosen, zero_pool)
    return selected_by_order


def _source_assemble_selected_occurrences(valid_pos, test_pos, selected_by_order, k, seed):
    grouped = {
        "valid": torch.empty((valid_pos.size(0), 2 * int(k), 2), dtype=torch.long),
        "test": torch.empty((test_pos.size(0), 2 * int(k), 2), dtype=torch.long),
    }
    fallback_rng = np.random.RandomState(int(seed))
    valid_occurrences = 2 * int(valid_pos.size(0))
    occurrence_count = 2 * (int(valid_pos.size(0)) + int(test_pos.size(0)))
    for occurrence_order in range(occurrence_count):
        (endpoint, chosen, zero_pool) = selected_by_order[occurrence_order]
        split_name = "valid" if occurrence_order < valid_occurrences else "test"
        split_order = occurrence_order if split_name == "valid" else occurrence_order - valid_occurrences
        (row, side) = divmod(split_order, 2)
        fallback_used = zero_pool is not None
        if fallback_used:
            need = int(k) - int(chosen.numel())
            sampled = _source_sample_zero_pool_with_replacement(fallback_rng, zero_pool, need)
            chosen = torch.cat([chosen, sampled]).to(torch.long).contiguous()
        block = grouped[split_name][row, side * int(k) : (side + 1) * int(k)]
        block[:, side] = int(endpoint)
        block[:, 1 - side] = chosen
    return (grouped["valid"].view(-1, 2).contiguous(), grouped["test"].view(-1, 2).contiguous())


def _source_faithful_grouped_negatives(valid_pos, test_pos, score_rows, tr_rowptr, tr_col, tv_rowptr, tv_col, k, seed, num_nodes, tie_namespace="unspecified"):
    occurrences = _source_occurrence_schedule(valid_pos, test_pos)
    filters = {"valid": (tr_rowptr, tr_col), "test": (tv_rowptr, tv_col)}
    selected_by_order = {}
    for endpoint, endpoint_occurrences in occurrences.items():
        selected_by_order.update(_source_select_endpoint_occurrences(
            int(endpoint), endpoint_occurrences, score_rows[int(endpoint)], filters, k, num_nodes, tie_namespace))
    return _source_assemble_selected_occurrences(valid_pos, test_pos, selected_by_order, k, seed)


def _stream_compact_endpoint_filter_batch(*, split_name, split_nodes, score_matrices, invalid_mask, filter_rowptr, filter_col, schedules, positives, k, tie_priority, writer, fallback_records, stats):
    metric_batches = [_compact_metric_shortlists_by_column(values, invalid_mask, k, tie_priority) for values in score_matrices]
    for batch_state in metric_batches:
        batch_state["summary_cpu"] = {key: batch_state[key].detach().cpu() for key in ("max_count", "max_rank", "unique_max_removed_rank")}
    evidence = ~invalid_mask
    positive_any = torch.zeros_like(invalid_mask, dtype=torch.bool)
    for values in score_matrices:
        positive_any |= values.gt(0) & torch.isfinite(values)
    evidence_counts = (evidence & positive_any).sum(dim=0).to(torch.long).cpu()
    del evidence, positive_any
    for column, endpoint in enumerate(split_nodes):
        endpoint = int(endpoint)
        metric_states = [_compact_metric_shortlist_column(batch, column) for batch in metric_batches]
        prepared = _compact_endpoint_rank_state(metric_states, int(evidence_counts[column]), tie_priority)
        fallback_capable = int(prepared["evidence_count"]) <= int(k)
        filter_start = int(filter_rowptr[endpoint])
        filter_end = int(filter_rowptr[endpoint + 1])
        evidence_state = None
        if fallback_capable:
            has_evidence = (prepared["member"] & prepared["scores"].gt(0)).any(dim=0)
            evidence_ids = prepared["union"][has_evidence].detach().cpu().contiguous()
            cpu_state = _compact_state_to_cpu(prepared)
            evidence_state = {
                "num_nodes": int(cpu_state["num_nodes"]),
                "evidence_ids": evidence_ids,
                "invalid_ref": {"col": filter_col, "start": filter_start, "end": filter_end, "endpoint": endpoint},
            }
        for side in (0, 1):
            schedule = schedules[split_name][side]
            start = int(schedule["rowptr"][endpoint])
            end = int(schedule["rowptr"][endpoint + 1])
            rows = schedule["rows"][start:end].to(torch.long)
            if not rows.numel():
                continue
            for row_start in range(0, int(rows.numel()), _HEART_ENDPOINT_GROUP_MAX_ROWS):
                block_rows = rows[row_start : row_start + _HEART_ENDPOINT_GROUP_MAX_ROWS].contiguous()
                counterparts = positives[block_rows, 1 - side].to(torch.long)
                counterparts_device = counterparts.to(device=invalid_mask.device, non_blocking=True)
                counterpart_scores = torch.stack([values[counterparts_device, column] for values in score_matrices], dim=0)
                counterpart_scores.masked_fill_(invalid_mask[counterparts_device, column].view(1, -1), -1.0)
                if fallback_capable:
                    fallback_records.append({
                        "split": split_name, "side": side, "endpoint": endpoint, "rows": block_rows,
                        "counterparts": counterparts.contiguous(),
                        "counterpart_scores": counterpart_scores.detach().cpu().contiguous(),
                        "state": cpu_state, "evidence_state": evidence_state,
                    })
                    stats["fallback_capable_occurrences"] += int(block_rows.numel())
                    continue
                (local_order, hard_counts) = _compact_occurrence_candidates(prepared, counterparts_device, counterpart_scores)
                hard_counts_cpu = hard_counts.detach().cpu()
                candidate_nodes = prepared["union"][local_order]
                _append_endpoint_candidate_rows(writer, side=side, endpoint=endpoint, row_ids=block_rows,
                                                candidate_nodes=candidate_nodes, hard_prefix_count=hard_counts_cpu, stats=stats)
        del prepared, metric_states


def _build_heart_negatives_gpu(adj, x, valid_pos, test_pos, filters, k2, seed, device, ppr_eps, tie_namespace="unspecified", endpoint_writers=None, endpoint_cache_path=None):
    device = torch.device(device)
    release_mutating_cosine = _uses_release_cosine(tie_namespace)
    if release_mutating_cosine and endpoint_writers is not None:
        raise RuntimeError(
            "Fixed-input Planetoid cosine compatibility requires the dense candidate representation. Planetoid splits are well below the streaming threshold."
        )
    alpha = _HEART_PPR_ALPHA
    eps_ppr = float(ppr_eps)
    endpoint_groups = _source_score_endpoint_groups(valid_pos, test_pos)
    nodes_eval = [node for _, group_nodes in endpoint_groups for node in group_nodes]
    split_endpoints = {"valid": set(valid_pos.reshape(-1).tolist()), "test": set(test_pos.reshape(-1).tolist())}
    (cpu_rowptr, cpu_col, _) = adj.csr()
    cpu_degree = cpu_rowptr[1:] - cpu_rowptr[:-1]
    print(
        f"Building Andersen/local-push PPR rows in bounded endpoint batches for {len(nodes_eval)} evaluation endpoints (eps={eps_ppr:g}).",
        flush=True,
    )
    with _strict_heart_math(), torch.no_grad():
        adj_device = adj.to(device)
        tie_priority_device = _heart_seeded_tie_priority(int(adj_device.size(0))).to(device=device)
        (rowptr_device, col_device, _) = adj_device.csr()
        deg = (rowptr_device[1:] - rowptr_device[:-1]).to(torch.float32)
        inv_deg = torch.zeros_like(deg)
        inv_deg[deg > 0] = 1.0 / deg[deg > 0]
        ra_projector = (adj_device * inv_deg.view(1, -1)).t()
        x_device = x.to(device=device, dtype=_HEART_SCORE_DTYPE, non_blocking=True)
        x_norm = x_device / x_device.norm(dim=1, keepdim=True).clamp_min(1e-15)
        if endpoint_writers is None:
            source_occurrences = _source_occurrence_schedule(valid_pos, test_pos)
            source_selected = {}
        else:
            source_schedules = {
                name: {
                    side: _endpoint_side_occurrence_schedule(pos, int(adj_device.size(0)), side)
                    for side in (0, 1)
                }
                for name, pos in (("valid", valid_pos), ("test", test_pos))
            }
            source_fallback_records = []
            stat_fields = ("groups", "occurrences", "union_nodes", "max_group_rows", "max_union_nodes", "fallback_capable_occurrences")
            source_stream_stats = {split_name: dict.fromkeys(stat_fields, 0) for split_name in ("valid", "test")}
        total = len(nodes_eval)
        batch_size = int(_HEART_SCORE_BATCH_SIZE)
        print(f"Building scalable GPU HeaRT negatives: endpoints={total} batch={batch_size} device={device}", flush=True)
        scored = 0
        for group_name, group_nodes in endpoint_groups:
            for start in range(0, len(group_nodes), batch_size):
                nodes = [int(v) for v in group_nodes[start : start + batch_size]]
                b = len(nodes)
                batch_exact_ppr = _andersen_ppr_for_eval_nodes(nodes, cpu_rowptr, cpu_col, cpu_degree, alpha, eps_ppr)
                score_shape = (int(adj_device.size(0)), b)
                w = torch.zeros(score_shape, dtype=_HEART_SCORE_DTYPE, device=device)
                for j, u in enumerate(nodes):
                    (s, e) = (int(cpu_rowptr[u]), int(cpu_rowptr[u + 1]))
                    if e > s:
                        neighbors = col_device[s:e]
                        w[neighbors, j] = 1.0
                ra = _source_compatible_ra(ra_projector, w)
                ppr = torch.zeros(score_shape, dtype=_HEART_SCORE_DTYPE, device=device)
                for column, node in enumerate(nodes):
                    (ids, values) = batch_exact_ppr[node]
                    ppr[ids.to(device), column] = values.to(device)
                node_tensor = torch.tensor(nodes, dtype=torch.long, device=device)
                cos = x_norm @ x_norm[node_tensor].t()
                for split_name, (filter_rowptr, filter_col) in filters.items():
                    split_columns = [column for column, node in enumerate(nodes) if node in split_endpoints[split_name]]
                    if not split_columns:
                        continue
                    column_ids = torch.tensor(split_columns, dtype=torch.long, device=device)
                    split_nodes = [nodes[column] for column in split_columns]
                    (invalid_mask, filter_bounds) = _endpoint_invalid_mask(split_nodes, filter_rowptr, filter_col, adj_device.size(0), device)
                    score_matrices = [torch.index_select(values, 1, column_ids) for values in (ra, ppr)]
                    if not release_mutating_cosine:
                        score_matrices.append(torch.index_select(cos, 1, column_ids))
                    if endpoint_writers is not None:
                        _stream_compact_endpoint_filter_batch(
                            split_name=split_name, split_nodes=split_nodes, score_matrices=score_matrices,
                            invalid_mask=invalid_mask, filter_rowptr=filter_rowptr, filter_col=filter_col,
                            schedules=source_schedules, positives=valid_pos if split_name == "valid" else test_pos,
                            k=k2, tie_priority=tie_priority_device, writer=endpoint_writers[split_name],
                            fallback_records=source_fallback_records, stats=source_stream_stats[split_name])
                        del score_matrices, invalid_mask, column_ids, filter_bounds
                        continue
                    metric_states = _prepare_masked_metric_rank_matrices(score_matrices, invalid_mask)
                    evidence_states = None
                    if not release_mutating_cosine:
                        evidence_states = [
                            _source_prepare_filter_evidence_state(metric_states, column, invalid_mask, k2,
                                filter_col=filter_col, filter_bounds=filter_bounds[column], endpoint=node)
                            for column, node in enumerate(split_nodes)
                        ]
                    for column, node in enumerate(split_nodes):
                        for occurrence in source_occurrences[node]:
                            if occurrence[1] != split_name:
                                continue
                            (occurrence_order, _, counterpart) = occurrence
                            if release_mutating_cosine:
                                (chosen, zero_pool) = _source_occurrence_candidates_with_mutating_cosine(
                                    metric_states, column, invalid_mask, counterpart, k2, cos[:, split_columns[column]],
                                    filter_col=filter_col, filter_bounds=filter_bounds[column], endpoint=node)
                            else:
                                (chosen, zero_pool) = _source_occurrence_candidates_from_metric_ranks(
                                    metric_states, column, invalid_mask, counterpart, k2, evidence_states[column]
                                )
                            source_selected[int(occurrence_order)] = (node, chosen, zero_pool)
                    del score_matrices, metric_states, invalid_mask, column_ids, evidence_states, filter_bounds
                scored += b
                print(f"  scored {scored}/{total} endpoint nodes ({group_name})", flush=True)
                del w, ra, ppr, batch_exact_ppr
                del cos
        del x_device, x_norm
        del adj_device, tie_priority_device
        torch.cuda.empty_cache()
    if endpoint_writers is None:
        return _source_assemble_selected_occurrences(valid_pos, test_pos, source_selected, k2, seed)
    _finalize_deferred_fallback_records(source_fallback_records, endpoint_writers, {"valid": valid_pos, "test": test_pos},
                                        seed, endpoint_cache_path, source_stream_stats)
    return source_stream_stats


def _build_heart_negatives_dense(adj, x, valid_pos, test_pos, filters, k2, seed, ppr_eps, tie_namespace="unspecified"):
    num_nodes = int(adj.size(0))
    (rowptr, col, _) = adj.csr()
    deg = (rowptr[1:] - rowptr[:-1]).to(_HEART_SCORE_DTYPE)
    inv_deg = torch.zeros_like(deg)
    inv_deg[deg > 0] = 1.0 / deg[deg > 0]
    ra_projector = (adj * inv_deg.view(1, -1)).t()
    x_f = x.to(dtype=_HEART_SCORE_DTYPE, device="cpu")
    x_norm = x_f / x_f.norm(dim=1, keepdim=True).clamp_min(1e-15)
    endpoint_groups = _source_score_endpoint_groups(valid_pos, test_pos)
    nodes_eval = [node for (_, group_nodes) in endpoint_groups for node in group_nodes]
    occurrences = _source_occurrence_schedule(valid_pos, test_pos)
    selected_by_order = {}
    alpha = _HEART_PPR_ALPHA
    eps_ppr = float(ppr_eps)
    for index, u in enumerate(nodes_eval, 1):
        u = int(u)
        w = torch.zeros(num_nodes, dtype=_HEART_SCORE_DTYPE)
        (s, e) = (int(rowptr[u]), int(rowptr[u + 1]))
        neighbors = col[s:e]
        if neighbors.numel():
            w[neighbors] = 1.0
        ra_base = _source_compatible_ra(ra_projector, w.view(-1, 1)).view(-1).to(_HEART_SCORE_DTYPE)
        ppr_base = torch.zeros(num_nodes, dtype=_HEART_SCORE_DTYPE)
        (ppr_ids, ppr_values) = _andersen_ppr_for_node(u, rowptr, col, deg, alpha=alpha, eps=eps_ppr)
        ppr_base[ppr_ids] = ppr_values.to(_HEART_SCORE_DTYPE)
        cos_base = torch.mv(x_norm, x_norm[u]).to(_HEART_SCORE_DTYPE)
        selected_by_order.update(_source_select_endpoint_occurrences(
            u, occurrences[u], (ra_base, ppr_base, cos_base), filters, k2, num_nodes, tie_namespace))
        if index % 25 == 0 or index == len(nodes_eval):
            print(f"  ranked {index}/{len(nodes_eval)} endpoint nodes", flush=True)
    return _source_assemble_selected_occurrences(valid_pos, test_pos, selected_by_order, k2, seed)


def _select_ranked_evaluation_positives(all_valid_pos, all_test_pos, eval_cap, default_eval_cap, seed):
    cap = int(eval_cap or 0)
    if cap <= 0:
        cap = int(default_eval_cap or 0)
    if cap > 0 and int(all_valid_pos.size(0)) > cap:
        return (_sample_rows(all_valid_pos, cap, int(seed) + 100), all_test_pos, cap)
    return (all_valid_pos, all_test_pos, 0)


def _prepare_ranked_base(data_name, split, seed, root, eval_cap, default_eval_cap=0, positive_split=None):
    d = _load_dataset(data_name, root)
    num_nodes = int(d.num_nodes)
    x = d.x
    feature_source = "raw-pyg-dataset-x"
    feature_path = None
    train_rowptr = train_col = None
    print(f"the number of nodes in {data_name} is: ", num_nodes)
    if positive_split is None:
        split_values = _load_or_create_split(d, data_name, split, seed, root, return_metadata=True)
        (train_uv, valid_uv, test_uv, train_rowptr, train_col, split_metadata) = split_values
        positive_split_source = "seeded-pyg-edge-split"
        positive_split_dir = None
    else:
        def checked_positive(name):
            value = torch.as_tensor(positive_split[name], dtype=torch.long).cpu()
            if value.ndim != 2 or value.size(1) != 2:
                raise ValueError(f"{name} must have shape [N, 2], got {tuple(value.shape)}")
            if value.numel() and (int(value.min()) < 0 or int(value.max()) >= num_nodes):
                raise ValueError(f"{name} contains a node ID outside [0, {num_nodes})")
            if value.numel() and bool(value[:, 0].eq(value[:, 1]).any()):
                raise ValueError(f"{name} contains self-loop positives")
            return value.t().contiguous()

        train_uv, valid_uv, test_uv = (checked_positive(name) for name in ("train_pos", "all_valid_pos", "test_pos"))
        feature_dir = str(positive_split.get("artifact_dir") or "")
        (x, feature_path) = load_fixed_planetoid_features(feature_dir, num_nodes)
        if x is None:
            feature_path = os.path.join(feature_dir, "gnn_feature")
            raise FileNotFoundError(
                f"Generated Planetoid HeaRT requires gnn_feature next to the fixed positive split files: {feature_path}"
            )
        feature_source = "fixed-planetoid-gnn-feature"

        def canonical_ids(edges):
            left = torch.minimum(edges[0], edges[1])
            right = torch.maximum(edges[0], edges[1])
            return left * num_nodes + right

        supplied_ids = torch.cat([canonical_ids(edges) for edges in (train_uv, valid_uv, test_uv)])
        raw_edges = d.edge_index.to(device="cpu", dtype=torch.long)
        raw_edges = raw_edges[:, raw_edges[0] != raw_edges[1]]
        raw_ids = torch.unique(canonical_ids(raw_edges), sorted=True)
        supplied_unique = torch.unique(supplied_ids, sorted=True)
        if supplied_ids.numel() != supplied_unique.numel() or not torch.equal(supplied_unique, raw_ids):
            raise ValueError("Fixed positive split files do not form a duplicate-free partition of the local PyG Planetoid graph.")
        all_uv = torch.cat([train_uv, valid_uv, test_uv], dim=1)
        split_metadata = {
            "split_cache_digest_version": 1,
            "split_tensor_sha256": _split_tensor_sha256(train_uv, valid_uv, test_uv),
            "known_graph_sha256": _canonical_known_graph_sha256(all_uv, num_nodes),
            "known_graph_sha256_method": "full-sorted-canonical-undirected-edge-sha256-v1",
            "raw_graph_identity": _bounded_raw_graph_identity(d.edge_index, num_nodes),
            "raw_graph_identity_method": "bounded-canonical-raw-edge-sample-v1-probabilistic",
        }
        del all_uv
        positive_split_source = str(
            positive_split.get(
                "positive_split_source", "fixed-planetoid-positive-txt"
            )
        )
        positive_split_dir = positive_split.get("artifact_dir")
    adj = _make_adj(train_uv, num_nodes, train_rowptr, train_col)
    train_pos = train_uv.t().to(torch.long).contiguous()
    all_valid_pos = valid_uv.t().to(torch.long).contiguous()
    test_pos = test_uv.t().to(torch.long).contiguous()
    (valid_pos, test_pos, effective_cap) = _select_ranked_evaluation_positives(all_valid_pos, test_pos, eval_cap, default_eval_cap, seed)
    (tr_rowptr, tr_col, _) = adj.csr()
    known_positive_count = int(train_pos.size(0)) + int(all_valid_pos.size(0)) + int(test_pos.size(0))
    return {
        "adj": adj,
        "x": x,
        "num_nodes": num_nodes,
        "train_pos": train_pos,
        "train_val": _sample_rows(train_pos, valid_pos.size(0), seed + 3),
        "valid_pos": valid_pos,
        "test_pos": test_pos,
        "all_valid_pos": all_valid_pos,
        "all_test_pos": test_pos,
        "csr_train_rowptr": tr_rowptr,
        "csr_train_col": tr_col,
        "known_positive_count": known_positive_count,
        "split_cache_digest_version": split_metadata["split_cache_digest_version"],
        "split_tensor_sha256": split_metadata["split_tensor_sha256"],
        "known_graph_sha256": split_metadata["known_graph_sha256"],
        "known_graph_sha256_method": split_metadata["known_graph_sha256_method"],
        "raw_graph_identity": split_metadata["raw_graph_identity"],
        "raw_graph_identity_method": split_metadata["raw_graph_identity_method"],
        "split": tuple((float(value) for value in split)),
        "effective_eval_cap": effective_cap,
        "effective_validation_cap": effective_cap,
        "effective_test_cap": 0,
        "heart_positive_split_source": positive_split_source,
        "heart_positive_split_dir": positive_split_dir,
        "heart_fixed_positive_split": positive_split is not None,
        "heart_feature_source": feature_source,
        "heart_feature_path": feature_path,
    }


def _ensure_test_positive_filter(base):
    keys = ("csr_test_filter_rowptr", "csr_test_filter_col")
    if all((key in base for key in keys)):
        return (base[keys[0]], base[keys[1]])
    positives = torch.cat([base["train_pos"], base["all_valid_pos"]], dim=0)
    edge_index = positives.to(device="cpu", dtype=torch.long).t().contiguous()
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    adjacency = SparseTensor.from_edge_index(edge_index, sparse_sizes=(int(base["num_nodes"]), int(base["num_nodes"])))
    (rowptr, col, _) = adjacency.csr()
    (base[keys[0]], base[keys[1]]) = (rowptr, col)
    return (rowptr, col)


def _resolve_heart_backend(heart_backend, heart_ppr_iters, heart_device=None):
    backend = str(heart_backend or "auto").lower()
    if backend == "auto":
        device = torch.device(heart_device or ("cuda" if torch.cuda.is_available() else "cpu"))
        backend = "gpu" if device.type == "cuda" else "dense"
    if backend not in {"gpu", "dense"}:
        raise ValueError(f"Unknown HeaRT backend: {heart_backend!r}")
    if heart_ppr_iters not in (None, 0):
        print("Ignoring heart_ppr_iters: generated PyG HeaRT always uses the fixed Andersen local-push PPR protocol.", flush=True)
    return (backend, 0)


def _load_or_build_full_graph_heart(base, data_name, root, seed, requested_draw, heart_backend, heart_device, heart_batch_size, heart_ppr_iters):
    (backend, heart_ppr_iters) = _resolve_heart_backend(heart_backend, heart_ppr_iters, heart_device)
    draw_per_side = int(requested_draw)
    cache_path = _heart_pool_cache_file(root, data_name, base["split"], seed, base["effective_eval_cap"], base["num_nodes"],
                                        draw_per_side, backend, heart_ppr_iters, True)
    ppr_eps = _heart_ppr_eps(data_name)
    metadata = _heart_cache_metadata(base, data_name, seed, draw_per_side, backend, heart_ppr_iters, ppr_eps)
    storage = metadata["candidate_storage_format"]
    recipe = _heart_candidate_recipe_sha256(metadata)
    splits = ("valid", "test")
    positives = {name: base[f"{name}_pos"] for name in splits}
    manifests = {name: f"{cache_path}.{name}.endpoint.json" for name in splits}
    common_payload = {
        "metadata": metadata,
        "valid_pos": positives["valid"],
        "test_pos": positives["test"],
        "candidate_storage_format": storage,
        "candidate_recipe_sha256": recipe,
    }

    def validate_dense(candidates, known_filters=None):
        validated = {}
        for name in splits:
            known = (None, None) if known_filters is None else known_filters[name]
            validated[name] = _validate_generated_heart_negatives(
                "validation" if name == "valid" else "test", candidates[name], positives[name], draw_per_side,
                base["num_nodes"], *known)
        return validated

    def try_load_cache():
        if not os.path.exists(cache_path):
            return None
        try:
            payload = _torch_load(cache_path)
            stored_metadata = payload.get("metadata")
            if not isinstance(stored_metadata, dict):
                raise ValueError("missing cache metadata")
            cached_metadata = _normalize_heart_cache_metadata(stored_metadata)
            mismatches = [f"{key}={cached_metadata.get(key)!r} (expected {value!r})"
                          for key, value in metadata.items() if cached_metadata.get(key) != value]
            if mismatches:
                raise ValueError("cache metadata mismatch: " + "; ".join(mismatches))
            for name in splits:
                if not torch.equal(payload[f"{name}_pos"], positives[name].cpu()):
                    raise ValueError(f"cached {'validation' if name == 'valid' else 'test'} positives do not match split")

            if storage == ENDPOINT_GROUPED_FORMAT:
                stored = payload.get("endpoint_manifest_paths")
                if not isinstance(stored, dict):
                    raise ValueError("cached endpoint manifest paths are missing")
                paths = {name: os.path.abspath(str(stored.get(name, ""))) for name in splits}
                if paths != {name: os.path.abspath(manifests[name]) for name in splits}:
                    raise ValueError("cached endpoint manifest paths do not match")
                digest = _heart_endpoint_artifact_sha256(paths["valid"], paths["test"])
                if payload.get("candidate_artifact_sha256") != digest:
                    raise ValueError("cached endpoint artifact digest is missing or invalid")
                cached_recipe = payload.get("candidate_recipe_sha256") or _heart_candidate_recipe_sha256(stored_metadata)
                candidates = {name: load_streamed_grouped_negatives(paths[name], positives[name],
                    expected_recipe_sha256=cached_recipe, verify_shards="lazy") for name in splits}
                print(f"Loading cached endpoint-grouped HeaRT negatives: {cache_path}", flush=True)
                return (candidates["valid"], candidates["test"], digest, payload.get("stream_stats"))

            digest = _heart_candidate_tensor_sha256(payload["valid_neg"], payload["test_neg"])
            if payload.get("candidate_tensor_sha256") != digest:
                raise ValueError("cached candidate tensor digest is missing or does not match")
            candidates = validate_dense({name: payload[f"{name}_neg"] for name in splits})
            print(f"Loading cached HeaRT negatives: {cache_path}", flush=True)
            return (candidates["valid"], candidates["test"], digest, None)
        except Exception as exc:
            print(f"Ignoring invalid HeaRT cache {cache_path}: {exc}", flush=True)
            return None

    def build_cache():
        if backend == "gpu" and (torch.device(heart_device or "cuda").type != "cuda" or not torch.cuda.is_available()):
            raise RuntimeError("GPU HeaRT generation requires a CUDA device; an existing GPU cache may still be read on CPU.")
        filters = {
            "valid": (base["csr_train_rowptr"], base["csr_train_col"]),
            "test": _ensure_test_positive_filter(base),
        }
        generation_args = (base["adj"], base["x"], positives["valid"], positives["test"], filters,
                           draw_per_side, _HEART_CANDIDATE_SEED)
        print(
            f"Building HeaRT top-{draw_per_side} per side with {storage}; heuristics={'+'.join(_HEART_HEURISTICS)}; "
            "validation filter=train; test filter=train+uncapped-valid; exact query masking; released fallback RNG.",
            flush=True,
        )

        if storage == ENDPOINT_GROUPED_FORMAT:
            if backend != "gpu":
                raise RuntimeError("Large uncapped generated HeaRT candidates require the scalable GPU endpoint-grouped backend.")
            candidates = {}
            for name in splits:
                if not os.path.exists(manifests[name]):
                    continue
                try:
                    candidate = load_streamed_grouped_negatives(manifests[name], positives[name],
                                                                expected_recipe_sha256=recipe, verify_shards="eager")
                    for _ in candidate.iter_endpoint_group_chunks():
                        pass
                    candidates[name] = candidate
                except Exception as exc:
                    print(f"Ignoring incomplete endpoint-grouped HeaRT {name} manifest {manifests[name]}: {exc}", flush=True)
            if len(candidates) == 2:
                print("Reusing complete validation+test endpoint-grouped HeaRT manifest pair.", flush=True)
                stream_stats = {"resumed_complete_split_manifests": list(splits)}
            else:
                if len(candidates) == 1:
                    print("Only one endpoint manifest is reusable; rebuilding validation and test together as one semantic unit.", flush=True)
                writer_options = {
                    "num_nodes": base["num_nodes"], "candidate_recipe_sha256": recipe,
                    "negatives_per_side": draw_per_side,
                    "occurrence_rows_per_shard": _HEART_ENDPOINT_GROUP_MAX_ROWS,
                }
                with (
                    IndependentSideBufferedEndpointGroupedNegativeWriter(
                        manifests["valid"], positives["valid"], split="valid", **writer_options
                    ) as valid_writer,
                    IndependentSideBufferedEndpointGroupedNegativeWriter(
                        manifests["test"], positives["test"], split="test", **writer_options
                    ) as test_writer,
                ):
                    writers = {"valid": valid_writer, "test": test_writer}
                    stream_stats = _build_heart_negatives_gpu(*generation_args, heart_device or "cuda", ppr_eps,
                        tie_namespace=str(data_name).lower(), endpoint_writers=writers, endpoint_cache_path=cache_path)
                    candidates = {name: writer.finish(verify_shards="lazy") for name, writer in writers.items()}
            digest = _heart_endpoint_artifact_sha256(manifests["valid"], manifests["test"])
            _atomic_save(
                {
                    **common_payload,
                    "endpoint_manifest_paths": {name: os.path.abspath(manifests[name]) for name in splits},
                    "candidate_artifact_sha256": digest,
                    "candidate_artifact_sha256_method": "sha256-of-valid-test-manifest-bytes-v1",
                    "stream_stats": stream_stats,
                },
                cache_path,
            )
            print(f"Saved endpoint-grouped HeaRT cache pair: {cache_path}", flush=True)
            return (candidates["valid"], candidates["test"], digest, stream_stats)

        if backend == "gpu":
            try:
                generated = _build_heart_negatives_gpu(*generation_args, heart_device or "cuda", ppr_eps,
                                                        tie_namespace=str(data_name).lower())
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        f"PyG HeaRT generation exhausted GPU memory at its fixed reproducible score batch={_HEART_SCORE_BATCH_SIZE}."
                    ) from exc
                raise
        else:
            generated = _build_heart_negatives_dense(*generation_args, ppr_eps, tie_namespace=str(data_name).lower())
        candidates = validate_dense(dict(zip(splits, generated)), filters)
        digest = _heart_candidate_tensor_sha256(candidates["valid"], candidates["test"])
        try:
            _atomic_save({**common_payload, "valid_neg": candidates["valid"], "test_neg": candidates["test"],
                          "candidate_tensor_sha256": digest}, cache_path)
            print(f"Saved HeaRT negative cache: {cache_path}", flush=True)
        except Exception as exc:
            print(f"WARNING: could not save HeaRT cache: {exc}", flush=True)
        return (candidates["valid"], candidates["test"], digest, None)

    cached = try_load_cache()
    if cached is None:
        with _exclusive_cache_build(cache_path):
            cached = try_load_cache() or build_cache()
    (valid_neg, test_neg, candidate_digest, stream_stats) = cached
    returned_metadata = dict(metadata)
    if storage == ENDPOINT_GROUPED_FORMAT:
        returned_metadata.update(
            {
                "candidate_artifact_sha256": candidate_digest,
                "candidate_artifact_sha256_method": "sha256-of-valid-test-manifest-bytes-v1",
                "candidate_tensor_sha256": candidate_digest,
                "candidate_tensor_sha256_method": (
                    "compatibility-alias-of-candidate-artifact-sha256;no-dense-tensor-materialized"
                ),
                "candidate_stream_stats": stream_stats,
            }
        )
    else:
        returned_metadata.update(
            {
                "candidate_tensor_sha256": candidate_digest,
                "candidate_tensor_sha256_method": "full-logical-dense-int64-tensor-sha256-v1",
            }
        )
    return (valid_neg, test_neg, draw_per_side, backend, cache_path, returned_metadata)
