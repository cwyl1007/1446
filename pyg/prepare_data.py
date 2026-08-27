import os
import torch
from .data_core import (
    _full_integer_tensor_sha256,
    _load_dataset,
    _resolve_pool_request,
    parse_pool_argument,
)
from .heart_generation import _load_or_build_full_graph_heart, _prepare_ranked_base
from .planetoid_inputs import (
    FIXED_PLANETOID_POSITIVE_SPLIT_DATASETS,
    load_fixed_planetoid_positive_split,
    missing_fixed_planetoid_positive_split_message,
)
from utils.heart_protocol import heart_negative_count_metadata, warn_if_custom_heart_negative_count
from ogbl.ranked_candidates import load_or_build_ranked_valid_pool

_PYG_HEART_VALIDATION_MAX = 100000
_REDDIT_HEART_EVAL_CAP = _PYG_HEART_VALIDATION_MAX


def _reddit_heart_query_metadata(data_name, base, seed):
    if str(data_name).strip().lower() != "reddit":
        return {}
    effective_cap = int(base.get("effective_validation_cap", base.get("effective_eval_cap", 0)) or 0)
    valid_population = int(base["all_valid_pos"].size(0))
    test_population = int(base["all_test_pos"].size(0))
    valid_selected = int(base["valid_pos"].size(0))
    test_selected = int(base["test_pos"].size(0))
    valid_sampled = valid_selected < valid_population
    test_sampled = test_selected < test_population
    return {
        "heart_query_scope": (
            "reddit-validation-deterministic-100000-cap-test-complete"
            if effective_cap == _REDDIT_HEART_EVAL_CAP and (not test_sampled)
            else (
                "reddit-complete-positive-splits"
                if not valid_sampled and (not test_sampled)
                else "reddit-generated-deterministic-custom-cap"
            )
        ),
        "heart_query_sampling": (
            "validation-uniform-without-replacement;test-complete-ordered"
            if valid_sampled and (not test_sampled)
            else (
                "complete-ordered-positive-splits" if not valid_sampled and (not test_sampled) else "split-specific-positive-query-sampling"
            )
        ),
        "heart_query_validation_sampling": (
            "uniform-without-replacement-torch-generator-v1" if valid_sampled else "complete-ordered-positive-split"
        ),
        "heart_query_test_sampling": (
            "uniform-without-replacement-torch-generator-v1" if test_sampled else "complete-ordered-positive-split"
        ),
        "heart_query_cap": effective_cap,
        "heart_query_validation_cap": effective_cap,
        "heart_query_test_cap": test_selected if test_sampled else 0,
        "heart_query_valid_seed": int(seed) + 100 if valid_sampled else None,
        "heart_query_test_seed": int(seed) + 101 if test_sampled else None,
        "heart_query_valid_population_count": valid_population,
        "heart_query_test_population_count": test_population,
        "heart_query_valid_selected_count": valid_selected,
        "heart_query_test_selected_count": test_selected,
        "heart_query_selected_tensor_sha256_method": "full-integer-tensor-sha256-v1" if valid_sampled or test_sampled else None,
        "heart_query_valid_selected_tensor_sha256": _full_integer_tensor_sha256(base["valid_pos"]) if valid_sampled else None,
        "heart_query_test_selected_tensor_sha256": _full_integer_tensor_sha256(base["test_pos"]) if test_sampled else None,
        "heart_query_released_reference_protocol": False,
    }


def resolve_pyg_eval_cap(eval_cap, mode, data_name):
    normalized_mode = str(mode).strip().lower()
    if normalized_mode not in {"heart", "all"}:
        raise ValueError(f"Unsupported PyG mode: {mode}. Use heart or all.")
    if normalized_mode == "heart":
        explicit_cap = None if eval_cap is None else int(eval_cap)
        if explicit_cap is not None and explicit_cap < 0:
            raise ValueError("--eval-cap must be non-negative.")
        if explicit_cap in (None, 0):
            return _PYG_HEART_VALIDATION_MAX
        return min(explicit_cap, _PYG_HEART_VALIDATION_MAX)
    if eval_cap is not None:
        return int(eval_cap)
    return 500


def read_pyg_all_test(
    data_name,
    split=(0.7, 0.15, 0.15),
    seed=0,
    root="dataset",
    eval_cap=0,
    pool=10000,
    heart_backend="auto",
    heart_device=None,
    heart_batch_size=128,
    heart_ppr_iters=None,
):
    probe = _load_dataset(data_name, root)
    num_nodes = int(probe.num_nodes)
    del probe
    (requested_pool, pool_request, full_graph_pool) = _resolve_pool_request(pool, num_nodes)
    original_requested_pool = int(requested_pool)
    requested_pool = min(original_requested_pool, 10000)
    if requested_pool < original_requested_pool:
        print(
            f"all-mode pool request reduced from {original_requested_pool} to the strict maximum of 10000 candidates per side.", flush=True
        )
    base = _prepare_ranked_base(data_name, split, seed, root, eval_cap, default_eval_cap=500)
    (valid_pool, test_pool, backend, cache_path) = load_or_build_ranked_valid_pool(
        base,
        data_name,
        requested_pool,
        seed,
        base["effective_eval_cap"],
        heart_backend,
        os.path.join(root, "lp_cache"),
        True,
        device=heart_device,
        batch_size=heart_batch_size,
        ppr_iters=heart_ppr_iters,
    )
    side_counts = torch.cat([valid_pool.side_counts.reshape(-1), test_pool.side_counts.reshape(-1)])
    row_counts = torch.cat([valid_pool.rowptr[1:] - valid_pool.rowptr[:-1], test_pool.rowptr[1:] - test_pool.rowptr[:-1]])
    side_min = int(side_counts.min()) if side_counts.numel() else 0
    side_max = int(side_counts.max()) if side_counts.numel() else 0
    side_mean = float(side_counts.to(torch.float64).mean()) if side_counts.numel() else 0.0
    total_min = int(row_counts.min()) if row_counts.numel() else 0
    total_max = int(row_counts.max()) if row_counts.numel() else 0
    total_mean = float(row_counts.to(torch.float64).mean()) if row_counts.numel() else 0.0
    result = {
        "adj": base["adj"],
        "train_pos": base["train_pos"],
        "train_val": base["train_val"],
        "valid_pos": base["valid_pos"],
        "valid_neg": valid_pool,
        "test_pos": base["test_pos"],
        "test_neg": test_pool,
        "x": base["x"],
        "csr_train_rowptr": base["csr_train_rowptr"],
        "csr_train_col": base["csr_train_col"],
        "mode": "all",
        "pool_setting": pool_request,
        "pool_request": pool_request,
        "pool_full_graph": full_graph_pool,
        "pool_cap_applied": True,
        "pool_sampling": "positive-ra-ppr-ranked-valid-unpadded",
        "pool_graph_nodes_per_side": base["num_nodes"],
        "pool_per_side": side_max,
        "pool_per_side_min": side_min,
        "pool_per_side_mean": side_mean,
        "pool_per_side_max": side_max,
        "pool_requested_per_side": requested_pool,
        "pool_user_requested_per_side": original_requested_pool,
        "pool_total": total_max,
        "pool_total_min": total_min,
        "pool_total_mean": total_mean,
        "pool_total_max": total_max,
        "pool_requested_total": 2 * requested_pool,
        "pool_user_requested_total": 2 * original_requested_pool,
        "all_negatives": total_max,
        "all_negatives_requested": 2 * requested_pool,
        "ranked_backend": backend,
        "negative_cache_path": cache_path,
    }
    return result


def read_pyg_heart(
    data_name,
    split=(0.7, 0.15, 0.15),
    seed=0,
    root="dataset",
    eval_cap=None,
    heart_backend="auto",
    heart_device=None,
    heart_batch_size=128,
    heart_ppr_iters=None,
    heart_negatives=500,
    pool=10000,
    planetoid_input_root=None,
):
    resolved_eval_cap = (
        resolve_pyg_eval_cap(None, "heart", data_name)
        if eval_cap is None
        else int(eval_cap)
    )
    eval_cap = resolved_eval_cap
    validation_cap = _PYG_HEART_VALIDATION_MAX if resolved_eval_cap <= 0 else min(resolved_eval_cap, _PYG_HEART_VALIDATION_MAX)
    requested_total = int(heart_negatives)
    requested_per_side = requested_total // 2
    fixed_split_dataset = (
        str(data_name).lower() in FIXED_PLANETOID_POSITIVE_SPLIT_DATASETS
    )
    positive_split = None
    if fixed_split_dataset:
        positive_split = load_fixed_planetoid_positive_split(
            data_name, root=root, input_root=planetoid_input_root
        )
        if positive_split is None:
            raise FileNotFoundError(
                missing_fixed_planetoid_positive_split_message(
                    data_name, root=root, input_root=planetoid_input_root
                )
            )
        print(
            "HeaRT negatives=generated-online with fixed positive split; "
            f"split_dir={positive_split['artifact_dir']} "
            "feature_source=fixed-planetoid-gnn-feature",
            flush=True,
        )
    base = _prepare_ranked_base(data_name, split, seed, root, validation_cap, default_eval_cap=0, positive_split=positive_split)
    del pool
    warn_if_custom_heart_negative_count(requested_total, effective_total=requested_total, source="generated")
    try:
        (valid_neg, test_neg, draw_per_side, backend, cache_path, cache_metadata) = _load_or_build_full_graph_heart(
            base, data_name, root, seed, requested_per_side, heart_backend, heart_device, heart_batch_size, heart_ppr_iters
        )
    finally:
        base.pop("csr_test_filter_rowptr", None)
        base.pop("csr_test_filter_col", None)
    selection_protocol = str(cache_metadata["selection_protocol"])
    hard_topk_per_side = int(cache_metadata["selection_hard_topk_per_side"])
    heuristic_set = list(cache_metadata["heuristic_set"])
    tie_break_seed = cache_metadata["selection_tie_break_seed"]
    print(
        f"HeaRT selection={selection_protocol} hard_topk_per_side={hard_topk_per_side} selected_total={2 * draw_per_side} selected_per_side={draw_per_side}",
        flush=True,
    )
    reddit_query_metadata = _reddit_heart_query_metadata(data_name, base, seed)
    result = {
        "adj": base["adj"],
        "train_pos": base["train_pos"],
        "train_val": base["train_val"],
        "valid_pos": base["valid_pos"],
        "valid_neg": valid_neg,
        "test_pos": base["test_pos"],
        "test_neg": test_neg,
        "x": base["x"],
        "csr_train_rowptr": base["csr_train_rowptr"],
        "csr_train_col": base["csr_train_col"],
        "mode": "heart",
        "heart_negatives_per_side": draw_per_side,
        "heart_negatives_requested_per_side": requested_per_side,
        "heart_negatives_total": 2 * draw_per_side,
        "heart_negatives_requested_total": requested_total,
        **heart_negative_count_metadata(requested_total, effective_total=2 * draw_per_side),
        "heart_selection": selection_protocol,
        "heart_candidate_universe": "full-legal-graph",
        "heart_candidate_graph_nodes": base["num_nodes"],
        "heart_source": "generated-online",
        "heart_positive_split_source": base["heart_positive_split_source"],
        "heart_positive_split_dir": base["heart_positive_split_dir"],
        "heart_fixed_positive_split": base["heart_fixed_positive_split"],
        "heart_feature_source": base["heart_feature_source"],
        "heart_feature_path": base["heart_feature_path"],
        "heart_hard_topk_per_side": hard_topk_per_side,
        "heart_selection_selected_per_side": draw_per_side,
        "heart_known_positive_count": base["known_positive_count"],
        "heart_heuristic_set": heuristic_set,
        "heart_feature_used_for_selection": "cosine" in heuristic_set,
        **reddit_query_metadata,
        "ranked_backend": backend,
        "negative_cache_path": cache_path,
    }
    direct_keys = (
        "rank_aggregation",
        "candidate_slot_capacity_policy",
        "selection_policy",
        "metric_zero_rank_policy",
        "metric_invalid_rank_policy",
        "selection_tie_break",
        "selection_zero_evidence_fill",
        "query_counterpart_mask_stage",
        "query_counterpart_rank_correction",
        "candidate_filter_scope",
        "validation_filter_scope",
        "test_filter_scope",
        "validation_filter_fingerprint",
        "validation_filter_fingerprint_method",
        "test_filter_fingerprint",
        "test_filter_fingerprint_method",
        "selection_calibration",
        "complete_positive_graph_fingerprint",
        "complete_positive_graph_fingerprint_method",
        "split_tensor_sha256",
        "raw_graph_identity",
        "raw_graph_identity_method",
        "graph_sample_sha256",
        "selector_implementation_sha256",
        "cache_validation_contract",
        "candidate_storage_policy",
        "candidate_storage_format",
        "candidate_tensor_sha256",
        "candidate_tensor_sha256_method",
        "cosine_state_policy",
        "feature_fingerprint_method",
        "reference_fallback_sampling_deviation",
        "released_artifact_exact_reason",
        "eligibility_policy",
        "reference_eligibility_deviation",
        "ppr_method",
        "endpoint_score_schedule",
        "score_dtype",
        "score_math",
        "ra_semantics",
        "fallback_sampling",
        "duplicate_candidates_scope",
    )
    int_keys = (
        "split_cache_digest_version",
        "candidate_logical_dense_bytes",
        "candidate_dense_logical_byte_limit",
        "ppr_iters",
        "score_batch_size",
        "fallback_rng_seed",
    )
    bool_keys = (
        "released_cosine_compat_exact",
        "reference_candidate_count_exact",
        "reference_rank_fusion_exact",
        "reference_hard_topk_exact",
        "reference_hard_topk_core_exact",
        "reference_fallback_sampling_exact",
        "released_artifact_exact",
        "reference_source_filter_exact",
        "unique_candidates_per_positive_side",
    )
    result.update({f"heart_{key}": cache_metadata[key] for key in direct_keys})
    result.update({f"heart_{key}": int(cache_metadata[key]) for key in int_keys})
    result.update({f"heart_{key}": bool(cache_metadata[key]) for key in bool_keys})
    result.update({f"heart_{key}": float(cache_metadata[key]) for key in ("ppr_alpha", "ppr_eps")})
    result.update(
        {
            f"heart_{key}": cache_metadata.get(key)
            for key in ("candidate_artifact_sha256", "candidate_artifact_sha256_method", "candidate_stream_stats")
        }
    )
    result.update(
        {
            "heart_selection_tie_break_seed": None if tie_break_seed is None else int(tie_break_seed),
            "heart_recipe_version": int(cache_metadata["candidate_cache_version"]),
            "heart_cache_metadata_version": int(cache_metadata["metadata_version"]),
            "heart_selector_recipe_version": int(cache_metadata["selector_recipe_version"]),
            "heart_feature_fingerprint": cache_metadata["feature_sample_sha256"],
        }
    )
    return result


def read_data(
    data,
    mode,
    eval_cap=None,
    seed=0,
    root="dataset",
    heart_backend="auto",
    heart_device=None,
    heart_batch_size=128,
    heart_ppr_iters=None,
    heart_negatives=500,
    pool=10000,
    planetoid_input_root=None,
):
    mode = str(mode).strip().lower()
    eval_cap = resolve_pyg_eval_cap(eval_cap, mode, data)
    split = (0.85, 0.05, 0.1)
    if mode == "all":
        out = read_pyg_all_test(
            data,
            split=split,
            seed=seed,
            root=root,
            eval_cap=eval_cap,
            pool=pool,
            heart_backend=heart_backend,
            heart_device=heart_device,
            heart_batch_size=heart_batch_size,
            heart_ppr_iters=heart_ppr_iters,
        )
    else:
        out = read_pyg_heart(
            data,
            split=split,
            seed=seed,
            root=root,
            eval_cap=eval_cap,
            heart_backend=heart_backend,
            heart_device=heart_device,
            heart_batch_size=heart_batch_size,
            heart_ppr_iters=heart_ppr_iters,
            heart_negatives=heart_negatives,
            pool=pool,
            planetoid_input_root=planetoid_input_root,
        )
    out["mode"] = mode
    return out
