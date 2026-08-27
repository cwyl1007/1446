import copy
import os
import random
import time
from typing import Any, Dict
import numpy as np
import torch

RUNTIME_LIMIT_SEC = 24 * 60 * 60
_REFERENCE_HEART_EVAL_CAPS = {"ogbl-collab": 0, "ogbl-ddi": 0, "ogbl-ppa": 100000, "ogbl-citation2": 100000}
_HEART_VALIDATION_MAX = 100000
_NATIVE_HEART_VALIDATION_COUNTS = {"ogbl-collab": 60084, "ogbl-ddi": 133489, "ogbl-ppa": 6062562, "ogbl-citation2": 86596}
_OFFICIAL_SELECTION_METRICS = {"ogbl-collab": "Hits@50", "ogbl-ddi": "Hits@20", "ogbl-ppa": "Hits@100", "ogbl-citation2": "MRR"}
def resolve_ppa_query_panel_mode(value=None):
    from .ppa_query_panel import resolve_ppa_query_panel_mode as resolve

    return resolve(value)


def _ppa_local_scope():
    from .ppa_query_panel import PPA_LOCAL_SCOPE

    return PPA_LOCAL_SCOPE


def _dataset_key(dataset: str) -> str:
    return str(dataset).strip().lower()


def normalize_ogbl_mode(mode: str) -> str:
    value = str(mode).strip().lower()
    if value not in {"heart", "all"}:
        raise ValueError(f"Unsupported OGB evaluation mode: {mode!r}.")
    return value


def resolve_ogbl_device(value=None):
    cuda_available = torch.cuda.is_available()
    device = torch.device(value or ("cuda" if cuda_available else "cpu"))
    if device.type == "cuda" and not cuda_available:
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable.")
    return device


def set_seed(seed, deterministic_cudnn=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic_cudnn:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def runtime_exceeded(start):
    return time.time() - start >= RUNTIME_LIMIT_SEC


def checkpoint_metric(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def save_model_checkpoint(
    project_root,
    state_dict,
    *,
    framework,
    mode,
    dataset,
    model_name,
    run_number,
    seed,
    epoch,
    timed_out,
    metric_name,
    best_val,
    best_test,
    args,
    model_config,
    checkpoint_type="final_model_state",
    extra_metadata=None,
):
    checkpoint_dir = os.path.join(
        getattr(args, "checkpoint_root", None) or os.path.join(project_root, "checkpoints"),
        str(mode),
        str(dataset),
        str(model_name),
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint{int(run_number)}")
    checkpoint = {
        "format_version": 1,
        "checkpoint_type": str(checkpoint_type),
        "framework": str(framework),
        "mode": str(mode),
        "dataset": str(dataset),
        "model": str(model_name),
        "run": int(run_number),
        "seed": int(seed),
        "epoch": int(epoch),
        "timed_out": bool(timed_out),
        "selection_metric": str(metric_name),
        "best_validation_metric": checkpoint_metric(best_val),
        "best_test_metric": checkpoint_metric(best_test),
        "arguments": dict(vars(args)),
        "model_config": dict(model_config),
        "model_state_dict": state_dict,
    }
    checkpoint.update(extra_metadata or {})
    checkpoint["heart_source_resolved"] = getattr(args, "heart_source_resolved", None)
    checkpoint["heart_candidate_metadata"] = dict(getattr(args, "heart_candidate_metadata", {}))
    temporary_path = checkpoint_path + ".tmp"
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return checkpoint_path


def snapshot_state_dict_cpu(state_dict):
    snapshot = state_dict.__class__()
    for key, value in state_dict.items():
        snapshot[key] = value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        snapshot._metadata = copy.deepcopy(metadata)
    return snapshot


def mean_std(values):
    if not values:
        return (0.0, 0.0)
    return (float(np.mean(values)), float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)


def hit_sort_key(name):
    value = str(name).strip().lower()
    try:
        if value.startswith("hits@"):
            return int(value.split("@", 1)[1])
        if value.startswith("mrr_hit"):
            return int(value[len("mrr_hit") :])
    except Exception:
        pass
    return 10**9


def write_summary(log_path, lines):
    if log_path is not None:
        with open(log_path, "w") as handle:
            handle.write("\n".join(lines) + "\n")


def should_compute_auc(option, mode=None):
    del mode
    return str(option).strip().lower() != "no"


def print_ogbl_protocol(args, mode, metadata, metric_key):
    lines = (
        f"Selection/reporting metric requested: {args.metric}",
        f"Selection/reporting metric effective: {metric_key}",
        f"Evaluation mode: {mode}",
        f"eval_cap={args.eval_cap}",
        f"evaluation_positive_scope={metadata['evaluation_positive_scope']}",
        f"reference_evaluation_scope={str(metadata['reference_evaluation_scope']).lower()}",
        f"candidate_artifact_compatibility={metadata['candidate_artifact_compatibility']}",
        f"all_negatives_legacy_override={args.all_negatives}",
        f"pool={args.pool}",
        "heart_candidate_source=generated-only",
        f"heart_negatives_total_requested={args.heart_negatives}",
        f"ranked_negatives_backend={args.ranked_negatives_backend}",
        f"negative_cache_dir={args.negative_cache_dir}",
        f"negative_cache_enabled={not args.no_negative_cache}",
        f"compute_auc={args.compute_auc}",
        f"runtime_limit_sec={RUNTIME_LIMIT_SEC}",
        f"runtime_limit_hours={RUNTIME_LIMIT_SEC / 3600:.2f}",
    )
    print("\n".join(lines))


def bind_protocol_metadata(args, metadata, metric_key):
    args.selection_metric_effective = metric_key
    for attribute in (
        "evaluation_positive_scope",
        "evaluation_protocol",
        "reference_query_scope",
        "reference_evaluation_scope",
        "candidate_artifact_compatibility",
    ):
        setattr(args, attribute, metadata[attribute])
    return args


def log_protocol_summary(log, args, mode, metadata, metric_key, bundle, device):
    for line in (
        f"evaluation_mode: {mode}",
        f"evaluation_positive_cap: {args.eval_cap}",
        f"evaluation_positive_scope: {metadata['evaluation_positive_scope']}",
        f"reference_evaluation_scope: {str(metadata['reference_evaluation_scope']).lower()}",
        f"candidate_artifact_compatibility: {metadata['candidate_artifact_compatibility']}",
        f"selection_metric_requested: {args.metric}",
        f"selection_metric_effective: {metric_key}",
        f"ranked_backend: {bundle.get('ranked_backend', 'not-applicable')}",
        f"ranking_device: {device}",
        f"negative_cache_path: {bundle.get('negative_cache_path') or 'not-applicable'}",
        f"heart_candidate_source: {bundle.get('heart_source', 'generated-online')}",
        f"heart_negatives_total_effective: {bundle.get('heart_negatives_total')}",
    ):
        log(line)


def log_run_statistics(log, timings, peaks, records):
    for label in ("train", "test", "eval", "inference", "testing", "mrr", "auc"):
        mean, std = mean_std(timings[label])
        log(f"{label}_sec_per_run mean ± std: {mean:.2f} ± {std:.2f}")
    for label, values in peaks.items():
        log(f"{label}_max: {(max(values) if values else 0.0):.2f}")
    log("\nPer-run train/test timing and peak memory")
    for record in records:
        prefix = f"run_{record['run']}_seed_{record['seed']}"
        log(f"{prefix}_test_completed: {str(record['test_completed']).lower()}")
        for label in ("train_time_sec", "test_time_sec"):
            log(f"{prefix}_{label}: {record[label]:.6f}")
        for label in (
            "train_peak_cpu_rss_mb",
            "test_peak_cpu_rss_mb",
            "train_peak_cuda_allocated_mb",
            "test_peak_cuda_allocated_mb",
            "train_peak_cuda_reserved_mb",
            "test_peak_cuda_reserved_mb",
        ):
            log(f"{prefix}_{label}: {record[label]:.2f}")


def log_aggregate_results(log, dataset, seed, metric_key, selected, aucs, aps, mrrs, hits):
    count = len(selected)
    if not count:
        log("No runs produced a valid selected metric — nothing to aggregate.")
        log("=" * 80)
        return False
    selected_mean, selected_std = mean_std(selected)
    log("\n" + "=" * 80)
    log(f"Aggregate results over {count} runs on {dataset} (seeds {seed}..{seed + count - 1})")
    log(f"Test {metric_key} mean ± std: {100 * selected_mean:.6f} ± {100 * selected_std:.6f} (percent)")
    for label, values in (("AUC", aucs), ("AP", aps), ("MRR", mrrs)):
        if values:
            mean, std = mean_std(values)
            log(f"Test {label} mean ± std: {100 * mean:.6f} ± {100 * std:.6f}")
    for name in sorted(hits, key=hit_sort_key):
        mean, std = mean_std(hits[name])
        log(f"{name} mean ± std: {100 * mean:.6f} ± {100 * std:.6f}")
    log("=" * 80)
    return True


def resolve_ogbl_eval_cap(eval_cap, mode: str, dataset: str) -> int:
    value = None if eval_cap is None else int(eval_cap)
    if value is not None and value < 0:
        raise ValueError("--eval-cap must be non-negative.")
    mode_key = normalize_ogbl_mode(mode)
    del dataset
    if mode_key == "heart":
        if value is None or value == 0 or value >= _HEART_VALIDATION_MAX:
            return _HEART_VALIDATION_MAX
        return value
    if value is not None:
        return value
    return 500


def resolve_ogbl_metric(metric, dataset: str) -> str:
    requested = str(metric or "auto").strip().lower()
    if requested == "auto":
        return _OFFICIAL_SELECTION_METRICS.get(_dataset_key(dataset), "Hits@50")
    if requested in {"auc", "ap", "mrr"}:
        return requested.upper()
    if requested.startswith("hits@"):
        suffix = requested.split("@", 1)[1]
        if suffix.isdigit() and int(suffix) > 0:
            return f"Hits@{int(suffix)}"
        raise ValueError(f"Invalid metric '{metric}'. Expected hits@K with positive integer K.")
    if requested.startswith("mrr_hit"):
        suffix = requested[len("mrr_hit") :]
        if suffix.isdigit() and int(suffix) > 0:
            return f"Hits@{int(suffix)}"
    raise ValueError(f"Unsupported metric '{metric}'. Use: auto, auc, ap, mrr, hits@K.")


def ogbl_protocol_metadata(*, dataset: str, mode: str, eval_cap: int, selection_metric: str) -> Dict[str, Any]:
    dataset_key = _dataset_key(dataset)
    mode_key = normalize_ogbl_mode(mode)
    cap = int(eval_cap)
    if mode_key == "heart" and cap >= _HEART_VALIDATION_MAX:
        cap = _HEART_VALIDATION_MAX
    generated = mode_key == "heart"
    ppa_query_panel_mode = resolve_ppa_query_panel_mode() if generated and dataset_key == "ogbl-ppa" else None
    generated_reference_ppa_panel = ppa_query_panel_mode == "reference" and cap in {0, 100000}
    native_validation_count = _NATIVE_HEART_VALIDATION_COUNTS.get(dataset_key)
    complete_native_validation = mode_key == "heart" and (
        cap == 0 or (native_validation_count is not None and cap >= int(native_validation_count))
    )
    if complete_native_validation:
        positive_scope = "validation-full-split;test-full-split"
    elif mode_key == "heart" and cap == _HEART_VALIDATION_MAX:
        positive_scope = "validation-max-100000;test-full-split"
    elif mode_key == "heart" and cap > 0:
        positive_scope = f"validation-capped-{cap};test-full-split"
    elif mode_key == "heart":
        positive_scope = "validation-full-split;test-full-split"
    elif cap == 0:
        positive_scope = "full-split"
    else:
        positive_scope = f"capped-{cap}"
    reference_cap = _REFERENCE_HEART_EVAL_CAPS.get(dataset_key)
    reference_query_scope = (
        mode_key == "heart"
        and reference_cap is not None
        and (
            dataset_key != "ogbl-ppa"
            and (cap == int(reference_cap) or complete_native_validation)
            or (dataset_key == "ogbl-ppa" and generated_reference_ppa_panel and (cap in {0, int(reference_cap)}))
        )
    )
    reference_scope = False
    if generated:
        candidate_compatibility = (
            (
                "generated-source-style-reference-fixed-query-panel"
                if generated_reference_ppa_panel
                else "generated-local-seeded-query-panel-custom"
            )
            if dataset_key == "ogbl-ppa"
            else "generated-released-filter-source-style" if dataset_key == "ogbl-collab" else "generated-source-style-not-artifact-exact"
        )
    else:
        candidate_compatibility = "generated-ranked-pool"
    return {
        "evaluation_protocol": (
            "ogbl-heart-generated-reference-fixed-query"
            if generated_reference_ppa_panel
            else "ogbl-heart-generated-grouped" if generated else f"ogbl-{mode_key}"
        ),
        "evaluation_positive_scope": positive_scope,
        "evaluation_positive_cap": cap,
        "evaluation_validation_positive_cap": cap,
        "evaluation_test_positive_cap": 0,
        "heart_source_effective": "generated-only" if generated else "generated-ranked-pool",
        "selection_metric_effective": str(selection_metric),
        "reference_query_scope": bool(reference_query_scope),
        "reference_evaluation_scope": bool(reference_scope),
        "candidate_artifact_compatibility": candidate_compatibility,
        "ppa_query_panel_mode": ppa_query_panel_mode,
        "ppa_query_panel_scope": (
            "reference-fixed-100000-released-index-order"
            if generated_reference_ppa_panel
            else _ppa_local_scope() if ppa_query_panel_mode == "local-seeded" else None
        ),
    }
