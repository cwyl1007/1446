import argparse
import copy
import gc
import json
import math
import os
import time
import random
import numpy as np
import torch
from tqdm import tqdm
from ogb.linkproppred import Evaluator
from . import train_eval
from .train_eval import all_train, test, test_only, validation_only
from .prepare_data import parse_pool_argument, read_data, resolve_pyg_eval_cap
from model.pairwise_models import get_model, reset_reference_planetoid_model
from model.feature_aggregation import aggregated_mlp_method, aggregated_mlp_recipe, is_aggregated_mlp, preprocess_aggregated_mlp
from utils.profiling import StageProfiler, current_cpu_rss_mb, peak_cpu_rss_mb
from utils.heart_protocol import persist_heart_candidate_metadata

RUNTIME_LIMIT_SEC = 24 * 60 * 60
_CUDA_ATOMIC_REPLAY_MODELS = frozenset({"gat", "seal", "buddy", "neognn", "ncn", "ncnc", "nbfnet", "peg", "lpformer", "lpf"})
_MAX_CUDA_ATOMIC_REPLAY_ABS_DELTA = 0.01
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _configure_cuda_matmul_precision(device, dataset, model_name):
    if torch.device(device).type != "cuda":
        return "cpu-default"
    normalized_model = str(model_name).strip().lower().replace("-", "").replace("_", "")
    normalized_dataset = str(dataset).strip().lower()
    exact_planetoid_models = {"mf", "mlp", "gcn", "gat", "sage", "gae", "buddy", "neognn", "ncn", "ncnc", "peg", "n2v", "node2vec"}
    reference_planetoid_full_fp32 = normalized_dataset in {"cora", "citeseer", "pubmed"} and normalized_model in exact_planetoid_models
    allow_tf32 = not reference_planetoid_full_fp32
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    torch.backends.cudnn.deterministic = not allow_tf32
    torch.backends.cudnn.benchmark = False
    torch.set_float32_matmul_precision("high" if allow_tf32 else "highest")
    return "tf32-high" if allow_tf32 else "full-fp32-planetoid-reference"


def _checkpoint_metric(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def _snapshot_state_dict_cpu(state_dict):
    snapshot = state_dict.__class__()
    for key, value in state_dict.items():
        snapshot[key] = value.detach().cpu().clone() if torch.is_tensor(value) else copy.deepcopy(value)
    metadata = getattr(state_dict, "_metadata", None)
    if metadata is not None:
        snapshot._metadata = copy.deepcopy(metadata)
    return snapshot


def _merge_test_metrics(results, test_metrics):
    merged = dict(results)
    for key, test_value in test_metrics.items():
        previous = merged.get(key)
        if isinstance(previous, (tuple, list)) and len(previous) == 3:
            merged[key] = (previous[0], previous[1], test_value)
        else:
            merged[key] = (None, None, test_value)
    return merged


def _record_test_metrics(selected, metrics, selected_values, metric_values, hit_values):
    if selected is None:
        return False
    selected_values.append(float(selected))
    for name, values in metric_values.items():
        if name in metrics:
            values.append(float(metrics[name][2]))
    for name, triple in metrics.items():
        if isinstance(name, str) and (name.startswith("Hits@") or name.startswith("mrr_hit")):
            hit_values.setdefault(name, []).append(float(triple[2]))
    return True


def _run_resource_profile(run, seed, epochs, timed_out, times, peaks, test_completed, **extra):
    profile = {
        "run": int(run), "seed": int(seed), "epochs_completed": int(epochs),
        "status": "timed_out" if timed_out else "completed",
    }
    profile.update(dict(zip(("train_time_sec", "test_time_sec", "eval_model_selection_time_sec"), times)))
    for index, suffix in enumerate(("peak_cpu_rss_mb", "peak_cuda_allocated_mb", "peak_cuda_reserved_mb")):
        profile.update({f"{stage}_{suffix}": peaks[stage_index][index]
                        for stage_index, stage in enumerate(("train", "test", "eval"))})
    profile["test_completed"] = bool(test_completed)
    profile.update(extra)
    return profile


def _reconcile_restored_validation_metrics(validation_metrics, selected_key, selected_value, *, allow_cuda_atomic_mismatch=False, maximum_abs_delta=_MAX_CUDA_ATOMIC_REPLAY_ABS_DELTA):
    reconciled = dict(validation_metrics)
    selected = reconciled.get(selected_key)
    if not isinstance(selected, (tuple, list)) or len(selected) != 3:
        raise RuntimeError(f"Restored best checkpoint did not reproduce a three-way validation result for {selected_key!r}.")
    observed = selected[1]
    try:
        observed = None if observed is None else float(observed)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            f"Restored best checkpoint produced a non-numeric validation value for {selected_key!r}: {selected[1]!r}."
        ) from exc
    if observed is None or not np.isfinite(observed):
        raise RuntimeError(f"Restored best checkpoint produced no finite validation value for {selected_key!r}: {selected[1]!r}.")
    try:
        authoritative = float(selected_value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"The checkpoint-selection validation value is non-numeric: {selected_value!r}.") from exc
    if not np.isfinite(authoritative):
        raise RuntimeError(f"The checkpoint-selection validation value is not finite: {selected_value!r}.")
    absolute_delta = abs(observed - authoritative)
    mismatched = absolute_delta != 0.0
    if mismatched and (not allow_cuda_atomic_mismatch):
        raise RuntimeError(
            f"Restored best checkpoint changed the selected validation value on a strict replay path: {observed!r} != {authoritative!r}."
        )
    if mismatched and absolute_delta > float(maximum_abs_delta):
        raise RuntimeError(
            f"Restored best checkpoint validation replay mismatch exceeds the guarded CUDA-atomic allowance: abs_delta={absolute_delta!r}, maximum={maximum_abs_delta!r}, replay={observed!r}, selection={authoritative!r}."
        )
    if mismatched:
        reconciled[selected_key] = (selected[0], authoritative, selected[2])
    return (reconciled, observed, absolute_delta, mismatched)


def _save_model_checkpoint(state_dict, *, framework, mode, dataset, model_name, run_number, seed, epoch, timed_out, metric_name, best_val, best_test, args, model_config, checkpoint_type="final_model_state"):
    checkpoint_dir = os.path.join(
        getattr(args, "checkpoint_root", None) or os.path.join(PROJECT_ROOT, "checkpoints"), str(mode), str(dataset), str(model_name)
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f"model_checkpoint{int(run_number)}")
    checkpoint_arguments = dict(vars(args))
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
        "best_validation_metric": _checkpoint_metric(best_val),
        "best_test_metric": _checkpoint_metric(best_test),
        "heart_source_resolved": getattr(args, "heart_source_resolved", None),
        "heart_backend_resolved": getattr(args, "heart_backend_resolved", None),
        "heart_candidate_metadata": copy.deepcopy(getattr(args, "heart_candidate_metadata", {})),
        "arguments": checkpoint_arguments,
        "model_config": dict(model_config),
        "model_state_dict": state_dict,
    }
    if hasattr(args, "n2v_protocol_effective"):
        checkpoint["n2v_protocol"] = args.n2v_protocol_effective
        checkpoint["n2v_embedding_path"] = getattr(args, "reference_embedding_path_resolved", None)
    temporary_path = checkpoint_path + ".tmp"
    try:
        torch.save(checkpoint, temporary_path)
        os.replace(temporary_path, checkpoint_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
    return checkpoint_path


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _resolve_device(value=None):
    cuda_available = torch.cuda.is_available()
    device = torch.device(value or ("cuda" if cuda_available else "cpu"))
    if device.type == "cuda" and not cuda_available:
        raise RuntimeError(f"CUDA device {device} was requested but CUDA is unavailable.")
    return device


def _make_optimizer(model, lr, weight_decay, device):
    lr = float(getattr(model, "reference_optimizer_lr", lr))
    weight_decay = float(getattr(model, "reference_optimizer_weight_decay", weight_decay))
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if bool(getattr(model, "reference_optimizer", False)):
        kwargs["foreach"] = False
        return torch.optim.Adam(parameters, **kwargs)
    if str(device).startswith("cuda"):
        try:
            return torch.optim.Adam(parameters, fused=True, **kwargs)
        except (TypeError, RuntimeError):
            pass
    return torch.optim.Adam(parameters, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Run best config on dataset")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--metric", type=str, default="mrr")
    parser.add_argument("--compute-auc", choices=["yes", "no"], default="yes")
    parser.add_argument("--mode", choices=["heart", "all", "ranked-selector"], default="heart")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None, help="Positive-query cap; generated Reddit HeaRT defaults to 100000.")
    parser.add_argument("--train-samples-per-epoch", type=int, default=None, help="Positive training-edge cap per epoch; 0 uses the full split.")
    parser.add_argument("--pool", type=parse_pool_argument, default=10000, help="Per-side all-mode candidate maximum.")
    parser.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, choices=[500], default=500, help="Total negatives per positive.")
    parser.add_argument(
        "--planetoid-input-root",
        type=str,
        default=None,
        help="Fixed Planetoid positive-split and gnn_feature root.",
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device")
    parser.add_argument("--base-seed", "--seed", dest="seed", type=int, default=0, help="First-run seed.")
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--selector-depth", type=int)
    parser.add_argument("--selector-hidden-channels", type=int)
    parser.add_argument("--selector-dropout", type=float)
    parser.add_argument("--selector-lr", "--selector-learning-rate", dest="selector_lr", type=float)
    parser.add_argument("--selector-weight-decay", type=float)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--results-root")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--heart-backend", default="auto", choices=["auto", "gpu", "dense"])
    parser.add_argument("--heart-batch-size", type=int, default=0)
    parser.add_argument("--heart-ppr-iters", type=int, default=None)
    parser.add_argument("--save-log", action="store_true", default=True)
    return parser.parse_args()


def _runtime_exceeded(t0):
    return time.time() - t0 >= RUNTIME_LIMIT_SEC


def _resolve_eval_cap(eval_cap, mode, dataset=None):
    if str(mode).strip().lower() == "ranked-selector":
        cap = 100000 if eval_cap in (None, 0) else int(eval_cap)
        if cap != 100000:
            raise ValueError("Ranked-selector training requires --eval-cap 100000.")
        return cap
    return resolve_pyg_eval_cap(eval_cap, mode, dataset)


def _default_heart_batch_size(device):
    if device.type != "cuda":
        return 1
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if total_gb >= 60:
        return 1024
    if total_gb >= 32:
        return 512
    if total_gb >= 16:
        return 256
    return 128


def _default_batches(dataset, model, device):
    name = str(model).lower().replace("-", "").replace("_", "")
    if str(dataset).lower() != "reddit" or device.type != "cuda":
        return (1024, 1024)
    total_gb = torch.cuda.get_device_properties(device).total_memory / 1024**3
    if name in {"mf", "mlp", "mlpip", "ppr", "concat", "concatip", "gcn", "gat", "sage", "gae"}:
        if total_gb >= 60:
            return (524288, 1048576)
        if total_gb >= 24:
            return (262144, 524288)
        return (131072, 262144)
    if name in {"lpformer", "lpf"}:
        return (4096, 8192) if total_gb >= 60 else (2048, 4096)
    if name in {"seal", "ncnc"}:
        return (8192, 16384) if total_gb >= 60 else (4096, 8192)
    if name == "peg":
        return (65536, 131072) if total_gb >= 60 else (32768, 65536)
    return (32768, 65536) if total_gb >= 60 else (16384, 32768)


def _default_train_samples(dataset, model, device):
    del device
    name = str(model).lower().replace("-", "").replace("_", "")
    if str(dataset).lower() != "reddit" or name in {
        "mf",
        "mlp",
        "mlpip",
        "ppr",
        "concat",
        "concatip",
        "gcn",
        "gat",
        "sage",
        "gae",
    }:
        return 0
    if name in {"lpformer", "lpf"}:
        return 65536
    if name in {"seal", "ncnc"}:
        return 131072
    if name == "peg":
        return 524288
    return 262144


def _find_result_key(results, metric_name):
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


def _install_metric_timers():
    stats = {"mrr_sec": 0.0, "auc_sec": 0.0}
    orig_mrr = train_eval.evaluate_mrr
    orig_mrr_only = train_eval.evaluate_mrr_only
    orig_auc = train_eval.evaluate_auc

    def timed_mrr(*args, **kwargs):
        t0 = time.time()
        out = orig_mrr(*args, **kwargs)
        stats["mrr_sec"] += time.time() - t0
        return out

    def timed_auc(*args, **kwargs):
        t0 = time.time()
        out = orig_auc(*args, **kwargs)
        stats["auc_sec"] += time.time() - t0
        return out

    def timed_mrr_only(*args, **kwargs):
        t0 = time.time()
        out = orig_mrr_only(*args, **kwargs)
        stats["mrr_sec"] += time.time() - t0
        return out

    train_eval.evaluate_mrr = timed_mrr
    train_eval.evaluate_mrr_only = timed_mrr_only
    train_eval.evaluate_auc = timed_auc
    return (stats, orig_mrr, orig_mrr_only, orig_auc)


def _restore_metric_timers(orig_mrr, orig_mrr_only, orig_auc):
    train_eval.evaluate_mrr = orig_mrr
    train_eval.evaluate_mrr_only = orig_mrr_only
    train_eval.evaluate_auc = orig_auc


def _configure_ip_selector_evaluator(model, model_name):
    if str(model_name).strip().lower() not in {"mlpip", "concatip"}:
        return None
    model.shared_ip_training_contract = True
    model.reference_optimizer = False
    model.reference_probability_loss = False
    model.reference_random_endpoint_negatives = False
    model.strict_train_negatives = True
    model.training_protocol = "shared-ip-strict-negative-bce-with-logits-v1"
    model.train_negative_protocol = "strict-unobserved-nonself-edge"
    model.training_loss_protocol = "binary-cross-entropy-with-logits"
    model.protocol_fidelity = "requested-inner-product-selector-evaluator"
    return {
        "ip_training_contract": model.training_protocol,
        "ip_training_loss": model.training_loss_protocol,
        "ip_train_negative_protocol": model.train_negative_protocol,
        "ip_strict_train_negatives": True,
        "ip_reference_probability_loss": False,
        "ip_reference_random_endpoint_negatives": False,
        "ip_reference_optimizer": False,
    }


def _mean_std(values):
    if not values:
        return (0.0, 0.0)
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    return (mean, std)


def _hit_sort_key(name):
    s = str(name).strip().lower()
    if s.startswith("hits@"):
        try:
            return int(s.split("@", 1)[1])
        except Exception:
            return 10**9
    if s.startswith("mrr_hit"):
        try:
            return int(s[len("mrr_hit") :])
        except Exception:
            return 10**9
    return 10**9


def _write_summary(log_path, summary_lines):
    if log_path is not None:
        with open(log_path, "w") as f:
            f.write("\n".join(summary_lines) + "\n")


def _write_timing_summary(*, log_path, header_lines, summary_values, per_run_series, run_profiles, profile_value_keys=(), profile_bool_keys=(), extra_lines=(), test_selected, test_aucs, test_aps, test_mrrs, test_hits, dataset, base_seed, metric):
    lines = []

    def log(line=""):
        print(line)
        lines.append(str(line))

    for line in header_lines:
        log(line)
    for key, value in summary_values.items():
        log(f"{key}: {value:.2f}")
    for line in extra_lines:
        log(line)
    for key, values in per_run_series.items():
        mean, std = _mean_std(values)
        log(f"{key} mean ± std: {mean:.2f} ± {std:.2f}")
    for line in ("\nPer-run timing and peak memory", "memory_units: MiB", "cpu_memory: peak stage-local resident set size (RSS)",
                 "cuda_memory: peak allocated tensor and allocator-reserved memory",
                 "test_measurement: held-out test split only, using the best-validation state"):
        log(line)
    for profile in run_profiles:
        log(f"Run {profile['run']} (seed={profile['seed']}, status={profile['status']}, test_completed={profile['test_completed']}, epochs_completed={profile['epochs_completed']})")
        for key in ("train_time_sec", "test_time_sec", "eval_model_selection_time_sec"):
            log(f"  {key}: {profile[key]:.2f}")
        for key in profile_value_keys:
            log(f"  {key}: {profile[key]}")
        for item in profile_bool_keys:
            label, key = item if isinstance(item, tuple) else (item, item)
            log(f"  {label}: {profile[key]}")
        for stage in ("train", "test", "eval"):
            for suffix in ("peak_cpu_rss_mb", "peak_cuda_allocated_mb", "peak_cuda_reserved_mb"):
                key = f"{stage}_{suffix}"
                log(f"  {key}: {profile[key]:.2f}")
    count = len(test_selected)
    if not count:
        log("No runs produced a valid selected metric — nothing to aggregate.")
    else:
        mean, std = _mean_std(test_selected)
        log("\n" + "=" * 80)
        log(f"Aggregate results over {count} runs on {dataset} (seeds {base_seed}..{base_seed + count - 1})")
        log(f"Test {metric} mean ± std: {100 * mean:.6f} ± {100 * std:.6f} (percent)")
        for name, values in (("AUC", test_aucs), ("AP", test_aps), ("MRR", test_mrrs)):
            if values:
                mean, std = _mean_std(values)
                log(f"Test {name} mean ± std: {100 * mean:.6f} ± {100 * std:.6f}")
        for name in sorted(test_hits, key=_hit_sort_key):
            mean, std = _mean_std(test_hits[name])
            log(f"{name} mean ± std: {100 * mean:.6f} ± {100 * std:.6f}")
    log("=" * 80)
    _write_summary(log_path, lines)


def _print_candidate_metadata(data):
    fields = []
    if data.get("pool_per_side") is not None:
        fields += [
            ("pool_setting", "pool_setting"), ("pool_full_graph", "pool_full_graph"), ("pool_cap_applied", "pool_cap_applied"),
            ("pool_sampling", "pool_sampling"), ("pool_requested_per_side", "pool_requested_per_side"),
            ("pool_requested_total", "pool_requested_total"), ("pool_per_side_effective", "pool_per_side"),
            ("pool_total_effective", "pool_total"),
        ]
        fields += [(key, key) for key in ("pool_per_side_min", "pool_per_side_mean", "pool_per_side_max", "pool_total_min", "pool_total_mean", "pool_total_max") if data.get(key) is not None]
    if data.get("heart_candidate_universe") is not None:
        fields += [
            ("heart_candidate_universe", "heart_candidate_universe"), ("heart_candidate_graph_nodes", "heart_candidate_graph_nodes"),
            ("heart_selection", "heart_selection"), ("heart_negatives_requested_per_side", "heart_negatives_requested_per_side"),
            ("heart_negatives_requested_total", "heart_negatives_requested_total"),
            ("heart_negatives_per_side_effective", "heart_negatives_per_side"), ("heart_negatives_total_effective", "heart_negatives_total"),
        ]
    for label, key in fields:
        print(f"{label}={data.get(key)}", flush=True)


def _read_run_data(args, device, heart_batch_size=None):
    if args.mode == "ranked-selector":
        from eval_modes.evaluator_helpers import (
            ensure_complete_ranked_positive_splits,
            ensure_pyg_full_known_positive_filter,
            load_ranked_selector_bundle,
        )
        from eval_modes.ranked_helpers import build_neutral_selector_validation_negatives

        bundle = load_ranked_selector_bundle("pyg", args.dataset, args.root, args.seed)
        ensure_complete_ranked_positive_splits(
            bundle, framework="pyg", dataset=args.dataset, root=args.root, seed=args.seed
        )
        ensure_pyg_full_known_positive_filter(
            bundle, dataset=args.dataset, root=args.root, seed=args.seed
        )
        validation = build_neutral_selector_validation_negatives(
            bundle, "pyg", args.dataset, seed=args.seed, negatives=500
        )
        bundle["valid_neg"] = validation.negatives
        bundle["test_neg"] = None
        bundle["selector_validation_metadata"] = dict(validation.metadata)
        return bundle
    return read_data(
        args.dataset,
        args.mode,
        root=args.root,
        eval_cap=args.eval_cap,
        seed=args.seed,
        heart_backend=args.heart_backend,
        heart_device=str(device),
        heart_batch_size=args.heart_batch_size if heart_batch_size is None else heart_batch_size,
        heart_ppr_iters=args.heart_ppr_iters,
        heart_negatives=args.heart_negatives,
        pool=args.pool,
        planetoid_input_root=args.planetoid_input_root,
    )


def main():
    program_t0 = time.time()
    args = parse_args()
    device = _resolve_device(args.device)
    args.device = str(device)
    normalized_model = str(args.model).strip().lower().replace("-", "").replace("_", "")
    selector_training_mode = args.mode == "ranked-selector"
    if selector_training_mode and normalized_model != "concat":
        raise ValueError("--mode ranked-selector is reserved for training the Concat selector.")
    if selector_training_mode and args.num_runs != 1:
        raise ValueError("Ranked-selector training requires --num-runs 1.")
    if selector_training_mode and args.planetoid_input_root is not None:
        raise ValueError(
            "--planetoid-input-root does not apply to ranked-selector "
            "training; use --root."
        )
    matmul_precision = _configure_cuda_matmul_precision(device, args.dataset, args.model)
    print(f"matmul_precision={matmul_precision}", flush=True)
    args.eval_cap = _resolve_eval_cap(args.eval_cap, args.mode, args.dataset)
    timed_out = False
    startup_lines = [
        f"Using device: {device}", f"Selection/reporting metric: {args.metric}", f"Evaluation mode: {args.mode}",
        f"eval_cap={args.eval_cap}", f"runtime_limit_sec={RUNTIME_LIMIT_SEC}",
        f"runtime_limit_hours={RUNTIME_LIMIT_SEC / 3600:.2f}",
    ]
    if selector_training_mode:
        startup_lines.append("selector_validation=deterministic-neutral-legal-fixed-250-per-side")
    else:
        startup_lines.extend(
            (
                f"pool={args.pool}",
                "heart_negatives=generated-online",
                f"heart_negatives_total_requested={args.heart_negatives}",
            )
        )
    for line in startup_lines:
        print(line)
    config_path = os.path.join("configs", f"{args.model}_{args.dataset}_config.json")
    if (is_aggregated_mlp(args.model) or normalized_model == "mlpip") and (not os.path.isfile(config_path)):
        config_path = os.path.join("configs", f"mlp_{args.dataset}_config.json")
    print(f"Loading config from: {config_path}")
    with open(config_path, "r") as f:
        payload = json.load(f)
    best_config = dict(payload["best_config"])
    if normalized_model in {"mlpip", "concatip"}:
        best_config.update(
            {
                "pred_layers": 0,
                "decoder_type": "inner-product",
                "predictor_depth": 0,
                "decoder_output": "raw-inner-product-logit",
                "ranking_score": "raw-inner-product-logit",
                "probability_transform": "sigmoid",
            }
        )
    selector_override_requested = any(
        value is not None
        for value in (
            args.selector_depth,
            args.selector_hidden_channels,
            args.selector_dropout,
            args.selector_lr,
            args.selector_weight_decay,
        )
    )
    if selector_override_requested:
        if normalized_model not in {"concat", "concatip", "mlpip"} or args.checkpoint_root is None or args.results_root is None:
            raise ValueError("Concat/ConcatIP/MLPIP selector overrides require isolated checkpoint and result roots.")
        if args.selector_depth is not None:
            if args.selector_depth <= 0:
                raise ValueError("Selector depth must be positive.")
            best_config["layers"] = args.selector_depth
            best_config["pred_layers"] = 0 if normalized_model in {"concatip", "mlpip"} else args.selector_depth
        if args.selector_hidden_channels is not None:
            if args.selector_hidden_channels <= 0:
                raise ValueError("Selector hidden channels must be positive.")
            best_config["emb_size"] = args.selector_hidden_channels
        if args.selector_dropout is not None:
            if not math.isfinite(args.selector_dropout) or not 0.0 <= args.selector_dropout < 1.0:
                raise ValueError("Selector dropout must satisfy 0 <= dropout < 1.")
            best_config["dropout"] = args.selector_dropout
        if args.selector_lr is not None:
            if not math.isfinite(args.selector_lr) or args.selector_lr <= 0.0:
                raise ValueError("Selector learning rate must be positive.")
            best_config["lr"] = args.selector_lr
        if args.selector_weight_decay is not None:
            if not math.isfinite(args.selector_weight_decay) or args.selector_weight_decay < 0.0:
                raise ValueError("Selector weight decay must be non-negative.")
            best_config["weight_decay"] = args.selector_weight_decay
    if is_aggregated_mlp(args.model):
        feature_method = aggregated_mlp_method(args.model)
        feature_recipe = aggregated_mlp_recipe(args.model)
        best_config[f"{feature_method}_preprocessing"] = feature_recipe
        implementation_name = normalized_model if normalized_model == "concatip" else feature_method
        best_config["model_implementation"] = f"{implementation_name}-{feature_recipe}"
        best_config.pop("alpha", None)
    output_model_name = args.model
    log_path = None
    if args.save_log:
        log_dir = os.path.join(args.results_root or os.path.join(PROJECT_ROOT, "results"), "pyg", args.mode, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{output_model_name}.txt")
    epochs = int(args.epochs if args.epochs is not None else best_config.get("epochs", 500))
    args.epochs = epochs
    (default_train_batch, default_eval_batch) = _default_batches(args.dataset, args.model, device)
    batch_size = int(args.batch_size if args.batch_size is not None else best_config.get("train_batch_size", default_train_batch))
    eval_batch_size = int(
        args.eval_batch_size if args.eval_batch_size is not None else best_config.get("eval_batch_size", default_eval_batch)
    )
    default_train_samples = _default_train_samples(args.dataset, args.model, device)
    train_samples_per_epoch = max(
        0,
        int(
            args.train_samples_per_epoch
            if args.train_samples_per_epoch is not None
            else best_config.get("train_samples_per_epoch", default_train_samples)
        ),
    )
    best_config["train_samples_per_epoch"] = train_samples_per_epoch
    heart_batch_size = (
        None
        if selector_training_mode
        else int(args.heart_batch_size) if int(args.heart_batch_size) > 0 else _default_heart_batch_size(device)
    )
    base_seed = args.seed
    eval_steps = int(args.eval_steps if args.eval_steps is not None else best_config.get("eval_steps", 5))
    patience = int(args.patience if args.patience is not None else best_config.get("patience", 10))
    args.eval_steps = eval_steps
    args.patience = patience
    emb_size = best_config["emb_size"]
    layers = best_config["layers"]
    dropout = best_config["dropout"]
    pred_layers_val = best_config["pred_layers"]
    lr = best_config["lr"]
    weight_decay = best_config["weight_decay"]
    print("Loaded best config:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    training_summary = (
        f"\nUsing dataset={args.dataset}, epochs={epochs}, base_seed={base_seed}, num_runs={args.num_runs},"
        f" eval_steps={eval_steps}, patience={patience}, batch_size={batch_size}, eval_batch_size={eval_batch_size},"
        f" train_samples_per_epoch={train_samples_per_epoch}"
    )
    if not selector_training_mode:
        training_summary += f", heart_batch_size={heart_batch_size}"
    print(training_summary)
    metric_key = args.metric.strip()
    compute_auc = args.compute_auc == "yes"
    test_selected_metrics = []
    test_aucs = []
    test_aps = []
    test_mrrs = []
    test_hits_any = {}
    run_train_secs = []
    run_eval_secs = []
    run_test_secs = []
    run_inference_secs = []
    run_testing_secs = []
    run_mrr_secs = []
    run_auc_secs = []
    run_resource_profiles = []
    total_train_sec = 0.0
    total_eval_sec = 0.0
    total_test_sec = 0.0
    total_inference_sec = 0.0
    total_testing_sec = 0.0
    total_mrr_sec = 0.0
    total_auc_sec = 0.0
    train_peak_cpu = eval_peak_cpu = test_peak_cpu = 0.0
    train_peak_cuda = eval_peak_cuda = test_peak_cuda = 0.0
    train_peak_cuda_reserved = eval_peak_cuda_reserved = 0.0
    test_peak_cuda_reserved = 0.0
    t_data = time.time()
    set_seed(base_seed)
    data = _read_run_data(args, device, heart_batch_size)
    if selector_training_mode:
        selector_validation_metadata = dict(data["selector_validation_metadata"])
        best_config.update(selector_validation_metadata)
        args.selector_validation_metadata = dict(selector_validation_metadata)
        args.heart_source_resolved = None
        args.heart_candidate_metadata = {}
        heart_candidate_metadata = {}
    else:
        selector_validation_metadata = {}
        heart_candidate_metadata = persist_heart_candidate_metadata(args, data)
    validation_only_mode = args.mode in {"heart", "ranked-selector"}
    selection_requires_auc = metric_key.lower() in {"auc", "ap"}
    selection_requires_hits = metric_key.lower() != "mrr"
    if selection_requires_auc and not compute_auc:
        raise ValueError("--metric AUC/AP requires --compute-auc yes.")
    print(f"compute_auc_effective={compute_auc}", flush=True)
    for key, value in heart_candidate_metadata.items():
        print(f"{key}={(value if value is not None else 'not-applicable')}", flush=True)
    print(
        "model_selection_evaluation="
        + (
            "neutral-validation-only"
            if selector_training_mode
            else "validation-final-test-only" if validation_only_mode else "validation-and-test"
        ),
        flush=True,
    )
    _print_candidate_metadata(data)
    data_load_sec = time.time() - t_data
    print(f"data_load_sec={data_load_sec:.2f}", flush=True)
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED during data loading: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    if "adj" in data and device.type == "cuda":
        data["adj"] = data["adj"].to(device)
    x = data["x"].to(device, non_blocking=True)
    if is_aggregated_mlp(args.model):
        feature_method = aggregated_mlp_method(args.model)
        feature_recipe = aggregated_mlp_recipe(args.model)
        base_feature_dim = int(x.size(-1))
        x = preprocess_aggregated_mlp(args.model, args.dataset, x, data.get("adj", data["train_pos"]))
        data["x"] = x
        best_config[f"{feature_method}_base_feature_dim"] = base_feature_dim
        best_config[f"{feature_method}_output_feature_dim"] = int(x.size(-1))
        self_loops = "none" if feature_method == "concat" else "once"
        print(
            f"feature_preprocessing={feature_method} recipe={feature_recipe} graph=train-only binary=true self_loops={self_loops} output_width={x.size(-1)} cached=true",
            flush=True,
        )
    keep_train_cpu = train_samples_per_epoch > 0
    train_pos = data["train_pos"] if keep_train_cpu else data["train_pos"].to(device)
    if keep_train_cpu:
        print("Keeping the full training edge list on CPU and transferring only the sampled epoch edges", flush=True)
    streamed_grouped_eval = any((bool(getattr(data.get(key), "is_streamed_grouped_negative", False)) for key in ("valid_neg", "test_neg")))
    if streamed_grouped_eval:
        print("Keeping evaluation positive rows on CPU for bounded streamed grouped-candidate scoring", flush=True)
    if device.type == "cuda":
        for key in (
            "train_val",
            "valid_pos",
            "valid_neg",
            "test_pos",
            "test_neg",
            "csr_train_rowptr",
            "csr_train_col",
            "csr_tv_rowptr",
            "csr_tv_col",
        ):
            if key in data and torch.is_tensor(data[key]):
                if streamed_grouped_eval and key in ("train_val", "valid_pos", "test_pos"):
                    continue
                data[key] = data[key].to(device=device, dtype=torch.long, non_blocking=True)
    num_nodes = x.size(0)
    input_channel = x.size(1)
    base_model_x = x
    evaluator_hit = Evaluator(name="ogbl-collab")
    evaluator_mrr = Evaluator(name="ogbl-citation2")
    for run_idx in range(args.num_runs):
        if timed_out or _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED before run {run_idx + 1}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
        run_seed = base_seed + run_idx
        set_seed(run_seed)
        x = base_model_x
        params = {
            **best_config,
            "in_channels": input_channel,
            "emb_size": emb_size,
            "layers": layers,
            "pred_layers": pred_layers_val,
            "dropout": dropout,
            "num_nodes": num_nodes,
            "dataset_name": args.dataset,
            "evaluation_mode": args.mode,
            "train_samples_per_epoch": train_samples_per_epoch,
        }
        model = get_model(args.model, params).to(device)
        ip_evaluator_metadata = _configure_ip_selector_evaluator(model, normalized_model)
        if ip_evaluator_metadata is not None:
            best_config.update(ip_evaluator_metadata)
        if normalized_model == "concat" and (selector_override_requested or selector_training_mode):
            from eval_modes.ranked_helpers import configure_ranked_selector_training

            best_config.update(configure_ranked_selector_training(model, args.dataset))
        if normalized_model not in {"mlpip", "concatip"}:
            reset_reference_planetoid_model(
                model,
                model_name=normalized_model,
                dataset_name=args.dataset,
                seed=run_seed,
                seed_fn=set_seed,
                device=device,
                emb_size=emb_size,
                pred_layers=pred_layers_val,
                dropout=dropout,
            )
        implementation_name = getattr(model, "implementation_name", None)
        if implementation_name is not None:
            print(f"model_implementation={implementation_name}", flush=True)
            best_config["model_implementation"] = str(implementation_name)
        for metadata_key in (
            "execution_path",
            "training_protocol",
            "protocol_fidelity",
            "train_negative_protocol",
            "training_loss_protocol",
        ):
            metadata_value = getattr(model, metadata_key, None)
            if metadata_value is not None:
                print(f"{metadata_key}={metadata_value}", flush=True)
                best_config[metadata_key] = str(metadata_value)
        optimizer = _make_optimizer(model, lr=lr, weight_decay=weight_decay, device=device)
        last_epoch = 0
        best_val_selected = float("-inf")
        best_test_selected_for_run = None
        best_test_other_metrics = {}
        metric_label = metric_key
        patience_counter = 0
        best_state_dict = None
        best_epoch = None
        run_train_sec = 0.0
        run_eval_sec = 0.0
        run_inference_sec = 0.0
        run_testing_sec = 0.0
        run_mrr_sec = 0.0
        run_auc_sec = 0.0
        run_train_peak_cpu = 0.0
        run_eval_peak_cpu = 0.0
        run_train_peak_cuda = 0.0
        run_eval_peak_cuda = 0.0
        run_train_peak_cuda_reserved = 0.0
        run_eval_peak_cuda_reserved = 0.0
        run_test_sec = 0.0
        run_test_peak_cpu = 0.0
        run_test_peak_cuda = 0.0
        run_test_peak_cuda_reserved = 0.0
        run_test_completed = False
        validation_selection_value = None
        validation_replay_value = None
        validation_replay_abs_delta = None
        validation_replay_mismatch = False
        pbar = tqdm(range(1, epochs + 1), desc=f"Run {run_idx + 1}/{args.num_runs} (seed={run_seed})", leave=False)
        for epoch in pbar:
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED before epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            train_profiler = StageProfiler(device)
            train_profiler.start()
            train_loss = all_train(
                model,
                train_pos,
                x,
                optimizer,
                batch_size,
                adj_t=data.get("adj", None),
                csr_rowptr=data.get("csr_train_rowptr"),
                csr_col=data.get("csr_train_col"),
            )
            last_epoch = epoch
            train_info = train_profiler.stop()
            epoch_train_sec = train_info["sec"]
            run_train_peak_cpu = max(run_train_peak_cpu, train_info["cpu_peak_rss_mb"])
            run_train_peak_cuda = max(run_train_peak_cuda, train_info["cuda_peak_allocated_mb"])
            run_train_peak_cuda_reserved = max(run_train_peak_cuda_reserved, train_info["cuda_peak_reserved_mb"])
            train_peak_cpu = max(train_peak_cpu, train_info["cpu_peak_rss_mb"])
            train_peak_cuda = max(train_peak_cuda, train_info["cuda_peak_allocated_mb"])
            train_peak_cuda_reserved = max(train_peak_cuda_reserved, train_info["cuda_peak_reserved_mb"])
            run_train_sec += epoch_train_sec
            total_train_sec += epoch_train_sec
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after train epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                pbar.set_postfix(loss=f"{train_loss:.6f}")
                break
            if epoch % eval_steps != 0 and epoch != epochs:
                pbar.set_postfix(loss=f"{train_loss:.6f}")
                continue
            (stats, orig_mrr, orig_mrr_only, orig_auc) = _install_metric_timers()
            eval_profile = {}
            eval_profiler = StageProfiler(device)
            eval_profiler.start()
            try:
                if validation_only_mode:
                    results_rank = validation_only(
                        model,
                        data,
                        x,
                        eval_batch_size,
                        profile=eval_profile,
                        include_auc=selection_requires_auc,
                        include_hits=selection_requires_hits,
                    )
                else:
                    (results_rank, _) = test(
                        model,
                        data,
                        x,
                        evaluator_hit,
                        evaluator_mrr,
                        eval_batch_size,
                        profile=eval_profile,
                        return_scores=False,
                        include_auc=compute_auc,
                    )
            finally:
                _restore_metric_timers(orig_mrr, orig_mrr_only, orig_auc)
            eval_info = eval_profiler.stop()
            epoch_eval_sec = eval_info["sec"]
            epoch_inference_sec = float(eval_profile.get("inference_sec", max(0.0, epoch_eval_sec - stats["mrr_sec"] - stats["auc_sec"])))
            epoch_testing_sec = float(eval_profile.get("testing_sec", stats["mrr_sec"] + stats["auc_sec"]))
            run_eval_peak_cpu = max(run_eval_peak_cpu, eval_info["cpu_peak_rss_mb"])
            run_eval_peak_cuda = max(run_eval_peak_cuda, eval_info["cuda_peak_allocated_mb"])
            run_eval_peak_cuda_reserved = max(run_eval_peak_cuda_reserved, eval_info["cuda_peak_reserved_mb"])
            eval_peak_cpu = max(eval_peak_cpu, eval_info["cpu_peak_rss_mb"])
            eval_peak_cuda = max(eval_peak_cuda, eval_info["cuda_peak_allocated_mb"])
            eval_peak_cuda_reserved = max(eval_peak_cuda_reserved, eval_info["cuda_peak_reserved_mb"])
            run_eval_sec += epoch_eval_sec
            run_inference_sec += epoch_inference_sec
            run_testing_sec += epoch_testing_sec
            total_eval_sec += epoch_eval_sec
            total_inference_sec += epoch_inference_sec
            total_testing_sec += epoch_testing_sec
            run_mrr_sec += stats["mrr_sec"]
            total_mrr_sec += stats["mrr_sec"]
            run_auc_sec += stats["auc_sec"]
            total_auc_sec += stats["auc_sec"]
            selected_key = _find_result_key(results_rank, metric_key)
            if selected_key is None:
                raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(results_rank.keys())}")
            (_, val_selected, test_selected) = results_rank[selected_key]
            metric_label = selected_key
            improved = val_selected is not None and float(val_selected) > best_val_selected
            if improved:
                best_val_selected = float(val_selected)
                best_test_selected_for_run = None if test_selected is None else float(test_selected)
                best_test_other_metrics = dict(results_rank)
                best_state_dict = _snapshot_state_dict_cpu(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            pbar.set_postfix(
                loss=f"{train_loss:.6f}",
                eval=f"{epoch_eval_sec:.2f}s",
                infer=f"{epoch_inference_sec:.2f}s",
                test=f"{epoch_testing_sec:.2f}s",
            )
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after eval epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            if patience_counter >= patience:
                break
        if best_state_dict is not None and (not timed_out):
            model.load_state_dict(best_state_dict)
            best_state_dict = None
            model.zero_grad(set_to_none=True)
            optimizer = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if validation_only_mode and (not selection_requires_auc):
                (stats, orig_mrr, orig_mrr_only, orig_auc) = _install_metric_timers()
                deferred_profile = {}
                deferred_profiler = StageProfiler(device)
                deferred_profiler.start()
                try:
                    final_validation_metrics = validation_only(
                        model,
                        data,
                        x,
                        eval_batch_size,
                        profile=deferred_profile,
                        include_auc=compute_auc,
                    )
                finally:
                    _restore_metric_timers(orig_mrr, orig_mrr_only, orig_auc)
                    deferred_info = deferred_profiler.stop()
                deferred_selected_key = _find_result_key(final_validation_metrics, metric_key)
                if deferred_selected_key is None:
                    raise RuntimeError(f"Restored best checkpoint did not reproduce the selection metric {metric_key!r}.")
                (final_validation_metrics, validation_replay_value, validation_replay_abs_delta, validation_replay_mismatch) = (
                    _reconcile_restored_validation_metrics(
                        final_validation_metrics,
                        deferred_selected_key,
                        best_val_selected,
                        allow_cuda_atomic_mismatch=device.type == "cuda" and normalized_model in _CUDA_ATOMIC_REPLAY_MODELS,
                    )
                )
                validation_selection_value = float(best_val_selected)
                if validation_replay_mismatch:
                    tqdm.write(
                        f"RESTORED_VALIDATION_REPLAY_MISMATCH: metric={deferred_selected_key} selection_value={best_val_selected!r} replay_value={validation_replay_value!r} abs_delta={validation_replay_abs_delta!r}; preserving the selection-time value."
                    )
                best_test_other_metrics = dict(final_validation_metrics)
                deferred_sec = float(deferred_info["sec"])
                deferred_inference_sec = float(deferred_profile.get("inference_sec", 0.0))
                deferred_testing_sec = float(deferred_profile.get("testing_sec", 0.0))
                run_eval_sec += deferred_sec
                total_eval_sec += deferred_sec
                run_inference_sec += deferred_inference_sec
                total_inference_sec += deferred_inference_sec
                run_testing_sec += deferred_testing_sec
                total_testing_sec += deferred_testing_sec
                run_mrr_sec += stats["mrr_sec"]
                total_mrr_sec += stats["mrr_sec"]
                run_auc_sec += stats["auc_sec"]
                total_auc_sec += stats["auc_sec"]
                run_eval_peak_cpu = max(run_eval_peak_cpu, float(deferred_info["cpu_peak_rss_mb"]))
                run_eval_peak_cuda = max(run_eval_peak_cuda, float(deferred_info["cuda_peak_allocated_mb"]))
                run_eval_peak_cuda_reserved = max(run_eval_peak_cuda_reserved, float(deferred_info["cuda_peak_reserved_mb"]))
                eval_peak_cpu = max(eval_peak_cpu, float(deferred_info["cpu_peak_rss_mb"]))
                eval_peak_cuda = max(eval_peak_cuda, float(deferred_info["cuda_peak_allocated_mb"]))
                eval_peak_cuda_reserved = max(eval_peak_cuda_reserved, float(deferred_info["cuda_peak_reserved_mb"]))
            if not selector_training_mode:
                final_profile = {}
                final_profiler = StageProfiler(device)
                final_profiler.start()
                try:
                    final_test_metrics = test_only(
                        model,
                        data,
                        x,
                        eval_batch_size,
                        profile=final_profile,
                        include_auc=compute_auc,
                    )
                finally:
                    final_info = final_profiler.stop()
                run_test_sec = float(final_info["sec"])
                run_test_peak_cpu = float(final_info["cpu_peak_rss_mb"])
                run_test_peak_cuda = float(final_info["cuda_peak_allocated_mb"])
                run_test_peak_cuda_reserved = float(final_info["cuda_peak_reserved_mb"])
                run_test_completed = True
                total_test_sec += run_test_sec
                final_inference_sec = float(final_profile.get("inference_sec", 0.0))
                final_testing_sec = float(final_profile.get("testing_sec", 0.0))
                run_inference_sec += final_inference_sec
                run_testing_sec += final_testing_sec
                total_inference_sec += final_inference_sec
                total_testing_sec += final_testing_sec
                test_peak_cpu = max(test_peak_cpu, run_test_peak_cpu)
                test_peak_cuda = max(test_peak_cuda, run_test_peak_cuda)
                test_peak_cuda_reserved = max(test_peak_cuda_reserved, run_test_peak_cuda_reserved)
                best_test_other_metrics = _merge_test_metrics(best_test_other_metrics, final_test_metrics)
                final_selected_key = _find_result_key(final_test_metrics, metric_key)
                if final_selected_key is not None:
                    final_selected = final_test_metrics[final_selected_key]
                    if final_selected is not None:
                        best_test_selected_for_run = float(final_selected)
                        metric_label = final_selected_key
        run_train_secs.append(run_train_sec)
        run_eval_secs.append(run_eval_sec)
        run_test_secs.append(run_test_sec)
        run_inference_secs.append(run_inference_sec)
        run_testing_secs.append(run_testing_sec)
        run_mrr_secs.append(run_mrr_sec)
        run_auc_secs.append(run_auc_sec)
        run_resource_profiles.append(_run_resource_profile(
            run_idx + 1, run_seed, last_epoch, timed_out, (run_train_sec, run_test_sec, run_eval_sec),
            ((run_train_peak_cpu, run_train_peak_cuda, run_train_peak_cuda_reserved),
             (run_test_peak_cpu, run_test_peak_cuda, run_test_peak_cuda_reserved),
             (run_eval_peak_cpu, run_eval_peak_cuda, run_eval_peak_cuda_reserved)), run_test_completed,
            validation_selection_value=validation_selection_value, validation_replay_value=validation_replay_value,
            validation_replay_abs_delta=validation_replay_abs_delta, validation_replay_mismatch=validation_replay_mismatch))
        recorded = _record_test_metrics(best_test_selected_for_run, best_test_other_metrics, test_selected_metrics,
                                        {"AUC": test_aucs, "AP": test_aps, "MRR": test_mrrs}, test_hits_any)
        if recorded:
            tqdm.write(f"\n[RUN {run_idx + 1}] Best Val {metric_label}: {100 * best_val_selected:.6f}")
            tqdm.write(f"[RUN {run_idx + 1}] Test  {metric_label}: {100 * best_test_selected_for_run:.6f}")
        elif selector_training_mode and best_epoch is not None:
            tqdm.write(f"\n[RUN {run_idx + 1}] Best neutral validation {metric_label}: {100 * best_val_selected:.6f}")
        else:
            tqdm.write(f"\n[RUN {run_idx + 1}] No valid selection metric computed; skipping recording for this run.")
        tqdm.write(
            f"[RUN {run_idx + 1}] train_time_sec={run_train_sec:.2f} test_time_sec={run_test_sec:.2f}"
            f" eval_model_selection_sec={run_eval_sec:.2f} train_peak_cpu_rss_mb={run_train_peak_cpu:.2f}"
            f" test_peak_cpu_rss_mb={run_test_peak_cpu:.2f} train_peak_cuda_allocated_mb={run_train_peak_cuda:.2f}"
            f" test_peak_cuda_allocated_mb={run_test_peak_cuda:.2f}"
            f" train_peak_cuda_reserved_mb={run_train_peak_cuda_reserved:.2f}"
            f" test_peak_cuda_reserved_mb={run_test_peak_cuda_reserved:.2f} inference_sec={run_inference_sec:.2f}"
            f" metric_computation_sec={run_testing_sec:.2f} mrr_sec={run_mrr_sec:.2f} auc_sec={run_auc_sec:.2f}"
        )
        if best_epoch is None:
            checkpoint_state_dict = _snapshot_state_dict_cpu(model.state_dict())
            checkpoint_epoch = last_epoch
            checkpoint_type = "final_model_state"
        else:
            checkpoint_state_dict = (
                best_state_dict if best_state_dict is not None else _snapshot_state_dict_cpu(model.state_dict())
            )
            checkpoint_epoch = best_epoch
            checkpoint_type = "best_validation_model_state"
        checkpoint_path = _save_model_checkpoint(
            checkpoint_state_dict,
            framework="pyg",
            mode=args.mode,
            dataset=args.dataset,
            model_name=output_model_name,
            run_number=run_seed + 1,
            seed=run_seed,
            epoch=checkpoint_epoch,
            timed_out=timed_out,
            metric_name=metric_label,
            best_val=best_val_selected,
            best_test=best_test_selected_for_run,
            args=args,
            model_config=best_config,
            checkpoint_type=checkpoint_type,
        )
        tqdm.write(f"[RUN {run_idx + 1}] Saved checkpoint: {checkpoint_path}")
        del checkpoint_state_dict
        optimizer = None
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if timed_out:
            print(f"Stopping remaining runs because runtime exceeded 24 hours. elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
    total_wall_sec = time.time() - program_t0
    header_lines = [
        "\n" + "=" * 80, "Timing summary", f"dataset: {args.dataset}", f"mode: {args.mode}", f"model: {output_model_name}",
        f"device: {device}", f"evaluation_positive_cap: {args.eval_cap}",
        f"compute_auc: {args.compute_auc}",
        "evaluation_scope: " + (
            "deterministic_neutral_validation_rows"
            if selector_training_mode
            else "configured_heart_validation_and_test_rows" if validation_only_mode else "configured_validation_and_test_rows"
        ),
        "model_selection_evaluation: " + (
            "neutral_validation_only"
            if selector_training_mode
            else "validation_final_test_only" if validation_only_mode else "validation_and_test"
        ),
        f"train_samples_per_epoch: {train_samples_per_epoch}", f"model_implementation: {best_config.get('model_implementation', 'unspecified')}",
        f"model_execution_path: {best_config.get('execution_path', 'unspecified')}",
    ]
    if selector_training_mode:
        header_lines += [f"{key}: {value}" for key, value in sorted(selector_validation_metadata.items())]
    header_lines += [f"{key}: {(value if value is not None else 'not-applicable')}" for key, value in heart_candidate_metadata.items()]
    header_lines += [f"ranking_device: {device}", f"runtime_limit_exceeded: {timed_out}",
                     "status: " + ("exceeded 24 hour runtime limit" if timed_out else "completed within 24 hour runtime limit")]
    summary_values = {
        "runtime_limit_sec": RUNTIME_LIMIT_SEC, "data_load_sec": data_load_sec, "train_total_sec": total_train_sec,
        "test_total_sec": total_test_sec, "eval_total_sec": total_eval_sec, "inference_total_sec": total_inference_sec,
        "testing_total_sec": total_testing_sec, "mrr_total_sec": total_mrr_sec, "auc_total_sec": total_auc_sec,
        "total_wall_sec": total_wall_sec, "cpu_rss_mb_current": current_cpu_rss_mb(), "cpu_rss_mb_peak_process": peak_cpu_rss_mb(),
        "train_peak_cpu_rss_mb_max": train_peak_cpu, "test_peak_cpu_rss_mb_max": test_peak_cpu, "eval_peak_cpu_rss_mb_max": eval_peak_cpu,
        "train_peak_cuda_allocated_mb_max": train_peak_cuda, "test_peak_cuda_allocated_mb_max": test_peak_cuda,
        "eval_peak_cuda_allocated_mb_max": eval_peak_cuda, "train_peak_cuda_reserved_mb_max": train_peak_cuda_reserved,
        "test_peak_cuda_reserved_mb_max": test_peak_cuda_reserved, "eval_peak_cuda_reserved_mb_max": eval_peak_cuda_reserved,
    }
    per_run_series = {
        "train_sec_per_run": run_train_secs, "test_sec_per_run": run_test_secs, "eval_sec_per_run": run_eval_secs,
        "inference_sec_per_run": run_inference_secs, "testing_sec_per_run": run_testing_secs,
        "mrr_sec_per_run": run_mrr_secs, "auc_sec_per_run": run_auc_secs,
        "train_peak_cpu_rss_mb_per_run": [p["train_peak_cpu_rss_mb"] for p in run_resource_profiles],
        "test_peak_cpu_rss_mb_per_run": [p["test_peak_cpu_rss_mb"] for p in run_resource_profiles],
        "train_peak_cuda_allocated_mb_per_run": [p["train_peak_cuda_allocated_mb"] for p in run_resource_profiles],
        "test_peak_cuda_allocated_mb_per_run": [p["test_peak_cuda_allocated_mb"] for p in run_resource_profiles],
    }
    _write_timing_summary(
        log_path=log_path, header_lines=header_lines, summary_values=summary_values, per_run_series=per_run_series,
        run_profiles=run_resource_profiles,
        profile_value_keys=("validation_selection_value", "validation_replay_value", "validation_replay_abs_delta"),
        profile_bool_keys=(("validation_replay_accepted_mismatch", "validation_replay_mismatch"),),
        extra_lines=(f"validation_replay_accepted_mismatch_runs: {sum(bool(p['validation_replay_mismatch']) for p in run_resource_profiles)}",),
        test_selected=test_selected_metrics, test_aucs=test_aucs, test_aps=test_aps, test_mrrs=test_mrrs,
        test_hits=test_hits_any, dataset=args.dataset, base_seed=base_seed, metric=metric_key,
    )


if __name__ == "__main__":
    main()
