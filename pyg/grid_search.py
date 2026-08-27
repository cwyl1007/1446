import argparse
import gc
import itertools
import json
import math
import os
import random
import time
from typing import Any, Dict, List, Optional, Tuple
import torch
from tqdm import tqdm
from .train_eval import all_train, validation_only
from .prepare_data import parse_pool_argument
from .data_core import SUPPORTED_PYG_DATASETS
from .main import (
    _configure_cuda_matmul_precision,
    _default_batches,
    _default_heart_batch_size,
    _default_train_samples,
    _make_optimizer,
    _read_run_data,
    _resolve_device,
    _resolve_eval_cap,
    set_seed,
)
from model.pairwise_models import get_model, reset_reference_planetoid_model
from model.feature_aggregation import aggregated_mlp_method, aggregated_mlp_recipe, is_aggregated_mlp, preprocess_aggregated_mlp
from utils.heart_protocol import persist_heart_candidate_metadata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GRID_MODELS = (
    "mf", "mlp", "ppr", "concat", "gcn", "gat", "sage", "gae",
    "seal", "buddy", "neo-gnn", "ncn", "ncnc", "nbfnet", "peg", "lpformer",
)
_GRID_MODEL_ALIASES = {
    name.replace("-", "").replace("_", ""): name for name in _GRID_MODELS
}
_GRID_MODEL_ALIASES["lpf"] = "lpformer"
_STALE_CONFIG_KEYS = {
    "alpha",
    "model",
    "model_implementation",
    "implementation_name",
    "dataset",
    "mode",
    "metric",
    "device",
    "run",
    "base_seed",
    "epochs",
    "max_epochs",
    "eval_steps",
    "patience",
    "seed",
    "num_runs",
    "batch_size",
    "eval_cap",
    "pool",
    "heart_negatives",
    "planetoid_input_root",
    "heart_backend",
    "heart_batch_size",
    "heart_ppr_iters",
    "in_channels",
    "num_nodes",
    "train_edge_index",
    "dataset_name",
    "evaluation_mode",
    "config_source",
    "config_protocol",
    "execution_path",
    "protocol_fidelity",
    "training_protocol",
    "train_negative_protocol",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Successive Halving search (fast)")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--mode", choices=["heart"], default="heart")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    parser.add_argument("--pool", type=parse_pool_argument, default=10000)
    parser.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, choices=[500], default=500)
    parser.add_argument("--planetoid-input-root", type=str, default=None)
    parser.add_argument("--heart-backend", choices=["auto", "gpu", "dense"], default="auto")
    parser.add_argument("--heart-batch-size", type=int, default=0)
    parser.add_argument("--heart-ppr-iters", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--metric", type=str, default="mrr")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=60)
    parser.add_argument("--output-config", "--output", dest="output", type=str, default=None)
    return parser.parse_args()


def _canonicalize_scope(args: argparse.Namespace) -> None:
    dataset = str(args.dataset).strip().lower()
    if dataset not in SUPPORTED_PYG_DATASETS:
        raise ValueError(
            f"Unsupported PyG dataset {args.dataset!r}. Supported datasets: "
            + ", ".join(SUPPORTED_PYG_DATASETS)
        )
    compact_model = str(args.model).strip().lower().replace("-", "").replace("_", "")
    if compact_model in {"n2v", "node2vec", "heuristics"}:
        raise ValueError(f"{args.model!r} uses its own entry point and is not tuned by pyg.grid_search.")
    if compact_model in {"mlpip", "concatip"}:
        raise ValueError(f"{args.model!r} is an evaluator selector configured by its batch runner, not pyg.grid_search.")
    try:
        model = _GRID_MODEL_ALIASES[compact_model]
    except KeyError:
        raise ValueError(f"Unsupported PyG grid-search model {args.model!r}. Supported models: {', '.join(_GRID_MODELS)}") from None
    args.dataset = dataset
    args.model = model


def _metric_key_from_arg(metric: str) -> str:
    m = metric.strip().lower()
    if m in ("auc", "ap", "mrr"):
        return m.upper()
    if m.startswith("hits@"):
        k = m.split("@", 1)[1]
        if not k.isdigit():
            raise ValueError(f"Invalid metric '{metric}'. Expected hits@K with integer K.")
        return f"Hits@{int(k)}"
    if m.startswith("mrr_hit"):
        k = m[len("mrr_hit") :]
        if not k.isdigit():
            raise ValueError(f"Invalid metric '{metric}'. Expected mrr_hitK with integer K.")
        return f"Hits@{int(k)}"
    raise ValueError(f"Unsupported metric '{metric}'. Use: auc, ap, mrr, hits@K, mrr_hitK.")


def _reusable_config(config: Dict[str, Any]) -> Dict[str, Any]:
    return {
        key: value
        for key, value in config.items()
        if key not in _STALE_CONFIG_KEYS and not key.startswith(("best_", "search_", "config_", "model_"))
    }


def _load_existing_config(path: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if not os.path.isfile(path):
        return ({}, {})
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"Existing config payload must be a JSON object: {path}")
    best_config = payload.get("best_config", {})
    if not isinstance(best_config, dict):
        raise ValueError(f"Existing best_config must be a JSON object: {path}")
    return (dict(payload), _reusable_config(dict(best_config)))


def _jsonable_metrics(results: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    output: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for key, value in results.items():
        if not isinstance(value, (tuple, list)) or len(value) != 3:
            continue
        converted = []
        for item in value:
            converted.append(None if item is None else float(item))
        output[str(key)] = tuple(converted)
    return output


def _atomic_json_dump(payload: Dict[str, Any], path: str) -> None:
    absolute_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(absolute_path), exist_ok=True)
    temporary_path = f"{absolute_path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
        os.replace(temporary_path, absolute_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _make_stage_budgets(max_epochs: int) -> List[int]:
    b1 = max(1, max_epochs // 10)
    b2 = max(1, max_epochs // 3)
    budgets = sorted(set([b1, b2, max_epochs]))
    return budgets


def _stage_eval_steps(stage_epochs: int, final_epochs: int, base_eval_steps: int) -> int:
    if stage_epochs < final_epochs:
        if stage_epochs <= 30:
            return stage_epochs
        if stage_epochs <= 100:
            return max(10, stage_epochs // 5)
        return max(10, stage_epochs // 10)
    return base_eval_steps


def _stage_patience(stage_epochs: int, final_epochs: int, base_patience: int) -> int:
    if stage_epochs < final_epochs:
        return 1 if stage_epochs <= 30 else 3
    return base_patience


def _run_one_config(
    *,
    args: argparse.Namespace,
    device: torch.device,
    metric_key: str,
    data: Dict[str, Any],
    x: torch.Tensor,
    train_pos: torch.Tensor,
    num_nodes: int,
    in_channels: int,
    base_config: Dict[str, Any],
    lr: float,
    dropout: float,
    weight_decay: float,
    layers: int,
    pred_layers_val: int,
    emb_size: int,
    max_epochs: int,
    eval_steps_this: int,
    patience_this: int,
) -> Tuple[float, Optional[float], Dict[str, Tuple[float, float, float]]]:
    set_seed(args.seed)
    normalized_model = str(args.model).strip().lower().replace("-", "").replace("_", "")
    params: Dict[str, Any] = {
        **base_config,
        "in_channels": in_channels,
        "emb_size": emb_size,
        "layers": layers,
        "pred_layers": pred_layers_val,
        "dropout": dropout,
        "num_nodes": num_nodes,
        "dataset_name": args.dataset,
        "evaluation_mode": args.mode,
        "train_samples_per_epoch": args.train_samples_per_epoch,
    }
    if normalized_model == "buddy" and str(args.dataset).lower() in {"cora", "citeseer", "pubmed"}:
        params["label_dropout"] = dropout
        params["feature_dropout"] = dropout
    model = get_model(args.model, params).to(device)
    reset_reference_planetoid_model(
        model,
        model_name=args.model,
        dataset_name=args.dataset,
        seed=args.seed,
        seed_fn=set_seed,
        device=device,
        emb_size=emb_size,
        pred_layers=pred_layers_val,
        dropout=dropout,
    )
    optimizer = _make_optimizer(model, lr=lr, weight_decay=weight_decay, device=device)
    best_val = float("-inf")
    best_test: Optional[float] = None
    best_metrics_at_best: Dict[str, Tuple[float, float, float]] = {}
    patience_counter = 0
    for epoch in range(1, max_epochs + 1):
        _ = all_train(
            model,
            train_pos,
            x,
            optimizer,
            args.batch_size,
            adj_t=data.get("adj"),
            csr_rowptr=data.get("csr_train_rowptr"),
            csr_col=data.get("csr_train_col"),
        )
        if epoch % eval_steps_this == 0 or epoch == max_epochs:
            results_rank = validation_only(
                model,
                data,
                x,
                args.eval_batch_size,
                include_auc=metric_key in {"AUC", "AP"},
                include_hits=metric_key.startswith("Hits@"),
            )
            if metric_key not in results_rank:
                raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(results_rank.keys())}")
            (_, val_selected, _) = results_rank[metric_key]
            improved = val_selected is not None and float(val_selected) > best_val
            if improved:
                best_val = float(val_selected)
                best_test = None
                best_metrics_at_best = {k: v for (k, v) in results_rank.items()}
                patience_counter = 0
            else:
                patience_counter += 1
            if patience_counter >= patience_this:
                break
    del optimizer, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return (best_val, best_test, best_metrics_at_best)


def main() -> None:
    args = parse_args()
    _canonicalize_scope(args)
    device = _resolve_device(args.device)
    args.device = str(device)
    metric_key = _metric_key_from_arg(args.metric)
    if args.epochs is None:
        args.epochs = 500
    if args.eval_steps is None:
        args.eval_steps = 5
    if args.patience is None:
        args.patience = 10
    requested_batch_size = args.batch_size
    requested_eval_batch_size = args.eval_batch_size
    requested_train_samples = args.train_samples_per_epoch
    default_config_path = os.path.join(PROJECT_ROOT, "configs", f"{args.model}_{args.dataset}_config.json")
    fallback_config_path = os.path.join(PROJECT_ROOT, "configs", f"mlp_{args.dataset}_config.json")
    out_path = os.path.abspath(os.path.expanduser(args.output)) if args.output else default_config_path
    if os.path.isfile(out_path):
        existing_path = out_path
    elif os.path.isfile(default_config_path):
        existing_path = default_config_path
    elif is_aggregated_mlp(args.model):
        existing_path = fallback_config_path
    else:
        existing_path = default_config_path
    (existing_payload, base_config) = _load_existing_config(existing_path)
    matmul_precision = _configure_cuda_matmul_precision(device, args.dataset, args.model)
    print(f"matmul_precision={matmul_precision}", flush=True)
    (default_train_batch, default_eval_batch) = _default_batches(args.dataset, args.model, device)
    args.batch_size = int(
        requested_batch_size if requested_batch_size is not None else base_config.get("train_batch_size", default_train_batch)
    )
    args.eval_batch_size = int(
        requested_eval_batch_size
        if requested_eval_batch_size is not None
        else requested_batch_size if requested_batch_size is not None else base_config.get("eval_batch_size", default_eval_batch)
    )
    default_train_samples = _default_train_samples(args.dataset, args.model, device)
    args.train_samples_per_epoch = int(
        requested_train_samples
        if requested_train_samples is not None
        else base_config.get("train_samples_per_epoch", default_train_samples)
    )
    args.eval_cap = _resolve_eval_cap(args.eval_cap, args.mode, args.dataset)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.eval_steps <= 0:
        raise ValueError("--eval-steps must be positive.")
    if args.patience <= 0:
        raise ValueError("--patience must be positive.")
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive.")
    if args.eval_batch_size <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    if args.train_samples_per_epoch < 0:
        raise ValueError("--train-samples-per-epoch must be non-negative.")
    if args.max_configs < 0:
        raise ValueError("--max-configs must be non-negative.")
    if args.eval_cap < 0:
        raise ValueError("--eval-cap must be non-negative.")
    if args.heart_batch_size < 0:
        raise ValueError("--heart-batch-size must be non-negative.")
    if args.heart_ppr_iters is not None and args.heart_ppr_iters <= 0:
        raise ValueError("--heart-ppr-iters must be positive when provided.")
    if args.heart_batch_size == 0:
        args.heart_batch_size = _default_heart_batch_size(device)
    tqdm.write(f"Using device: {device}")
    tqdm.write(f"Selection metric: {metric_key}")
    tqdm.write(f"Evaluation mode: {args.mode}")
    tqdm.write(f"eval_cap: {args.eval_cap}")
    tqdm.write(f"train_batch_size: {args.batch_size}")
    tqdm.write(f"eval_batch_size: {args.eval_batch_size}")
    tqdm.write(f"train_samples_per_epoch: {args.train_samples_per_epoch}")
    if base_config:
        tqdm.write(f"Preserving unsearched config fields from: {existing_path}")
    set_seed(args.seed)
    data = _read_run_data(args, device)
    heart_candidate_metadata = persist_heart_candidate_metadata(args, data)
    for key, value in heart_candidate_metadata.items():
        tqdm.write(f"{key}={(value if value is not None else 'not-applicable')}")
    if "adj" in data and device.type == "cuda":
        data["adj"] = data["adj"].to(device)
    x = data["x"].to(device, non_blocking=True)
    preprocessing_base_feature_dim = None
    if is_aggregated_mlp(args.model):
        feature_method = aggregated_mlp_method(args.model)
        feature_recipe = aggregated_mlp_recipe(args.model)
        preprocessing_base_feature_dim = int(x.size(-1))
        x = preprocess_aggregated_mlp(args.model, args.dataset, x, data.get("adj", data["train_pos"]))
        data["x"] = x
        self_loops = "none" if feature_method == "concat" else "once"
        tqdm.write(
            f"feature_preprocessing={feature_method} recipe={feature_recipe} graph=train-only binary=true self_loops={self_loops} output_width={x.size(-1)} cached=true"
        )
    keep_train_cpu = args.train_samples_per_epoch > 0
    train_pos = data["train_pos"] if keep_train_cpu else data["train_pos"].to(device, non_blocking=True)
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
                data[key] = data[key].to(device=device, dtype=torch.long, non_blocking=True)
    num_nodes = int(x.size(0))
    in_channels = int(x.size(1))
    output_config_overrides: Dict[str, Any] = {}
    if is_aggregated_mlp(args.model):
        feature_method = aggregated_mlp_method(args.model)
        feature_recipe = aggregated_mlp_recipe(args.model)
        output_config_overrides[f"{feature_method}_preprocessing"] = feature_recipe
        output_config_overrides["model_implementation"] = f"{feature_method}-{feature_recipe}"
        output_config_overrides[f"{feature_method}_base_feature_dim"] = int(preprocessing_base_feature_dim)
        output_config_overrides[f"{feature_method}_output_feature_dim"] = int(x.size(-1))
    if requested_batch_size is not None:
        output_config_overrides["train_batch_size"] = args.batch_size
        if requested_eval_batch_size is None:
            output_config_overrides["eval_batch_size"] = args.eval_batch_size
    if requested_eval_batch_size is not None:
        output_config_overrides["eval_batch_size"] = args.eval_batch_size
    if requested_train_samples is not None:
        output_config_overrides["train_samples_per_epoch"] = args.train_samples_per_epoch
    lrs = [0.01, 0.001]
    dropouts = [0.1, 0.3, 0.5]
    weight_decays = [0.0001, 1e-07, 0.0]
    model_layers = [1, 2, 3]
    pred_layers = [1, 2, 3]
    emb_sizes = [128, 256]
    search_space = list(itertools.product(lrs, dropouts, weight_decays, model_layers, pred_layers, emb_sizes))
    rng = random.Random(args.seed)
    rng.shuffle(search_space)
    if args.max_configs > 0 and args.max_configs < len(search_space):
        search_space = search_space[: args.max_configs]
    tqdm.write(f"Total configs (after subsample): {len(search_space)}")
    budgets = _make_stage_budgets(args.epochs)
    eta = 3
    tqdm.write(f"Successive Halving budgets: {budgets} | eta={eta}")
    tqdm.write("HeaRT grid search selects configurations from validation metrics only.")
    candidates = list(search_space)
    best_overall_val = float("-inf")
    best_overall_test = None
    best_overall_config: Dict[str, Any] = {}
    best_overall_metrics_at_best: Dict[str, Tuple[float, float, float]] = {}
    for stage_i, stage_epochs in enumerate(budgets):
        final_stage = stage_i == len(budgets) - 1
        eval_steps_this = _stage_eval_steps(stage_epochs, budgets[-1], args.eval_steps)
        patience_this = _stage_patience(stage_epochs, budgets[-1], args.patience)
        tqdm.write("\n" + "#" * 80)
        tqdm.write(
            f"STAGE {stage_i + 1}/{len(budgets)}: epochs={stage_epochs} | eval_steps={eval_steps_this} | patience={patience_this} | configs={len(candidates)} | test_eval=off"
        )
        stage_results: List[
            Tuple[float, Optional[float], Tuple[float, float, float, int, int, int], Dict[str, Tuple[float, float, float]]]
        ] = []
        pbar = tqdm(candidates, desc=f"Stage {stage_i + 1}", leave=True, dynamic_ncols=True, mininterval=0.2)
        for lr, dropout, weight_decay, layers, pred_layers_val, emb_size in pbar:
            pbar.set_description(
                f"S{stage_i + 1} lr={lr} drop={dropout} wd={weight_decay} L={layers} predL={pred_layers_val} emb={emb_size}"
            )
            (best_val, best_test, best_metrics) = _run_one_config(
                args=args,
                device=device,
                metric_key=metric_key,
                data=data,
                x=x,
                train_pos=train_pos,
                num_nodes=num_nodes,
                in_channels=in_channels,
                base_config=base_config,
                lr=lr,
                dropout=dropout,
                weight_decay=weight_decay,
                layers=layers,
                pred_layers_val=pred_layers_val,
                emb_size=emb_size,
                max_epochs=stage_epochs,
                eval_steps_this=eval_steps_this,
                patience_this=patience_this,
            )
            stage_results.append((best_val, best_test, (lr, dropout, weight_decay, layers, pred_layers_val, emb_size), best_metrics))
            if final_stage and best_val > best_overall_val:
                best_overall_val = best_val
                best_overall_test = best_test
                best_overall_metrics_at_best = best_metrics
                best_overall_config = {
                    **base_config,
                    "emb_size": emb_size,
                    "layers": layers,
                    "dropout": dropout,
                    "pred_layers": pred_layers_val,
                    "lr": lr,
                    "weight_decay": weight_decay,
                    **output_config_overrides,
                }
                tqdm.write(f"*** New final-budget best: Val {metric_key}={best_overall_val:.4f} ***")
        pbar.close()
        stage_results.sort(key=lambda t: t[0], reverse=True)
        if stage_i < len(budgets) - 1:
            keep_n = max(1, math.ceil(len(stage_results) / eta))
            kept = stage_results[:keep_n]
            candidates = [cfg for (_, _, cfg, _) in kept]
            tqdm.write(f"Keeping top {keep_n}/{len(stage_results)} configs for next stage")
        else:
            tqdm.write("Final stage complete (no further pruning)")
    tqdm.write("\n" + "#" * 80)
    tqdm.write("SUCCESSIVE HALVING COMPLETE")
    if not best_overall_config:
        tqdm.write("No best result stored.")
        return
    tqdm.write("Best configuration:")
    for k, v in best_overall_config.items():
        tqdm.write(f"  {k}: {v}")
    tqdm.write(f"\nBest Val {metric_key} (search):  {best_overall_val:.4f}")
    tqdm.write("Best Test metric (search): not evaluated; run the selected configuration normally for held-out test results.")
    payload = dict(existing_payload)
    payload.update(
        {
            "best_config": best_overall_config,
            "best_val_metric": best_overall_val,
            "best_test_metric": best_overall_test,
            "test_evaluated_during_search": False,
            "best_metric_name": metric_key,
            "metrics_at_best": _jsonable_metrics(best_overall_metrics_at_best),
            "search_timestamp": int(time.time()),
            "search_protocol": args.mode,
            "search_seed": int(args.seed),
            "search_epochs": int(args.epochs),
            "search_eval_cap": int(args.eval_cap),
            "search_stage_budgets": budgets,
            "search_train_batch_size": int(args.batch_size),
            "search_eval_batch_size": int(args.eval_batch_size),
            "search_train_samples_per_epoch": int(args.train_samples_per_epoch),
            "heart_candidate_metadata": heart_candidate_metadata,
        }
    )
    _atomic_json_dump(payload, out_path)
    tqdm.write(f"\nBest config saved to: {out_path}")


if __name__ == "__main__":
    main()
