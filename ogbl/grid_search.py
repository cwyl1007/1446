import argparse
import itertools
import json
import os
import random
import time
from typing import Any, Dict, Optional, Tuple
import torch
from tqdm import tqdm
from utils.profiling import configure_torch_cpu_threads
from .prepare_data import parse_pool_argument, read_data
from .fast_negatives import prepare_ddi_grouped_eval_edges
from .protocol import (
    bind_protocol_metadata,
    ogbl_protocol_metadata,
    resolve_ogbl_device,
    resolve_ogbl_eval_cap,
    resolve_ogbl_metric,
    set_seed,
)
from .train_eval import (
    cache_eval_edges_on_device,
    evaluate_ogbl_validation,
    find_result_key,
    move_graph_data_to_device,
    prepare_ogbl_evaluation,
    recommended_decode_batch_size,
    recommended_train_samples_per_epoch,
    release_ogbl_evaluation,
    train_one_epoch_ogbl as train_one_epoch_fast,
)
from .training import make_ogbl_optimizer as _make_optimizer
from model.pairwise_models import get_model
from model.feature_aggregation import aggregated_mlp_method, aggregated_mlp_recipe, is_aggregated_mlp, preprocess_aggregated_mlp

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OGBL_DATASETS = ("ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation2")
_GRID_MODELS = (
    "mf", "mlp", "ppr", "concat", "gcn", "gat", "sage", "gae",
    "seal", "buddy", "neo-gnn", "ncn", "ncnc", "nbfnet", "peg", "lpformer",
)
_GRID_MODEL_ALIASES = {
    name.replace("-", "").replace("_", ""): name for name in _GRID_MODELS
}
_GRID_MODEL_ALIASES["lpf"] = "lpformer"
_SEARCH_CONFIG_KEYS = {"emb_size", "layers", "dropout", "pred_layers", "lr", "weight_decay"}
_RUNTIME_CONFIG_KEYS = {
    "alpha",
    "model",
    "dataset",
    "epochs",
    "max_epochs",
    "eval_steps",
    "patience",
    "seed",
    "base_seed",
    "run",
    "run_number",
    "num_runs",
    "mode",
    "metric",
    "device",
    "root",
    "eval_cap",
    "batch_size",
    "train_batch_size",
    "train_decode_batch_size",
    "model_decode_batch_size",
    "eval_batch_size",
    "citation2_query_batch_size",
    "training_path",
    "train_negative_sampler",
    "max_train_batches",
    "cache_eval_edges",
    "test_eval_policy",
    "in_channels",
    "num_nodes",
    "train_edge_index",
    "use_node_emb",
    "dataset_name",
    "evaluation_mode",
    "train_samples_per_epoch",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Grid search for OGBL link prediction")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--mode", choices=["heart"], default="heart")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device")
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--metric", type=str, default="auto")
    parser.add_argument("--batch-size", "--train-batch-size", dest="batch_size", type=int, default=None)
    parser.add_argument("--train-decode-batch-size", type=int, default=None)
    parser.add_argument("--model-decode-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--train-negative-sampler", choices=["auto", "random", "fast", "pyg"], default="fast")
    parser.add_argument("--training-path", choices=["auto", "reference", "full-graph", "legacy"], default="auto")
    parser.add_argument("--cache-eval-edges", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--citation2-query-batch-size", type=int, default=512)
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--max-configs", type=int, default=0)
    parser.add_argument("--heart-negatives", type=int, default=500)
    parser.add_argument("--pool", type=parse_pool_argument, default=10000)
    parser.add_argument("--all-negatives", type=int, default=None)
    parser.add_argument("--ranked-negatives-backend", choices=["auto", "official", "batched", "fast", "dense"], default="auto")
    parser.add_argument("--negative-cache-dir", type=str, default=None)
    parser.add_argument("--no-negative-cache", action="store_true")
    parser.add_argument("--output-config", type=str, default=None)
    return parser.parse_args()


def _canonicalize_scope(args: argparse.Namespace) -> None:
    dataset = str(args.dataset).strip().lower()
    if dataset not in _OGBL_DATASETS:
        raise ValueError(f"Unsupported OGB grid-search dataset {args.dataset!r}. Supported datasets: {', '.join(_OGBL_DATASETS)}")
    compact_model = str(args.model).strip().lower().replace("-", "").replace("_", "")
    if compact_model in {"n2v", "node2vec", "heuristics"}:
        raise ValueError(f"{args.model!r} uses its own entry point and is not tuned by ogbl.grid_search.")
    if compact_model in {"mlpip", "concatip"}:
        raise ValueError(f"{args.model!r} is an evaluator selector configured by its batch runner, not ogbl.grid_search.")
    try:
        model = _GRID_MODEL_ALIASES[compact_model]
    except KeyError:
        raise ValueError(f"Unsupported OGB grid-search model {args.model!r}. Supported models: {', '.join(_GRID_MODELS)}") from None
    args.dataset = dataset
    args.model = model


def _default_config_path(model: str, dataset: str) -> str:
    return os.path.join(PROJECT_ROOT, "configs", f"{model}_{dataset}_config.json")


def _resolve_config_paths(args: argparse.Namespace) -> Tuple[str, Optional[str]]:
    default_path = _default_config_path(args.model, args.dataset)
    output_path = os.path.abspath(os.path.expanduser(args.output_config)) if args.output_config else default_path
    if os.path.isfile(output_path):
        input_path: Optional[str] = output_path
    elif os.path.isfile(default_path):
        input_path = default_path
    elif is_aggregated_mlp(args.model):
        mlp_path = _default_config_path("mlp", args.dataset)
        input_path = mlp_path if os.path.isfile(mlp_path) else None
    else:
        input_path = None
    return (output_path, input_path)


def _load_existing_best_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    with open(path, "r") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object.")
    best_config = payload.get("best_config", {})
    if not isinstance(best_config, dict):
        raise ValueError(f"{path} must contain a JSON object at 'best_config'.")
    return dict(best_config)


def _static_model_config(config: Dict[str, Any]) -> Dict[str, Any]:
    excluded = _SEARCH_CONFIG_KEYS | _RUNTIME_CONFIG_KEYS
    static = {}
    for key, value in config.items():
        if key in excluded:
            continue
        if key in {"execution_path", "implementation_name", "protocol_fidelity", "training_protocol", "train_negative_protocol"}:
            continue
        if key.startswith(("best_", "search_", "config_", "model_")):
            continue
        static[key] = value
    return static


def _resolved_model_options(model) -> Dict[str, Any]:
    options = {}
    for attribute, key in {
        "implementation_name": "model_implementation",
        "training_protocol": "model_training_protocol",
        "train_negative_protocol": "model_train_negative_protocol",
        "protocol_fidelity": "model_protocol_fidelity",
        "reference_mf_embedding_initialization": "model_embedding_initialization",
        "reference_mf_gradient_clipping": "model_gradient_clipping",
        "reference_evaluation_transform": "model_evaluation_transform",
        "reference_probability_loss": "model_probability_domain_loss",
        "optimizer_protocol": "model_optimizer_protocol",
        "gradient_clipping_protocol": "model_gradient_clipping_protocol",
        "gat_heads": "model_gat_heads",
        "dense_gae_loss": "model_dense_gae_loss",
        "dense_gae_evaluation": "model_dense_gae_evaluation",
        "effective_training_path": "model_effective_training_path",
    }.items():
        value = getattr(model, attribute, None)
        if value is not None:
            options[key] = value
    return options


def _build_eval_edges(bundle: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "pos_train_edge": bundle["train_pos"],
        "train_val_edge": bundle["train_val"],
        "pos_valid_edge": bundle["valid_pos"],
        "neg_valid_edge": bundle["valid_neg"],
        "pos_test_edge": bundle["test_pos"],
        "neg_test_edge": bundle["test_neg"],
    }


def _validate_args(args: argparse.Namespace) -> None:
    if not str(args.dataset).startswith("ogbl-"):
        raise ValueError("This script is for OGBL datasets only (dataset must start with 'ogbl-').")
    for name in ("epochs", "eval_steps", "patience"):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    for name in (
        "batch_size",
        "train_decode_batch_size",
        "model_decode_batch_size",
        "eval_batch_size",
        "citation2_query_batch_size",
        "heart_negatives",
    ):
        value = getattr(args, name)
        if value is not None and int(value) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive.")
    if int(args.max_configs) < 0:
        raise ValueError("--max-configs must be non-negative.")
    if int(args.heart_negatives) % 2:
        raise ValueError("--heart-negatives must be an even total.")
    if args.train_samples_per_epoch is not None and int(args.train_samples_per_epoch) < 0:
        raise ValueError("--train-samples-per-epoch must be non-negative.")


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    temporary_path = f"{path}.tmp.{os.getpid()}"
    try:
        with open(temporary_path, "w") as f:
            json.dump(payload, f, indent=2)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _jsonable_metrics(results: Dict[str, Any]) -> Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]]:
    out: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    for k, v in results.items():
        if isinstance(v, (tuple, list)) and len(v) == 3:
            (a, b, c) = v
            out[k] = (None if a is None else float(a), None if b is None else float(b), None if c is None else float(c))
    return out


def _is_pairwise_model(model_name: str) -> bool:
    name = str(model_name).strip().lower().replace("-", "").replace("_", "")
    return name in {"seal", "buddy", "neognn", "ncn", "ncnc", "nbfnet", "peg", "lpformer", "lpf"}


def main() -> None:
    args = parse_args()
    _canonicalize_scope(args)
    cpu_threads = configure_torch_cpu_threads()
    print(f"torch_cpu_threads={cpu_threads}", flush=True)
    _validate_args(args)
    device = resolve_ogbl_device(args.device)
    args.device = str(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    batch_was_explicit = args.batch_size is not None
    ppa_cuda = args.dataset == "ogbl-ppa" and device.type == "cuda"
    recommended_decode = recommended_decode_batch_size(args.dataset, args.model)
    if args.batch_size is None:
        args.batch_size = recommended_decode if ppa_cuda else 65536
    if args.train_decode_batch_size is None:
        args.train_decode_batch_size = recommended_decode if ppa_cuda and (not batch_was_explicit) else int(args.batch_size)
    if args.model_decode_batch_size is None and ppa_cuda:
        args.model_decode_batch_size = int(args.train_decode_batch_size)
    if args.eval_batch_size is None:
        if batch_was_explicit:
            args.eval_batch_size = int(args.batch_size)
        elif device.type == "cuda":
            args.eval_batch_size = 262144 if _is_pairwise_model(args.model) else 1048576
        else:
            args.eval_batch_size = int(args.batch_size)
    if args.train_samples_per_epoch is None:
        args.train_samples_per_epoch = recommended_train_samples_per_epoch(args.dataset, args.model)
    args.eval_cap = resolve_ogbl_eval_cap(args.eval_cap, args.mode, args.dataset)
    metric_key = resolve_ogbl_metric(args.metric, args.dataset)
    protocol_metadata = ogbl_protocol_metadata(
        dataset=args.dataset,
        mode=args.mode,
        eval_cap=args.eval_cap,
        selection_metric=metric_key,
    )
    bind_protocol_metadata(args, protocol_metadata, metric_key)
    (output_path, existing_config_path) = _resolve_config_paths(args)
    existing_best_config = _load_existing_best_config(existing_config_path)
    static_model_config = _static_model_config(existing_best_config)
    tqdm.write(f"Using device: {device}")
    tqdm.write(f"Dataset: {args.dataset}")
    tqdm.write(f"Evaluation mode: {args.mode}")
    tqdm.write(f"eval_cap: {args.eval_cap}")
    tqdm.write(f"evaluation_positive_scope: {protocol_metadata['evaluation_positive_scope']}")
    tqdm.write(f"reference_evaluation_scope: {str(protocol_metadata['reference_evaluation_scope']).lower()}")
    tqdm.write(f"candidate_artifact_compatibility: {protocol_metadata['candidate_artifact_compatibility']}")
    tqdm.write("heart_candidate_source: generated-only")
    tqdm.write(f"Selection metric requested: {args.metric}")
    tqdm.write(f"Selection metric effective: {metric_key}")
    tqdm.write(f"train_samples_per_epoch: {args.train_samples_per_epoch}")
    tqdm.write(f"batch_size: {args.batch_size}")
    tqdm.write(f"train_decode_batch_size: {args.train_decode_batch_size}")
    tqdm.write(f"model_decode_batch_size: {args.model_decode_batch_size}")
    tqdm.write(f"eval_batch_size: {args.eval_batch_size}")
    if existing_config_path is None:
        tqdm.write("Existing config: none (using model defaults)")
    else:
        tqdm.write(f"Existing config: {existing_config_path}")
    tqdm.write(f"Output config: {output_path}")
    set_seed(args.seed)
    mode_bundle = read_data(
        args.dataset,
        args.mode,
        eval_cap=args.eval_cap,
        seed=args.seed,
        root=args.root,
        all_negatives=args.all_negatives,
        ranked_backend=args.ranked_negatives_backend,
        negative_cache_dir=args.negative_cache_dir,
        cache_negatives=not args.no_negative_cache,
        heart_negatives=args.heart_negatives,
        pool=args.pool,
    )
    data = mode_bundle["data"]
    eval_edges = _build_eval_edges(mode_bundle)
    ddi_dedup_t0 = time.time()
    (eval_edges, ddi_dedup_summaries) = prepare_ddi_grouped_eval_edges(
        eval_edges, dataset_name=args.dataset, model_name=args.model, num_nodes=int(data.num_nodes), source_bundle=mode_bundle
    )
    ddi_dedup_prepare_sec = time.time() - ddi_dedup_t0
    for summary in ddi_dedup_summaries:
        tqdm.write(
            f"ddi_eval_dedup key={summary['key']} original_edges={summary['original_edges']} unique_edges={summary['unique_edges']} decode_fraction={summary['decode_fraction']:.6f} canonical_undirected={summary['canonical_undirected']} cpu_storage_bytes={summary['storage_nbytes']}"
        )
    if ddi_dedup_summaries:
        tqdm.write(f"ddi_eval_dedup_prepare_sec={ddi_dedup_prepare_sec:.2f}")
    if data.x.dtype != torch.float:
        data.x = data.x.to(torch.float)
    data = move_graph_data_to_device(data, device)
    preprocessing_feature_dim = None
    preprocessing_feature_seed = 0
    preprocessing_base_feature_dim = None
    if is_aggregated_mlp(args.model):
        feature_method = aggregated_mlp_method(args.model)
        feature_recipe = aggregated_mlp_recipe(args.model)
        if args.dataset == "ogbl-ddi":
            preprocessing_feature_dim = int(existing_best_config.get("emb_size", 256))
            preprocessing_base_feature_dim = int(preprocessing_feature_dim)
        else:
            preprocessing_base_feature_dim = int(data.x.size(-1))
        training_graph = getattr(data, "adj_t", None)
        if training_graph is None:
            training_graph = mode_bundle.get("adj", eval_edges["pos_train_edge"])
        data.x = preprocess_aggregated_mlp(
            args.model,
            args.dataset,
            data.x,
            training_graph,
            featureless_dim=preprocessing_feature_dim,
            featureless_seed=preprocessing_feature_seed,
        )
        if "x" in mode_bundle:
            mode_bundle["x"] = data.x
        self_loops = "none" if feature_method == "concat" else "once"
        tqdm.write(
            f"feature_preprocessing={feature_method} recipe={feature_recipe} graph=train-only binary=true self_loops={self_loops} output_width={data.x.size(-1)} cached=true"
        )
    (eval_edges, eval_cached, eval_bytes) = cache_eval_edges_on_device(eval_edges, device, option=args.cache_eval_edges)
    tqdm.write(f"eval_edges_cached_on_device: {eval_cached} ({eval_bytes} bytes)")
    in_channels = int(data.x.size(-1))
    num_nodes = int(data.num_nodes)
    train_edge_index = getattr(data, "edge_index", None)
    if args.epochs is None:
        args.epochs = 500
    if args.eval_steps is None:
        args.eval_steps = 5
    if args.patience is None:
        args.patience = 10
    lrs = [0.01, 0.001]
    dropouts = [0.0, 0.1, 0.3, 0.5]
    weight_decays = [0.0]
    model_layers = [int(existing_best_config.get("layers", 3))]
    pred_layers = [int(existing_best_config.get("pred_layers", 3))]
    emb_sizes = [int(existing_best_config.get("emb_size", 256))]
    search_space = list(itertools.product(lrs, dropouts, weight_decays, model_layers, pred_layers, emb_sizes))
    if args.max_configs and args.max_configs > 0:
        rng = random.Random(args.seed)
        rng.shuffle(search_space)
        search_space = search_space[: args.max_configs]
    tqdm.write(f"Total configs: {len(search_space)}")
    best_overall_val = float("-inf")
    best_overall_test: Optional[float] = None
    best_overall_config: Dict[str, Any] = {}
    best_overall_metrics_at_best: Dict[str, Tuple[Optional[float], Optional[float], Optional[float]]] = {}
    best_overall_resolved_model: Dict[str, Any] = {}
    config_pbar = tqdm(search_space, desc="Grid search", leave=True, dynamic_ncols=True, mininterval=0.2)
    eval_bsz = args.eval_batch_size
    for lr, dropout, weight_decay, layers, pred_layers_val, emb_size in config_pbar:
        config_pbar.set_description(f"cfg lr={lr} drop={dropout} wd={weight_decay} L={layers} predL={pred_layers_val} emb={emb_size}")
        set_seed(args.seed)
        candidate_config: Dict[str, Any] = {
            **static_model_config,
            "emb_size": emb_size,
            "layers": layers,
            "dropout": dropout,
            "pred_layers": pred_layers_val,
            "lr": lr,
            "weight_decay": weight_decay,
        }
        if is_aggregated_mlp(args.model):
            feature_method = aggregated_mlp_method(args.model)
            feature_recipe = aggregated_mlp_recipe(args.model)
            candidate_config[f"{feature_method}_preprocessing"] = feature_recipe
            candidate_config[f"{feature_method}_base_feature_dim"] = int(preprocessing_base_feature_dim)
            candidate_config[f"{feature_method}_output_feature_dim"] = int(data.x.size(-1))
            if args.dataset == "ogbl-ddi":
                candidate_config[f"{feature_method}_feature_dim"] = int(preprocessing_feature_dim)
                candidate_config[f"{feature_method}_feature_seed"] = int(preprocessing_feature_seed)
        if args.model_decode_batch_size is not None:
            candidate_config["decode_batch_size"] = int(args.model_decode_batch_size)
        params: Dict[str, Any] = {
            **candidate_config,
            "in_channels": in_channels,
            "num_nodes": num_nodes,
            "train_edge_index": train_edge_index,
            "use_node_emb": args.dataset == "ogbl-ddi" and (not is_aggregated_mlp(args.model)),
            "dataset_name": args.dataset,
            "evaluation_mode": args.mode,
            "train_samples_per_epoch": int(args.train_samples_per_epoch),
        }
        model = get_model(args.model, params).to(device)
        if bool(getattr(model, "reference_ogbl_mf", False)):
            set_seed(args.seed)
            model.reset_parameters()
        resolved_model_config = _resolved_model_options(model)
        optimizer = _make_optimizer(model, lr=lr, weight_decay=weight_decay, device=device)
        best_val_config = float("-inf")
        best_validation_results_config: Dict[str, Any] = {}
        patience_counter = 0
        epoch_pbar = tqdm(range(1, args.epochs + 1), desc="epochs", leave=False, dynamic_ncols=True, mininterval=0.2)
        for epoch in epoch_pbar:
            train_loss = train_one_epoch_fast(
                model=model,
                optimizer=optimizer,
                data=data,
                pos_train_edge=eval_edges["pos_train_edge"],
                device=device,
                batch_size=args.batch_size,
                negative_sampler=args.train_negative_sampler,
                train_decode_batch_size=args.train_decode_batch_size,
                seed=args.seed + epoch * 1009,
                training_path=args.training_path,
            )
            if epoch % args.eval_steps != 0 and epoch != args.epochs:
                epoch_pbar.set_postfix(loss=f"{train_loss:.4f}")
                continue
            context = prepare_ogbl_evaluation(
                model=model,
                data=data,
                eval_edges=eval_edges,
                dataset_name=args.dataset,
                device=device,
                batch_size=eval_bsz,
                citation2_query_batch_size=args.citation2_query_batch_size,
            )
            try:
                compute_selection_auc = metric_key in {"AUC", "AP"}
                validation_results = evaluate_ogbl_validation(context, compute_auc=compute_selection_auc)
                selected_key = find_result_key(validation_results, metric_key)
                if selected_key is None:
                    raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(validation_results.keys())}")
                (_, val_selected, _) = validation_results[selected_key]
                improved = val_selected is not None and float(val_selected) > best_val_config
            finally:
                release_ogbl_evaluation(context)
            if improved:
                best_val_config = float(val_selected)
                best_validation_results_config = validation_results
                patience_counter = 0
            else:
                patience_counter += 1
            epoch_pbar.set_postfix(
                loss=f"{train_loss:.4f}", val=f"{float(val_selected):.4f}" if val_selected is not None else "-", pat=str(patience_counter)
            )
            if patience_counter >= args.patience:
                break
        epoch_pbar.close()
        resolved_model_config.update(_resolved_model_options(model))
        if not best_validation_results_config:
            raise RuntimeError("Grid-search trial completed without a selectable validation checkpoint.")
        trial_improves_overall = best_val_config > best_overall_val
        del optimizer, model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if trial_improves_overall:
            best_overall_val = best_val_config
            best_overall_test = None
            best_overall_metrics_at_best = _jsonable_metrics(best_validation_results_config)
            best_overall_config = dict(candidate_config)
            best_overall_resolved_model = dict(resolved_model_config)
            tqdm.write(f"*** New best config: Val {metric_key}={best_overall_val:.4f} ***")
    tqdm.write("\n" + "#" * 80)
    tqdm.write("GRID SEARCH COMPLETE")
    if not best_overall_config:
        tqdm.write("No best result stored.")
        return
    tqdm.write("Best configuration:")
    for k, v in best_overall_config.items():
        tqdm.write(f"  {k}: {v}")
    tqdm.write(f"\nBest Val {metric_key} (search):  {best_overall_val:.4f}")
    tqdm.write("Best Test metric (search): not evaluated; run the selected configuration normally for held-out test results.")
    if best_overall_metrics_at_best:
        if "AUC" in best_overall_metrics_at_best:
            (_, v, _) = best_overall_metrics_at_best["AUC"]
            tqdm.write(f"At best config: Val AUC={v}")
        if "AP" in best_overall_metrics_at_best:
            (_, v, _) = best_overall_metrics_at_best["AP"]
            tqdm.write(f"At best config: Val AP={v}")
        if "MRR" in best_overall_metrics_at_best:
            (_, v, _) = best_overall_metrics_at_best["MRR"]
            tqdm.write(f"At best config: Val MRR={v}")
        for k, triple in best_overall_metrics_at_best.items():
            if isinstance(k, str) and (k.startswith("Hits@") or k.startswith("mrr_hit")):
                (_, v, _) = triple
                tqdm.write(f"At best config: Val {k}={v}")
    payload = {
        "best_config": best_overall_config,
        "best_val_metric": best_overall_val,
        "best_test_metric": best_overall_test,
        "best_metric_name": metric_key,
        "metrics_at_best": best_overall_metrics_at_best,
        "resolved_model": best_overall_resolved_model,
        "test_evaluated_during_search": False,
        "search_mode": args.mode,
        "search_eval_cap": args.eval_cap,
        "search_evaluation_positive_scope": protocol_metadata["evaluation_positive_scope"],
        "search_reference_evaluation_scope": protocol_metadata["reference_evaluation_scope"],
        "search_candidate_artifact_compatibility": protocol_metadata["candidate_artifact_compatibility"],
        "search_heart_candidate_source": "generated-only",
        "search_seed": args.seed,
        "search_selection_protocol": "validation-only",
        "search_epochs": int(args.epochs),
        "search_eval_steps": int(args.eval_steps),
        "search_patience": int(args.patience),
        "search_batch_size": int(args.batch_size),
        "search_train_decode_batch_size": int(args.train_decode_batch_size),
        "search_model_decode_batch_size": (
            None if args.model_decode_batch_size is None else int(args.model_decode_batch_size)
        ),
        "search_eval_batch_size": int(args.eval_batch_size),
        "search_train_samples_per_epoch": int(args.train_samples_per_epoch),
        "search_train_negative_sampler": args.train_negative_sampler,
        "search_training_path": args.training_path,
        "search_max_configs": int(args.max_configs),
        "search_timestamp": int(time.time()),
    }
    _atomic_write_json(output_path, payload)
    tqdm.write(f"\nBest config saved to: {output_path}")


if __name__ == "__main__":
    main()
