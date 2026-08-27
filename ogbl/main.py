import argparse
import gc
import json
import math
import os
import time
import torch
from tqdm import tqdm
from .prepare_data import load_ogbl_splits, parse_pool_argument, read_data
from .fast_negatives import prepare_ddi_grouped_eval_edges
from .protocol import (
    RUNTIME_LIMIT_SEC,
    bind_protocol_metadata,
    log_aggregate_results,
    log_protocol_summary,
    log_run_statistics,
    ogbl_protocol_metadata,
    print_ogbl_protocol,
    resolve_ogbl_device,
    resolve_ogbl_eval_cap,
    resolve_ogbl_metric,
    runtime_exceeded as _runtime_exceeded,
    save_model_checkpoint,
    set_seed,
    should_compute_auc as _should_compute_auc,
    snapshot_state_dict_cpu as _snapshot_state_dict_cpu,
    write_summary as _write_summary,
)
from .mf_protocol import reference_mf_runtime_metadata
from .train_eval import (
    cache_eval_edges_on_device,
    evaluate_ogbl_test,
    evaluate_ogbl_validation,
    find_result_key,
    merge_ogbl_results,
    move_graph_data_to_device,
    prepare_ogbl_evaluation,
    recommended_decode_batch_size,
    recommended_train_samples_per_epoch,
    release_ogbl_evaluation,
    train_one_epoch_ogbl,
)
from .training import make_ogbl_optimizer as _make_optimizer
from utils.profiling import StageProfiler, configure_torch_cpu_threads, current_cpu_rss_mb, peak_cpu_rss_mb
from model.pairwise_models import get_model
from utils.heart_protocol import persist_heart_candidate_metadata
from model.feature_aggregation import aggregated_mlp_method, aggregated_mlp_recipe, is_aggregated_mlp, preprocess_aggregated_mlp


def _checkpoint_state_dict(model):
    compact = getattr(model, "checkpoint_state_dict", None)
    return compact() if callable(compact) else model.state_dict()


def _load_checkpoint_state_dict(model, state_dict):
    compact_loader = getattr(model, "load_checkpoint_state_dict", None)
    if callable(compact_loader):
        return compact_loader(state_dict, strict=True)
    return model.load_state_dict(state_dict)


_MODEL_PROVENANCE_ATTRIBUTES = {
    "implementation_name": "model_implementation",
    "training_protocol": "model_training_protocol",
    "train_negative_protocol": "model_train_negative_protocol",
    "protocol_fidelity": "model_protocol_fidelity",
    "reference_mf_embedding_initialization": "model_embedding_initialization",
    "reference_mf_gradient_clipping": "model_gradient_clipping",
    "reference_evaluation_transform": "model_evaluation_transform",
    "reference_probability_loss": "model_probability_domain_loss",
    "reference_mf_rng_protocol": "model_rng_protocol",
    "optimizer_protocol": "model_optimizer_protocol",
    "gradient_clipping_protocol": "model_gradient_clipping_protocol",
    "gat_heads": "model_gat_heads",
    "dense_gae_loss": "model_dense_gae_loss",
    "dense_gae_evaluation": "model_dense_gae_evaluation",
    "training_loss_protocol": "model_training_loss_protocol",
}


def _resolved_model_provenance(model):
    provenance = {}
    for attribute, key in _MODEL_PROVENANCE_ATTRIBUTES.items():
        value = getattr(model, attribute, None)
        if value is not None:
            provenance[key] = value
    effective_path = getattr(model, "effective_training_path", None)
    if effective_path is not None:
        provenance["model_effective_training_path"] = str(effective_path)
    return provenance


def _configure_ip_selector_evaluator(model, model_name):
    if str(model_name).strip().lower().replace("-", "").replace("_", "") not in {"mlpip", "concatip"}:
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run best config on OGBL dataset")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--metric", type=str, default="auto")
    parser.add_argument("--mode", choices=["heart", "all", "ranked-selector"], default="heart")
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device")
    parser.add_argument("--base-seed", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("--num-runs", "--runs", dest="num_runs", type=int, default=None)
    parser.add_argument("--selector-depth", type=int)
    parser.add_argument("--selector-hidden-channels", type=int)
    parser.add_argument("--selector-dropout", type=float)
    parser.add_argument("--selector-lr", "--selector-learning-rate", dest="selector_lr", type=float)
    parser.add_argument("--selector-weight-decay", type=float)
    parser.add_argument("--checkpoint-root")
    parser.add_argument("--results-root")
    parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--batch-size", "--batch_size", dest="batch_size", type=int, default=None)
    parser.add_argument("--train-decode-batch-size", type=int, default=None)
    parser.add_argument("--model-decode-batch-size", type=int, default=None)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--train-samples-per-epoch", type=int, default=None)
    parser.add_argument("--training-path", choices=["auto", "reference", "full-graph", "legacy"], default="auto")
    parser.add_argument("--cache-eval-edges", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--test-eval-policy", choices=["auto", "final", "improvement", "every"], default="auto")
    parser.add_argument("--train-negative-sampler", choices=["auto", "random", "fast", "pyg"], default="fast")
    parser.add_argument("--train-batch-progress", action="store_true")
    parser.add_argument("--max-train-batches", type=int, default=None)
    parser.add_argument("--citation2-query-batch-size", type=int, default=512)
    parser.add_argument("--all-negatives", type=int, default=None)
    parser.add_argument("--ranked-negatives-backend", choices=["auto", "official", "batched", "fast", "dense"], default="auto")
    parser.add_argument("--negative-cache-dir", type=str, default=None)
    parser.add_argument("--no-negative-cache", action="store_true")
    parser.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, default=500)
    parser.add_argument("--pool", type=parse_pool_argument, default=10000)
    parser.add_argument("--compute-auc", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--save-log", action="store_true", default=True)
    return parser.parse_args()


def _is_pairwise_model(model_name):
    name = str(model_name).strip().lower().replace("-", "").replace("_", "")
    return name in {"seal", "buddy", "neognn", "ncn", "ncnc", "nbfnet", "peg", "lpformer", "lpf"}


def _resolve_run_defaults(args, dataset):
    dataset = str(dataset).strip().lower()
    model_name = str(args.model).strip().lower()
    cuda_requested = str(args.device).startswith("cuda")
    ppa_cuda = dataset == "ogbl-ppa" and cuda_requested
    batch_was_explicit = args.batch_size is not None
    if args.epochs is None:
        args.epochs = 500
    if args.num_runs is None:
        args.num_runs = 5
    if args.eval_steps is None:
        args.eval_steps = 5
    if args.patience is None:
        args.patience = 10
    recommended_decode = recommended_decode_batch_size(dataset, model_name)
    if args.batch_size is None:
        if ppa_cuda:
            args.batch_size = recommended_decode
        else:
            args.batch_size = 65536
    if args.train_decode_batch_size is None:
        args.train_decode_batch_size = recommended_decode if ppa_cuda and (not batch_was_explicit) else int(args.batch_size)
    if args.model_decode_batch_size is None and ppa_cuda:
        args.model_decode_batch_size = int(args.train_decode_batch_size)
    if args.eval_batch_size is None:
        if batch_was_explicit:
            args.eval_batch_size = int(args.batch_size)
        elif cuda_requested:
            args.eval_batch_size = 262144 if _is_pairwise_model(model_name) else 1048576
        else:
            args.eval_batch_size = int(args.batch_size)
    if args.train_samples_per_epoch is None:
        args.train_samples_per_epoch = recommended_train_samples_per_epoch(dataset, model_name)
    if args.test_eval_policy == "auto":
        args.test_eval_policy = "final"
    if getattr(args, "max_train_batches", None) is None:
        args.max_train_batches = 0
    args.max_train_batches = max(0, int(args.max_train_batches))
    return args


def _build_eval_edges(bundle):
    return {
        "pos_train_edge": bundle["train_pos"],
        "train_val_edge": bundle["train_val"],
        "pos_valid_edge": bundle["valid_pos"],
        "neg_valid_edge": bundle["valid_neg"],
        "pos_test_edge": bundle["test_pos"],
        "neg_test_edge": bundle["test_neg"],
    }


def _ranked_selector_protocol_metadata(metric_key):
    return {
        "evaluation_protocol": "ranked-selector-neutral-validation-v1",
        "evaluation_positive_scope": "deterministic-validation-max-100000",
        "evaluation_positive_cap": 100000,
        "evaluation_validation_positive_cap": 100000,
        "evaluation_test_positive_cap": None,
        "reference_query_scope": False,
        "reference_evaluation_scope": False,
        "candidate_artifact_compatibility": "not-applicable-neutral-selector-validation",
        "ppa_query_panel_mode": None,
        "ppa_query_panel_scope": None,
        "selection_metric_effective": str(metric_key),
    }


def _read_ranked_selector_training_bundle(args):
    from eval_modes.evaluator_helpers import (
        ensure_complete_ranked_positive_splits,
        ensure_ogb_full_known_positive_filter,
        load_ranked_selector_bundle,
    )
    from eval_modes.ranked_helpers import build_neutral_selector_validation_negatives

    bundle = load_ranked_selector_bundle("ogb", args.dataset, args.root, args.seed)
    ensure_complete_ranked_positive_splits(
        bundle, framework="ogb", dataset=args.dataset, root=args.root, seed=args.seed
    )
    ensure_ogb_full_known_positive_filter(bundle, dataset=args.dataset)
    validation = build_neutral_selector_validation_negatives(
        bundle, "ogb", args.dataset, seed=args.seed, negatives=500
    )
    bundle["valid_neg"] = validation.negatives
    bundle["test_neg"] = None
    bundle["selector_validation_metadata"] = dict(validation.metadata)
    return bundle


def main():
    program_t0 = time.time()
    args = parse_args()
    cpu_threads = configure_torch_cpu_threads()
    device = resolve_ogbl_device(args.device)
    args.device = str(device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.set_float32_matmul_precision("high")
    mode = args.mode
    normalized_model = str(args.model).strip().lower().replace("-", "").replace("_", "")
    selector_training_mode = mode == "ranked-selector"
    if selector_training_mode and normalized_model != "concat":
        raise ValueError("--mode ranked-selector is reserved for training the Concat selector.")
    if selector_training_mode:
        requested_cap = 100000 if args.eval_cap in (None, 0) else int(args.eval_cap)
        if requested_cap != 100000:
            raise ValueError("Ranked-selector training requires --eval-cap 100000.")
        args.eval_cap = requested_cap
    else:
        args.eval_cap = resolve_ogbl_eval_cap(args.eval_cap, mode, args.dataset)
    args = _resolve_run_defaults(args, args.dataset)
    if selector_training_mode:
        if args.num_runs != 1:
            raise ValueError("Ranked-selector training requires --num-runs 1.")
        args.test_eval_policy = "final"
    metric_key = resolve_ogbl_metric(args.metric, args.dataset)
    protocol_metadata = (
        _ranked_selector_protocol_metadata(metric_key)
        if selector_training_mode
        else ogbl_protocol_metadata(
            dataset=args.dataset,
            mode=mode,
            eval_cap=args.eval_cap,
            selection_metric=metric_key,
        )
    )
    bind_protocol_metadata(args, protocol_metadata, metric_key)
    timed_out = False
    log_path = None
    if args.save_log:
        log_dir = os.path.join(args.results_root or "results", "ogbl", args.mode, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{args.model}.txt")
    print(f"Using device: {device}")
    print(f"torch_cpu_threads={cpu_threads}")
    if device.type == "cuda":
        print(f"cuda_tf32_enabled={torch.backends.cuda.matmul.allow_tf32}", flush=True)
    if selector_training_mode:
        print(
            "\n".join(
                (
                    f"Selection/reporting metric requested: {args.metric}",
                    f"Selection/reporting metric effective: {metric_key}",
                    f"Evaluation mode: {mode}",
                    f"eval_cap={args.eval_cap}",
                    "selector_validation=deterministic-neutral-legal-fixed-250-per-side",
                    f"runtime_limit_sec={RUNTIME_LIMIT_SEC}",
                    f"runtime_limit_hours={RUNTIME_LIMIT_SEC / 3600:.2f}",
                )
            )
        )
    else:
        print_ogbl_protocol(args, mode, protocol_metadata, metric_key)
    for key in ("training_path", "train_samples_per_epoch", "train_decode_batch_size", "model_decode_batch_size", "eval_batch_size", "cache_eval_edges", "test_eval_policy", "train_negative_sampler", "train_batch_progress", "max_train_batches"):
        print(f"{key}={getattr(args, key)}")
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
            raise ValueError("Selector overrides require an evaluator model and isolated checkpoint and result roots.")
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
        if args.dataset == "ogbl-ddi":
            best_config[f"{feature_method}_feature_dim"] = int(best_config["emb_size"])
            best_config[f"{feature_method}_feature_seed"] = 0
    best_config["train_samples_per_epoch"] = int(args.train_samples_per_epoch)
    print("Loaded best config:")
    for k, v in best_config.items():
        print(f"  {k}: {v}")
    print(
        f"\nUsing dataset={args.dataset}, epochs={args.epochs}, base_seed={args.seed}, num_runs={args.num_runs}, eval_steps={args.eval_steps}, patience={args.patience}, batch_size={args.batch_size}, train_decode_batch_size={args.train_decode_batch_size}"
    )
    test_selected_metrics, test_aucs, test_aps, test_mrrs = [], [], [], []
    test_hits_any = {}
    (
        run_train_secs, run_eval_secs, run_test_secs, run_inference_secs, run_testing_secs, run_mrr_secs, run_auc_secs,
        run_train_peak_cpu_rss_mbs, run_eval_peak_cpu_rss_mbs, run_train_peak_cuda_allocated_mbs,
        run_eval_peak_cuda_allocated_mbs, run_train_peak_cuda_reserved_mbs, run_eval_peak_cuda_reserved_mbs,
        run_test_peak_cpu_rss_mbs, run_test_peak_cuda_allocated_mbs, run_test_peak_cuda_reserved_mbs,
    ) = [[] for _ in range(16)]
    run_resource_records = []
    total_train_sec, total_eval_sec, total_test_sec, total_inference_sec, total_testing_sec, total_mrr_sec, total_auc_sec = (0.0,) * 7
    set_seed(args.seed)
    t_data = time.time()
    if selector_training_mode:
        mode_bundle = _read_ranked_selector_training_bundle(args)
        selector_validation_metadata = dict(mode_bundle["selector_validation_metadata"])
        best_config.update(selector_validation_metadata)
        args.selector_validation_metadata = dict(selector_validation_metadata)
        args.heart_source_resolved = None
        args.heart_candidate_metadata = {}
        heart_candidate_provenance = {}
    else:
        selector_validation_metadata = {}
        mode_bundle = read_data(
            args.dataset,
            mode,
            eval_cap=args.eval_cap,
            seed=args.seed,
            root=args.root,
            heart_negatives=args.heart_negatives,
            pool=args.pool,
            all_negatives=args.all_negatives,
            ranked_backend=args.ranked_negatives_backend,
            negative_cache_dir=args.negative_cache_dir,
            cache_negatives=not args.no_negative_cache,
        )
        heart_candidate_provenance = persist_heart_candidate_metadata(args, mode_bundle)
        if args.heart_source_resolved is None:
            args.heart_source_resolved = mode_bundle.get("heart_source", "generated-online")
    data = mode_bundle.get("data")
    if data is None:
        (data, _split_edge, _base_eval_edges) = load_ogbl_splits(name=args.dataset, root=args.root)
    if getattr(data, "x", None) is not None and data.x.dtype != torch.float:
        data.x = data.x.to(torch.float)
    eval_edges_base = _build_eval_edges(mode_bundle)
    ddi_dedup_t0 = time.time()
    (eval_edges_base, ddi_dedup_summaries) = prepare_ddi_grouped_eval_edges(
        eval_edges_base, dataset_name=args.dataset, model_name=args.model, num_nodes=int(data.num_nodes), source_bundle=mode_bundle
    )
    ddi_dedup_prepare_sec = time.time() - ddi_dedup_t0
    data_load_sec = time.time() - t_data
    print(f"data_load_sec={data_load_sec:.2f}", flush=True)
    for summary in ddi_dedup_summaries:
        print(
            f"ddi_eval_dedup key={summary['key']} original_edges={summary['original_edges']} unique_edges={summary['unique_edges']} decode_fraction={summary['decode_fraction']:.6f} canonical_undirected={summary['canonical_undirected']} cpu_storage_bytes={summary['storage_nbytes']}",
            flush=True,
        )
    if ddi_dedup_summaries:
        print(f"ddi_eval_dedup_prepare_sec={ddi_dedup_prepare_sec:.2f}", flush=True)
    if mode_bundle.get("heart_source"):
        print(f"heart_source={mode_bundle.get('heart_source')}", flush=True)
    if mode_bundle.get("pool_per_side") is not None:
        pool_fields = (
            ("pool_setting", "pool_setting"), ("pool_full_graph", "pool_full_graph"), ("pool_cap_applied", "pool_cap_applied"),
            ("pool_sampling", "pool_sampling"), ("pool_requested_per_side", "pool_requested_per_side"),
            ("pool_requested_total", "pool_requested_total"), ("pool_per_side_effective", "pool_per_side"),
            ("pool_total_effective", "pool_total"),
        )
        for label, key in pool_fields:
            print(f"{label}={mode_bundle.get(key)}", flush=True)
        if mode_bundle.get("pool_per_side_min") is not None:
            for key in ("pool_per_side_min", "pool_per_side_mean", "pool_per_side_max", "pool_total_min", "pool_total_mean", "pool_total_max"):
                print(f"{key}={mode_bundle.get(key)}", flush=True)
    if mode_bundle.get("heart_candidate_universe") is not None:
        for label, key in (
            ("heart_candidate_universe", "heart_candidate_universe"), ("heart_candidate_graph_nodes", "heart_candidate_graph_nodes"),
            ("heart_selection", "heart_selection"), ("heart_negatives_requested_per_side", "heart_negatives_requested_per_side"),
            ("heart_negatives_requested_total", "heart_negatives_requested_total"),
            ("heart_negatives_per_side_effective", "heart_negatives_per_side"), ("heart_negatives_total_effective", "heart_negatives_total"),
        ):
            print(f"{label}={mode_bundle.get(key)}", flush=True)
    if mode_bundle.get("negative_cache_path"):
        print(f"negative_cache_path={mode_bundle.get('negative_cache_path')}", flush=True)
    compute_auc = _should_compute_auc(args.compute_auc, mode)
    print(f"compute_auc_effective={compute_auc}", flush=True)
    selection_compute_auc = compute_auc and metric_key.lower() in {"auc", "ap"}
    print(f"selection_compute_auc_effective={selection_compute_auc}", flush=True)
    t_device = time.time()
    data = move_graph_data_to_device(data, device)
    if is_aggregated_mlp(args.model):
        feature_method = aggregated_mlp_method(args.model)
        feature_recipe = aggregated_mlp_recipe(args.model)
        feature_dim_key = f"{feature_method}_feature_dim"
        feature_seed_key = f"{feature_method}_feature_seed"
        base_feature_dim = int(best_config[feature_dim_key]) if args.dataset == "ogbl-ddi" else int(data.x.size(-1))
        training_graph = getattr(data, "adj_t", None)
        if training_graph is None:
            training_graph = mode_bundle.get("adj", eval_edges_base["pos_train_edge"])
        data.x = preprocess_aggregated_mlp(
            args.model,
            args.dataset,
            data.x,
            training_graph,
            featureless_dim=best_config.get(feature_dim_key),
            featureless_seed=best_config.get(feature_seed_key, 0),
        )
        if "x" in mode_bundle:
            mode_bundle["x"] = data.x
        best_config[f"{feature_method}_base_feature_dim"] = base_feature_dim
        best_config[f"{feature_method}_output_feature_dim"] = int(data.x.size(-1))
        feature_source = "fixed-identity-sketch" if args.dataset == "ogbl-ddi" else "dataset"
        self_loops = "none" if feature_method == "concat" else "once"
        print(
            f"feature_preprocessing={feature_method} recipe={feature_recipe} features={feature_source} graph=train-only binary=true self_loops={self_loops} output_width={data.x.size(-1)} cached=true",
            flush=True,
        )
    (eval_edges_base, eval_edges_cached, eval_edge_bytes) = cache_eval_edges_on_device(
        eval_edges_base, device, option=args.cache_eval_edges
    )
    device_prepare_sec = time.time() - t_device
    print(f"device_prepare_sec={device_prepare_sec:.2f}", flush=True)
    print(f"eval_edges_cached_on_device={eval_edges_cached}", flush=True)
    print(f"eval_edge_tensor_bytes={eval_edge_bytes}", flush=True)
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED during data loading: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    in_channels = int(data.x.size(-1))
    num_nodes = int(data.num_nodes)
    train_edge_index = getattr(data, "edge_index", None)
    fixed_feature_model = is_aggregated_mlp(args.model)
    use_node_emb = args.dataset == "ogbl-ddi" and (not fixed_feature_model)
    eval_batch_size = int(args.eval_batch_size)
    for run_idx in range(args.num_runs):
        if timed_out or _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED before run {run_idx + 1}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
        run_seed = args.seed + run_idx
        set_seed(run_seed)
        eval_edges = dict(eval_edges_base)
        pos_train = eval_edges["pos_train_edge"]
        pos_valid = eval_edges["pos_valid_edge"]
        g = torch.Generator(device=pos_train.device).manual_seed(run_seed + 3)
        if pos_train.size(0) >= pos_valid.size(0):
            idx = torch.randperm(pos_train.size(0), generator=g, device=pos_train.device)[: pos_valid.size(0)]
        else:
            idx = torch.randint(0, pos_train.size(0), (pos_valid.size(0),), generator=g, device=pos_train.device)
        eval_edges["train_val_edge"] = pos_train[idx]
        params = {
            **best_config,
            "in_channels": in_channels,
            "emb_size": best_config["emb_size"],
            "layers": best_config["layers"],
            "pred_layers": best_config["pred_layers"],
            "dropout": best_config["dropout"],
            "num_nodes": num_nodes,
            "train_edge_index": train_edge_index,
            "use_node_emb": use_node_emb,
            "dataset_name": args.dataset,
            "evaluation_mode": mode,
            "train_samples_per_epoch": int(args.train_samples_per_epoch),
        }
        if args.model_decode_batch_size is not None:
            params["decode_batch_size"] = int(args.model_decode_batch_size)
        model = get_model(args.model, params).to(device)
        ip_evaluator_metadata = _configure_ip_selector_evaluator(model, normalized_model)
        if ip_evaluator_metadata is not None:
            best_config.update(ip_evaluator_metadata)
        if normalized_model == "concat" and (selector_override_requested or selector_training_mode):
            from eval_modes.ranked_helpers import configure_ranked_selector_training

            best_config.update(configure_ranked_selector_training(model, args.dataset))
        if bool(getattr(model, "reference_ogbl_mf", False)):
            model.training_protocol = "heart-ogbl-mf-random-endpoint-minibatch-v3-source-continuous-rng"
            model.reference_mf_rng_protocol = "continuous-cpu-dataloader-shuffle+continuous-device-negative-rng"
            best_config.update(
                reference_mf_runtime_metadata(
                    args.dataset,
                    epochs=args.epochs,
                    eval_steps=args.eval_steps,
                    patience=args.patience,
                    optimizer_batch_size=args.batch_size,
                )
            )
            set_seed(run_seed)
            model.reset_parameters()
        implementation_name = getattr(model, "implementation_name", None)
        if implementation_name is not None:
            print(f"model_implementation={implementation_name}", flush=True)
            best_config["model_implementation"] = str(implementation_name)
        best_config.update(_resolved_model_provenance(model))
        optimizer = _make_optimizer(model, lr=best_config["lr"], weight_decay=best_config["weight_decay"], device=device)
        last_epoch = 0
        best_val_selected = float("-inf")
        best_test_selected_for_run = None
        best_test_other_metrics = {}
        metric_label = metric_key
        patience_counter = 0
        best_state_dict = best_epoch = None
        (
            run_train_sec, run_eval_sec, run_inference_sec, run_testing_sec, run_mrr_sec, run_auc_sec,
            run_train_peak_cpu_rss_mb, run_eval_peak_cpu_rss_mb, run_train_peak_cuda_allocated_mb,
            run_eval_peak_cuda_allocated_mb, run_train_peak_cuda_reserved_mb, run_eval_peak_cuda_reserved_mb,
            run_test_sec, run_test_peak_cpu_rss_mb, run_test_peak_cuda_allocated_mb, run_test_peak_cuda_reserved_mb,
        ) = (0.0,) * 16
        run_test_completed = False
        pbar = tqdm(range(1, args.epochs + 1), desc=f"Run {run_idx + 1}/{args.num_runs} (seed={run_seed})", leave=False)
        for epoch in pbar:
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED before epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            train_profiler = StageProfiler(device)
            train_profiler.start()
            train_loss = train_one_epoch_ogbl(
                model,
                optimizer,
                data,
                eval_edges["pos_train_edge"],
                device,
                batch_size=args.batch_size,
                negative_sampler=args.train_negative_sampler,
                train_decode_batch_size=args.train_decode_batch_size,
                show_batch_progress=args.train_batch_progress,
                seed=run_seed + epoch * 1009,
                max_batches=args.max_train_batches,
                training_path=args.training_path,
            )
            last_epoch = epoch
            train_info = train_profiler.stop()
            epoch_train_sec = train_info["sec"]
            run_train_sec += epoch_train_sec
            total_train_sec += epoch_train_sec
            run_train_peak_cpu_rss_mb = max(run_train_peak_cpu_rss_mb, train_info["cpu_peak_rss_mb"])
            run_train_peak_cuda_allocated_mb = max(run_train_peak_cuda_allocated_mb, train_info["cuda_peak_allocated_mb"])
            run_train_peak_cuda_reserved_mb = max(run_train_peak_cuda_reserved_mb, train_info["cuda_peak_reserved_mb"])
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after train epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                pbar.set_postfix(loss=f"{train_loss:.6f}")
                break
            if epoch % args.eval_steps != 0 and epoch != args.epochs:
                pbar.set_postfix(loss=f"{train_loss:.6f}")
                continue
            stats = {"mrr_sec": 0.0, "auc_sec": 0.0}
            eval_profile = {}
            eval_profiler = StageProfiler(device)
            eval_profiler.start()
            eval_context = None
            try:
                eval_context = prepare_ogbl_evaluation(
                    model=model,
                    data=data,
                    eval_edges=eval_edges,
                    dataset_name=args.dataset,
                    device=device,
                    batch_size=eval_batch_size,
                    citation2_query_batch_size=args.citation2_query_batch_size,
                    profile=eval_profile,
                )
                validation_results = evaluate_ogbl_validation(eval_context, compute_auc=selection_compute_auc, profile=eval_profile)
                selected_key = find_result_key(validation_results, metric_key)
                if selected_key is None:
                    raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(validation_results.keys())}")
                (_train_selected, val_selected, _) = validation_results[selected_key]
                improved = val_selected is not None and float(val_selected) > best_val_selected
                evaluate_test_now = args.test_eval_policy == "every" or (args.test_eval_policy == "improvement" and improved)
                if evaluate_test_now:
                    test_results = evaluate_ogbl_test(eval_context, compute_auc=selection_compute_auc, profile=eval_profile)
                    results_rank = merge_ogbl_results(validation_results, test_results)
                else:
                    results_rank = validation_results
            finally:
                release_ogbl_evaluation(eval_context)
            eval_info = eval_profiler.stop()
            stats["mrr_sec"] = float(eval_profile.get("mrr_sec", 0.0))
            stats["auc_sec"] = float(eval_profile.get("auc_sec", 0.0))
            epoch_eval_sec = eval_info["sec"]
            epoch_inference_sec = float(eval_profile.get("inference_sec", max(0.0, epoch_eval_sec - stats["mrr_sec"] - stats["auc_sec"])))
            epoch_testing_sec = float(eval_profile.get("testing_sec", stats["mrr_sec"] + stats["auc_sec"]))
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
            run_eval_peak_cpu_rss_mb = max(run_eval_peak_cpu_rss_mb, eval_info["cpu_peak_rss_mb"])
            run_eval_peak_cuda_allocated_mb = max(run_eval_peak_cuda_allocated_mb, eval_info["cuda_peak_allocated_mb"])
            run_eval_peak_cuda_reserved_mb = max(run_eval_peak_cuda_reserved_mb, eval_info["cuda_peak_reserved_mb"])
            (_train_selected, val_selected, test_selected) = results_rank[selected_key]
            metric_label = selected_key
            if improved:
                best_val_selected = float(val_selected)
                best_test_selected_for_run = None if test_selected is None else float(test_selected)
                best_test_other_metrics = dict(results_rank)
                best_state_dict = _snapshot_state_dict_cpu(_checkpoint_state_dict(model))
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            pbar.set_postfix(
                loss=f"{train_loss:.6f}",
                eval=f"{epoch_eval_sec:.2f}s",
                infer=f"{epoch_inference_sec:.2f}s",
                test=f"{epoch_testing_sec:.2f}s",
                mrr=f"{stats['mrr_sec']:.2f}s",
                auc=f"{stats['auc_sec']:.2f}s",
            )
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after eval epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            if patience_counter >= args.patience:
                break
        if best_state_dict is not None and (not timed_out):
            _load_checkpoint_state_dict(model, best_state_dict)
            if hasattr(model, "configure_epoch"):
                model.configure_epoch(best_epoch, args.epochs)
            best_state_dict = None
            model.zero_grad(set_to_none=True)
            optimizer = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if not selector_training_mode:
                final_test_profile = {}
                final_test_profiler = StageProfiler(device)
                final_test_context = None
                final_test_profiler.start()
                try:
                    final_test_context = prepare_ogbl_evaluation(
                        model=model,
                        data=data,
                        eval_edges=eval_edges,
                        dataset_name=args.dataset,
                        device=device,
                        batch_size=eval_batch_size,
                        citation2_query_batch_size=args.citation2_query_batch_size,
                        profile=final_test_profile,
                        test_only=True,
                    )
                    final_test_results = evaluate_ogbl_test(
                        final_test_context, compute_auc=compute_auc, profile=final_test_profile
                    )
                finally:
                    release_ogbl_evaluation(final_test_context)
                    final_test_info = final_test_profiler.stop()
                run_test_sec = float(final_test_info["sec"])
                run_test_peak_cpu_rss_mb = float(final_test_info["cpu_peak_rss_mb"])
                run_test_peak_cuda_allocated_mb = float(final_test_info["cuda_peak_allocated_mb"])
                run_test_peak_cuda_reserved_mb = float(final_test_info["cuda_peak_reserved_mb"])
                run_test_completed = True
                total_test_sec += run_test_sec
                final_inference_sec = float(final_test_profile.get("inference_sec", 0.0))
                final_mrr_sec = float(final_test_profile.get("mrr_sec", 0.0))
                final_auc_sec = float(final_test_profile.get("auc_sec", 0.0))
                final_testing_sec = float(final_test_profile.get("testing_sec", final_mrr_sec + final_auc_sec))
                run_inference_sec += final_inference_sec
                run_testing_sec += final_testing_sec
                run_mrr_sec += final_mrr_sec
                run_auc_sec += final_auc_sec
                total_inference_sec += final_inference_sec
                total_testing_sec += final_testing_sec
                total_mrr_sec += final_mrr_sec
                total_auc_sec += final_auc_sec
                best_test_other_metrics = merge_ogbl_results(best_test_other_metrics, final_test_results)
                final_selected_key = find_result_key(best_test_other_metrics, metric_key)
                if final_selected_key is not None:
                    (_, _, final_test_selected) = best_test_other_metrics[final_selected_key]
                    if final_test_selected is not None:
                        best_test_selected_for_run = float(final_test_selected)
                        metric_label = final_selected_key
        run_train_secs.append(run_train_sec)
        run_eval_secs.append(run_eval_sec)
        run_test_secs.append(run_test_sec)
        run_inference_secs.append(run_inference_sec)
        run_testing_secs.append(run_testing_sec)
        run_mrr_secs.append(run_mrr_sec)
        run_auc_secs.append(run_auc_sec)
        run_train_peak_cpu_rss_mbs.append(run_train_peak_cpu_rss_mb)
        run_eval_peak_cpu_rss_mbs.append(run_eval_peak_cpu_rss_mb)
        run_train_peak_cuda_allocated_mbs.append(run_train_peak_cuda_allocated_mb)
        run_eval_peak_cuda_allocated_mbs.append(run_eval_peak_cuda_allocated_mb)
        run_train_peak_cuda_reserved_mbs.append(run_train_peak_cuda_reserved_mb)
        run_eval_peak_cuda_reserved_mbs.append(run_eval_peak_cuda_reserved_mb)
        run_test_peak_cpu_rss_mbs.append(run_test_peak_cpu_rss_mb)
        run_test_peak_cuda_allocated_mbs.append(run_test_peak_cuda_allocated_mb)
        run_test_peak_cuda_reserved_mbs.append(run_test_peak_cuda_reserved_mb)
        run_resource_records.append(
            {
                "run": run_idx + 1,
                "seed": run_seed,
                "train_time_sec": run_train_sec,
                "test_time_sec": run_test_sec,
                "train_peak_cpu_rss_mb": run_train_peak_cpu_rss_mb,
                "test_peak_cpu_rss_mb": run_test_peak_cpu_rss_mb,
                "train_peak_cuda_allocated_mb": run_train_peak_cuda_allocated_mb,
                "test_peak_cuda_allocated_mb": run_test_peak_cuda_allocated_mb,
                "train_peak_cuda_reserved_mb": run_train_peak_cuda_reserved_mb,
                "test_peak_cuda_reserved_mb": run_test_peak_cuda_reserved_mb,
                "test_completed": run_test_completed,
            }
        )
        if best_test_selected_for_run is not None:
            test_selected_metrics.append(best_test_selected_for_run)
            for key, values in (("AUC", test_aucs), ("AP", test_aps), ("MRR", test_mrrs)):
                if key in best_test_other_metrics:
                    values.append(float(best_test_other_metrics[key][2]))
            for k, triple in best_test_other_metrics.items():
                if isinstance(k, str) and (k.startswith("Hits@") or k.startswith("mrr_hit")):
                    (_, _, t) = triple
                    test_hits_any.setdefault(k, []).append(float(t))
            tqdm.write(f"\n[RUN {run_idx + 1}] Best Val {metric_label}: {100 * best_val_selected:.6f}")
            tqdm.write(f"[RUN {run_idx + 1}] Test  {metric_label}: {100 * best_test_selected_for_run:.6f}")
        elif selector_training_mode and best_epoch is not None:
            tqdm.write(f"\n[RUN {run_idx + 1}] Best neutral validation {metric_label}: {100 * best_val_selected:.6f}")
        else:
            tqdm.write(f"\n[RUN {run_idx + 1}] No valid selection metric computed; skipping recording for this run.")
        tqdm.write(
            f"[RUN {run_idx + 1}] train_sec={run_train_sec:.2f} test_sec={run_test_sec:.2f} eval_sec={run_eval_sec:.2f}"
            f" inference_sec={run_inference_sec:.2f} testing_sec={run_testing_sec:.2f} mrr_sec={run_mrr_sec:.2f}"
            f" auc_sec={run_auc_sec:.2f} train_peak_cpu_rss_mb={run_train_peak_cpu_rss_mb:.2f}"
            f" test_peak_cpu_rss_mb={run_test_peak_cpu_rss_mb:.2f}"
            f" train_peak_cuda_allocated_mb={run_train_peak_cuda_allocated_mb:.2f}"
            f" test_peak_cuda_allocated_mb={run_test_peak_cuda_allocated_mb:.2f}"
        )
        if best_epoch is None:
            checkpoint_state_dict = _snapshot_state_dict_cpu(_checkpoint_state_dict(model))
            checkpoint_epoch = last_epoch
            checkpoint_type = "final_model_state"
        else:
            checkpoint_state_dict = (
                best_state_dict if best_state_dict is not None else _snapshot_state_dict_cpu(_checkpoint_state_dict(model))
            )
            checkpoint_epoch = best_epoch
            checkpoint_type = "best_validation_model_state"
        best_config.update(_resolved_model_provenance(model))
        checkpoint_path = save_model_checkpoint(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            checkpoint_state_dict,
            framework="ogbl",
            mode=mode,
            dataset=args.dataset,
            model_name=args.model,
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
        del optimizer, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if timed_out:
            print(f"Stopping remaining runs because runtime exceeded 24 hours. elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
    total_wall_sec = time.time() - program_t0
    summary_lines = []

    def log(line=""):
        print(line)
        summary_lines.append(str(line))

    log("\n" + "=" * 80)
    log("Timing summary")
    log(f"torch_cpu_threads: {cpu_threads}")
    log(f"dataset: {args.dataset}")
    log(f"model: {args.model}")
    log(f"device: {device}")
    log(f"model_implementation: {best_config.get('model_implementation', 'unspecified')}")
    for key in (
        "model_training_protocol",
        "model_train_negative_protocol",
        "model_protocol_fidelity",
        "model_embedding_initialization",
        "model_gradient_clipping",
        "model_evaluation_transform",
        "model_probability_domain_loss",
        "model_rng_protocol",
        "model_optimizer_protocol",
        "model_gradient_clipping_protocol",
        "model_gat_heads",
        "model_dense_gae_loss",
        "model_dense_gae_evaluation",
        "model_effective_training_path",
        "config_source",
        "config_protocol",
        "reference_mf_source_epochs",
        "reference_mf_source_eval_steps",
        "reference_mf_source_patience",
        "reference_mf_effective_epochs",
        "reference_mf_effective_eval_steps",
        "reference_mf_effective_patience",
        "reference_mf_training_schedule_faithful",
        "reference_mf_source_optimizer_batch_size",
        "reference_mf_effective_optimizer_batch_size",
        "reference_mf_optimizer_batch_faithful",
        "reference_mf_optimizer_schedule_fidelity",
    ):
        if best_config.get(key) is not None:
            log(f"{key}: {best_config.get(key)}")
    log(f"train_samples_per_epoch: {args.train_samples_per_epoch}")
    log(f"runtime_limit_exceeded: {timed_out}")
    log("status: exceeded 24 hour runtime limit" if timed_out else "status: completed within 24 hour runtime limit")
    log(f"runtime_limit_sec: {RUNTIME_LIMIT_SEC:.2f}")
    if selector_training_mode:
        log("evaluation_mode: ranked-selector")
        log("evaluation_positive_scope: deterministic-neutral-validation-only")
        log(f"selection_metric_requested: {args.metric}")
        log(f"selection_metric_effective: {metric_key}")
        for key, value in sorted(selector_validation_metadata.items()):
            log(f"{key}: {value}")
    else:
        log_protocol_summary(log, args, mode, protocol_metadata, metric_key, mode_bundle, device)
    for key, value in sorted(heart_candidate_provenance.items()):
        if value is not None and key != "heart_source":
            log(f"{key}: {value}")
    for summary in ddi_dedup_summaries:
        prefix = str(summary["key"]).replace("neg_", "").replace("_edge", "")
        log(f"ddi_{prefix}_negative_edges_original: {summary['original_edges']}")
        log(f"ddi_{prefix}_negative_edges_unique: {summary['unique_edges']}")
        log(f"ddi_{prefix}_decode_fraction: {summary['decode_fraction']:.6f}")
        log(f"ddi_{prefix}_canonical_undirected: {str(summary['canonical_undirected']).lower()}")
    if ddi_dedup_summaries:
        log(f"ddi_eval_dedup_prepare_sec: {ddi_dedup_prepare_sec:.2f}")
    log(f"data_load_sec: {data_load_sec:.2f}")
    log(f"device_prepare_sec: {device_prepare_sec:.2f}")
    log(f"train_total_sec: {total_train_sec:.2f}")
    log(f"test_total_sec: {total_test_sec:.2f}")
    log(f"eval_total_sec: {total_eval_sec:.2f}")
    log(f"inference_total_sec: {total_inference_sec:.2f}")
    log(f"testing_total_sec: {total_testing_sec:.2f}")
    log(f"mrr_total_sec: {total_mrr_sec:.2f}")
    log(f"auc_total_sec: {total_auc_sec:.2f}")
    log(f"total_wall_sec: {total_wall_sec:.2f}")
    log(f"cpu_rss_mb_current: {current_cpu_rss_mb():.2f}")
    log(f"cpu_rss_mb_peak_process: {peak_cpu_rss_mb():.2f}")
    log_run_statistics(
        log,
        {
            "train": run_train_secs,
            "test": run_test_secs,
            "eval": run_eval_secs,
            "inference": run_inference_secs,
            "testing": run_testing_secs,
            "mrr": run_mrr_secs,
            "auc": run_auc_secs,
        },
        {
            "train_peak_cpu_rss_mb": run_train_peak_cpu_rss_mbs,
            "eval_peak_cpu_rss_mb": run_eval_peak_cpu_rss_mbs,
            "train_peak_cuda_allocated_mb": run_train_peak_cuda_allocated_mbs,
            "eval_peak_cuda_allocated_mb": run_eval_peak_cuda_allocated_mbs,
            "train_peak_cuda_reserved_mb": run_train_peak_cuda_reserved_mbs,
            "eval_peak_cuda_reserved_mb": run_eval_peak_cuda_reserved_mbs,
            "test_peak_cpu_rss_mb": run_test_peak_cpu_rss_mbs,
            "test_peak_cuda_allocated_mb": run_test_peak_cuda_allocated_mbs,
            "test_peak_cuda_reserved_mb": run_test_peak_cuda_reserved_mbs,
        },
        run_resource_records,
    )
    if not log_aggregate_results(
        log, args.dataset, args.seed, metric_key, test_selected_metrics, test_aucs, test_aps, test_mrrs, test_hits_any
    ):
        _write_summary(log_path, summary_lines)
        return
    _write_summary(log_path, summary_lines)


if __name__ == "__main__":
    main()
