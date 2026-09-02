"""Policy-neutral mechanics shared by ranked evaluator dispatch modules."""

from __future__ import annotations

import hashlib
import fcntl
import json
import math
import os
import re
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterator, Mapping, NamedTuple, Optional, Sequence

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OVERRIDE_FIELDS = ("depth", "hidden_channels", "dropout", "lr", "weight_decay")
_CHECKPOINT_NAME = re.compile(r"model_checkpoint(\d+)$")

RANKED_SELECTOR_TRAINING_PROTOCOL = "ranked-selector-strict-train-negative-bce-with-logits-v1"
RANKED_SELECTOR_NEGATIVE_PROTOCOL = "strict-nonself-non-train-positive"
RANKED_SELECTOR_LOSS_PROTOCOL = "binary-cross-entropy-with-logits"
NEUTRAL_SELECTOR_VALIDATION_PROTOCOL = "deterministic-affine-legal-validation-fixed-250-per-side-v1"
NEUTRAL_SELECTOR_VALIDATION_CONTRACT_FIELDS = (
    "selector_validation_protocol",
    "selector_validation_source",
    "selector_validation_split",
    "selector_validation_seed",
    "selector_validation_negatives_total",
    "selector_validation_negatives_per_side",
    "selector_validation_legality_policy",
    "selector_validation_filter_scope",
    "selector_validation_directionality",
    "selector_validation_query_sha256",
    "selector_validation_blocked_role_sha256",
    "selector_validation_candidate_nodes_sha256",
)
RANKED_SELECTOR_CHECKPOINT_FIELDS = {
    "selector_training_contract": RANKED_SELECTOR_TRAINING_PROTOCOL,
    "selector_training_loss": RANKED_SELECTOR_LOSS_PROTOCOL,
    "selector_train_negative_protocol": RANKED_SELECTOR_NEGATIVE_PROTOCOL,
    "selector_strict_train_negatives": True,
    "selector_rejects_self_loops": True,
    "selector_rejects_train_positives": True,
    "selector_filters_validation_positives": False,
    "selector_filters_test_positives": False,
    "selector_reference_optimizer": False,
    "selector_reference_probability_loss": False,
    "selector_reference_random_endpoint_negatives": False,
}
IP_TRAINING_CONTRACT = {
    "ip_training_contract": "shared-ip-strict-negative-bce-with-logits-v1",
    "ip_training_loss": "binary-cross-entropy-with-logits",
    "ip_train_negative_protocol": "strict-unobserved-nonself-edge",
    "ip_strict_train_negatives": True,
    "ip_reference_probability_loss": False,
    "ip_reference_random_endpoint_negatives": False,
    "ip_reference_optimizer": False,
}
IP_MODEL_CONTRACT = {
    "pred_layers": 0,
    "predictor_depth": 0,
    "decoder_type": "inner-product",
    "decoder_output": "raw-inner-product-logit",
    "ranking_score": "raw-inner-product-logit",
    "probability_transform": "sigmoid",
}


def configure_ranked_selector_training(model, dataset: str) -> dict[str, Any]:
    """Apply the shared, leak-free training contract for ranked selectors."""
    directed = str(dataset).strip().lower() == "ogbl-citation2"
    model.ranked_selector_training_contract = True
    model.reference_optimizer = False
    model.reference_probability_loss = False
    model.reference_random_endpoint_negatives = False
    model.strict_train_negatives = True
    model.directed = directed
    model.training_protocol = RANKED_SELECTOR_TRAINING_PROTOCOL
    model.training_loss_protocol = RANKED_SELECTOR_LOSS_PROTOCOL
    model.train_negative_protocol = RANKED_SELECTOR_NEGATIVE_PROTOCOL
    metadata = {
        "selector_training_contract": RANKED_SELECTOR_TRAINING_PROTOCOL,
        "selector_training_loss": RANKED_SELECTOR_LOSS_PROTOCOL,
        "selector_train_negative_protocol": RANKED_SELECTOR_NEGATIVE_PROTOCOL,
        "selector_strict_train_negatives": True,
        "selector_rejects_self_loops": True,
        "selector_rejects_train_positives": True,
        "selector_filters_validation_positives": False,
        "selector_filters_test_positives": False,
        "selector_training_directionality": "directed" if directed else "canonical-undirected",
        "selector_reference_optimizer": False,
        "selector_reference_probability_loss": False,
        "selector_reference_random_endpoint_negatives": False,
    }
    model.ranked_selector_training_metadata = dict(metadata)
    return metadata


def validate_ranked_selector_training_config(config: Mapping[str, Any], label: str) -> None:
    """Reject checkpoints trained under a different selector-negative/loss contract."""
    for key, expected in RANKED_SELECTOR_CHECKPOINT_FIELDS.items():
        if config.get(key) != expected:
            raise ValueError(f"{label} has {key}={config.get(key)!r}, expected {expected!r}.")


def project_path(value: str | os.PathLike[str], project_root: str | os.PathLike[str] | None = None) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else Path(project_root or PROJECT_ROOT) / path


def selector_overrides(args, policy: str) -> tuple[Any, ...]:
    return tuple(getattr(args, f"{policy}_selector_{field}", None) for field in OVERRIDE_FIELDS)


def checkpoint_run(path: str | os.PathLike[str]) -> int:
    match = _CHECKPOINT_NAME.fullmatch(Path(path).name)
    return int(match.group(1)) if match else 10**9


def load_checkpoint_with_identity(path: str | os.PathLike[str]) -> tuple[dict[str, Any], str, float]:
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Selector checkpoint not found: {checkpoint_path}")
    started = time.perf_counter()
    identity = _sha256_file(checkpoint_path)
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    seconds = round(time.perf_counter() - started, 3)
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Selector checkpoint is not a mapping: {checkpoint_path}")
    return checkpoint, identity, seconds


def validate_common_cli(args, policy: str) -> int:
    if str(getattr(args, "split", "test")).strip().lower() != "test":
        raise ValueError(f"--candidate-policy {policy} is defined for test positives only; use --split test.")
    cap = str(getattr(args, "eval_cap", "checkpoint")).strip().lower().replace("_", "-")
    if cap not in {"checkpoint", "all", "full", "entire", "0"}:
        try:
            numeric_cap = int(cap)
        except ValueError as exc:
            raise ValueError(f"{policy.upper()} evaluation does not support auxiliary --eval-cap {cap!r}.") from exc
        if numeric_cap != 0:
            raise ValueError(
                f"{policy.upper()} evaluation uses --test-positive-cap; auxiliary --eval-cap must be checkpoint/all/0."
            )
    return 0


def validate_candidate_count(value, option: str, *, default: int = 500, even: bool = False, exact: int | None = None) -> int:
    count = int(default if value is None else value)
    if exact is not None and count != int(exact):
        raise ValueError(f"--{option} must be exactly {int(exact)} (250 per endpoint side).")
    if count <= 0 or (even and count % 2):
        requirement = "a positive even number" if even else "positive"
        raise ValueError(f"--{option} must be {requirement}.")
    return count


def validate_score_batch_size(value, option: str) -> int:
    size = int(value)
    if size <= 0:
        raise ValueError(f"--{option} must be positive.")
    return size


def validate_selector_overrides(args, policy: str) -> tuple[Any, ...]:
    depth, hidden, dropout, lr, weight_decay = selector_overrides(args, policy)
    if depth is not None and int(depth) <= 0:
        raise ValueError(f"--{policy}-selector-depth must be positive.")
    if hidden is not None and int(hidden) <= 0:
        raise ValueError(f"--{policy}-selector-hidden-channels must be positive.")
    if dropout is not None and (not math.isfinite(float(dropout)) or not 0.0 <= float(dropout) < 1.0):
        raise ValueError(f"--{policy}-selector-dropout must be finite and in [0, 1).")
    if lr is not None and (not math.isfinite(float(lr)) or float(lr) <= 0.0):
        raise ValueError(f"--{policy}-selector-lr must be finite and positive.")
    if weight_decay is not None and (not math.isfinite(float(weight_decay)) or float(weight_decay) < 0.0):
        raise ValueError(f"--{policy}-selector-weight-decay must be finite and non-negative.")
    return depth, hidden, dropout, lr, weight_decay


def validate_checkpoint_identity(
    checkpoint, path, framework, dataset, policy: str, expected_run, expected_mode: str = "heart"
) -> int:
    if not isinstance(checkpoint, Mapping):
        raise ValueError(f"{policy.upper()} selector {path} is not a checkpoint mapping.")
    checks = (
        ("framework", _normal_framework(checkpoint.get("framework", "")), _normal_framework(framework)),
        ("dataset", str(checkpoint.get("dataset", "")).strip().lower(), str(dataset).strip().lower()),
        ("model", _normal_name(checkpoint.get("model", "")), _normal_name(policy)),
        ("mode", str(checkpoint.get("mode", "")).strip().lower(), str(expected_mode).strip().lower()),
    )
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(f"{policy.upper()} selector {path} has {name}={actual!r}, expected {expected!r}.")
    actual_run = int(checkpoint.get("run", checkpoint_run(path)))
    if expected_run is not None and actual_run != int(expected_run):
        raise ValueError(
            f"{policy.upper()} selector {path} declares run={actual_run}, expected fixed run={int(expected_run)}."
        )
    return actual_run


def validate_frozen_checkpoint_contract(
    checkpoint,
    path,
    framework,
    dataset,
    policy,
    expected_run,
    *,
    mode="heart",
    schedule: Mapping[str, Any] | None = None,
    config_contract: Mapping[str, Any] | None = None,
    predictor_depth: int | None = None,
    overrides: Sequence[Any] = (),
    require_fast_ogbl: bool = True,
) -> tuple[Mapping[str, Any], Mapping[str, torch.Tensor]]:
    """Validate the shared identity, schedule, and architecture of a frozen selector."""
    schedule = dict(schedule or {})
    declared_run = validate_checkpoint_identity(
        checkpoint, path, framework, dataset, policy, expected_run, expected_mode=mode
    )
    if checkpoint.get("checkpoint_type") != "best_validation_model_state" or checkpoint.get("timed_out", False):
        raise ValueError(f"{policy} selector {path} is not a completed best-validation checkpoint.")
    if "run" in schedule and declared_run != int(schedule["run"]):
        raise ValueError(f"{policy} selector {path} has run={declared_run}, expected {int(schedule['run'])}.")
    arguments = checkpoint.get("arguments")
    if not isinstance(arguments, Mapping):
        raise ValueError(f"{policy} selector {path} has no arguments provenance.")
    for key, expected in schedule.items():
        if key == "run":
            continue
        try:
            values = (int(checkpoint[key]), int(arguments[key])) if key == "seed" else (int(arguments[key]),)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{policy} selector {path} has invalid {key}.") from exc
        if any(actual != int(expected) for actual in values):
            raise ValueError(f"{policy} selector {path} has {key}={values!r}, expected {int(expected)}.")
    if require_fast_ogbl and _normal_framework(framework) == "ogbl" and str(
        arguments.get("train_negative_sampler", "")
    ).strip().lower() != "fast":
        raise ValueError(f"{policy} OGB selector must use the fast train-negative sampler.")
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping) or not config:
        raise ValueError(f"{policy} selector {path} has no model_config.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError(f"{policy} selector {path} has no model state.")
    for key, expected in dict(config_contract or {}).items():
        if config.get(key) != expected:
            raise ValueError(f"{policy} selector {path} has {key}={config.get(key)!r}, expected {expected!r}.")
    depth, hidden, dropout, lr, weight_decay = (*overrides, None, None, None, None, None)[:5]
    if depth is not None:
        expected_layers = (int(depth), int(depth) if predictor_depth is None else int(predictor_depth))
        if (config.get("layers"), config.get("pred_layers")) != expected_layers:
            raise ValueError(f"{policy} selector {path} has the wrong encoder/predictor depth.")
    for key, expected in (("emb_size", hidden), ("dropout", dropout), ("lr", lr), ("weight_decay", weight_decay)):
        if expected is not None and float(config.get(key)) != float(expected):
            raise ValueError(f"{policy} selector {path} has {key}={config.get(key)!r}, expected {expected!r}.")
    return config, state


def resolved_cache_dir(args, policy: str) -> tuple[str | None, str]:
    configured = getattr(args, f"{policy}_cache_dir", None)
    return (str(project_path(configured).resolve()) if configured else None, str(configured))


def known_positive_filter(information, policy: str, label: str | None = None) -> dict[str, Any]:
    metadata = information.get("metadata")
    display = label or policy.upper()
    if not isinstance(metadata, Mapping):
        raise ValueError(f"{display} candidate metadata is missing.")
    keys = {
        "policy": "eligibility_policy",
        "positive_scope": "eligibility_scope",
        "directionality": "eligibility_directionality",
    }
    missing = [source for source in keys.values() if not str(metadata.get(source, "")).strip()]
    if missing:
        raise ValueError(f"{display} candidate metadata is missing its legal-edge contract: " + ", ".join(missing))
    output = {target: metadata[source] for target, source in keys.items()}
    output["selector_model"] = str(metadata.get("selector_model", policy))
    return output


def result_fields(bundle, policy: str) -> dict[str, Any]:
    return {f"{policy}_candidates": dict(bundle[f"{policy}_candidates"])}


def output_root(args, project_root, policy: str, artifact_tag: str | None = None) -> Path:
    root = Path(project_root) / "results" / policy
    if artifact_tag:
        root /= artifact_tag
    design = []
    for field, label in (
        ("depth", "depth"),
        ("hidden_channels", "hidden"),
        ("dropout", "dropout"),
        ("lr", "lr"),
        ("weight_decay", "weight_decay"),
    ):
        value = getattr(args, f"{policy}_selector_{field}", None)
        if value is not None:
            rendered = str(int(value)) if field in {"depth", "hidden_channels"} else format(float(value), ".12g")
            design.append(f"{label}_{rendered}")
    return root / f"selector_{'_'.join(design)}" if design else root


_TRAINING_DEFAULTS = {"epochs": 300, "eval_steps": 5, "patience": 10, "batch_size": 1024}


def _ranked_extra(spec, key, *args):
    value = spec.get(key)
    return dict((value(*args) if callable(value) else value) or {})


def _training_defaults(spec):
    return {key: int(spec.get(f"default_{key}", value)) for key, value in _TRAINING_DEFAULTS.items()}


def add_ranked_arguments(parser, *, spec, fresh=False):
    policy, add = str(spec["policy"]), parser.add_argument
    if not fresh:
        add(f"--{policy}-checkpoint", f"--{policy}-selector-checkpoint", dest=f"{policy}_checkpoint")
        add(f"--{policy}-checkpoint-root")
        add(f"--{policy}-checkpoint-run", f"--{policy}-selector-run", dest=f"{policy}_checkpoint_run",
            type=int, default=int(spec.get("default_run", 1)))
    add(f"--{policy}-negatives", f"--{policy}-k", dest=f"{policy}_negatives", type=int)
    add(f"--{policy}-score-batch-size", type=int, default=int(spec.get("default_score_batch_size", 65536)))
    add(f"--{policy}-cache-dir", default=str(spec.get("default_cache_dir", f"{policy}_sets")))
    if fresh:
        add(f"--{policy}-selector-seed", type=int, default=int(spec.get("default_seed", 42)))
        for field, default in _training_defaults(spec).items():
            add(f"--{policy}-selector-{field.replace('_', '-')}", type=int, default=default)
        add(f"--{policy}-selector-metric")
    for field, kind in (("depth", int), ("hidden-channels", int), ("dropout", float), ("lr", float),
                        ("weight-decay", float)):
        add(f"--{policy}-selector-{field}", type=kind)
    return parser


def validate_ranked_cli(args, dataset, *, spec, fresh=False):
    del dataset
    policy = str(spec["policy"])
    validate_common_cli(args, policy)
    default_k = int(spec.get("default_k", 500))
    validate_candidate_count(getattr(args, f"{policy}_negatives", None), f"{policy}-negatives",
                             default=default_k, even=bool(spec.get("require_even_k", False)),
                             exact=spec.get("exact_k"))
    validate_score_batch_size(getattr(args, f"{policy}_score_batch_size", 65536), f"{policy}-score-batch-size")
    depth, *_ = validate_selector_overrides(args, policy)
    if fresh:
        for field, default in _training_defaults(spec).items():
            if int(getattr(args, f"{policy}_selector_{field}", default)) <= 0:
                raise ValueError(f"--{policy}-selector-{field.replace('_', '-')} must be positive.")
    else:
        if int(getattr(args, f"{policy}_checkpoint_run", spec.get("default_run", 1))) <= 0:
            raise ValueError(f"--{policy}-checkpoint-run must be positive.")
        fixed_depth = spec.get("fixed_depth")
        if depth is not None and fixed_depth is not None and int(depth) != int(fixed_depth):
            raise ValueError(f"--{policy}-selector-depth must be {int(fixed_depth)}.")
        if callable(spec.get("cli_extra")):
            spec["cli_extra"](args)
    return 0


def ranked_evaluator_context(args, dataset, framework, device, *, spec, fresh=False):
    del framework, device
    validate_ranked_cli(args, dataset, spec=spec, fresh=fresh)
    policy = str(spec["policy"])
    cache_dir, cache_key = resolved_cache_dir(args, policy)
    default_k = int(spec.get("default_k", 500))
    context = {
        "policy": policy, "path": None, "run": None,
        "requested_k": int(getattr(args, f"{policy}_negatives", None) or default_k),
        "score_batch_size": int(getattr(args, f"{policy}_score_batch_size", 65536)),
        "load_seconds": 0.0, "cache_dir": cache_dir, "_cache_dir_key": cache_key,
        "_overrides": selector_overrides(args, policy),
    }
    if fresh:
        context.update(seed=int(getattr(args, f"{policy}_selector_seed", spec.get("default_seed", 42))),
                       _metric=getattr(args, f"{policy}_selector_metric", None))
        context.update({f"_{field}": int(getattr(args, f"{policy}_selector_{field}", default))
                        for field, default in _training_defaults(spec).items()})
        return context
    run = int(getattr(args, f"{policy}_checkpoint_run", spec.get("default_run", 1)))
    explicit = getattr(args, f"{policy}_checkpoint", None)
    root = getattr(args, f"{policy}_checkpoint_root", None) or getattr(args, "checkpoint_root", "checkpoints")
    path = (project_path(explicit) if explicit else project_path(root) / str(spec.get("checkpoint_mode", "heart"))
            / str(dataset) / policy / f"model_checkpoint{run}").resolve()
    checkpoint, digest, seconds = load_checkpoint_with_identity(path)
    arguments = checkpoint.get("arguments") or {}
    context.update(path=path, run=int(checkpoint.get("run", checkpoint_run(path))),
                   seed=int(arguments.get("seed", checkpoint.get("seed", 0))), load_seconds=seconds,
                   _checkpoint=checkpoint, _selector_sha256=digest)
    return context


def ranked_data_seed(context, target_seed, *, fresh=False):
    if fresh:
        return int(target_seed)
    if int(context["seed"]) != int(target_seed):
        raise ValueError("Target and frozen selector data seeds differ.")
    return int(target_seed)


def ranked_resolve_cap(args, checkpoint, framework, dataset, *, spec, fresh=False):
    del checkpoint, framework
    if fresh:
        validate_ranked_cli(args, dataset, spec=spec, fresh=True)
        return 0
    from eval_modes.evaluator_helpers import resolve_ranked_selector_cap
    return resolve_ranked_selector_cap(args.eval_cap, dataset)


def load_ranked_bundle(framework, dataset, root, seed, cap=None, device=None, checkpoint=None, args=None, context=None):
    del cap, device, checkpoint, args, context
    from eval_modes.evaluator_helpers import load_ranked_selector_bundle
    return load_ranked_selector_bundle(framework, dataset, root, seed)


def ranked_cache_key(args, context, data_seed, cap, checkpoint=None, *, spec, fresh=False):
    del checkpoint
    policy = str(spec["policy"])
    if fresh:
        training = tuple(context[f"_{field}"] for field in _training_defaults(spec))
        return (policy, str(spec["selector_identity"]), int(context["seed"]), *training, context["_metric"],
                *context["_overrides"], int(data_seed), cap, int(context["requested_k"]),
                int(context["score_batch_size"]), context["_cache_dir_key"])
    return (policy, str(Path(context["path"]).resolve()), int(context["seed"]), *context["_overrides"], cap,
            int(context["requested_k"]), int(context["score_batch_size"]), context["_cache_dir_key"],
            *_ranked_extra(spec, "cache_key_extra", args, context).items())


def prepare_ranked_bundle(bundle, *, fresh=False):
    if not fresh:
        bundle.pop("valid_neg", None)
    bundle.pop("test_neg", None)
    return bundle


def install_ranked_candidates(bundle, framework, dataset, device, args, context, data_seed, root,
                              *, spec, loader, fresh=False):
    from pyg.main import _configure_cuda_matmul_precision
    policy = str(spec["policy"])
    precision = _configure_cuda_matmul_precision(device, dataset, policy)
    training = tuple(context[f"_{field}"] for field in _training_defaults(spec)) if fresh else ()
    request = ((str(spec["selector_identity"]), int(context["seed"]), *training, context["_metric"])
               if fresh else (str(Path(context["path"]).resolve()), int(context["run"])))
    request += (*context["_overrides"], int(context["requested_k"]), int(context["score_batch_size"]),
                context["cache_dir"], precision, *_ranked_extra(spec, "cache_key_extra", args, context).items())
    request_key, information_key = f"_{policy}_candidate_request", f"{policy}_candidates"
    if bundle.get(request_key) == request and information_key in bundle:
        information = dict(bundle[information_key]); information["memory_reused"] = True
        return information
    depth, hidden, dropout, lr, weight_decay = context["_overrides"]
    common = dict(k=context["requested_k"], score_batch_size=context["score_batch_size"],
                  cache_dir=context["cache_dir"], precision_contract=precision, selector_depth=depth,
                  selector_hidden_channels=hidden, selector_dropout=dropout, selector_lr=lr,
                  selector_weight_decay=weight_decay)
    if fresh:
        result = loader(framework, dataset, bundle, device, **common, seed=context["seed"],
                        epochs=context["_epochs"], eval_steps=context["_eval_steps"], patience=context["_patience"],
                        batch_size=context["_batch_size"], metric=context["_metric"], data_root=root,
                        data_seed=int(data_seed))
    else:
        result = loader(framework, dataset, bundle, context["path"], device, **common,
                        _checkpoint_payload=context.pop("_checkpoint", None),
                        _selector_sha256=context.get("_selector_sha256"),
                        **_ranked_extra(spec, "loader_kwargs", args, context))
    effective_k = int(result.negatives.size(1))
    if spec.get("require_output_width") and effective_k != int(context["requested_k"]):
        raise ValueError(f"{spec['display_name']} requires exactly {int(context['requested_k'])} negatives.")
    bundle["test_neg"] = result.negatives.contiguous()
    information = {
        "protocol": str(spec["candidate_protocol"]), "effective_k": effective_k,
        "cache_path": result.cache_path, "cache_hit": bool(result.cache_hit), "memory_reused": False,
        "metadata": dict(result.metadata),
    }
    bundle[information_key], bundle[request_key] = information, request
    return dict(information)


def ranked_selector_metadata(args, context, data_seed, *, spec, fresh=False):
    policy = str(spec["policy"])
    depth, hidden, dropout, lr, weight_decay = context["_overrides"]
    metadata = {
        "checkpoint": None if fresh else str(context["path"]), "run": None if fresh else int(context["run"]),
        "selector_depth": depth, "selector_hidden_channels": hidden, "selector_dropout": dropout,
        "selector_lr": lr, "selector_weight_decay": weight_decay, "data_seed": int(data_seed),
        "requested_k": int(context["requested_k"]),
        "per_side_k": None if spec.get("global_top_k") else int(context["requested_k"]) // 2,
        "score_batch_size": int(context["score_batch_size"]), "cache_dir": context["cache_dir"],
    }
    if fresh:
        metadata.update(training_protocol=str(spec["training_protocol"]), selector_seed=int(context["seed"]))
    else:
        metadata.update(_ranked_extra(spec, "selector_metadata_extra", args, context))
    return metadata


def ranked_candidate_pool(bundle, context, both_sides, *, spec):
    policy = str(spec["policy"])
    information = bundle[f"{policy}_candidates"]
    effective_k = int(information["effective_k"])
    return {
        "setting": policy,
        "candidate_universe": str(spec.get("candidate_universe", "all-legal-endpoint-corruptions")),
        "both_corruption_sides_combined": bool(both_sides), "selection": str(spec["candidate_selection"]),
        "per_side_quota": None if spec.get("global_top_k") else effective_k // 2,
        "effective_total": effective_k,
        "known_positive_filter": known_positive_filter(information, policy, str(spec["display_name"])),
        "positive_query_scope": bundle["test_positive_scope"]["policy"],
    }


CONCAT_SHARD_FORMAT = "endpoint-side-node-int32-shards-v1"
MAX_UNCACHED_MATERIALIZED_BYTES = 2 * 1024**3
_SHARED_MAX_CACHE_K = {"learnedfeat": 2000}


class RankedNegativeResult(NamedTuple):
    negatives: Any
    metadata: dict[str, Any]
    cache_path: Optional[str]
    cache_hit: bool

class RankedTestFilterLayout(NamedTuple):
    ranking_keys: torch.Tensor
    rowptr: torch.Tensor
    col: torch.Tensor
    left_rows: torch.Tensor
    right_rows: torch.Tensor
    capacities: torch.Tensor
    blocked_ids: torch.Tensor
    right_role_offset: int
    directionality: str
    filter_scope: str


class NeutralSelectorValidationNegatives:
    """Compact, streamed grouped negatives for selector validation."""

    is_streaming_negative = True
    is_streamed_grouped_negative = True
    is_streaming_grouped_negative = True

    def __init__(self, positives: torch.Tensor, candidate_nodes: torch.Tensor, num_nodes: int, pos_chunk_size: int = 256):
        self.positives = _edge_rows(positives, "neutral selector validation positives").cpu()
        self.candidate_nodes = candidate_nodes.detach().to(device="cpu", dtype=torch.int32).contiguous()
        if self.candidate_nodes.dim() != 2:
            raise ValueError("Neutral selector validation candidates must be a rank-2 node tensor.")
        self.num_nodes = int(num_nodes)
        self.pos_chunk_size = max(1, int(pos_chunk_size))
        self.num_rows = int(self.positives.size(0))
        self.negatives_per_row = int(self.candidate_nodes.size(1))
        if tuple(self.candidate_nodes.shape[:1]) != (self.num_rows,) or self.negatives_per_row <= 0:
            raise ValueError("Neutral selector validation candidates do not align with their positives.")
        if self.negatives_per_row % 2:
            raise ValueError("Neutral selector validation candidates must split evenly across endpoint sides.")
        if self.candidate_nodes.numel() and (
            int(self.candidate_nodes.min().item()) < 0 or int(self.candidate_nodes.max().item()) >= self.num_nodes
        ):
            raise ValueError("Neutral selector validation candidates contain an out-of-range node.")
        self.shape = (self.num_rows, self.negatives_per_row, 2)

    @property
    def negatives_per_side(self) -> int:
        return self.negatives_per_row // 2

    @property
    def storage_nbytes(self) -> int:
        return int(self.positives.numel() * self.positives.element_size()) + int(
            self.candidate_nodes.numel() * self.candidate_nodes.element_size()
        )

    def dim(self) -> int:
        return 3

    @property
    def ndim(self) -> int:
        return 3

    def size(self, dim: Optional[int] = None):
        return torch.Size(self.shape) if dim is None else self.shape[int(dim)]

    def numel(self) -> int:
        return self.num_rows * self.negatives_per_row * 2

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def contiguous(self):
        return self

    def iter_chunks(self, pos_chunk_size: Optional[int] = None) -> Iterator[tuple[int, int, torch.Tensor]]:
        chunk_size = self.pos_chunk_size if pos_chunk_size is None else max(1, int(pos_chunk_size))
        side_k = self.negatives_per_side
        for start in range(0, self.num_rows, chunk_size):
            end = min(start + chunk_size, self.num_rows)
            positives = self.positives[start:end]
            nodes = self.candidate_nodes[start:end].to(torch.long)
            edges = torch.empty((end - start, self.negatives_per_row, 2), dtype=torch.long)
            edges[:, :side_k, 0] = positives[:, 0:1]
            edges[:, :side_k, 1] = nodes[:, :side_k]
            edges[:, side_k:, 0] = nodes[:, side_k:]
            edges[:, side_k:, 1] = positives[:, 1:2]
            yield (start, end, edges)

    def iter_grouped_chunks(self) -> Iterator[tuple[int, int, torch.Tensor]]:
        yield from self.iter_chunks()

    def materialize(self) -> torch.Tensor:
        output = torch.empty(self.shape, dtype=torch.long)
        for start, end, edges in self.iter_chunks():
            output[start:end] = edges
        return output.contiguous()

    def __repr__(self) -> str:
        return (
            f"NeutralSelectorValidationNegatives(shape={self.shape}, num_nodes={self.num_nodes}, "
            f"candidate_dtype={self.candidate_nodes.dtype})"
        )


class NeutralSelectorValidationResult(NamedTuple):
    negatives: NeutralSelectorValidationNegatives
    metadata: dict[str, Any]


@dataclass(frozen=True)
class _NegativeShard:
    start: int
    end: int
    filename: str


class RankedNegativeShards:
    is_streaming_negative = True
    candidate_summary_prevalidated = True

    def __init__(
        self,
        *,
        manifest_path: str,
        queries: torch.Tensor,
        num_nodes: int,
        effective_k: int,
        shards: Sequence[_NegativeShard],
        default_chunk_size: int = 2048,
        candidate_summary: Optional[Mapping[str, Any]] = None,
        storage_k: Optional[int] = None,
    ) -> None:
        self.manifest_path = str(Path(manifest_path).resolve())
        self.queries = _edge_rows(queries, "ranked-negative queries")
        self.num_nodes = int(num_nodes)
        self.effective_k = int(effective_k)
        self.storage_k = self.effective_k if storage_k is None else int(storage_k)
        self.shards = tuple(shards)
        self.default_chunk_size = max(1, int(default_chunk_size))
        self.candidate_summary = dict(candidate_summary) if candidate_summary is not None else None
        self.shape = (int(self.queries.size(0)), int(self.effective_k), 2)
        if self.num_nodes <= 1 or self.effective_k <= 0 or self.effective_k > self.storage_k:
            raise ValueError("Invalid ranked-negative shard dimensions.")
        expected = 0
        for shard in self.shards:
            if shard.start != expected or shard.end <= shard.start:
                raise ValueError("Ranked-negative shards must cover rows contiguously.")
            expected = shard.end
        if expected != self.shape[0]:
            raise ValueError("Ranked-negative shards do not cover every test query.")

    def dim(self) -> int:
        return 3

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return torch.Size(self.shape)
        return self.shape[int(dim)]

    def numel(self) -> int:
        return int(self.shape[0]) * int(self.shape[1]) * 2

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def contiguous(self):
        return self

    def _shard_path(self, shard: _NegativeShard) -> str:
        return str(Path(self.manifest_path).parent / shard.filename)

    def _load_encoded(self, shard: _NegativeShard) -> torch.Tensor:
        path = self._shard_path(shard)
        payload = _load_torch_payload_mmap(path)
        if not isinstance(payload, Mapping):
            raise ValueError("Ranked-negative shard payload is not a mapping.")
        encoded = payload.get("encoded")
        expected_shape = (shard.end - shard.start, self.storage_k)
        if not torch.is_tensor(encoded) or encoded.dtype != torch.int32 or tuple(encoded.shape) != expected_shape:
            raise ValueError("Ranked-negative shard has the wrong compact tensor layout.")
        return encoded

    def iter_chunks(self, pos_chunk_size: Optional[int] = None) -> Iterator[tuple[int, int, torch.Tensor]]:
        chunk_size = self.default_chunk_size if pos_chunk_size is None else max(1, int(pos_chunk_size))
        for shard in self.shards:
            encoded = self._load_encoded(shard)
            for local_start in range(0, int(encoded.size(0)), chunk_size):
                local_end = min(local_start + chunk_size, int(encoded.size(0)))
                start = shard.start + local_start
                end = shard.start + local_end
                compact = encoded[local_start:local_end, : self.effective_k].to(torch.long)
                sides = torch.div(compact, self.num_nodes, rounding_mode="floor")
                nodes = compact.remainder(self.num_nodes)
                if compact.numel() and (int(sides.min().item()) < 0 or int(sides.max().item()) > 1):
                    raise ValueError("Ranked-negative shard contains an invalid side.")
                query = self.queries[start:end]
                left = sides == 0
                edges = torch.empty((end - start, self.effective_k, 2), dtype=torch.long)
                edges[:, :, 0] = torch.where(left, query[:, 0:1], nodes)
                edges[:, :, 1] = torch.where(left, nodes, query[:, 1:2])
                yield (start, end, edges)

    def prefix(self, effective_k: int):
        width = int(effective_k)
        if width == self.effective_k:
            return self
        if width <= 0 or width > self.effective_k:
            raise ValueError("Ranked-negative prefix width is out of range.")
        view = RankedNegativeShards(
            manifest_path=self.manifest_path,
            queries=self.queries,
            num_nodes=self.num_nodes,
            effective_k=width,
            storage_k=self.storage_k,
            shards=self.shards,
            default_chunk_size=self.default_chunk_size,
        )
        left_min, left_max, left_sum = width, 0, 0
        for shard in self.shards:
            encoded = view._load_encoded(shard)[:, :width]
            if encoded.size(0):
                counts = (encoded < self.num_nodes).sum(dim=1)
                left_min = min(left_min, int(counts.min().item()))
                left_max = max(left_max, int(counts.max().item()))
                left_sum += int(counts.sum().item())
        view.candidate_summary = _candidate_summary(
            int(self.queries.size(0)), width, left_min, left_max, left_sum
        )
        return view

    def materialize(self) -> torch.Tensor:
        output = torch.empty(self.shape, dtype=torch.long)
        for start, end, block in self.iter_chunks():
            output[start:end] = block
        return output.contiguous()

    def __repr__(self) -> str:
        return f"RankedNegativeShards(shape={self.shape}, num_nodes={self.num_nodes}, shards={len(self.shards)})"


def _shared_cache_prefix(
    result: RankedNegativeResult, *, requested_k: int, effective_k: int, cache_k: int
) -> RankedNegativeResult:
    stored_k = int(result.negatives.size(1))
    if int(effective_k) > stored_k:
        raise ValueError("Shared ranked-negative cache is narrower than the requested prefix.")
    if isinstance(result.negatives, RankedNegativeShards):
        negatives = result.negatives.prefix(effective_k)
    elif torch.is_tensor(result.negatives):
        negatives = result.negatives[:, :effective_k].contiguous()
    else:
        raise TypeError("Shared ranked-negative cache does not support prefix selection.")
    metadata = dict(result.metadata)
    metadata.update(
        requested_k=int(requested_k),
        effective_total_across_both_endpoint_sides=int(effective_k),
        backing_cache_requested_k=int(cache_k),
        backing_cache_effective_k=stored_k,
    )
    return RankedNegativeResult(negatives, metadata, result.cache_path, result.cache_hit)


def _normal_framework(value: str) -> str:
    key = str(value).strip().lower().replace("-", "").replace("_", "")
    if key == "pyg":
        return "pyg"
    if key in {"ogb", "ogbl"}:
        return "ogbl"
    raise ValueError("framework must be 'pyg', 'ogb', or 'ogbl'.")


def _normal_name(value: Any) -> str:
    return str(value).strip().lower().replace("-", "").replace("_", "")


def _normal_selector_model(value: Any) -> str:
    model = _normal_name(value)
    if not model:
        raise ValueError("selector_model must not be empty.")
    return model


def _selector_label(selector_model: str) -> str:
    return selector_model if selector_model in {"concat", "concatip"} else selector_model.upper()

def _load_torch_payload(path: str | os.PathLike[str]):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_torch_payload_mmap(path: str | os.PathLike[str]):
    try:
        return torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        return _load_torch_payload(path)


def _sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def selector_state_sha256(state: Mapping[str, torch.Tensor], label: str = "selector-state") -> str:
    digest = hashlib.sha256()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            raise TypeError(f"Selector state value {key!r} is not a tensor.")
        digest.update(str(key).encode("utf8"))
        digest.update(_sha256_tensor(value, f"{label}:{key}").encode("ascii"))
    return digest.hexdigest()


def selector_runtime_sha256(framework: str, *owner_files: str) -> str:
    framework_key = _normal_framework(framework)
    common = (
        *owner_files,
        "model/models.py",
        "model/pairwise_models.py",
        "model/decoder_training.py",
    )
    framework_files = (
        ("pyg/prepare_data.py", "pyg/heart_generation.py", "pyg/grouped_negatives.py", "pyg/training.py", "pyg/train_eval.py")
        if framework_key == "pyg"
        else ("ogbl/prepare_data.py", "ogbl/data_core.py", "ogbl/training.py", "ogbl/train_eval.py")
    )
    digest = hashlib.sha256()
    digest.update(b"ranked-selector-runtime-v1")
    digest.update(RANKED_SELECTOR_TRAINING_PROTOCOL.encode("utf8"))
    digest.update(NEUTRAL_SELECTOR_VALIDATION_PROTOCOL.encode("utf8"))
    for relative in (*common, *framework_files):
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(f"Selector runtime dependency does not exist: {path}")
        digest.update(relative.encode("utf8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()


def default_selector_metric(framework: str, dataset: str) -> str:
    if _normal_framework(framework) == "pyg":
        return "MRR"
    return {
        "ogbl-collab": "Hits@50",
        "ogbl-ddi": "Hits@20",
        "ogbl-ppa": "Hits@100",
        "ogbl-citation2": "MRR",
    }.get(str(dataset).strip().lower(), "Hits@50")


def _ranked_selection_scope(bundle, framework, dataset, selector_model, *, force_node_embeddings=False):
    raw_x = _raw_features(
        bundle,
        framework,
        selector_model=selector_model,
        dataset=dataset,
        force_node_embeddings=force_node_embeddings,
    )
    num_nodes = int(raw_x.size(0))
    queries, query_source = _test_queries(bundle)
    valid_pos, valid_source = _full_positive_split(bundle, "valid")
    test_pos, test_source = _full_positive_split(bundle, "test")
    for edges, source in ((queries, query_source), (valid_pos, valid_source), (test_pos, test_source)):
        _validate_edge_range(edges, num_nodes, source)
    training_graph, graph_source = _resolve_training_graph(bundle, framework)
    train_pos, train_source = _training_positive_edges(bundle, num_nodes, training_graph, graph_source)
    directed = dataset == "ogbl-citation2"
    layout = build_ranked_test_filter_layout(bundle, framework, dataset, queries, num_nodes)
    eligibility = bundle.get("full_graph_known_positive_filter")
    if not isinstance(eligibility, Mapping) or not str(eligibility.get("policy", "")).strip():
        raise ValueError("Authoritative legal-edge metadata has no policy.")
    return (
        raw_x, num_nodes, queries, query_source, valid_pos, valid_source, test_pos, test_source,
        training_graph, train_pos, train_source, directed, layout, str(eligibility["policy"]).strip(),
    )


def _sha256_tensor(tensor: torch.Tensor, label: str) -> str:
    value = tensor.detach().to(device="cpu").contiguous()
    digest = hashlib.sha256()
    digest.update(str(label).encode("utf8"))
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
    if value.numel():
        byte_view = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
        block_size = 64 * 1024 * 1024
        for start in range(0, len(byte_view), block_size):
            digest.update(byte_view[start : start + block_size])
    return digest.hexdigest()


def _edge_rows(value: Any, name: str) -> torch.Tensor:
    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a tensor.")
    edges = value.detach().to(device="cpu", dtype=torch.long)
    if edges.ndim != 2:
        raise ValueError(f"{name} must be rank 2, got shape {tuple(edges.shape)}.")
    if edges.size(1) == 2:
        return edges.contiguous()
    if edges.size(0) == 2:
        return edges.t().contiguous()
    raise ValueError(f"{name} must have shape [E,2] or [2,E].")


def _validate_edge_range(edges: torch.Tensor, num_nodes: int, name: str) -> None:
    if edges.numel() == 0:
        return
    minimum = int(edges.min().item())
    maximum = int(edges.max().item())
    if minimum < 0 or maximum >= int(num_nodes):
        raise ValueError(f"{name} contains node ids [{minimum}, {maximum}] outside [0, {int(num_nodes) - 1}].")


def _authoritative_filter_tensor(bundle: Mapping[str, Any], key: str, *, num_nodes: int, rowptr: bool) -> torch.Tensor:
    value = bundle.get(key)
    if not torch.is_tensor(value):
        raise ValueError(
            f"Ranked candidate generation requires the evaluator's authoritative legal filter tensor bundle[{key!r}]."
        )
    value = value.detach().to(device="cpu", dtype=torch.long).contiguous()
    if rowptr:
        if tuple(value.shape) != (int(num_nodes) + 1,):
            raise ValueError(f"Authoritative filter {key!r} must have shape {(int(num_nodes) + 1,)}.")
        if int(value[0].item()) != 0 or bool((value[1:] < value[:-1]).any().item()):
            raise ValueError(f"Authoritative filter {key!r} is not a valid CSR rowptr.")
    elif value.numel() and (int(value.min().item()) < 0 or int(value.max().item()) >= int(num_nodes)):
        raise ValueError(f"Authoritative filter {key!r} has out-of-range nodes.")
    return value


def _build_ranked_filter_layout(
    bundle: Mapping[str, Any], framework: str, dataset: str, queries: torch.Tensor, num_nodes: int, split: str
) -> RankedTestFilterLayout:
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping.")
    framework_key = _normal_framework(framework)
    dataset_key = str(dataset).strip().lower()
    split = str(split).strip().lower()
    if split not in {"valid", "test"}:
        raise ValueError("Ranked filter split must be 'valid' or 'test'.")
    nodes = int(num_nodes)
    if nodes <= 1:
        raise ValueError("Ranked filter layout requires at least two nodes.")
    queries = _edge_rows(queries, f"ranked {split}-filter queries")
    _validate_edge_range(queries, nodes, f"ranked {split}-filter queries")
    if framework_key == "pyg":
        out_rowptr_key = in_rowptr_key = f"csr_{split}_known_rowptr"
        out_col_key = in_col_key = f"csr_{split}_known_col"
        fallback_directionality = "canonical-undirected"
    else:
        out_rowptr_key = f"heart_{split}_out_rowptr"
        out_col_key = f"heart_{split}_out_col"
        in_rowptr_key = f"heart_{split}_in_rowptr"
        in_col_key = f"heart_{split}_in_col"
        fallback_directionality = "directed-shared-released-row-view" if dataset_key == "ogbl-citation2" else "undirected-canonical"
    filter_metadata = bundle.get("full_graph_known_positive_filter")
    if isinstance(filter_metadata, Mapping):
        directionality = str(filter_metadata.get("directionality", fallback_directionality))
        filter_scope = str(filter_metadata.get("positive_scope", f"authoritative-{split}-filter"))
    else:
        directionality = fallback_directionality
        filter_scope = f"authoritative-{split}-filter"
    out_rowptr = _authoritative_filter_tensor(bundle, out_rowptr_key, num_nodes=nodes, rowptr=True)
    out_col = _authoritative_filter_tensor(bundle, out_col_key, num_nodes=nodes, rowptr=False)
    in_rowptr = _authoritative_filter_tensor(bundle, in_rowptr_key, num_nodes=nodes, rowptr=True)
    in_col = _authoritative_filter_tensor(bundle, in_col_key, num_nodes=nodes, rowptr=False)
    if int(out_rowptr[-1].item()) != int(out_col.numel()):
        raise ValueError("Authoritative outgoing legal-filter CSR is inconsistent.")
    if int(in_rowptr[-1].item()) != int(in_col.numel()):
        raise ValueError("Authoritative incoming legal-filter CSR is inconsistent.")
    declared_shared = framework_key == "pyg" or "shared" in directionality.lower() or "undirected" in directionality.lower()
    if declared_shared and (not torch.equal(out_rowptr, in_rowptr) or not torch.equal(out_col, in_col)):
        raise ValueError("Authoritative metadata declares a shared legal filter, but its outgoing and incoming CSR views differ.")
    shared_views = declared_shared or (out_rowptr.data_ptr() == in_rowptr.data_ptr() and out_col.data_ptr() == in_col.data_ptr())
    right_role_offset = 0 if shared_views else nodes
    left_keys = queries[:, 0].to(torch.long).cpu().contiguous()
    right_keys = queries[:, 1].to(torch.long).cpu().contiguous() + int(right_role_offset)
    ranking_keys = torch.unique(torch.cat([left_keys, right_keys], dim=0), sorted=True)
    left_rows = torch.searchsorted(ranking_keys, left_keys)
    right_rows = torch.searchsorted(ranking_keys, right_keys)
    source_rows = ranking_keys.remainder(nodes)
    incoming = ranking_keys >= nodes
    starts = out_rowptr[source_rows].clone()
    ends = out_rowptr[source_rows + 1].clone()
    if bool(incoming.any().item()):
        starts[incoming] = in_rowptr[source_rows[incoming]]
        ends[incoming] = in_rowptr[source_rows[incoming] + 1]
    lengths = ends - starts
    rowptr = torch.empty(int(ranking_keys.numel()) + 1, dtype=torch.long)
    rowptr[0] = 0
    rowptr[1:] = lengths.cumsum(0)
    total = int(rowptr[-1].item())
    col = torch.empty(total, dtype=torch.long)
    blocked_ids = torch.empty(total, dtype=torch.long)
    self_is_blocked = torch.zeros(int(ranking_keys.numel()), dtype=torch.long)
    rows_per_chunk = 4096
    for first in range(0, int(ranking_keys.numel()), rows_per_chunk):
        last = min(first + rows_per_chunk, int(ranking_keys.numel()))
        chunk_lengths = lengths[first:last]
        chunk_total = int(chunk_lengths.sum().item())
        if chunk_total == 0:
            continue
        local_rowptr = torch.empty(int(chunk_lengths.numel()) + 1, dtype=torch.long)
        local_rowptr[0] = 0
        local_rowptr[1:] = chunk_lengths.cumsum(0)
        repeated_starts = torch.repeat_interleave(starts[first:last], chunk_lengths)
        repeated_bases = torch.repeat_interleave(local_rowptr[:-1], chunk_lengths)
        indices = repeated_starts + torch.arange(chunk_total) - repeated_bases
        repeated_incoming = torch.repeat_interleave(incoming[first:last], chunk_lengths)
        values = torch.empty(chunk_total, dtype=torch.long)
        outgoing = ~repeated_incoming
        if bool(outgoing.any().item()):
            values[outgoing] = out_col[indices[outgoing]]
        if bool(repeated_incoming.any().item()):
            values[repeated_incoming] = in_col[indices[repeated_incoming]]
        destination_start = int(rowptr[first].item())
        destination_end = int(rowptr[last].item())
        col[destination_start:destination_end] = values
        roles = torch.repeat_interleave(ranking_keys[first:last], chunk_lengths)
        blocked_ids[destination_start:destination_end] = roles * nodes + values
        self_entries = values == roles.remainder(nodes)
        if bool(self_entries.any().item()):
            compact_rows = torch.repeat_interleave(torch.arange(first, last, dtype=torch.long), chunk_lengths)
            self_is_blocked.scatter_add_(0, compact_rows[self_entries], torch.ones_like(compact_rows[self_entries]))
    capacities = nodes - 1 - lengths + self_is_blocked
    if bool((capacities < 0).any().item()):
        raise ValueError("Authoritative legal filter has an impossible row degree.")
    if blocked_ids.numel() > 1 and bool((blocked_ids[1:] <= blocked_ids[:-1]).any().item()):
        raise ValueError("Authoritative legal-filter rows must be sorted and duplicate-free.")
    return RankedTestFilterLayout(
        ranking_keys=ranking_keys,
        rowptr=rowptr,
        col=col,
        left_rows=left_rows,
        right_rows=right_rows,
        capacities=capacities,
        blocked_ids=blocked_ids,
        right_role_offset=int(right_role_offset),
        directionality=directionality,
        filter_scope=filter_scope,
    )


def build_ranked_test_filter_layout(
    bundle: Mapping[str, Any], framework: str, dataset: str, queries: torch.Tensor, num_nodes: int
) -> RankedTestFilterLayout:
    return _build_ranked_filter_layout(bundle, framework, dataset, queries, num_nodes, "test")


def _neutral_validation_scope(bundle: Mapping[str, Any], framework: str, dataset: str):
    framework_key = _normal_framework(framework)
    dataset_key = str(dataset).strip().lower()
    (queries, _source) = _lookup_mapping(bundle, ("valid_pos", "pos_valid_edge", "valid_edge"))
    if queries is None:
        queries, _source = _split_edge_rows(bundle, "valid")
    if queries is None:
        raise KeyError("Neutral selector validation requires validation-positive queries.")
    queries = _edge_rows(queries, "neutral selector validation positives").cpu()
    if not int(queries.size(0)):
        raise ValueError("Neutral selector validation requires at least one positive query.")
    if bool((queries[:, 0] == queries[:, 1]).any().item()):
        raise ValueError("Neutral selector validation does not support self-loop positive queries.")
    raw_nodes = bundle.get("num_nodes")
    data = bundle.get("data")
    if raw_nodes is None and data is not None:
        raw_nodes = getattr(data, "num_nodes", None)
    features = bundle.get("x")
    if raw_nodes is None and torch.is_tensor(features):
        raw_nodes = int(features.size(0))
    if raw_nodes is None:
        raise ValueError("Neutral selector validation cannot determine the graph's node count.")
    num_nodes = int(raw_nodes)
    _validate_edge_range(queries, num_nodes, "neutral selector validation positives")
    layout = _build_ranked_filter_layout(bundle, framework_key, dataset_key, queries, num_nodes, "valid")
    eligibility = bundle.get("full_graph_known_positive_filter")
    policy = str(eligibility.get("policy", "released-legal-positive-filter")) if isinstance(eligibility, Mapping) else "released-legal-positive-filter"
    return queries, num_nodes, layout, policy


def _neutral_selector_validation_identity(
    queries: torch.Tensor,
    num_nodes: int,
    layout: RankedTestFilterLayout,
    policy: str,
    *,
    seed: int,
    negatives: int,
) -> dict[str, Any]:
    total = int(negatives)
    if total <= 0 or total % 2:
        raise ValueError("Neutral selector validation negatives must be a positive even number.")
    requested_side, effective_side = _balanced_side_layout(
        queries=queries, filter_layout=layout, num_nodes=num_nodes, k=total
    )
    if effective_side != requested_side:
        raise ValueError(
            f"Neutral selector validation requires {requested_side} legal candidates on each endpoint side; "
            f"the common legal capacity supports only {effective_side}."
        )
    return {
        "selector_validation_protocol": NEUTRAL_SELECTOR_VALIDATION_PROTOCOL,
        "selector_validation_source": "deterministic-neutral-legal-filter",
        "selector_validation_split": "valid",
        "selector_validation_seed": int(seed),
        "selector_validation_negatives_total": total,
        "selector_validation_negatives_per_side": requested_side,
        "selector_validation_legality_policy": policy,
        "selector_validation_filter_scope": layout.filter_scope,
        "selector_validation_directionality": layout.directionality,
        "selector_validation_num_nodes": num_nodes,
        "selector_validation_positive_rows": int(queries.size(0)),
        "selector_validation_query_sha256": _sha256_tensor(queries, "neutral-selector-validation-queries"),
        "selector_validation_blocked_role_sha256": _sha256_tensor(
            layout.blocked_ids, "neutral-selector-validation-blocked-role-ids"
        ),
    }


def neutral_selector_validation_identity(
    bundle: Mapping[str, Any], framework: str, dataset: str, *, seed: int, negatives: int = 500
) -> dict[str, Any]:
    """Return the deterministic identity of the neutral legal validation pool."""
    queries, num_nodes, layout, policy = _neutral_validation_scope(bundle, framework, dataset)
    return _neutral_selector_validation_identity(
        queries, num_nodes, layout, policy, seed=int(seed), negatives=int(negatives)
    )


def _coprime_validation_stride(num_nodes: int, seed: int) -> int:
    stride = (104729 + 2 * (abs(int(seed)) % 104729)) % int(num_nodes)
    stride = stride or 1
    while math.gcd(stride, int(num_nodes)) != 1:
        stride = (stride + 1) % int(num_nodes) or 1
    return stride


def _neutral_validation_side_nodes(
    *,
    role_keys: torch.Tensor,
    endpoints: torch.Tensor,
    counterparts: torch.Tensor,
    blocked_role_ids: torch.Tensor,
    num_nodes: int,
    per_side: int,
    seed: int,
    side: int,
    output: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    rows = int(role_keys.numel())
    expected_shape = (rows, int(per_side))
    if output is None:
        output = torch.empty(expected_shape, dtype=torch.int32)
    elif output.dtype != torch.int32 or tuple(output.shape) != expected_shape:
        raise ValueError("Neutral validation output buffer has the wrong shape or dtype.")
    stride = _coprime_validation_stride(num_nodes, seed + side * 1000003)
    scan_width = max(512, int(per_side) + 64)
    row_chunk = 2048
    for first in range(0, rows, row_chunk):
        last = min(first + row_chunk, rows)
        roles = role_keys[first:last].to(torch.long).cpu()
        fixed = endpoints[first:last].to(torch.long).cpu()
        counterpart = counterparts[first:last].to(torch.long).cpu()
        offsets = torch.remainder(
            roles * 1103515245 + (int(seed) + 1 + side * 214013) * 12345,
            int(num_nodes),
        )
        filled = torch.zeros(last - first, dtype=torch.long)
        local = output[first:last]
        for scan_start in range(0, int(num_nodes), scan_width):
            width = min(scan_width, int(num_nodes) - scan_start)
            positions = torch.arange(scan_start, scan_start + width, dtype=torch.long)
            candidates = torch.remainder(offsets[:, None] + positions[None, :] * stride, int(num_nodes))
            legal = (candidates != fixed[:, None]) & (candidates != counterpart[:, None])
            role_candidate_ids = roles[:, None] * int(num_nodes) + candidates
            legal &= ~_sorted_membership(blocked_role_ids, role_candidate_ids.reshape(-1)).view_as(legal)
            ranks = legal.cumsum(dim=1)
            remaining = int(per_side) - filled
            take = legal & (ranks <= remaining[:, None]) & (remaining[:, None] > 0)
            selected_rows, selected_cols = torch.nonzero(take, as_tuple=True)
            if selected_rows.numel():
                destination = filled[selected_rows] + ranks[selected_rows, selected_cols] - 1
                local[selected_rows, destination] = candidates[selected_rows, selected_cols].to(torch.int32)
            filled += take.sum(dim=1)
            if bool((filled == int(per_side)).all().item()):
                break
        if bool((filled != int(per_side)).any().item()):
            raise RuntimeError("Neutral selector validation could not fill one legal endpoint-side pool.")
    return output


def build_neutral_selector_validation_negatives(
    bundle: Mapping[str, Any],
    framework: str,
    dataset: str,
    *,
    seed: int,
    negatives: int = 500,
    pos_chunk_size: int = 256,
) -> NeutralSelectorValidationResult:
    """Build deterministic 250/250 validation negatives without heuristic ranking."""
    queries, num_nodes, layout, policy = _neutral_validation_scope(bundle, framework, dataset)
    identity = _neutral_selector_validation_identity(
        queries, num_nodes, layout, policy, seed=int(seed), negatives=int(negatives)
    )
    per_side = int(identity["selector_validation_negatives_per_side"])
    left_roles = queries[:, 0]
    right_roles = queries[:, 1] + int(layout.right_role_offset)
    candidate_nodes = torch.empty((int(queries.size(0)), int(negatives)), dtype=torch.int32)
    _neutral_validation_side_nodes(
        role_keys=left_roles,
        endpoints=queries[:, 0],
        counterparts=queries[:, 1],
        blocked_role_ids=layout.blocked_ids,
        num_nodes=num_nodes,
        per_side=per_side,
        seed=int(seed),
        side=0,
        output=candidate_nodes[:, :per_side],
    )
    _neutral_validation_side_nodes(
        role_keys=right_roles,
        endpoints=queries[:, 1],
        counterparts=queries[:, 0],
        blocked_role_ids=layout.blocked_ids,
        num_nodes=num_nodes,
        per_side=per_side,
        seed=int(seed),
        side=1,
        output=candidate_nodes[:, per_side:],
    )
    grouped = NeutralSelectorValidationNegatives(queries, candidate_nodes, num_nodes, pos_chunk_size)
    metadata = {
        **identity,
        "selector_validation_candidate_nodes_sha256": _sha256_tensor(
            candidate_nodes, "neutral-selector-validation-candidate-nodes"
        ),
        "selector_validation_storage": "streamed-int32-endpoint-nodes",
        "selector_validation_storage_bytes": grouped.storage_nbytes,
    }
    return NeutralSelectorValidationResult(grouped, metadata)


def _sorted_membership(sorted_ids: torch.Tensor, query_ids: torch.Tensor) -> torch.Tensor:
    positions = torch.searchsorted(sorted_ids, query_ids)
    inside = positions < int(sorted_ids.numel())
    if sorted_ids.numel():
        safe = positions.clamp_max(int(sorted_ids.numel()) - 1)
        inside &= sorted_ids[safe] == query_ids
    return inside


def _query_side_capacities(queries, filter_layout, num_nodes):
    queries = _edge_rows(queries, "Concat selection queries")
    if not int(queries.size(0)):
        raise ValueError("Concat selection requires at least one test query.")
    left_ids = queries[:, 0] * int(num_nodes) + queries[:, 1]
    right_ids = (queries[:, 1] + int(filter_layout.right_role_offset)) * int(num_nodes) + queries[:, 0]
    left_capacity = filter_layout.capacities[filter_layout.left_rows] - (
        ~_sorted_membership(filter_layout.blocked_ids, left_ids)
    ).to(torch.long)
    right_capacity = filter_layout.capacities[filter_layout.right_rows] - (
        ~_sorted_membership(filter_layout.blocked_ids, right_ids)
    ).to(torch.long)
    return left_capacity, right_capacity


def _balanced_side_layout(
    *, queries: torch.Tensor, filter_layout: RankedTestFilterLayout, num_nodes: int, k: int
) -> tuple[int, int]:
    if int(k) <= 0 or int(k) % 2:
        raise ValueError("Concat total k must be a positive even number split across two endpoint sides.")
    left_capacity, right_capacity = _query_side_capacities(queries, filter_layout, num_nodes)
    requested_side_k = int(k) // 2
    effective_side_k = min(requested_side_k, int(left_capacity.min().item()), int(right_capacity.min().item()))
    if effective_side_k <= 0:
        raise ValueError("No legal Concat negative exists on one endpoint side.")
    return (requested_side_k, effective_side_k)


def _global_top_k_layout(
    *, queries: torch.Tensor, filter_layout: RankedTestFilterLayout, num_nodes: int, k: int
) -> int:
    if int(k) <= 0:
        raise ValueError("Concat total k must be positive.")
    left_capacity, right_capacity = _query_side_capacities(queries, filter_layout, num_nodes)
    effective_k = min(int(k), int((left_capacity + right_capacity).min().item()))
    if effective_k <= 0:
        raise ValueError("No legal Concat endpoint-corruption negative exists for a test query.")
    return effective_k


def _lookup_mapping(bundle: Mapping[str, Any], keys: tuple[str, ...]):
    for key in keys:
        value = bundle.get(key)
        if torch.is_tensor(value):
            return (value, f"bundle.{key}")
    eval_edges = bundle.get("eval_edges")
    if isinstance(eval_edges, Mapping):
        for key in keys:
            value = eval_edges.get(key)
            if torch.is_tensor(value):
                return (value, f"bundle.eval_edges.{key}")
    return (None, None)


def _split_edge_rows(bundle: Mapping[str, Any], split: str):
    split_edge = bundle.get("split_edge")
    if not isinstance(split_edge, Mapping):
        return (None, None)
    payload = split_edge.get(split)
    if not isinstance(payload, Mapping):
        return (None, None)
    edge = payload.get("edge")
    if torch.is_tensor(edge):
        return (edge, f"bundle.split_edge.{split}.edge")
    source = payload.get("source_node")
    target = payload.get("target_node")
    if torch.is_tensor(source) and torch.is_tensor(target):
        if source.numel() != target.numel():
            raise ValueError(f"split_edge[{split!r}] source and target lengths differ.")
        return (torch.stack([source.reshape(-1), target.reshape(-1)], dim=1), f"bundle.split_edge.{split}.source-target")
    return (None, None)


def _full_positive_split(bundle: Mapping[str, Any], split: str):
    if split == "valid":
        preferred = ("concat_full_valid_pos", "all_valid_pos", "full_valid_pos", "valid_pos_full", "uncapped_valid_pos", "valid_input_pos")
    elif split == "test":
        preferred = ("concat_full_test_pos", "all_test_pos", "full_test_pos", "test_pos_full", "uncapped_test_pos")
    else:
        raise ValueError(f"Unknown split: {split}")
    (value, source) = _lookup_mapping(bundle, preferred)
    if value is None:
        (value, source) = _split_edge_rows(bundle, split)
    if value is None:
        raise KeyError(
            f"concat requires the complete {split} positive split. Provide bundle['full_{split}_pos'] (or an OGB split_edge payload)."
        )
    return (_edge_rows(value, source), source)


def _test_queries(bundle: Mapping[str, Any]) -> tuple[torch.Tensor, str]:
    (value, source) = _lookup_mapping(bundle, ("test_pos", "pos_test_edge", "test_edge"))
    if value is None:
        (value, source) = _split_edge_rows(bundle, "test")
    if value is None:
        raise KeyError("concat requires bundle['test_pos'] test queries.")
    queries = _edge_rows(value, source)
    if queries.size(0) == 0:
        raise ValueError("concat requires at least one test-positive query.")
    if bool((queries[:, 0] == queries[:, 1]).any().item()):
        raise ValueError("concat does not support self-loop test-positive queries.")
    return (queries, source)


def _graph_edges(graph: Any, num_nodes: int, source: str) -> torch.Tensor:
    if hasattr(graph, "coo"):
        (row, col, _) = graph.coo()
        return torch.stack([row.detach().cpu().long(), col.detach().cpu().long()], dim=1).contiguous()
    if not torch.is_tensor(graph):
        raise TypeError(f"Cannot extract training edges from {source}.")
    if graph.layout == torch.sparse_coo:
        index = graph.detach().cpu().coalesce().indices()
        return index.t().to(torch.long).contiguous()
    if graph.layout in {torch.sparse_csr, torch.sparse_csc}:
        coo = graph.detach().cpu().to_sparse_coo().coalesce().indices()
        return coo.t().to(torch.long).contiguous()
    if graph.ndim == 2 and tuple(graph.shape) == (int(num_nodes), int(num_nodes)):
        return torch.nonzero(graph.detach().cpu(), as_tuple=False).to(torch.long)
    return _edge_rows(graph, source)


def _resolve_training_graph(bundle: Mapping[str, Any], framework: str):
    graph = bundle.get("adj")
    if graph is not None:
        return (graph, "bundle.adj")
    data = bundle.get("data")
    if data is not None:
        graph = getattr(data, "adj_t", None)
        if graph is not None:
            return (graph, "bundle.data.adj_t")
        graph = getattr(data, "edge_index", None)
        if graph is not None:
            return (graph, "bundle.data.edge_index")
    (train, source) = _lookup_mapping(bundle, ("train_pos", "pos_train_edge", "train_edge"))
    if train is None:
        raise KeyError("concat requires a training adjacency or train_pos edges.")
    rows = _edge_rows(train, source)
    return (rows.t().contiguous(), source)


def _training_positive_edges(bundle: Mapping[str, Any], num_nodes: int, training_graph: Any, graph_source: str) -> tuple[torch.Tensor, str]:
    (edges, source) = _lookup_mapping(bundle, ("train_pos", "pos_train_edge", "train_edge"))
    if edges is None:
        (edges, source) = _split_edge_rows(bundle, "train")
    if edges is not None:
        edges = _edge_rows(edges, source)
    else:
        edges = _graph_edges(training_graph, num_nodes, graph_source)
        source = graph_source
    _validate_edge_range(edges, num_nodes, source)
    return (edges, source)


def _raw_features(
    bundle: Mapping[str, Any],
    framework: str,
    *,
    selector_model: str = "concat",
    dataset: str = "",
    force_node_embeddings: bool = False,
) -> torch.Tensor:
    if framework == "pyg":
        x = bundle.get("x")
    else:
        data = bundle.get("data")
        x = getattr(data, "x", None) if data is not None else bundle.get("x")
    if force_node_embeddings or (not torch.is_tensor(x) and str(dataset).strip().lower() == "ogbl-ddi"):
        data = bundle.get("data")
        node_count = getattr(data, "num_nodes", None) if data is not None else None
        if node_count is None:
            node_count = bundle.get("num_nodes")
        if node_count is None and torch.is_tensor(x) and x.ndim == 2:
            node_count = int(x.size(0))
        if node_count is None:
            raise ValueError(f"{_selector_label(selector_model)} selection with trainable node features requires the graph's node count.")
        return torch.empty((int(node_count), 0), dtype=torch.float32)
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError(f"{_selector_label(selector_model)} selection requires a rank-2 target feature tensor.")
    return x


def _atomic_save(payload: Mapping[str, Any], path: str) -> None:
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    (descriptor, temporary) = tempfile.mkstemp(prefix=f".{os.path.basename(path)}.", suffix=".tmp", dir=directory)
    os.close(descriptor)
    try:
        torch.save(dict(payload), temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


@contextmanager
def _exclusive_cache_lock(cache_path: str):
    lock_path = f"{cache_path}.lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    handle = open(lock_path, "a+b")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        handle.close()


def _candidate_summary(
    num_queries: int, effective_k: int, left_min=None, left_max=None, left_sum=None,
) -> dict[str, Any]:
    rows = int(num_queries)
    k = int(effective_k)
    if rows and left_min is not None:
        left_stats = {"min": int(left_min), "mean": float(left_sum) / rows, "max": int(left_max)}
        right_stats = {"min": k - int(left_max), "mean": k - float(left_sum) / rows, "max": k - int(left_min)}
        both_sides = int(left_min) > 0 and int(left_max) < k
    elif rows:
        side = k // 2
        left_stats = {"min": side, "mean": float(side), "max": side}
        right_stats = dict(left_stats)
        both_sides = bool(side)
    else:
        left_stats = right_stats = {"min": 0, "mean": 0.0, "max": 0}
        both_sides = False
    zero_stats = {"min": 0, "mean": 0.0, "max": 0}
    total_stats = {"min": k, "mean": float(k), "max": k}
    return {
        "num_positive_edges": rows,
        "negative_group_shape": [rows, k, 2],
        "grouped_negatives_per_positive": k,
        "both_corruption_sides_combined": both_sides,
        "fixed_left_endpoint_candidates": left_stats,
        "fixed_right_endpoint_candidates": right_stats,
        "other_grouped_candidates": zero_stats,
        "total_grouped_candidates": total_stats,
        "total_legal_candidates": dict(total_stats),
    }


def _validate_compact_block(
    encoded: torch.Tensor,
    *,
    queries: torch.Tensor,
    known_ids: torch.Tensor,
    num_nodes: int,
    effective_k: int,
    directed: bool,
    selector_model: str,
    blocked_role_ids: Optional[torch.Tensor] = None,
    right_role_offset: int = 0,
) -> torch.Tensor:
    selector_label = _selector_label(_normal_selector_model(selector_model))
    if (
        not torch.is_tensor(encoded)
        or encoded.dtype != torch.int32
        or encoded.ndim != 2
        or (int(encoded.size(0)) != int(queries.size(0)))
        or (int(encoded.size(1)) != int(effective_k))
    ):
        raise ValueError(f"Cached {selector_label} compact shard has the wrong shape or dtype.")
    compact = encoded.to(torch.long)
    if compact.numel() and (int(compact.min().item()) < 0 or int(compact.max().item()) >= 2 * int(num_nodes)):
        raise ValueError(f"Cached {selector_label} compact shard contains an invalid encoding.")
    sides = torch.div(compact, int(num_nodes), rounding_mode="floor")
    nodes = compact.remainder(int(num_nodes))
    query = queries.to(torch.long).cpu()
    if bool(((sides == 0) & (nodes == query[:, 0:1]) | (sides == 1) & (nodes == query[:, 1:2])).any().item()):
        raise ValueError(f"Cached {selector_label} negatives contain a self-loop.")
    if bool(((sides == 0) & (nodes == query[:, 1:2]) | (sides == 1) & (nodes == query[:, 0:1])).any().item()):
        raise ValueError(f"Cached {selector_label} negatives contain their positive query edge.")
    left = sides == 0
    source = torch.where(left, query[:, 0:1], nodes)
    target = torch.where(left, nodes, query[:, 1:2])
    if directed:
        ids = source * int(num_nodes) + target
    else:
        ids = torch.minimum(source, target) * int(num_nodes) + torch.maximum(source, target)
    if blocked_role_ids is not None:
        role_keys = torch.where(left, query[:, 0:1], query[:, 1:2] + int(right_role_offset))
        filter_ids = blocked_role_ids
        flat_filter_ids = (role_keys * int(num_nodes) + nodes).reshape(-1)
    else:
        filter_ids = known_ids
        flat_filter_ids = ids.reshape(-1)
    positions = torch.searchsorted(filter_ids, flat_filter_ids)
    inside = positions < int(filter_ids.numel())
    if filter_ids.numel():
        safe = positions.clamp_max(int(filter_ids.numel()) - 1)
        if bool((inside & (filter_ids[safe] == flat_filter_ids)).any().item()):
            raise ValueError(f"Cached {selector_label} negatives contain a known positive.")
    ordered = torch.sort(ids, dim=1).values
    if int(effective_k) > 1 and bool((ordered[:, 1:] == ordered[:, :-1]).any().item()):
        raise ValueError(f"Cached {selector_label} negatives contain duplicates within a query.")
    return sides


def _load_and_validate_ranked_shards(
    *,
    cache_path: str,
    storage: Any,
    queries: torch.Tensor,
    known_ids: torch.Tensor,
    num_nodes: int,
    effective_k: int,
    directed: bool,
    selector_model: str,
    blocked_role_ids: Optional[torch.Tensor] = None,
    right_role_offset: int = 0,
    expected_per_side_k: Optional[int] = None,
) -> RankedNegativeShards:
    if not isinstance(storage, Mapping):
        raise ValueError("Ranked-negative cache storage manifest is missing.")
    if storage.get("format") != CONCAT_SHARD_FORMAT:
        raise ValueError("Ranked-negative cache has an unsupported shard format.")
    if storage.get("shape") != [int(queries.size(0)), int(effective_k), 2]:
        raise ValueError("Ranked-negative cache storage shape mismatch.")
    if int(storage.get("num_nodes", -1)) != int(num_nodes):
        raise ValueError("Ranked-negative cache node count mismatch.")
    raw_shards = storage.get("shards")
    if not isinstance(raw_shards, list) or not raw_shards:
        raise ValueError("Ranked-negative cache shard list is missing.")
    descriptors = []
    expected_start = 0
    manifest_parent = Path(cache_path).resolve().parent
    for raw in raw_shards:
        if not isinstance(raw, Mapping):
            raise ValueError("Ranked-negative shard descriptor is invalid.")
        shard = _NegativeShard(
            start=int(raw.get("start", -1)),
            end=int(raw.get("end", -1)),
            filename=str(raw.get("filename", "")),
        )
        candidate_path = (manifest_parent / shard.filename).resolve()
        try:
            candidate_path.relative_to(manifest_parent)
        except ValueError as exc:
            raise ValueError("Ranked-negative shard path escapes its cache.") from exc
        if shard.start != expected_start or shard.end <= shard.start:
            raise ValueError("Ranked-negative shard row coverage is invalid.")
        expected_start = shard.end
        descriptors.append(shard)
    if expected_start != int(queries.size(0)):
        raise ValueError("Ranked-negative shards do not cover all query rows.")
    stream = RankedNegativeShards(
        manifest_path=cache_path,
        queries=queries,
        num_nodes=num_nodes,
        effective_k=effective_k,
        shards=descriptors,
        default_chunk_size=int(storage.get("default_chunk_size", 2048)),
        candidate_summary=None,
    )
    left_min, left_max, left_sum = int(effective_k), 0, 0
    for shard in descriptors:
        encoded = stream._load_encoded(shard)
        sides = _validate_compact_block(
            encoded,
            queries=queries[shard.start : shard.end],
            known_ids=known_ids,
            num_nodes=num_nodes,
            effective_k=effective_k,
            directed=directed,
            selector_model=selector_model,
            blocked_role_ids=blocked_role_ids,
            right_role_offset=right_role_offset,
        )
        left_counts = (sides == 0).sum(dim=1)
        if expected_per_side_k is not None:
            if not bool((left_counts == int(expected_per_side_k)).all().item()):
                raise ValueError("Ranked-negative cache row violated its fixed per-side quota.")
        else:
            left_min = min(left_min, int(left_counts.min().item()))
            left_max = max(left_max, int(left_counts.max().item()))
            left_sum += int(left_counts.sum().item())
    stream.candidate_summary = _candidate_summary(
        int(queries.size(0)), effective_k,
        None if expected_per_side_k is not None else left_min,
        None if expected_per_side_k is not None else left_max,
        None if expected_per_side_k is not None else left_sum,
    )
    return stream


def _ranked_shard_storage(negatives, descriptors, num_nodes):
    return {
        "format": CONCAT_SHARD_FORMAT,
        "shape": list(negatives.shape),
        "num_nodes": int(num_nodes),
        "default_chunk_size": int(negatives.default_chunk_size),
        "shards": [
            {"start": shard.start, "end": shard.end, "filename": shard.filename}
            for shard in descriptors
        ],
    }


def _load_cached_ranked_result(
    cache_path, identity, queries, num_nodes, effective_k, directed, selector_model, filter_layout,
    *, expected_per_side_k=None,
):
    cached = _load_torch_payload(cache_path)
    if not isinstance(cached, Mapping) or not isinstance(cached.get("metadata"), Mapping):
        raise ValueError("Cache payload or metadata is not a mapping.")
    metadata = dict(cached["metadata"])
    if metadata.get("cache_key") != identity.get("cache_key"):
        raise ValueError("Ranked-negative cache identity mismatch.")
    negatives = _load_and_validate_ranked_shards(
        cache_path=cache_path, storage=cached.get("storage"), queries=queries,
        known_ids=filter_layout.blocked_ids[:0], num_nodes=num_nodes, effective_k=effective_k,
        directed=directed, selector_model=selector_model, blocked_role_ids=filter_layout.blocked_ids,
        right_role_offset=filter_layout.right_role_offset, expected_per_side_k=expected_per_side_k,
    )
    return RankedNegativeResult(negatives, metadata, cache_path, True)


def _safe_filename(value: str) -> str:
    return re.sub("[^A-Za-z0-9_.-]+", "-", str(value)).strip("-.") or "dataset"


def _lexicographic_score_order(scores: torch.Tensor, sides: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(nodes, stable=True)
    order = order[torch.argsort(sides[order], stable=True)]
    order = order[torch.argsort(scores[order], descending=True, stable=True)]
    return order


def _exact_chunk_topk(scores: torch.Tensor, nodes: torch.Tensor, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    take = min(int(k), int(scores.numel()))
    if take <= 0:
        return (nodes[:0], scores[:0])
    if take == int(scores.numel()):
        return (nodes, scores)
    threshold = torch.topk(scores, take, largest=True, sorted=False).values.min()
    better = torch.nonzero(scores > threshold, as_tuple=False).reshape(-1)
    needed = take - int(better.numel())
    tied = torch.nonzero(scores == threshold, as_tuple=False).reshape(-1)
    chosen = torch.cat([better, tied[:needed]], dim=0)
    if int(chosen.numel()) != take:
        raise RuntimeError("Could not resolve the deterministic concat top-K tie.")
    return (nodes[chosen], scores[chosen])


def _endpoint_topk(
    endpoint: int,
    *,
    embedding: torch.Tensor,
    model: Any,
    blocked: torch.Tensor,
    num_nodes: int,
    k: int,
    score_batch_size: int,
    device: torch.device,
    reverse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    kept_nodes = torch.empty(0, dtype=torch.long)
    score_storage_dtype = torch.float64 if embedding.dtype == torch.float64 else torch.float32
    kept_scores = torch.empty(0, dtype=score_storage_dtype)
    for first in range(0, int(num_nodes), int(score_batch_size)):
        last = min(int(num_nodes), first + int(score_batch_size))
        nodes = torch.arange(first, last, dtype=torch.long)
        legal = nodes != int(endpoint)
        if blocked.numel():
            positions = torch.searchsorted(blocked, nodes)
            inside = positions < blocked.numel()
            safe_positions = positions.clamp_max(max(0, blocked.numel() - 1))
            legal &= ~(inside & (blocked[safe_positions] == nodes))
        nodes = nodes[legal]
        if not nodes.numel():
            continue
        candidate_nodes = nodes.to(device=device, non_blocking=True)
        fixed = torch.full_like(candidate_nodes, int(endpoint))
        edge_index = torch.stack([candidate_nodes, fixed], dim=0) if reverse else torch.stack([fixed, candidate_nodes], dim=0)
        with torch.inference_mode():
            scores = model.decode(embedding, edge_index).reshape(-1)
        if scores.numel() != candidate_nodes.numel():
            raise ValueError("Selector returned the wrong number of candidate scores.")
        if not bool(torch.isfinite(scores).all().item()):
            bad = int((~torch.isfinite(scores)).sum().item())
            raise ValueError(f"Selector produced {bad} non-finite scores for endpoint {endpoint}.")
        (chunk_nodes, chunk_scores) = _exact_chunk_topk(scores, candidate_nodes, k)
        chunk_nodes_cpu = chunk_nodes.detach().to(device="cpu", dtype=torch.long)
        scores_cpu = chunk_scores.detach().to(device="cpu", dtype=score_storage_dtype)
        all_nodes = torch.cat([kept_nodes, chunk_nodes_cpu], dim=0)
        all_scores = torch.cat([kept_scores, scores_cpu], dim=0)
        all_sides = torch.zeros(all_nodes.numel(), dtype=torch.long)
        order = _lexicographic_score_order(all_scores, all_sides, all_nodes)
        order = order[: min(int(k), int(order.numel()))]
        kept_nodes = all_nodes[order].contiguous()
        kept_scores = all_scores[order].contiguous()
    return (kept_nodes, kept_scores)


def _file_backed_rankings(
    *,
    scratch_dir: str,
    ranking_keys: torch.Tensor,
    rowptr: torch.Tensor,
    col: torch.Tensor,
    embedding: torch.Tensor,
    model: Any,
    num_nodes: int,
    effective_k: int,
    score_batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows = int(ranking_keys.numel())
    node_path = str(Path(scratch_dir) / "endpoint_nodes.bin")
    score_path = str(Path(scratch_dir) / "endpoint_scores.bin")
    node_matrix = torch.from_file(node_path, shared=True, size=rows * int(effective_k), dtype=torch.int32).view(rows, int(effective_k))
    score_dtype = torch.float64 if embedding.dtype == torch.float64 else torch.float32
    score_matrix = torch.from_file(score_path, shared=True, size=rows * int(effective_k), dtype=score_dtype).view(rows, int(effective_k))
    node_matrix.zero_()
    score_matrix.fill_(float("-inf"))
    for row, encoded_key in enumerate(ranking_keys.tolist()):
        reverse = int(encoded_key) >= int(num_nodes)
        endpoint = int(encoded_key) % int(num_nodes)
        start = int(rowptr[row].item())
        end = int(rowptr[row + 1].item())
        (nodes, scores) = _endpoint_topk(
            endpoint,
            embedding=embedding,
            model=model,
            blocked=col[start:end],
            num_nodes=num_nodes,
            k=effective_k,
            score_batch_size=score_batch_size,
            device=device,
            reverse=reverse,
        )
        count = min(int(effective_k), int(nodes.numel()))
        if count:
            node_matrix[row, :count] = nodes[:count].to(torch.int32)
            score_matrix[row, :count] = scores[:count].to(score_dtype)
    return (node_matrix, score_matrix)


def _merge_query_rows(
    *,
    left_nodes: torch.Tensor,
    left_scores: torch.Tensor,
    right_nodes: torch.Tensor,
    right_scores: torch.Tensor,
    queries: torch.Tensor,
    effective_k: int,
    num_nodes: int,
    global_top_k: bool = False,
) -> torch.Tensor:
    rows = int(left_nodes.size(0))
    queries = _edge_rows(queries, "ranked-negative merge queries")
    if int(queries.size(0)) != rows:
        raise ValueError("Ranked-negative merge queries have the wrong row count.")
    if int(effective_k) <= 0 or (not global_top_k and int(effective_k) % 2):
        message = "positive" if global_top_k else "positive and even for fixed per-side selection"
        raise ValueError(f"Concat effective k must be {message}.")

    def select_side(nodes: torch.Tensor, scores: torch.Tensor, counterparts: torch.Tensor) -> torch.Tensor:
        nodes = nodes.to(torch.long)
        if nodes.ndim != 2 or tuple(scores.shape) != tuple(nodes.shape) or int(nodes.size(0)) != rows:
            raise ValueError("Concat ranked nodes and scores do not align.")
        ranked_scores = scores.masked_fill(nodes == counterparts.view(-1, 1), float("-inf"))
        order = torch.argsort(ranked_scores, dim=1, descending=True, stable=True)[:, :side_k]
        chosen_scores = ranked_scores.gather(1, order)
        if tuple(order.shape) != (rows, side_k) or not bool(torch.isfinite(chosen_scores).all().item()):
            raise RuntimeError("Concat endpoint ranking did not retain enough candidates for its fixed per-side quota.")
        return nodes.gather(1, order)

    if global_top_k:
        left_nodes, right_nodes = left_nodes.to(torch.long), right_nodes.to(torch.long)
        nodes = torch.cat([left_nodes, right_nodes], dim=1)
        scores = torch.cat([
            left_scores.masked_fill(left_nodes == queries[:, 1:2], float("-inf")),
            right_scores.masked_fill(right_nodes == queries[:, 0:1], float("-inf")),
        ], dim=1)
        sides = torch.cat([torch.zeros_like(left_nodes), torch.ones_like(right_nodes)], dim=1)
        # Endpoint rows are score-desc/node-asc and left precedes right, so a
        # stable score sort gives score-desc/side-asc/node-asc globally.
        order = torch.argsort(scores, dim=1, descending=True, stable=True)[:, :int(effective_k)]
        if not bool(torch.isfinite(scores.gather(1, order)).all().item()):
            raise RuntimeError("Concat endpoint ranking did not retain enough global candidates.")
        encoded = nodes.gather(1, order) + sides.gather(1, order) * int(num_nodes)
    else:
        side_k = int(effective_k) // 2
        left = select_side(left_nodes, left_scores, queries[:, 1])
        right = select_side(right_nodes, right_scores, queries[:, 0]) + int(num_nodes)
        encoded = torch.cat([left, right], dim=1)
    if encoded.numel() and int(encoded.max().item()) > torch.iinfo(torch.int32).max:
        raise ValueError("Compact ranked-negative encoding exceeds the int32 node/side range.")
    return encoded.to(torch.int32).contiguous()


def _decode_compact_rows(encoded: torch.Tensor, queries: torch.Tensor, num_nodes: int) -> torch.Tensor:
    compact = encoded.to(torch.long)
    sides = torch.div(compact, int(num_nodes), rounding_mode="floor")
    nodes = compact.remainder(int(num_nodes))
    left = sides == 0
    output = torch.empty((*compact.shape, 2), dtype=torch.long)
    output[:, :, 0] = torch.where(left, queries[:, 0:1], nodes)
    output[:, :, 1] = torch.where(left, nodes, queries[:, 1:2])
    return output

def _validate_inner_product_selector(model: Any, selector_model: str) -> None:
    selector_model = _normal_selector_model(selector_model)
    decoder = getattr(model, "decoder", None)
    if decoder is None or decoder.__class__.__name__ != "DotProductDecoder":
        raise ValueError(f"{_selector_label(selector_model)} requires the parameter-free inner-product decoder.")
    if any((parameter.numel() for parameter in decoder.parameters())):
        raise ValueError(f"{_selector_label(selector_model)} decoder must not contain learned parameters.")
    if getattr(model, "decode_is_symmetric", None) is not True:
        raise ValueError(f"{_selector_label(selector_model)} inner-product decoder must declare symmetric scores.")
    contract = {
        "decoder_type": "inner-product",
        "predictor_depth": 0,
        "decoder_output": "raw-inner-product-logit",
        "ranking_score": "raw-inner-product-logit",
        "probability_transform": "sigmoid",
        "reference_evaluation_transform": "sigmoid",
    }
    for key, expected in contract.items():
        if getattr(model, key, None) != expected:
            raise ValueError(
                f"{_selector_label(selector_model)} selector {key}={getattr(model, key, None)!r}, "
                f"expected {expected!r}."
            )


def _select_ranked_queries(
    queries, filter_layout, embedding, model, k, effective_k, score_batch_size, device,
    global_top_k=False,
) -> torch.Tensor:
    num_nodes = int(embedding.size(0))
    if global_top_k:
        selected_k = _global_top_k_layout(
            queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=k
        )
    else:
        _, side_k = _balanced_side_layout(
            queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=k
        )
        selected_k = 2 * side_k
    if selected_k != int(effective_k):
        raise RuntimeError("concat legal-candidate capacity changed while constructing negatives.")
    output = torch.empty((int(queries.size(0)), int(effective_k), 2), dtype=torch.long)
    with tempfile.TemporaryDirectory(prefix="ranked-negative-scratch-") as scratch:
        (node_matrix, score_matrix) = _file_backed_rankings(
            scratch_dir=scratch,
            ranking_keys=filter_layout.ranking_keys,
            rowptr=filter_layout.rowptr,
            col=filter_layout.col,
            embedding=embedding,
            model=model,
            num_nodes=num_nodes,
            effective_k=effective_k + 1 if global_top_k else effective_k // 2 + 1,
            score_batch_size=score_batch_size,
            device=device,
        )
        row_chunk = max(1, min(2048, 2000000 // max(1, 2 * effective_k)))
        for start in range(0, int(queries.size(0)), row_chunk):
            end = min(start + row_chunk, int(queries.size(0)))
            encoded = _merge_query_rows(
                left_nodes=node_matrix[filter_layout.left_rows[start:end]],
                left_scores=score_matrix[filter_layout.left_rows[start:end]],
                right_nodes=node_matrix[filter_layout.right_rows[start:end]],
                right_scores=score_matrix[filter_layout.right_rows[start:end]],
                queries=queries[start:end],
                effective_k=effective_k,
                num_nodes=num_nodes,
                global_top_k=global_top_k,
            )
            output[start:end] = _decode_compact_rows(encoded, queries[start:end], num_nodes)
    return output.contiguous()


def _write_ranked_negative_shards(
    cache_path, queries, filter_layout, embedding, model, num_nodes, effective_k, score_batch_size, device,
    global_top_k=False,
) -> tuple[RankedNegativeShards, list[_NegativeShard]]:
    if int(effective_k) <= 0 or (not global_top_k and int(effective_k) % 2):
        message = "positive" if global_top_k else "positive and even for fixed per-side selection"
        raise ValueError(f"Concat effective k must be {message}.")
    manifest = Path(cache_path).resolve()
    manifest.parent.mkdir(parents=True, exist_ok=True)
    shard_directory = Path(tempfile.mkdtemp(prefix=f".{manifest.name}.shards-", dir=str(manifest.parent)))
    shard_directory_name = shard_directory.name
    selection_rows = max(1, min(2048, 2000000 // max(1, 2 * int(effective_k))))
    shard_rows = max(selection_rows, 8000000 // max(1, int(effective_k)) // selection_rows * selection_rows)
    descriptors: list[_NegativeShard] = []
    left_min, left_max, left_sum = int(effective_k), 0, 0
    with tempfile.TemporaryDirectory(prefix=f".{manifest.name}.scratch-", dir=str(manifest.parent)) as scratch:
        (node_matrix, score_matrix) = _file_backed_rankings(
            scratch_dir=scratch,
            ranking_keys=filter_layout.ranking_keys,
            rowptr=filter_layout.rowptr,
            col=filter_layout.col,
            embedding=embedding,
            model=model,
            num_nodes=num_nodes,
            effective_k=effective_k + 1 if global_top_k else effective_k // 2 + 1,
            score_batch_size=score_batch_size,
            device=device,
        )
        for shard_start in range(0, int(queries.size(0)), shard_rows):
            shard_end = min(shard_start + shard_rows, int(queries.size(0)))
            encoded_shard = torch.empty((shard_end - shard_start, int(effective_k)), dtype=torch.int32)
            for start in range(shard_start, shard_end, selection_rows):
                end = min(start + selection_rows, shard_end)
                encoded = _merge_query_rows(
                    left_nodes=node_matrix[filter_layout.left_rows[start:end]],
                    left_scores=score_matrix[filter_layout.left_rows[start:end]],
                    right_nodes=node_matrix[filter_layout.right_rows[start:end]],
                    right_scores=score_matrix[filter_layout.right_rows[start:end]],
                    queries=queries[start:end],
                    effective_k=effective_k,
                    num_nodes=num_nodes,
                    global_top_k=global_top_k,
                )
                if global_top_k:
                    left_counts = (encoded < int(num_nodes)).sum(dim=1)
                    left_min = min(left_min, int(left_counts.min().item()))
                    left_max = max(left_max, int(left_counts.max().item()))
                    left_sum += int(left_counts.sum().item())
                encoded_shard[start - shard_start : end - shard_start] = encoded
            filename = f"rows-{shard_start:012d}-{shard_end:012d}.pt"
            relative_filename = str(Path(shard_directory_name) / filename)
            _atomic_save({"encoded": encoded_shard}, str(shard_directory / filename))
            descriptors.append(_NegativeShard(shard_start, shard_end, relative_filename))
    summary = _candidate_summary(
        int(queries.size(0)), effective_k,
        left_min if global_top_k else None,
        left_max if global_top_k else None,
        left_sum if global_top_k else None,
    )
    stream = RankedNegativeShards(
        manifest_path=cache_path,
        queries=queries,
        num_nodes=num_nodes,
        effective_k=effective_k,
        shards=descriptors,
        default_chunk_size=selection_rows,
        candidate_summary=summary,
    )
    return stream, descriptors


def _required_positive_int(values: Mapping[str, Any], key: str, label: str) -> int:
    try:
        value = int(values[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{label} {key!r} must be an integer.") from exc
    if value <= 0:
        raise ValueError(f"{label} {key!r} must be positive.")
    return value


def _selector_state_dtype(state: Mapping[str, torch.Tensor]) -> torch.dtype:
    value = state.get("encoder.mlp.0.weight")
    if not torch.is_tensor(value) or not value.is_floating_point():
        raise ValueError("Selector state is missing floating tensor 'encoder.mlp.0.weight'.")
    return value.dtype


def _validate_concat_family_checkpoint(
    checkpoint,
    checkpoint_path,
    framework: str,
    dataset: str,
    raw_x: torch.Tensor,
    spec: Mapping[str, Any],
    overrides: Sequence[Any],
) -> tuple[dict[str, Any], Mapping[str, torch.Tensor], int, int]:
    """Validate storage/model invariants common to Concat-family encoders."""
    from model.feature_aggregation import aggregated_mlp_recipe

    policy = str(spec["policy"])
    label = str(spec.get("display_name", policy.upper()))
    spec["validator"](checkpoint, checkpoint_path, framework, dataset, None, *overrides)
    if int(checkpoint.get("format_version", -1)) != 1 or str(
        checkpoint.get("checkpoint_type", "")
    ).strip() != "best_validation_model_state":
        raise ValueError(f"{label} selection requires a best-validation selector checkpoint.")
    config = checkpoint.get("model_config")
    if not isinstance(config, Mapping):
        raise ValueError("Selector checkpoint model_config must be a mapping.")
    config = dict(config)
    if spec.get("requires_ranked_training_contract", False):
        validate_ranked_selector_training_config(config, f"{label} selector checkpoint")
    _required_positive_int(config, "emb_size", "Selector checkpoint")
    _required_positive_int(config, "layers", "Selector checkpoint")
    try:
        predictor_depth = int(config["pred_layers"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("Selector checkpoint 'pred_layers' must be an integer.") from exc
    exact_predictor = spec.get("predictor_depth")
    minimum_predictor = spec.get("minimum_predictor_depth")
    if exact_predictor is not None and predictor_depth != int(exact_predictor):
        raise ValueError(f"{label} selector checkpoint must declare pred_layers={int(exact_predictor)}.")
    if minimum_predictor is not None and predictor_depth < int(minimum_predictor):
        raise ValueError(f"{label} selector checkpoint pred_layers must be at least {int(minimum_predictor)}.")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("Selector checkpoint has no model_state_dict.")
    expected_recipe = aggregated_mlp_recipe("concat")
    if config.get("concat_preprocessing") != expected_recipe:
        raise ValueError(
            f"Selector checkpoint has the wrong concat preprocessing recipe: expected {expected_recipe!r}, "
            f"got {config.get('concat_preprocessing')!r}."
        )
    base_dim = _required_positive_int(config, "concat_base_feature_dim", "Selector checkpoint")
    output_dim = _required_positive_int(config, "concat_output_feature_dim", "Selector checkpoint")
    if output_dim != 2 * base_dim:
        raise ValueError("Selector concat dimensions are inconsistent: output must equal 2*base.")
    if spec.get("force_node_embeddings"):
        if int(config.get("emb_size", 0)) != base_dim:
            raise ValueError(f"{label} selector trainable feature width must equal emb_size.")
    elif str(dataset).strip().lower() == "ogbl-ddi":
        if _required_positive_int(config, "concat_feature_dim", "Selector checkpoint") != base_dim:
            raise ValueError("Selector DDI feature sketch width does not match concat base width.")
    elif int(raw_x.size(1)) != base_dim:
        raise ValueError(f"Target feature width is {raw_x.size(1)}, but the selector expects base width {base_dim}.")
    first_weight = state.get("encoder.mlp.0.weight")
    if not torch.is_tensor(first_weight) or first_weight.ndim != 2 or not first_weight.is_floating_point():
        raise ValueError("Selector state is missing encoder.mlp.0.weight.")
    if int(first_weight.size(1)) != output_dim:
        raise ValueError(
            f"Selector state input width {first_weight.size(1)} does not match the expected {label} input width {output_dim}."
        )
    return config, state, base_dim, output_dim


def _preprocess_concat_family_features(
    bundle, dataset, raw_x, training_graph, config, selector_dtype, device
) -> torch.Tensor:
    from model.feature_aggregation import preprocess_aggregated_mlp

    selector_x = raw_x.detach().to(device=device, dtype=selector_dtype, non_blocking=True)
    if selector_x.data_ptr() == raw_x.data_ptr():
        selector_x = selector_x.clone()
    processed = preprocess_aggregated_mlp(
        "concat",
        dataset,
        selector_x,
        training_graph,
        featureless_dim=config.get("concat_feature_dim"),
        featureless_seed=int(config.get("concat_feature_seed", 0)),
    )
    expected = (int(raw_x.size(0)), int(config["concat_output_feature_dim"]))
    if tuple(processed.shape) != expected:
        raise ValueError(f"Selector preprocessing produced shape {tuple(processed.shape)}, expected {expected}.")
    if not bool(torch.isfinite(processed).all().item()):
        raise ValueError("Selector preprocessing produced NaN or infinity.")
    return processed


def _build_concat_family_selector(config, state, dataset, num_nodes, output_dim, device, spec):
    from model.pairwise_models import get_model

    params = {
        **dict(config),
        "in_channels": int(output_dim),
        "num_nodes": int(num_nodes),
        "dataset_name": str(dataset),
        "evaluation_mode": str(spec.get("checkpoint_mode", "heart")),
        "use_node_emb": False,
        "train_samples_per_epoch": int(config.get("train_samples_per_epoch", 0)),
        "stage1_train_samples_per_epoch": int(config.get("stage1_train_samples_per_epoch", 0)),
    }
    model = get_model(str(spec["policy"]), params).to(dtype=_selector_state_dtype(state))
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"Selector state/config mismatch: {exc}") from exc
    if getattr(incompatible, "missing_keys", ()) or getattr(incompatible, "unexpected_keys", ()):
        raise ValueError("Selector state did not load strictly.")
    spec["model_validator"](model)
    model.requires_grad_(False)
    return model.to(device).eval()


def load_or_create_concat_family_test_negatives(
    framework,
    dataset,
    bundle,
    checkpoint_path,
    device,
    *,
    spec,
    k=None,
    score_batch_size=65536,
    cache_dir=None,
    precision_contract=None,
    selector_depth=None,
    selector_hidden_channels=None,
    selector_dropout=None,
    selector_lr=None,
    selector_weight_decay=None,
    _checkpoint_payload=None,
    _selector_sha256=None,
) -> RankedNegativeResult:
    """Generate/load ranked candidates for one declared Concat-family selector."""
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping.")
    policy = str(spec["policy"])
    label = str(spec.get("display_name", policy.upper()))
    precision_contract = str(precision_contract or "caller-unspecified").strip().lower()
    if not precision_contract:
        raise ValueError("precision_contract must not be empty.")
    framework_key = _normal_framework(framework)
    dataset_key = str(dataset).strip().lower()
    if not dataset_key:
        raise ValueError("dataset must not be empty.")
    k = int(spec.get("default_k", 500) if k is None else k)
    global_top_k = bool(spec.get("global_top_k"))
    exact_k = spec.get("exact_k")
    if exact_k is not None and k != int(exact_k):
        raise ValueError(f"{policy} requires exactly k={int(exact_k)}, selected equally per endpoint side.")
    if k <= 0 or (not global_top_k and k % 2):
        requirement = "positive" if global_top_k else "a positive even integer split equally across endpoint sides"
        raise ValueError(f"{label} k must be {requirement}.")
    score_batch_size = int(score_batch_size)
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be a positive integer.")
    device = torch.device(device)
    checkpoint_file = Path(checkpoint_path).expanduser().resolve()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"{label} selector checkpoint not found: {checkpoint_file}")
    selector_sha = str(_selector_sha256 or _sha256_file(checkpoint_file))
    checkpoint = _checkpoint_payload if _checkpoint_payload is not None else _load_torch_payload(checkpoint_file)
    (
        raw_x, num_nodes, queries, query_source, valid_pos, valid_source, test_pos_full, test_source,
        training_graph, train_pos, train_source, directed, filter_layout, eligibility_policy,
    ) = _ranked_selection_scope(
        bundle,
        framework_key,
        dataset_key,
        policy,
        force_node_embeddings=bool(spec.get("force_node_embeddings")),
    )
    if num_nodes <= 1:
        raise ValueError(f"{label} selection requires a graph with at least two nodes.")
    overrides = (selector_depth, selector_hidden_channels, selector_dropout, selector_lr, selector_weight_decay)
    config, state, base_dim, output_dim = _validate_concat_family_checkpoint(
        checkpoint, checkpoint_file, framework_key, dataset_key, raw_x, spec, overrides
    )
    if spec.get("requires_ranked_training_contract"):
        current_validation = neutral_selector_validation_identity(
            bundle,
            framework_key,
            dataset_key,
            seed=int(config["selector_validation_seed"]),
            negatives=int(config["selector_validation_negatives_total"]),
        )
        for key, expected in current_validation.items():
            if config.get(key) != expected:
                raise ValueError(
                    f"{label} selector checkpoint does not match the current validation split: "
                    f"{key}={config.get(key)!r}, expected {expected!r}."
                )
    if global_top_k:
        requested_side_k = effective_side_k = None
        effective_k = _global_top_k_layout(
            queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=k
        )
    else:
        requested_side_k, effective_side_k = _balanced_side_layout(
            queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=k
        )
        required_per_side = spec.get("required_per_side_k")
        if required_per_side is not None and effective_side_k != int(required_per_side):
            raise ValueError(
                f"{policy} requires {int(required_per_side)} legal negatives on each endpoint side; "
                f"the common legal capacity supports only {effective_side_k}."
            )
        effective_k = 2 * effective_side_k
    positive_protocol = spec["citation2_positive_protocol"] if directed else spec["positive_protocol"]
    split_protocol = (
        f"{positive_protocol};train={train_source};valid={valid_source};test={test_source};queries={query_source}"
    )
    identity = {
        "cache_version": int(spec["cache_version"]),
        "selection_protocol": str(spec["selection_protocol"]),
        "positive_protocol": split_protocol,
        "tie_break": str(spec["tie_break"]),
        "framework": framework_key,
        "dataset": dataset_key,
        "selector_sha256": selector_sha,
        "selector_model": policy,
        "selector_checkpoint_training_mode": str(spec.get("checkpoint_mode", "heart")),
        "selector_matmul_precision": precision_contract,
        "selection_scope": "global-across-both-endpoint-sides" if global_top_k else "independent-top-k-per-endpoint-side",
        "requested_k": k,
        "effective_k": effective_k,
        "num_nodes": num_nodes,
        "num_queries": int(queries.size(0)),
        "query_sha256": _sha256_tensor(queries, "ordered-test-queries"),
        "train_positive_sha256": _sha256_tensor(train_pos, "train-positive-edges"),
        "valid_positive_sha256": _sha256_tensor(valid_pos, "valid-positive-edges"),
        "test_positive_sha256": _sha256_tensor(test_pos_full, "test-positive-edges"),
        "selector_recipe": str(config["concat_preprocessing"]),
        "selector_base_feature_dim": base_dim,
        "selector_output_feature_dim": output_dim,
        "eligibility_policy": eligibility_policy,
        "eligibility_scope": filter_layout.filter_scope,
        "eligibility_directionality": filter_layout.directionality,
        "blocked_role_sha256": _sha256_tensor(
            filter_layout.blocked_ids, "authoritative-legal-test-blocked-role-ids"
        ),
        "eligibility_right_role_offset": int(filter_layout.right_role_offset),
    }
    if not spec.get("force_node_embeddings"):
        identity["raw_feature_sha256"] = _sha256_tensor(raw_x, "raw-node-features")
    if not global_top_k:
        identity.update(
            requested_per_side_k=requested_side_k,
            effective_per_side_k=effective_side_k,
        )
    identity_extra = spec.get("identity_extra")
    identity.update(dict(identity_extra(checkpoint, config, framework_key) if callable(identity_extra) else identity_extra or {}))
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    identity["cache_key"] = cache_key
    cache_path = None
    if cache_dir is not None:
        cache_path = str(
            Path(cache_dir).expanduser().resolve()
            / f"{policy}_test_neg_v{int(spec['cache_version'])}_{framework_key}_{_safe_filename(dataset_key)}_{cache_key[:24]}.pt"
        )
    elif int(queries.size(0)) * effective_k * 16 > MAX_UNCACHED_MATERIALIZED_BYTES:
        gib = int(queries.size(0)) * effective_k * 16 / 1024**3
        raise ValueError(
            f"Ranked evaluation without cache_dir would materialize approximately {gib:.2f} GiB; provide a cache directory."
        )

    def metadata():
        result = {
            "cache_key": cache_key,
            "selector_model": policy,
            "positive_protocol": split_protocol,
            "eligibility_policy": eligibility_policy,
            "eligibility_scope": filter_layout.filter_scope,
            "eligibility_directionality": filter_layout.directionality,
            "selection_scope": "global-across-both-endpoint-sides" if global_top_k else "independent-top-k-per-endpoint-side",
            "per_side_quota": effective_side_k,
            "effective_total_across_both_endpoint_sides": effective_k,
            "selector_checkpoint": str(checkpoint_file),
            "selector_checkpoint_run": int(checkpoint.get("run", 0)),
            "selector_depth": int(config["layers"]),
            "selector_predictor_depth": int(config["pred_layers"]),
            "selector_hidden_channels": int(config["emb_size"]),
            "selector_dropout": float(config["dropout"]),
            "selector_lr": float(config["lr"]),
            "selector_weight_decay": float(config["weight_decay"]),
        }
        extra = spec.get("candidate_metadata_extra")
        result.update(dict(extra(checkpoint, config) if callable(extra) else extra or {}))
        return result

    def load_or_generate():
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                return _load_cached_ranked_result(
                    cache_path, identity, queries, num_nodes, effective_k, directed, policy, filter_layout,
                    expected_per_side_k=None if global_top_k else effective_side_k,
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Existing {label} candidate cache is invalid; refusing implicit rebuild: {cache_path}."
                ) from exc
        custom_builder = spec.get("build_selector")
        if callable(custom_builder):
            model = custom_builder(
                config=config,
                state=state,
                dataset=dataset_key,
                num_nodes=num_nodes,
                output_dim=output_dim,
                training_graph=training_graph,
                device=device,
                spec=spec,
            )
            model_input = SimpleNamespace(
                x=raw_x.to(device), adj_t=training_graph, edge_index=training_graph
            )
        else:
            processed_x = _preprocess_concat_family_features(
                bundle, dataset_key, raw_x, training_graph, config, _selector_state_dtype(state), device
            )
            model = _build_concat_family_selector(
                config, state, dataset_key, num_nodes, output_dim, device, spec
            )
            model_input = SimpleNamespace(x=processed_x, adj_t=None, edge_index=None)
        with torch.inference_mode():
            embedding = model.embed(model_input)
        expected_embedding = (num_nodes, int(config["emb_size"]))
        if tuple(embedding.shape) != expected_embedding or not bool(torch.isfinite(embedding).all().item()):
            raise ValueError(f"Selector embedding is invalid; expected shape {expected_embedding}.")
        if cache_path is not None:
            negatives, descriptors = _write_ranked_negative_shards(
                cache_path, queries, filter_layout, embedding, model, num_nodes,
                effective_k, score_batch_size, device, global_top_k=global_top_k,
            )
            result_metadata = metadata()
            storage = _ranked_shard_storage(negatives, descriptors, num_nodes)
            _atomic_save({"metadata": result_metadata, "storage": storage}, cache_path)
            return RankedNegativeResult(negatives, result_metadata, cache_path, False)
        negatives = _select_ranked_queries(
            queries, filter_layout, embedding, model, k, effective_k, score_batch_size, device,
            global_top_k=global_top_k,
        )
        return RankedNegativeResult(negatives, metadata(), None, False)

    if cache_path is None:
        return load_or_generate()
    with _exclusive_cache_lock(cache_path):
        return load_or_generate()

def load_or_create_mlp_family_test_negatives(
    framework: str, dataset: str, bundle: Mapping[str, Any], device: str | torch.device,
    k: int | None = None, score_batch_size: int = 65536,
    cache_dir: str | os.PathLike[str] | None = None, precision_contract: str | None = None,
    seed: int = 42, epochs: int = 300, eval_steps: int = 5, patience: int = 10, batch_size: int = 1024,
    metric: Optional[str] = None, data_root: str | os.PathLike[str] = "dataset", data_seed: int = 0,
    selector_depth: Optional[int] = None,
    selector_hidden_channels: Optional[int] = None, selector_dropout: Optional[float] = None,
    selector_lr: Optional[float] = None, selector_weight_decay: Optional[float] = None,
    spec: Mapping[str, Any] | None = None,
    checkpoint_path: str | os.PathLike[str] | None = None,
    _checkpoint_payload: Optional[Mapping[str, Any]] = None, _selector_sha256: Optional[str] = None,
) -> RankedNegativeResult:
    if not isinstance(bundle, Mapping):
        raise TypeError("bundle must be a mapping.")
    if spec is None:
        raise TypeError("spec is required.")
    framework = _normal_framework(framework)
    dataset = str(dataset).strip().lower()
    frozen = bool(spec.get("frozen"))
    neutral_validation = bool(spec.get("neutral_validation", False))
    global_top_k = bool(spec.get("global_top_k", False))
    selector_model = str(spec["policy"])
    selector_label = str(spec.get("display_name", selector_model.upper()))
    device = torch.device(device)
    k = int(spec.get("default_k", 500)) if k is None else int(k)
    cache_k = max(k, _SHARED_MAX_CACHE_K.get(selector_model, k)) if cache_dir is not None else k
    score_batch_size = int(score_batch_size)
    seed = int(seed)
    epochs = int(epochs)
    eval_steps = int(eval_steps)
    patience = int(patience)
    batch_size = int(batch_size)
    if k <= 0:
        raise ValueError(f"{selector_model} total k must be positive.")
    exact_k = spec.get("exact_k")
    if exact_k is not None and k != int(exact_k):
        raise ValueError(
            f"{selector_model} requires exactly k={int(exact_k)}, selected equally per endpoint side."
        )
    if not global_top_k and k % 2:
        raise ValueError(f"{selector_model} requires an even k split equally per endpoint side.")
    if score_batch_size <= 0:
        raise ValueError("score_batch_size must be positive.")
    metric = str(metric or spec["default_metric"](framework, dataset))
    precision_contract = str(precision_contract or "caller-unspecified").strip().lower()
    if not precision_contract:
        raise ValueError("precision_contract must not be empty.")
    force_node_embeddings = bool(spec.get("force_node_embeddings", False))
    (
        raw_x, num_nodes, queries, query_source, valid_pos, valid_source, full_test_pos, test_source,
        training_graph, train_pos, train_source, directed, filter_layout, eligibility_policy,
    ) = _ranked_selection_scope(
        bundle,
        framework,
        dataset,
        selector_model,
        force_node_embeddings=force_node_embeddings,
    )
    checkpoint_file: Optional[Path] = None
    checkpoint: Optional[Mapping[str, Any]] = None
    state: Optional[Mapping[str, torch.Tensor]] = None
    if frozen:
        checkpoint_file = Path(checkpoint_path).expanduser().resolve()
        if not checkpoint_file.is_file():
            raise FileNotFoundError(f"{selector_model.upper()} selector checkpoint not found: {checkpoint_file}")
        checkpoint_value = _checkpoint_payload
        if checkpoint_value is None:
            checkpoint_value = _load_torch_payload(checkpoint_file)
        config, state = spec["validate"](
            checkpoint_value, framework=framework, dataset=dataset, raw_x=raw_x,
            selector_depth=int(spec["default_depth"]) if selector_depth is None else int(selector_depth),
            selector_hidden_channels=selector_hidden_channels, selector_dropout=selector_dropout,
            selector_lr=selector_lr, selector_weight_decay=selector_weight_decay)
        checkpoint = checkpoint_value
        config_sha = hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()
    else:
        config, config_sha, _ = spec["config_payload"](
            dataset, selector_depth=selector_depth, selector_hidden_channels=selector_hidden_channels,
            selector_dropout=selector_dropout, selector_lr=selector_lr,
            selector_weight_decay=selector_weight_decay)
    if global_top_k:
        requested_side_k = effective_side_k = None
        requested_effective_k = _global_top_k_layout(
            queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=k
        )
        effective_k = (
            requested_effective_k
            if cache_k == k
            else _global_top_k_layout(
                queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=cache_k
            )
        )
    else:
        requested_side_k, effective_side_k = _balanced_side_layout(
            queries=queries, filter_layout=filter_layout, num_nodes=num_nodes, k=k
        )
        required_side_k = int(spec.get("required_per_side_k", int(exact_k if exact_k is not None else k) // 2))
        if requested_side_k != required_side_k or effective_side_k != required_side_k:
            raise ValueError(
                f"{selector_model} requires {required_side_k} legal negatives on each endpoint side for every query; "
                f"the common legal capacity supports only {effective_side_k} per side."
            )
        effective_k = 2 * effective_side_k
        requested_effective_k = effective_k
    positive_protocol = str(spec["citation2_positive_protocol"] if directed else spec["positive_protocol"])
    split_protocol = f"{positive_protocol};train={train_source};valid={valid_source};test={test_source};queries={query_source}"
    validation_source_identity = (
        neutral_selector_validation_identity(
            bundle, framework, dataset, seed=int(data_seed), negatives=int(spec.get("validation_negatives", 500))
        )
        if neutral_validation
        else None
    )
    uses_node_embedding = force_node_embeddings or dataset == "ogbl-ddi"
    blocked_role_sha = _sha256_tensor(filter_layout.blocked_ids, "authoritative-legal-test-blocked-role-ids")
    cache_version = int(spec["cache_version"])
    selection_protocol = str(spec["selection_protocol"])
    training_protocol = str(spec["training_protocol"])
    tie_break = str(spec["tie_break"])
    identity = {
        "cache_version": cache_version,
        "selection_protocol": selection_protocol,
        "training_protocol": training_protocol,
        "positive_protocol": split_protocol,
        "tie_break": tie_break,
        "framework": framework,
        "dataset": dataset,
        "selector_implementation_sha256": (
            spec["implementation_sha256"](framework)
            if frozen
            else spec["implementation_sha256"](framework, selector_depth=selector_depth)
        ),
        "selector_config_sha256": config_sha,
        "selector_config": config,
        "selector_matmul_precision": precision_contract,
        "selection_scope": (
            "global-across-both-endpoint-sides"
            if global_top_k
            else "independent-top-k-per-endpoint-side"
        ),
        "selector_input_protocol": "trainable-node-identity-embedding-v1" if uses_node_embedding else "raw-node-features-v1",
        "eligibility_policy": eligibility_policy,
        "eligibility_scope": filter_layout.filter_scope,
        "eligibility_directionality": filter_layout.directionality,
        "blocked_role_sha256": blocked_role_sha,
        "eligibility_right_role_offset": int(filter_layout.right_role_offset),
        "requested_k": cache_k,
        "effective_k": effective_k,
        "num_nodes": num_nodes,
        "num_queries": int(queries.size(0)),
        "query_sha256": _sha256_tensor(queries, "ordered-test-queries"),
        "raw_feature_sha256": None if uses_node_embedding else _sha256_tensor(raw_x, "mlp-raw-node-features"),
        "train_positive_sha256": _sha256_tensor(train_pos, "train-positive-edges"),
        "valid_positive_sha256": _sha256_tensor(valid_pos, "valid-positive-edges"),
        "test_positive_sha256": _sha256_tensor(full_test_pos, "test-positive-edges"),
    }
    if not global_top_k:
        identity.update(
            requested_per_side_k=requested_side_k,
            effective_per_side_k=effective_side_k,
        )
    if "selector_factory_model" in spec:
        identity["selector_factory_model"] = str(spec["selector_factory_model"])
    if "input_protocol" in spec:
        identity["selector_input_protocol"] = str(spec["input_protocol"])
    if validation_source_identity is not None:
        identity["selector_validation_source_identity"] = validation_source_identity
    training_contract = spec.get("training_contract")
    if training_contract is not None:
        if not isinstance(training_contract, Mapping):
            raise TypeError("Ranked selector training_contract must be a mapping.")
        identity["selector_training_contract_identity"] = dict(training_contract)
    if frozen:
        assert checkpoint_file is not None and checkpoint is not None and state is not None
        identity.update(spec["cache_identity_fields"](
            checkpoint_file, checkpoint, state, framework, requested_side_k, effective_side_k,
            _selector_sha256))
    else:
        identity.update(
            {
                "selector_model": str(spec.get("selector_identity", "fresh-independent-mlp")),
                "selector_seed": seed,
                "selector_epochs": epochs,
                "selector_eval_steps": eval_steps,
                "selector_patience": patience,
                "selector_batch_size": batch_size,
                "selector_validation_metric": metric,
                "selector_data_seed": int(data_seed),
                "selector_data_root": str(Path(data_root).resolve()),
                "selector_validation_source": validation_source_identity["selector_validation_source"],
            }
        )
    legacy_identity = None
    if cache_k != k:
        legacy_identity = dict(identity)
        legacy_identity.update(requested_k=k, effective_k=requested_effective_k)
        legacy_key = hashlib.sha256(
            json.dumps(legacy_identity, sort_keys=True, separators=(",", ":")).encode("utf8")
        ).hexdigest()
        legacy_identity["cache_key"] = legacy_key
    cache_key = hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf8")).hexdigest()
    identity["cache_key"] = cache_key
    cache_path: Optional[str] = None
    legacy_cache_path: Optional[str] = None
    if cache_dir is not None:
        directory = Path(cache_dir).expanduser().resolve()
        cache_path = str(directory / f"{selector_model}_test_neg_v{cache_version}_{framework}_{_safe_filename(dataset)}_{cache_key[:24]}.pt")
        if legacy_identity is not None:
            legacy_cache_path = str(
                directory
                / f"{selector_model}_test_neg_v{cache_version}_{framework}_{_safe_filename(dataset)}_{legacy_key[:24]}.pt"
            )
    elif int(queries.size(0)) * effective_k * 2 * 8 > MAX_UNCACHED_MATERIALIZED_BYTES:
        raise ValueError(f"Large {selector_label} evaluation requires cache_dir for streaming negatives.")

    if frozen:
        assert checkpoint_file is not None and checkpoint is not None and state is not None
        selector_provenance = spec["selector_provenance"](checkpoint_file, checkpoint, state, framework, config_sha)
    else:
        selector_provenance = None

    def metadata(training: Mapping[str, Any]) -> dict[str, Any]:
        return {
            **dict(training),
            "cache_key": cache_key,
            "selector_model": selector_model,
            "positive_protocol": split_protocol,
            "eligibility_policy": eligibility_policy,
            "eligibility_scope": filter_layout.filter_scope,
            "eligibility_directionality": filter_layout.directionality,
            "selection_scope": (
                "global-across-both-endpoint-sides"
                if global_top_k
                else "independent-top-k-per-endpoint-side"
            ),
            "per_side_quota": None if global_top_k else int(effective_side_k),
            "effective_total_across_both_endpoint_sides": effective_k,
        }

    def serve(result: RankedNegativeResult) -> RankedNegativeResult:
        if cache_k == k:
            return result
        return _shared_cache_prefix(
            result,
            requested_k=k,
            effective_k=requested_effective_k,
            cache_k=cache_k,
        )

    def load_or_generate() -> RankedNegativeResult:
        if cache_path is not None and os.path.isfile(cache_path):
            try:
                return serve(
                    _load_cached_ranked_result(
                        cache_path, identity, queries, num_nodes, effective_k, directed, selector_model, filter_layout,
                        expected_per_side_k=None if global_top_k else effective_side_k,
                    )
                )
            except Exception as exc:
                raise RuntimeError(
                    f"Existing {selector_label} candidate cache is invalid; refusing implicit selector retraining/rebuild: "
                    f"{cache_path}. Remove or quarantine it explicitly before requesting a cold rebuild."
                ) from exc
        if legacy_cache_path is not None and (
            os.path.isfile(legacy_cache_path) or os.path.exists(f"{legacy_cache_path}.lock")
        ):
            with _exclusive_cache_lock(legacy_cache_path):
                if os.path.isfile(legacy_cache_path):
                    try:
                        return _load_cached_ranked_result(
                            legacy_cache_path,
                            legacy_identity,
                            queries,
                            num_nodes,
                            requested_effective_k,
                            directed,
                            selector_model,
                            filter_layout,
                        )
                    except Exception as exc:
                        raise RuntimeError(
                            f"Existing {selector_label} legacy candidate cache is invalid: {legacy_cache_path}."
                        ) from exc
        if frozen:
            assert state is not None and selector_provenance is not None
            model = spec["build"](config=config, state=state, dataset=dataset, num_nodes=num_nodes,
                                      raw_feature_dim=int(raw_x.size(1)), device=device)
            generation_provenance = selector_provenance
        else:
            train_kwargs = {
                "seed": seed,
                "epochs": epochs,
                "eval_steps": eval_steps,
                "patience": patience,
                "batch_size": batch_size,
                "metric": metric,
                "data_root": data_root,
                "data_seed": int(data_seed),
                "selector_depth": selector_depth,
                "selector_hidden_channels": selector_hidden_channels,
                "selector_dropout": selector_dropout,
                "selector_lr": selector_lr,
                "selector_weight_decay": selector_weight_decay,
            }
            trained = spec["train_selector"](framework, dataset, bundle, device, **train_kwargs)
            model = trained.model
            generation_provenance = trained.metadata
        raw_device_x = raw_x.to(device)
        model_input = SimpleNamespace(
            x=raw_device_x,
            adj_t=training_graph,
            edge_index=training_graph,
        )
        with torch.inference_mode():
            embedding = model.embed(model_input)
        if int(embedding.size(0)) != num_nodes or not bool(torch.isfinite(embedding).all().item()):
            raise ValueError(f"{selector_model.upper()} selector produced invalid node embeddings.")
        if cache_path is not None:
            negatives, descriptors = _write_ranked_negative_shards(
                cache_path, queries, filter_layout, embedding, model, num_nodes,
                effective_k, score_batch_size, device, global_top_k=global_top_k,
            )
            cache_metadata = metadata(generation_provenance)
            storage = _ranked_shard_storage(negatives, descriptors, num_nodes)
            _atomic_save({"metadata": cache_metadata, "storage": storage}, cache_path)
            return serve(RankedNegativeResult(negatives, cache_metadata, cache_path, False))
        negatives = _select_ranked_queries(
            queries, filter_layout, embedding, model, k, effective_k, score_batch_size, device,
            global_top_k=global_top_k,
        )
        return serve(RankedNegativeResult(negatives, metadata(generation_provenance), None, False))

    if cache_path is None:
        return load_or_generate()
    with _exclusive_cache_lock(cache_path):
        return load_or_generate()
