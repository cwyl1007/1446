"""Independent saved-checkpoint evaluator dispatch for MLPIP candidates."""

from __future__ import annotations

import hashlib
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from eval_modes import ranked_helpers as _ranked
from eval_modes.ranked_helpers import _validate_inner_product_selector


POLICY = "mlpip"
EVALUATION_PROTOCOL = "exact_grouped_mlpip_inner_product_fixed_250_per_side_negatives"
RESULT_FILENAME = "mlpip_candidate_evaluation.json"
FORMAT_VERSION = 7
GROUPED = True
TEST_ONLY = True
ORIENT_QUERY_ENDPOINTS = True
USES_HEART_ARTIFACTS = False
TARGET_CHECKPOINT_MODE = "heart"
CANDIDATE_LABEL = f"{POLICY}-selected"
ARTIFACT_TAG = "shared_ip_strict_bce_v1"
CACHE_VERSION = 2
SELECTION_PROTOCOL = "frozen-mlpip-best-validation-inner-product-fixed-250-per-endpoint-side-v2"
TIE_BREAK = "per-side-inner-product-desc-node-asc-left-before-right-v1"
SELECTOR_TRAINING_PROTOCOL = "frozen-mlpip-best-validation-checkpoint-inner-product-v1"
SCORING_PROTOCOL = "inner-product-logit-ranking-with-sigmoid-output-v1"
CHECKPOINT_PROTOCOL = {
    "seed": 0,
    "run": 1,
    "epochs": 500,
    "eval_steps": 5,
    "patience": 10,
    "num_runs": 1,
    "heart_negatives": 500,
}
TRAINING_CONTRACT = _ranked.IP_TRAINING_CONTRACT

_POSITIVE_PROTOCOL = "authoritative-heart-test-filter-undirected-query-only-test-v1"
_CITATION2_POSITIVE_PROTOCOL = "authoritative-heart-test-filter-directed-query-only-test-v1"


def _validated_checkpoint(
    checkpoint,
    path,
    framework,
    dataset,
    expected_run,
    expected_depth=None,
    expected_hidden_channels=None,
    expected_dropout=None,
    expected_lr=None,
    expected_weight_decay=None,
    *,
    raw_x: Optional[torch.Tensor] = None,
):
    required_depth = 2 if expected_depth is None else int(expected_depth)
    if required_depth != 2:
        raise ValueError("MLPIP implements the exact two-layer encoder contract; selector depth must be 2.")
    del raw_x
    config, state = _ranked.validate_frozen_checkpoint_contract(
        checkpoint,
        path,
        framework,
        dataset,
        POLICY,
        expected_run,
        schedule=CHECKPOINT_PROTOCOL,
        config_contract={**_ranked.IP_MODEL_CONTRACT, **TRAINING_CONTRACT},
        predictor_depth=0,
        overrides=(required_depth, expected_hidden_channels, expected_dropout, expected_lr, expected_weight_decay),
    )
    if any("decoder" in str(key).split(".") or "predictor" in str(key).split(".") for key in state):
        raise ValueError("MLPIP selector checkpoint contains learned decoder/predictor parameters.")
    return config, state


def validate_selector_checkpoint(
    checkpoint,
    path,
    framework,
    dataset,
    expected_run,
    expected_depth=None,
    expected_hidden_channels=None,
    expected_dropout=None,
    expected_lr=None,
    expected_weight_decay=None,
):
    """Validate the exact frozen MLPIP header, architecture, and training contract."""
    _validated_checkpoint(
        checkpoint,
        path,
        framework,
        dataset,
        expected_run,
        expected_depth,
        expected_hidden_channels,
        expected_dropout,
        expected_lr,
        expected_weight_decay,
    )


def _validate_frozen_mlpip_checkpoint(
    checkpoint,
    *,
    framework,
    dataset,
    raw_x,
    selector_depth=2,
    selector_hidden_channels=256,
    selector_dropout=None,
    selector_lr=None,
    selector_weight_decay=None,
):
    return _validated_checkpoint(
        checkpoint,
        "in-memory checkpoint",
        framework,
        dataset,
        CHECKPOINT_PROTOCOL["run"],
        selector_depth,
        selector_hidden_channels,
        selector_dropout,
        selector_lr,
        selector_weight_decay,
        raw_x=raw_x,
    )


def _build_frozen_mlpip_selector(*, config, state, dataset, num_nodes, raw_feature_dim, device):
    from model.pairwise_models import get_model

    params = {
        **dict(config),
        "in_channels": int(raw_feature_dim),
        "num_nodes": int(num_nodes),
        "dataset_name": dataset,
        "evaluation_mode": "heart",
        "use_node_emb": dataset == "ogbl-ddi",
        "train_samples_per_epoch": int(config.get("train_samples_per_epoch", 0)),
    }
    model = get_model(POLICY, params)
    try:
        incompatible = model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise ValueError(f"MLPIP selector state/config mismatch: {exc}") from exc
    if getattr(incompatible, "missing_keys", ()) or getattr(incompatible, "unexpected_keys", ()):
        raise ValueError("MLPIP selector state did not load strictly.")
    _validate_inner_product_selector(model, POLICY)
    model.requires_grad_(False)
    return model.to(device).eval()


def _selector_implementation_sha256(framework):
    from utils.cache_compat import relocated_mlpip_selector_fingerprint

    raw_digest = _ranked.selector_runtime_sha256(framework)
    digest = hashlib.sha256(f"{SCORING_PROTOCOL}:{raw_digest}".encode("utf8")).hexdigest()
    return relocated_mlpip_selector_fingerprint(framework, digest)


def _cache_identity_fields(
    checkpoint_path,
    checkpoint,
    state,
    framework,
    requested_per_side_k,
    effective_per_side_k,
    selector_sha256=None,
):
    return {
        "selector_checkpoint_sha256": selector_sha256 or _ranked._sha256_file(checkpoint_path),
        "selector_checkpoint_state_sha256": _ranked.selector_state_sha256(state, "mlp-selector-state"),
        "selector_checkpoint_run": CHECKPOINT_PROTOCOL["run"],
        **TRAINING_CONTRACT,
        "requested_per_side_k": int(requested_per_side_k),
        "effective_per_side_k": int(effective_per_side_k),
        "selector_scoring_protocol": SCORING_PROTOCOL,
    }


def _selector_provenance(checkpoint_path, checkpoint, state, framework, config_sha):
    return {
        "training_protocol": SELECTOR_TRAINING_PROTOCOL,
        "selector_model": POLICY,
        "selector_checkpoint": str(checkpoint_path),
        "selector_checkpoint_run": int(checkpoint.get("run", 0)),
        "selector_seed": int(checkpoint.get("seed", 0)),
        **TRAINING_CONTRACT,
        "selector_best_epoch": int(checkpoint.get("epoch", 0)),
        "selector_validation_metric": str(checkpoint.get("selection_metric", "")),
        "selector_best_validation_value": checkpoint.get("best_validation_metric"),
        "selector_state_sha256": _ranked.selector_state_sha256(state, "mlp-selector-state"),
        "selector_config_sha256": config_sha,
        "selector_scoring_protocol": SCORING_PROTOCOL,
    }


def _default_metric(framework, dataset):
    return _ranked.default_selector_metric(framework, dataset)


_MLPIP_FAMILY_SPEC = {
    "policy": POLICY,
    "frozen": True,
    "display_name": "MLPIP",
    "default_k": 500,
    "exact_k": 500,
    "default_depth": 2,
    "cache_version": CACHE_VERSION,
    "selection_protocol": SELECTION_PROTOCOL,
    "training_protocol": SELECTOR_TRAINING_PROTOCOL,
    "positive_protocol": _POSITIVE_PROTOCOL,
    "citation2_positive_protocol": _CITATION2_POSITIVE_PROTOCOL,
    "tie_break": TIE_BREAK,
    "negative_sha_label": "mlpip-balanced-inner-product-negatives",
    "default_metric": _default_metric,
    "implementation_sha256": _selector_implementation_sha256,
    "validate": _validate_frozen_mlpip_checkpoint,
    "build": _build_frozen_mlpip_selector,
    "cache_identity_fields": _cache_identity_fields,
    "selector_provenance": _selector_provenance,
    "candidate_protocol": "frozen-mlpip-inner-product-sigmoid-fixed-250-per-endpoint-side",
    "candidate_selection": "mlpip_inner_product_logit_top250_independently_per_side",
    "fixed_depth": 2,
    "default_cache_dir": "mlpip_sets",
    "require_output_width": True,
    "artifact_tag": ARTIFACT_TAG,
}


def load_or_create_mlpip_test_negatives(
    framework: str,
    dataset: str,
    bundle: Mapping[str, Any],
    checkpoint_path: str | os.PathLike[str],
    device: str | torch.device | None = None,
    k: int | None = 500,
    score_batch_size: int = 65536,
    cache_dir: str | os.PathLike[str] | None = None,
    precision_contract: str | None = None,
    selector_depth: Optional[int] = 2,
    selector_hidden_channels: Optional[int] = 256,
    selector_dropout: Optional[float] = None,
    selector_lr: Optional[float] = None,
    selector_weight_decay: Optional[float] = None,
    _checkpoint_payload: Optional[Mapping[str, Any]] = None,
    _selector_sha256: Optional[str] = None,
):
    if k is not None and int(k) != 500:
        raise ValueError("mlpip requires exactly k=500, selected as 250 negatives per endpoint side.")
    if device is None:
        raise TypeError("load_or_create_mlpip_test_negatives() requires a device after checkpoint_path.")
    return _ranked.load_or_create_mlp_family_test_negatives(
        framework,
        dataset,
        bundle,
        device,
        k=k,
        score_batch_size=score_batch_size,
        cache_dir=cache_dir,
        precision_contract=precision_contract,
        selector_depth=selector_depth,
        selector_hidden_channels=selector_hidden_channels,
        selector_dropout=selector_dropout,
        selector_lr=selector_lr,
        selector_weight_decay=selector_weight_decay,
        spec=_MLPIP_FAMILY_SPEC,
        checkpoint_path=checkpoint_path,
        _checkpoint_payload=_checkpoint_payload,
        _selector_sha256=_selector_sha256,
    )


_MLPIP_FAMILY_SPEC["selector_metadata_extra"] = {
    "decoder_type": "inner-product",
    "predictor_depth": 0,
    "probability_transform": "sigmoid",
    "training_contract": dict(TRAINING_CONTRACT),
}
add_evaluator_arguments = partial(_ranked.add_ranked_arguments, spec=_MLPIP_FAMILY_SPEC)
validate_cli = partial(_ranked.validate_ranked_cli, spec=_MLPIP_FAMILY_SPEC)
evaluator_context = partial(_ranked.ranked_evaluator_context, spec=_MLPIP_FAMILY_SPEC)
data_seed = _ranked.ranked_data_seed
resolve_cap = partial(_ranked.ranked_resolve_cap, spec=_MLPIP_FAMILY_SPEC)
load_bundle = _ranked.load_ranked_bundle
cache_key = partial(_ranked.ranked_cache_key, spec=_MLPIP_FAMILY_SPEC)
prepare_bundle = _ranked.prepare_ranked_bundle
install_evaluator_candidates = partial(
    _ranked.install_ranked_candidates, spec=_MLPIP_FAMILY_SPEC, loader=load_or_create_mlpip_test_negatives
)
selector_metadata = partial(_ranked.ranked_selector_metadata, spec=_MLPIP_FAMILY_SPEC)
candidate_pool = partial(_ranked.ranked_candidate_pool, spec=_MLPIP_FAMILY_SPEC)
result_fields = partial(_ranked.result_fields, policy=POLICY)
output_root = partial(_ranked.output_root, policy=POLICY, artifact_tag=ARTIFACT_TAG)


def main():
    from eval_modes.evaluator_helpers import main_for_mode

    return main_for_mode(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
