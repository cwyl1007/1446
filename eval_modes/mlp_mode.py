"""Independent fresh-MLP evaluator dispatch."""

from __future__ import annotations

import os
import sys
from functools import partial
from typing import Any, Mapping, Optional

import torch

from eval_modes import fresh_mlp_helpers as _fresh
from eval_modes import ranked_helpers as _ranked


POLICY = "mlp"
EVALUATION_PROTOCOL = "exact_grouped_mlp_fresh_selector_fixed_250_per_endpoint_side_negatives"
RESULT_FILENAME = "mlp_candidate_evaluation.json"
FORMAT_VERSION = 7
GROUPED = True
TEST_ONLY = True
ORIENT_QUERY_ENDPOINTS = True
USES_HEART_ARTIFACTS = False
TARGET_CHECKPOINT_MODE = None
DEFAULT_TARGET_CHECKPOINT_MODE = "heart"
CANDIDATE_LABEL = f"{POLICY}-selected"
MLP_CACHE_VERSION = 9
MLP_SELECTION_PROTOCOL = "fresh-mlp-best-validation-fixed-250-per-endpoint-side-v1"
MLP_POSITIVE_PROTOCOL = "authoritative-legal-test-filter-undirected-query-only-test-v2"
MLP_CITATION2_POSITIVE_PROTOCOL = "authoritative-legal-test-filter-directed-query-only-test-v2"
MLP_TIE_BREAK = "score-desc-node-asc-within-each-endpoint-side"
MLP_TRAINING_PROTOCOL = "ranked-selector-strict-bce-neutral-legal-validation-v1"
MLP_TRAINING_CONTRACT = {
    "protocol": MLP_TRAINING_PROTOCOL,
    "loss": _ranked.RANKED_SELECTOR_LOSS_PROTOCOL,
    "negative_protocol": _ranked.RANKED_SELECTOR_NEGATIVE_PROTOCOL,
    "reject_self_loops": True,
    "reject_train_positives": True,
    "filter_validation_positives": False,
    "filter_test_positives": False,
    "directionality": "directed-for-ogbl-citation2-otherwise-canonical-undirected",
    "ogbl_negative_sampler": "fast",
    "ogbl_training_path": "full-graph",
    "ogbl_epoch_seed": "selector-seed+epoch*1009",
    "validation_source": _ranked.NEUTRAL_SELECTOR_VALIDATION_PROTOCOL,
    "reference_probability_loss": False,
    "reference_random_endpoint_negatives": False,
    "reference_optimizer": False,
}
MLP_DEFAULT_TOTAL_NEGATIVES = 500
MLP_DEFAULT_SELECTOR_SEED = 0
MLP_DEFAULT_SELECTOR_EPOCHS = 500
FreshMlpSelector = _fresh.FreshMlpSelector
_edge_rows = _ranked._edge_rows
_normal_framework = _ranked._normal_framework
_raw_features = _ranked._raw_features


def _set_seed(seed: int) -> None:
    _fresh.set_seed(seed)


def _config_payload(
    dataset: str,
    *,
    selector_depth: Optional[int] = None,
    selector_hidden_channels: Optional[int] = None,
    selector_dropout: Optional[float] = None,
    selector_lr: Optional[float] = None,
    selector_weight_decay: Optional[float] = None,
):
    return _fresh.config_payload(
        dataset,
        selector_depth=selector_depth,
        selector_hidden_channels=selector_hidden_channels,
        selector_dropout=selector_dropout,
        selector_lr=selector_lr,
        selector_weight_decay=selector_weight_decay,
    )


def _raw_selector_implementation_sha256(framework: str) -> str:
    return _ranked.selector_runtime_sha256(framework)


def _selector_implementation_sha256(framework: str, selector_depth: Optional[int] = None) -> str:
    from utils.cache_compat import relocated_mlp_selector_fingerprint

    return relocated_mlp_selector_fingerprint(
        framework,
        _raw_selector_implementation_sha256(framework),
        selector_depth=selector_depth,
    )


def _metric_key(results: Mapping[str, Any], requested: str) -> str:
    return _fresh._metric_key(results, requested)


def _default_metric(framework: str, dataset: str) -> str:
    return _ranked.default_selector_metric(framework, dataset)


def _fresh_model(*, framework, dataset, bundle, config, device):
    return _fresh.fresh_model(
        framework=framework,
        dataset=dataset,
        bundle=bundle,
        config=config,
        device=device,
        selector_policy=POLICY,
        selector_factory_model="mlp",
    )


def _selector_training_bundle(framework, dataset, bundle, *, data_seed):
    return _fresh.selector_training_bundle(
        framework,
        dataset,
        bundle,
        data_seed=data_seed,
        selector_policy=POLICY,
    )


def _train_fresh_mlp_selector(
    framework: str,
    dataset: str,
    bundle: Mapping[str, Any],
    device: str | torch.device,
    *,
    seed: int = MLP_DEFAULT_SELECTOR_SEED,
    epochs: int = MLP_DEFAULT_SELECTOR_EPOCHS,
    eval_steps: int = 5,
    patience: int = 10,
    batch_size: int = 1024,
    metric: Optional[str] = None,
    data_root: str | os.PathLike[str] = "dataset",
    data_seed: int = 0,
    selector_depth: Optional[int] = None,
    selector_hidden_channels: Optional[int] = None,
    selector_dropout: Optional[float] = None,
    selector_lr: Optional[float] = None,
    selector_weight_decay: Optional[float] = None,
) -> FreshMlpSelector:
    return _fresh.train_fresh_mlp_selector(
        framework,
        dataset,
        bundle,
        device,
        selector_policy=POLICY,
        selector_factory_model="mlp",
        training_protocol=MLP_TRAINING_PROTOCOL,
        config_payload_fn=_config_payload,
        fresh_model_fn=_fresh_model,
        training_bundle_fn=_selector_training_bundle,
        seed=seed,
        epochs=epochs,
        eval_steps=eval_steps,
        patience=patience,
        batch_size=batch_size,
        metric=metric,
        data_root=data_root,
        data_seed=data_seed,
        selector_depth=selector_depth,
        selector_hidden_channels=selector_hidden_channels,
        selector_dropout=selector_dropout,
        selector_lr=selector_lr,
        selector_weight_decay=selector_weight_decay,
    )


_MLP_FAMILY_SPEC = {
    "policy": POLICY,
    "frozen": False,
    "display_name": "MLP",
    "default_k": MLP_DEFAULT_TOTAL_NEGATIVES,
    "exact_k": MLP_DEFAULT_TOTAL_NEGATIVES,
    "required_per_side_k": MLP_DEFAULT_TOTAL_NEGATIVES // 2,
    "require_even_k": True,
    "balanced_per_side": True,
    "cache_version": MLP_CACHE_VERSION,
    "selection_protocol": MLP_SELECTION_PROTOCOL,
    "training_protocol": MLP_TRAINING_PROTOCOL,
    "training_contract": MLP_TRAINING_CONTRACT,
    "positive_protocol": MLP_POSITIVE_PROTOCOL,
    "citation2_positive_protocol": MLP_CITATION2_POSITIVE_PROTOCOL,
    "tie_break": MLP_TIE_BREAK,
    "negative_sha_label": "mlp-balanced-per-side-negatives",
    "default_metric": _default_metric,
    "config_payload": _config_payload,
    "implementation_sha256": _selector_implementation_sha256,
    "neutral_validation": True,
    "validation_negatives": 500,
    "candidate_universe": "all-legal-endpoint-corruptions",
    "train_selector": _train_fresh_mlp_selector,
    "candidate_protocol": "fresh-mlp-fixed-250-per-endpoint-side",
    "candidate_selection": "fresh_mlp_score_independent_top250_per_endpoint_side",
    "selector_identity": "fresh-independent-mlp",
    "artifact_tag": "strict_bce_fixed_250_neutral_validation_v1",
    "default_seed": MLP_DEFAULT_SELECTOR_SEED,
    "default_epochs": MLP_DEFAULT_SELECTOR_EPOCHS,
    "default_cache_dir": "mlp_sets",
    "require_output_width": True,
}


def load_or_create_mlp_test_negatives(
    framework: str,
    dataset: str,
    bundle: Mapping[str, Any],
    device: str | torch.device,
    k: int | None = None,
    score_batch_size: int = 65536,
    cache_dir: str | os.PathLike[str] | None = None,
    precision_contract: str | None = None,
    seed: int = MLP_DEFAULT_SELECTOR_SEED,
    epochs: int = MLP_DEFAULT_SELECTOR_EPOCHS,
    eval_steps: int = 5,
    patience: int = 10,
    batch_size: int = 1024,
    metric: Optional[str] = None,
    data_root: str | os.PathLike[str] = "dataset",
    data_seed: int = 0,
    selector_depth: Optional[int] = None,
    selector_hidden_channels: Optional[int] = None,
    selector_dropout: Optional[float] = None,
    selector_lr: Optional[float] = None,
    selector_weight_decay: Optional[float] = None,
):
    return _ranked.load_or_create_mlp_family_test_negatives(
        framework,
        dataset,
        bundle,
        device,
        k=k,
        score_batch_size=score_batch_size,
        cache_dir=cache_dir,
        precision_contract=precision_contract,
        seed=seed,
        epochs=epochs,
        eval_steps=eval_steps,
        patience=patience,
        batch_size=batch_size,
        metric=metric,
        data_root=data_root,
        data_seed=data_seed,
        selector_depth=selector_depth,
        selector_hidden_channels=selector_hidden_channels,
        selector_dropout=selector_dropout,
        selector_lr=selector_lr,
        selector_weight_decay=selector_weight_decay,
        spec=_MLP_FAMILY_SPEC,
    )


def validate_selector_checkpoint(checkpoint, path, framework, dataset, expected_run, *overrides):
    del checkpoint, path, framework, dataset, expected_run, overrides
    raise ValueError("The MLP evaluator trains a fresh selector and does not accept a selector checkpoint.")


def add_evaluator_arguments(parser):
    _ranked.add_ranked_arguments(parser, spec=_MLP_FAMILY_SPEC, fresh=True)
    parser.set_defaults(mode=DEFAULT_TARGET_CHECKPOINT_MODE)
    return parser


_FRESH = {"spec": _MLP_FAMILY_SPEC, "fresh": True}
validate_cli = partial(_ranked.validate_ranked_cli, **_FRESH)
evaluator_context = partial(_ranked.ranked_evaluator_context, **_FRESH)
data_seed = partial(_ranked.ranked_data_seed, fresh=True)
resolve_cap = partial(_ranked.ranked_resolve_cap, **_FRESH)
load_bundle = _ranked.load_ranked_bundle
cache_key = partial(_ranked.ranked_cache_key, **_FRESH)
prepare_bundle = partial(_ranked.prepare_ranked_bundle, fresh=True)
install_evaluator_candidates = partial(
    _ranked.install_ranked_candidates,
    spec=_MLP_FAMILY_SPEC,
    loader=load_or_create_mlp_test_negatives,
    fresh=True,
)
selector_metadata = partial(_ranked.ranked_selector_metadata, **_FRESH)
candidate_pool = partial(_ranked.ranked_candidate_pool, spec=_MLP_FAMILY_SPEC)
result_fields = partial(_ranked.result_fields, policy=POLICY)
output_root = partial(
    _ranked.output_root,
    policy=POLICY,
    artifact_tag=_MLP_FAMILY_SPEC["artifact_tag"],
)


def main():
    from eval_modes.evaluator_helpers import main_for_mode

    return main_for_mode(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
