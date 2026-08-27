"""Standalone evaluator dispatch for frozen Concat-ranked candidates."""

from __future__ import annotations

import sys
from functools import partial
from typing import Mapping

from eval_modes import ranked_helpers as _ranked


POLICY = "concat"
EVALUATION_PROTOCOL = "exact_grouped_concat_neutral_selector_global_top_k_negatives"
RESULT_FILENAME = "concat_candidate_evaluation.json"
FORMAT_VERSION = 8
GROUPED = True
TEST_ONLY = True
ORIENT_QUERY_ENDPOINTS = True
USES_HEART_ARTIFACTS = False
TARGET_CHECKPOINT_MODE = None
DEFAULT_TARGET_CHECKPOINT_MODE = "heart"
SELECTOR_CHECKPOINT_MODE = "ranked-selector"
CANDIDATE_LABEL = f"{POLICY}-selected"
CONCAT_CACHE_VERSION = 8
CONCAT_SELECTION_PROTOCOL = "global-two-endpoint-top-k-v2"
CONCAT_POSITIVE_PROTOCOL = "query-role-authoritative-legal-test-filter-query-excluded-v4"
CONCAT_CITATION2_POSITIVE_PROTOCOL = "query-role-authoritative-directed-legal-test-filter-query-excluded-v3"
CONCAT_TIE_BREAK = "global-score-desc-side-asc-node-asc-v2"
CONCAT_ARTIFACT_TAG = "neutral_validation_strict_bce_global_top_k_v1"
CONCAT_CHECKPOINT_PROTOCOL = {
    "seed": 0,
    "run": 1,
    "epochs": 500,
    "eval_steps": 5,
    "patience": 10,
    "num_runs": 1,
}

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
    expected_validation = {
        "selector_validation_protocol": _ranked.NEUTRAL_SELECTOR_VALIDATION_PROTOCOL,
        "selector_validation_source": "deterministic-neutral-legal-filter",
        "selector_validation_split": "valid",
        "selector_validation_seed": 0,
        "selector_validation_negatives_total": 500,
        "selector_validation_negatives_per_side": 250,
    }
    config, _ = _ranked.validate_frozen_checkpoint_contract(
        checkpoint,
        path,
        framework,
        dataset,
        POLICY,
        expected_run,
        mode=SELECTOR_CHECKPOINT_MODE,
        schedule=CONCAT_CHECKPOINT_PROTOCOL,
        config_contract=expected_validation,
        overrides=(expected_depth, expected_hidden_channels, expected_dropout, expected_lr, expected_weight_decay),
    )
    _ranked.validate_ranked_selector_training_config(config, f"Concat selector {path}")
    expected_directionality = (
        "directed" if str(dataset).strip().lower() == "ogbl-citation2" else "canonical-undirected"
    )
    if config.get("selector_training_directionality") != expected_directionality:
        raise ValueError(
            f"Concat selector {path} has selector_training_directionality="
            f"{config.get('selector_training_directionality')!r}, expected {expected_directionality!r}."
        )
    missing_validation = [key for key in _ranked.NEUTRAL_SELECTOR_VALIDATION_CONTRACT_FIELDS if key not in config]
    if missing_validation:
        raise ValueError(f"Concat selector {path} is missing neutral validation provenance: {', '.join(missing_validation)}")
    for key in (
        "selector_validation_legality_policy",
        "selector_validation_filter_scope",
        "selector_validation_directionality",
    ):
        if not isinstance(config.get(key), str) or not config[key].strip():
            raise ValueError(f"Concat selector {path} has invalid {key}.")
    for key in (
        "selector_validation_query_sha256",
        "selector_validation_blocked_role_sha256",
        "selector_validation_candidate_nodes_sha256",
    ):
        value = config.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"Concat selector {path} has invalid {key}.")


def _validate_selector_model(model):
    if getattr(model, "decode_is_symmetric", None) is not True:
        raise ValueError("Concat selection requires a selector that declares a symmetric decoder.")


def _training_identity(_checkpoint, config, _framework):
    fields = (
        *_ranked.RANKED_SELECTOR_CHECKPOINT_FIELDS,
        "selector_training_directionality",
        *_ranked.NEUTRAL_SELECTOR_VALIDATION_CONTRACT_FIELDS,
    )
    return {key: config[key] for key in fields}


MODE_SPEC = {
    "policy": POLICY,
    "display_name": "Concat",
    "validator": validate_selector_checkpoint,
    "model_validator": _validate_selector_model,
    "default_k": 500,
    "global_top_k": True,
    "checkpoint_mode": SELECTOR_CHECKPOINT_MODE,
    "training_negatives_argument": None,
    "default_cache_dir": "concat_sets",
    "artifact_tag": CONCAT_ARTIFACT_TAG,
    "candidate_protocol": "frozen-neutral-concat-global-top-k-across-endpoint-sides",
    "candidate_selection": "neutral_concat_selector_score_global_top_k_across_endpoint_sides",
    "candidate_universe": "all-legal-endpoint-corruptions",
    "cache_version": CONCAT_CACHE_VERSION,
    "selection_protocol": CONCAT_SELECTION_PROTOCOL,
    "positive_protocol": CONCAT_POSITIVE_PROTOCOL,
    "citation2_positive_protocol": CONCAT_CITATION2_POSITIVE_PROTOCOL,
    "tie_break": CONCAT_TIE_BREAK,
    "minimum_predictor_depth": 1,
    "requires_ranked_training_contract": True,
    "identity_extra": _training_identity,
    "negative_sha_label": "concat-negatives",
}


load_or_create_concat_test_negatives = partial(
    _ranked.load_or_create_concat_family_test_negatives,
    spec=MODE_SPEC,
)

def add_evaluator_arguments(parser):
    _ranked.add_ranked_arguments(parser, spec=MODE_SPEC)
    parser.set_defaults(mode=DEFAULT_TARGET_CHECKPOINT_MODE)
    return parser


validate_cli = partial(_ranked.validate_ranked_cli, spec=MODE_SPEC)
evaluator_context = partial(_ranked.ranked_evaluator_context, spec=MODE_SPEC)
data_seed = _ranked.ranked_data_seed
resolve_cap = partial(_ranked.ranked_resolve_cap, spec=MODE_SPEC)
load_bundle = _ranked.load_ranked_bundle
cache_key = partial(_ranked.ranked_cache_key, spec=MODE_SPEC)
prepare_bundle = _ranked.prepare_ranked_bundle
install_evaluator_candidates = partial(
    _ranked.install_ranked_candidates, spec=MODE_SPEC, loader=load_or_create_concat_test_negatives
)
selector_metadata = partial(_ranked.ranked_selector_metadata, spec=MODE_SPEC)
candidate_pool = partial(_ranked.ranked_candidate_pool, spec=MODE_SPEC)
result_fields = partial(_ranked.result_fields, policy=POLICY)
output_root = partial(_ranked.output_root, policy=POLICY)


def main():
    from eval_modes.evaluator_helpers import main_for_mode

    return main_for_mode(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
