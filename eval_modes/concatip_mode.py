"""Independent saved-checkpoint evaluator dispatch for ConcatIP candidates."""

from __future__ import annotations

import sys
from functools import partial
from eval_modes import ranked_helpers as _ranked


POLICY = "concatip"
EVALUATION_PROTOCOL = "exact_grouped_concatip_inner_product_fixed_250_per_side_negatives"
RESULT_FILENAME = "concatip_candidate_evaluation.json"
FORMAT_VERSION = 7
GROUPED = True
TEST_ONLY = True
ORIENT_QUERY_ENDPOINTS = True
TARGET_CHECKPOINT_MODE = "heart"
CANDIDATE_LABEL = f"{POLICY}-selected"
ARTIFACT_TAG = "shared_ip_strict_bce_v1"
CACHE_VERSION = 2
SELECTION_PROTOCOL = "frozen-concatip-best-validation-inner-product-fixed-250-per-endpoint-side-v2"
POSITIVE_PROTOCOL = "query-role-authoritative-heart-test-filter-query-excluded-v3"
CITATION2_POSITIVE_PROTOCOL = "query-role-authoritative-released-heart-test-filter-query-excluded-v2"
TIE_BREAK = "per-side-inner-product-desc-node-asc-left-before-right-v1"
SCORING_PROTOCOL = "inner-product-logit-ranking-with-sigmoid-output-v1"
SELECTOR_TRAINING_PROTOCOL = "frozen-concatip-best-validation-checkpoint-inner-product-v1"
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
    _, state = _ranked.validate_frozen_checkpoint_contract(
        checkpoint,
        path,
        framework,
        dataset,
        POLICY,
        expected_run,
        schedule=CHECKPOINT_PROTOCOL,
        config_contract={**_ranked.IP_MODEL_CONTRACT, **TRAINING_CONTRACT},
        predictor_depth=0,
        overrides=(expected_depth, expected_hidden_channels, expected_dropout, expected_lr, expected_weight_decay),
    )
    if any("decoder" in str(key).split(".") for key in state):
        raise ValueError("ConcatIP selector must not contain learned decoder parameters.")


def _validate_selector_model(model):
    _ranked._validate_inner_product_selector(model, POLICY)


MODE_SPEC = {
    "policy": POLICY,
    "display_name": "ConcatIP",
    "validator": validate_selector_checkpoint,
    "model_validator": _validate_selector_model,
    "default_k": 500,
    "exact_k": 500,
    "required_per_side_k": 250,
    "predictor_depth": 0,
    "default_cache_dir": "concatip_sets",
    "candidate_protocol": "frozen-concatip-inner-product-sigmoid-fixed-250-per-endpoint-side",
    "candidate_selection": "concatip_inner_product_logit_top250_independently_per_side",
    "require_output_width": True,
    "artifact_tag": ARTIFACT_TAG,
    "cache_version": CACHE_VERSION,
    "selection_protocol": SELECTION_PROTOCOL,
    "positive_protocol": POSITIVE_PROTOCOL,
    "citation2_positive_protocol": CITATION2_POSITIVE_PROTOCOL,
    "tie_break": TIE_BREAK,
    "negative_sha_label": "concatip-negatives",
    "identity_extra": {
        "training_protocol": SELECTOR_TRAINING_PROTOCOL,
        **TRAINING_CONTRACT,
        "selector_scoring_protocol": SCORING_PROTOCOL,
    },
    "selector_metadata_extra": {
        "decoder_type": "inner-product",
        "predictor_depth": 0,
        "probability_transform": "sigmoid",
        "training_contract": dict(TRAINING_CONTRACT),
    },
}


load_or_create_concatip_test_negatives = partial(
    _ranked.load_or_create_concat_family_test_negatives,
    spec=MODE_SPEC,
    k=500,
)

add_evaluator_arguments = partial(_ranked.add_ranked_arguments, spec=MODE_SPEC)
validate_cli = partial(_ranked.validate_ranked_cli, spec=MODE_SPEC)
evaluator_context = partial(_ranked.ranked_evaluator_context, spec=MODE_SPEC)
data_seed = _ranked.ranked_data_seed
resolve_cap = partial(_ranked.ranked_resolve_cap, spec=MODE_SPEC)
load_bundle = _ranked.load_ranked_bundle
cache_key = partial(_ranked.ranked_cache_key, spec=MODE_SPEC)
prepare_bundle = _ranked.prepare_ranked_bundle
install_evaluator_candidates = partial(
    _ranked.install_ranked_candidates, spec=MODE_SPEC, loader=load_or_create_concatip_test_negatives
)
selector_metadata = partial(_ranked.ranked_selector_metadata, spec=MODE_SPEC)
candidate_pool = partial(_ranked.ranked_candidate_pool, spec=MODE_SPEC)
result_fields = partial(_ranked.result_fields, policy=POLICY)
output_root = partial(_ranked.output_root, policy=POLICY, artifact_tag=ARTIFACT_TAG)


def main():
    from eval_modes.evaluator_helpers import main_for_mode

    return main_for_mode(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
