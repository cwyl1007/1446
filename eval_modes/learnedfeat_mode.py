"""Concat evaluation with fresh learned node features."""

from __future__ import annotations

import sys
from functools import partial
from typing import Mapping

import torch
import torch.nn.functional as F

from eval_modes import fresh_mlp_helpers as _fresh
from eval_modes import ranked_helpers as _ranked
from model.models import LinkPredictor, MLPEncoder


POLICY = "learnedfeat"
EVALUATION_PROTOCOL = "exact_grouped_learnedfeat_fresh_trainable_concat_global_top_k_negatives"
RESULT_FILENAME = "learnedfeat_candidate_evaluation.json"
FORMAT_VERSION = 3
GROUPED = True
ORIENT_QUERY_ENDPOINTS = True
TARGET_CHECKPOINT_MODE = None
DEFAULT_TARGET_CHECKPOINT_MODE = "heart"
CANDIDATE_LABEL = f"{POLICY}-selected"
INPUT_PROTOCOL = "trainable-node-identity-embedding-v1"
TRAINING_PROTOCOL = "learnedfeat-concat-strict-bce-neutral-legal-validation-v1"
TRAINING_CONTRACT = {
    "protocol": TRAINING_PROTOCOL,
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


class LearnedFeatConcatSelector(LinkPredictor):
    """Learn node features and aggregate them with the training graph."""

    def __init__(self, config: Mapping, num_nodes: int, dataset: str):
        width = int(config["emb_size"])
        layers = int(config["layers"])
        pred_layers = int(config["pred_layers"])
        dropout = float(config["dropout"])
        if min(width, layers, pred_layers, int(num_nodes)) <= 0:
            raise ValueError("learnedfeat dimensions and layer counts must be positive.")
        super().__init__(
            MLPEncoder(2 * width, width, width, layers, dropout),
            pred_layers,
            dropout,
            dot=False,
        )
        self.node_features = torch.nn.Embedding(int(num_nodes), width)
        self.dataset_name = str(dataset).strip().lower()
        self.train_samples_per_epoch = max(0, int(config.get("train_samples_per_epoch", 0)))
        self._concat_train_adjacency = None
        _ranked.configure_ranked_selector_training(self, self.dataset_name)
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.node_features.weight)
        super().reset_parameters()

    def set_training_graph(self, training_graph):
        from model.concat import _literal_normalized_training_adjacency

        weight = self.node_features.weight
        with torch.no_grad():
            self._concat_train_adjacency = _literal_normalized_training_adjacency(
                training_graph,
                self.node_features.num_embeddings,
                dtype=weight.dtype,
                device=weight.device,
            )
        return self

    def _apply(self, fn):
        self._concat_train_adjacency = None
        return super()._apply(fn)

    def embed(self, data):
        if self._concat_train_adjacency is None:
            graph = getattr(data, "adj_t", None)
            if graph is None:
                graph = getattr(data, "edge_index", None)
            if graph is None:
                raise ValueError("learnedfeat requires a training graph.")
            self.set_training_graph(graph)
        local = F.normalize(self.node_features.weight, p=2, dim=1)
        adjacency = self._concat_train_adjacency
        neighbor = (
            torch.sparse.mm(adjacency, local)
            if torch.is_tensor(adjacency)
            else adjacency.matmul(local)
        )
        return self.encoder(torch.cat((local, neighbor), dim=1))


def build_learnedfeat_selector(
    config: Mapping,
    *,
    num_nodes: int,
    dataset: str,
    training_graph,
    device=None,
) -> LearnedFeatConcatSelector:
    """Build a fresh learned-feature Concat selector."""
    return LearnedFeatConcatSelector(config, num_nodes, dataset).to(
        torch.device(device or "cpu")
    ).set_training_graph(training_graph)


_config_payload = _fresh.config_payload


def _fresh_model(*, framework, dataset, bundle, config, device):
    node_count = _fresh.raw_features(
        bundle,
        framework,
        dataset,
        selector_policy=POLICY,
        force_node_embeddings=True,
    ).size(0)
    training_graph, _ = _ranked._resolve_training_graph(
        bundle, _ranked._normal_framework(framework)
    )
    model = build_learnedfeat_selector(
        config,
        num_nodes=int(node_count),
        dataset=dataset,
        training_graph=training_graph,
        device=device,
    )
    model.selector_factory_model = "concat"
    model.selector_input_protocol = INPUT_PROTOCOL
    return model


_selector_training_bundle = partial(
    _fresh.selector_training_bundle,
    force_node_embeddings=True,
    selector_policy=POLICY,
)
_train_fresh_learnedfeat_selector = partial(
    _fresh.train_fresh_mlp_selector,
    selector_policy=POLICY,
    selector_factory_model="concat",
    training_protocol=TRAINING_PROTOCOL,
    config_payload_fn=_config_payload,
    fresh_model_fn=_fresh_model,
    training_bundle_fn=_selector_training_bundle,
    force_node_embeddings=True,
)


def _selector_implementation_sha256(framework: str, selector_depth=None) -> str:
    del selector_depth
    return _ranked.selector_runtime_sha256(
        framework,
        "eval_modes/learnedfeat_mode.py",
        "eval_modes/fresh_mlp_helpers.py",
        "model/concat.py",
    )


MODE_SPEC = {
    "policy": POLICY,
    "frozen": False,
    "display_name": "learnedfeat",
    "default_k": 500,
    "global_top_k": True,
    "cache_version": 3,
    "selection_protocol": "fresh-learnedfeat-concat-global-two-endpoint-top-k-v1",
    "training_protocol": TRAINING_PROTOCOL,
    "training_contract": TRAINING_CONTRACT,
    "positive_protocol": "query-role-authoritative-legal-test-filter-query-excluded-v4",
    "citation2_positive_protocol": "query-role-authoritative-directed-legal-test-filter-query-excluded-v3",
    "tie_break": "global-score-desc-side-asc-node-asc-v2",
    "default_metric": _ranked.default_selector_metric,
    "config_payload": _config_payload,
    "implementation_sha256": _selector_implementation_sha256,
    "neutral_validation": True,
    "train_selector": _train_fresh_learnedfeat_selector,
    "candidate_protocol": "fresh-learnedfeat-concat-global-top-k-across-endpoint-sides",
    "candidate_selection": "learnedfeat_concat_score_global_top_k_across_endpoint_sides",
    "selector_identity": "fresh-independent-learnedfeat-concat",
    "selector_factory_model": "concat",
    "input_protocol": INPUT_PROTOCOL,
    "force_node_embeddings": True,
    "default_seed": 0,
    "default_epochs": 500,
}

load_or_create_learnedfeat_test_negatives = partial(
    _ranked.load_or_create_mlp_family_test_negatives,
    spec=MODE_SPEC,
)


def add_evaluator_arguments(parser):
    _ranked.add_ranked_arguments(parser, spec=MODE_SPEC, fresh=True)
    parser.set_defaults(mode=DEFAULT_TARGET_CHECKPOINT_MODE)
    return parser


evaluator_context = partial(
    _ranked.ranked_evaluator_context,
    spec=MODE_SPEC,
    fresh=True,
)
data_seed = partial(_ranked.ranked_data_seed, fresh=True)
resolve_cap = partial(_ranked.ranked_resolve_cap, spec=MODE_SPEC, fresh=True)


def load_bundle(*args, **kwargs):
    bundle = _ranked.load_ranked_bundle(*args, **kwargs)
    bundle["ranked_feature_source"] = "fresh-learned-node-embeddings"
    return bundle


cache_key = partial(_ranked.ranked_cache_key, spec=MODE_SPEC, fresh=True)
prepare_bundle = partial(_ranked.prepare_ranked_bundle, fresh=True)
install_evaluator_candidates = partial(
    _ranked.install_ranked_candidates,
    spec=MODE_SPEC,
    loader=load_or_create_learnedfeat_test_negatives,
    fresh=True,
)
selector_metadata = partial(_ranked.ranked_selector_metadata, spec=MODE_SPEC, fresh=True)
candidate_pool = partial(_ranked.ranked_candidate_pool, spec=MODE_SPEC)
result_fields = partial(_ranked.result_fields, policy=POLICY)
output_root = partial(_ranked.output_root, policy=POLICY)


def main():
    from eval_modes.evaluator_helpers import main_for_mode

    return main_for_mode(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
