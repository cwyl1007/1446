"""Shared construction and training for fresh MLP evaluator selectors."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import random
from pathlib import Path
from typing import Any, Callable, Mapping, NamedTuple, Optional

import numpy as np
import torch

from eval_modes import ranked_helpers as ranked


class FreshMlpSelector(NamedTuple):
    model: torch.nn.Module
    config: dict[str, Any]
    metadata: dict[str, Any]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def config_payload(
    dataset: str,
    *,
    selector_depth: Optional[int] = None,
    selector_hidden_channels: Optional[int] = None,
    selector_dropout: Optional[float] = None,
    selector_lr: Optional[float] = None,
    selector_weight_decay: Optional[float] = None,
) -> tuple[dict[str, Any], str, str]:
    path = Path(__file__).resolve().parents[1] / "configs" / f"mlp_{dataset}_config.json"
    if not path.is_file():
        raise FileNotFoundError(f"Cannot find MLP selector config: {path}")
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf8"))
    config = payload.get("best_config")
    if not isinstance(config, Mapping):
        raise ValueError(f"MLP selector config has no best_config mapping: {path}")
    config = dict(config)
    base_sha = hashlib.sha256(raw).hexdigest()
    values = {
        "selector_depth": selector_depth,
        "selector_hidden_channels": selector_hidden_channels,
        "selector_dropout": selector_dropout,
        "selector_lr": selector_lr,
        "selector_weight_decay": selector_weight_decay,
    }
    if not any(value is not None for value in values.values()):
        return config, base_sha, str(path.resolve())
    if selector_depth is not None:
        selector_depth = int(selector_depth)
        if selector_depth <= 0:
            raise ValueError("MLP selector depth must be positive.")
        config["layers"] = selector_depth
        config["pred_layers"] = selector_depth
    if selector_hidden_channels is not None:
        selector_hidden_channels = int(selector_hidden_channels)
        if selector_hidden_channels <= 0:
            raise ValueError("MLP selector hidden channels must be positive.")
        config["emb_size"] = selector_hidden_channels
    if selector_dropout is not None:
        selector_dropout = float(selector_dropout)
        if not math.isfinite(selector_dropout) or not 0.0 <= selector_dropout < 1.0:
            raise ValueError("MLP selector dropout must be finite and in [0, 1).")
        config["dropout"] = selector_dropout
    if selector_lr is not None:
        selector_lr = float(selector_lr)
        if not math.isfinite(selector_lr) or selector_lr <= 0.0:
            raise ValueError("MLP selector learning rate must be finite and positive.")
        config["lr"] = selector_lr
    if selector_weight_decay is not None:
        selector_weight_decay = float(selector_weight_decay)
        if not math.isfinite(selector_weight_decay) or selector_weight_decay < 0.0:
            raise ValueError("MLP selector weight decay must be finite and non-negative.")
        config["weight_decay"] = selector_weight_decay
    if selector_hidden_channels is None and all(
        value is None for value in (selector_dropout, selector_lr, selector_weight_decay)
    ):
        config_sha = hashlib.sha256(
            f"{base_sha}:{config['layers']}:{config['pred_layers']}".encode("utf8")
        ).hexdigest()
    else:
        values.update(
            selector_depth=selector_depth,
            selector_hidden_channels=selector_hidden_channels,
            selector_dropout=selector_dropout,
            selector_lr=selector_lr,
            selector_weight_decay=selector_weight_decay,
        )
        override = {
            "base_sha256": base_sha,
            **{key: value for key, value in values.items() if value is not None},
        }
        if selector_dropout is None and selector_lr is None and selector_weight_decay is None:
            override.update(
                selector_depth=selector_depth,
                selector_hidden_channels=selector_hidden_channels,
            )
        encoded = json.dumps(override, sort_keys=True, separators=(",", ":")).encode("utf8")
        config_sha = hashlib.sha256(encoded).hexdigest()
    return config, config_sha, str(path.resolve())


def raw_features(
    bundle: Mapping[str, Any],
    framework: str,
    dataset: str,
    *,
    selector_policy: str,
    force_node_embeddings: bool,
) -> torch.Tensor:
    return ranked._raw_features(
        bundle,
        framework,
        selector_model=selector_policy,
        dataset=dataset,
        force_node_embeddings=force_node_embeddings,
    )


def fresh_model(
    *,
    framework: str,
    dataset: str,
    bundle: Mapping[str, Any],
    config: Mapping[str, Any],
    device: torch.device,
    selector_policy: str,
    selector_factory_model: str = "mlp",
    force_node_embeddings: bool = False,
) -> torch.nn.Module:
    from model.pairwise_models import get_model

    raw_x = raw_features(
        bundle,
        framework,
        dataset,
        selector_policy=selector_policy,
        force_node_embeddings=force_node_embeddings,
    )
    num_nodes, in_channels = int(raw_x.size(0)), int(raw_x.size(1))
    data = bundle.get("data")
    train_edge_index = getattr(data, "edge_index", None) if data is not None else None
    use_node_emb = force_node_embeddings or dataset == "ogbl-ddi"
    model = get_model(
        selector_factory_model,
        {
            **dict(config),
            "in_channels": in_channels,
            "num_nodes": num_nodes,
            "train_edge_index": train_edge_index,
            "use_node_emb": use_node_emb,
            "dataset_name": dataset,
            "evaluation_mode": "ranked-selector",
        },
    ).to(device)
    ranked.configure_ranked_selector_training(model, dataset)
    if getattr(model, "decode_is_symmetric", None) is not True:
        raise ValueError("Fresh MLP selector must declare a symmetric decoder.")
    model.selector_factory_model = selector_factory_model
    model.selector_input_protocol = (
        "trainable-node-identity-embedding-v1" if use_node_emb else "raw-node-features-v1"
    )
    return model


def selector_training_bundle(
    framework: str,
    dataset: str,
    bundle: Mapping[str, Any],
    *,
    data_seed: int,
    force_node_embeddings: bool = False,
    selector_policy: str = "mlp",
) -> dict[str, Any]:
    training = dict(bundle)
    if force_node_embeddings:
        empty_x = raw_features(
            bundle,
            framework,
            dataset,
            selector_policy=selector_policy,
            force_node_embeddings=True,
        )
        if "x" in training:
            training["x"] = empty_x
        data = training.get("data")
        if data is not None:
            data = copy.copy(data)
            data.x = empty_x
            training["data"] = data
    validation = ranked.build_neutral_selector_validation_negatives(
        training,
        framework,
        dataset,
        seed=int(data_seed),
        negatives=500,
    )
    training["valid_neg"] = validation.negatives
    training["neutral_selector_validation"] = dict(validation.metadata)
    if framework == "ogbl":
        training["eval_edges"] = {
            "pos_train_edge": training["train_pos"],
            "train_val_edge": training["train_val"],
            "pos_valid_edge": training["valid_pos"],
            "neg_valid_edge": validation.negatives,
            "pos_test_edge": training.get("test_pos"),
            "neg_test_edge": None,
        }
    return training


def _metric_key(results: Mapping[str, Any], requested: str) -> str:
    wanted = str(requested).strip().lower()
    for key in results:
        if str(key).strip().lower() == wanted:
            return str(key)
    raise KeyError(f"MLP selector metric {requested!r} was not returned; available metrics: {list(results)}")


def train_fresh_mlp_selector(
    framework: str,
    dataset: str,
    bundle: Mapping[str, Any],
    device: str | torch.device,
    *,
    selector_policy: str,
    selector_factory_model: str,
    training_protocol: str,
    config_payload_fn: Callable[..., tuple[dict[str, Any], str, str]],
    fresh_model_fn: Callable[..., torch.nn.Module],
    training_bundle_fn: Callable[..., dict[str, Any]],
    force_node_embeddings: bool = False,
    seed: int = 0,
    epochs: int = 500,
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
    del data_root
    framework = ranked._normal_framework(framework)
    dataset = str(dataset).strip().lower()
    device = torch.device(device)
    epochs, eval_steps = int(epochs), int(eval_steps)
    patience, batch_size = int(patience), int(batch_size)
    if min(epochs, eval_steps, patience, batch_size) <= 0:
        raise ValueError("MLP selector training controls must be positive.")
    metric = str(metric or ranked.default_selector_metric(framework, dataset))
    config, config_sha, _ = config_payload_fn(
        dataset,
        selector_depth=selector_depth,
        selector_hidden_channels=selector_hidden_channels,
        selector_dropout=selector_dropout,
        selector_lr=selector_lr,
        selector_weight_decay=selector_weight_decay,
    )
    set_seed(seed)
    training = training_bundle_fn(framework, dataset, bundle, data_seed=int(data_seed))
    validation_metadata = dict(training["neutral_selector_validation"])
    model = fresh_model_fn(
        framework=framework,
        dataset=dataset,
        bundle=training,
        config=config,
        device=device,
    )
    training_contract = dict(model.ranked_selector_training_metadata)
    best_value, best_state, best_epoch = float("-inf"), None, None
    no_improve, stopped_epoch = 0, epochs

    if framework == "pyg":
        from pyg.main import _default_batches, _make_optimizer
        from pyg.train_eval import all_train, validation_only

        raw_x = raw_features(
            training,
            framework,
            dataset,
            selector_policy=selector_policy,
            force_node_embeddings=force_node_embeddings,
        )
        x = raw_x.to(device)
        train_pos = ranked._edge_rows(training.get("train_pos"), "selector_training.train_pos").to(device)
        selector_data = dict(training)
        selector_data["x"] = raw_x
        recommended_train, recommended_eval = _default_batches(dataset, selector_factory_model, device)
        train_batch = max(batch_size, int(recommended_train))
        eval_batch = max(batch_size, int(recommended_eval))
        optimizer = _make_optimizer(model, lr=config["lr"], weight_decay=config["weight_decay"], device=device)

        def train_epoch(epoch):
            all_train(
                model,
                train_pos,
                x,
                optimizer,
                train_batch,
                adj_t=selector_data.get("adj"),
                csr_rowptr=selector_data.get("csr_train_rowptr"),
                csr_col=selector_data.get("csr_train_col"),
            )

        def validate_epoch():
            results = validation_only(
                model,
                selector_data,
                x,
                eval_batch,
                include_auc=str(metric).strip().lower() in {"auc", "ap"},
                include_hits=True,
            )
            return results

    else:
        from ogbl.train_eval import (
            evaluate_ogbl_validation,
            prepare_ogbl_evaluation,
            release_ogbl_evaluation,
            train_one_epoch_ogbl,
        )
        from ogbl.training import make_ogbl_optimizer, recommended_decode_batch_size

        data, eval_edges = training.get("data"), training.get("eval_edges")
        if data is None or not isinstance(eval_edges, Mapping):
            raise ValueError("OGB MLP selector training requires data and eval_edges.")
        train_pos = eval_edges.get("pos_train_edge")
        if not torch.is_tensor(train_pos):
            train_pos = bundle.get("train_pos")
        if not torch.is_tensor(train_pos):
            raise ValueError("OGB MLP selector training requires train positives.")
        train_batch = max(batch_size, recommended_decode_batch_size(dataset, selector_factory_model))
        eval_batch = min(train_batch, 8192) if dataset == "ogbl-citation2" else train_batch
        optimizer = make_ogbl_optimizer(model, config["lr"], config["weight_decay"], device)

        def train_epoch(epoch):
            train_one_epoch_ogbl(
                model,
                optimizer,
                data,
                train_pos,
                device,
                batch_size=train_batch,
                negative_sampler="fast",
                train_decode_batch_size=train_batch,
                seed=int(seed) + epoch * 1009,
                training_path="full-graph",
            )

        def validate_epoch():
            context = prepare_ogbl_evaluation(
                model,
                data,
                eval_edges,
                dataset,
                device,
                batch_size=eval_batch,
                citation2_query_batch_size=min(2048, eval_batch),
            )
            try:
                return evaluate_ogbl_validation(
                    context,
                    compute_auc=str(metric).strip().lower() in {"auc", "ap"},
                )
            finally:
                release_ogbl_evaluation(context)

    for epoch in range(1, epochs + 1):
        if hasattr(model, "configure_epoch") and bool(model.configure_epoch(epoch, epochs)):
            no_improve = 0
        train_epoch(epoch)
        if epoch % eval_steps:
            continue
        results = validate_epoch()
        value = results[_metric_key(results, metric)]
        value = value[1] if isinstance(value, (tuple, list)) else value
        if value is not None and float(value) > best_value:
            best_value = float(value)
            best_state = copy.deepcopy(model.state_dict())
            best_epoch, no_improve = epoch, 0
        else:
            no_improve += 1
        if no_improve >= patience:
            stopped_epoch = epoch
            break

    if best_state is None or best_epoch is None:
        raise RuntimeError("Fresh MLP selector training never produced a valid state.")
    model.load_state_dict(best_state, strict=True)
    if hasattr(model, "clear_runtime_cache"):
        model.clear_runtime_cache()
    model.requires_grad_(False)
    model.eval()
    uses_node_embedding = force_node_embeddings or dataset == "ogbl-ddi"
    metadata = {
        **training_contract,
        "training_protocol": training_protocol,
        "selector_model": selector_policy,
        "selector_seed": int(seed),
        "selector_epochs_requested": epochs,
        "selector_eval_steps": eval_steps,
        "selector_patience": patience,
        "selector_batch_size": batch_size,
        "selector_effective_train_batch_size": train_batch,
        "selector_effective_eval_batch_size": eval_batch,
        "selector_epoch_seed_protocol": (
            "continuous-rng-from-selector-seed" if framework == "pyg" else "selector-seed+epoch*1009"
        ),
        "selector_validation_metric": metric,
        "selector_best_validation_value": best_value,
        "selector_best_epoch": best_epoch,
        "selector_stopped_epoch": stopped_epoch,
        "selector_state_sha256": ranked.selector_state_sha256(
            best_state, f"{selector_policy}-selector-state"
        ),
        "selector_config_sha256": config_sha,
        "selector_data_seed": int(data_seed),
        **validation_metadata,
    }
    if selector_policy != selector_factory_model:
        metadata.update(
            selector_factory_model=selector_factory_model,
            selector_input_protocol=(
                "trainable-node-identity-embedding-v1" if uses_node_embedding else "raw-node-features-v1"
            ),
            supplied_node_features_ignored=bool(force_node_embeddings),
        )
    return FreshMlpSelector(model=model, config=dict(config), metadata=metadata)
