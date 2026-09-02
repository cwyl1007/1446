#!/usr/bin/env python3
"""Policy-neutral checkpoint, scoring, metric, and serialization helpers."""
import argparse
import gc
import hashlib
import json
import os
import random
import re
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from types import MethodType, SimpleNamespace
import numpy as np
import torch
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from model.feature_aggregation import (
    aggregated_mlp_method,
    aggregated_mlp_recipe,
    is_aggregated_mlp,
    normalized_model_name,
    preprocess_aggregated_mlp,
)
from pyg.main import _configure_cuda_matmul_precision
from pyg.train_eval import _apply_reference_evaluation_transform, _reference_grouped_edge_scores
from utils.heart_protocol import heart_candidate_metadata
from utils.profiling import StageProfiler, configure_torch_cpu_threads

HITS_K = (1, 3, 5, 10, 20, 50, 100)
FINAL_METRICS = ("MRR", *(f"Hits@{k}" for k in HITS_K), "AUC")
METRIC_SIGNIFICANT_DIGITS = 8
CHECKPOINT_RE = re.compile("model_checkpoint(\\d+)$")
HEART_MODES = {"heart"}
HEART_BENCHMARK_PYG_DATASETS = {"cora", "citeseer", "pubmed"}
HEART_BENCHMARK_OGB_DATASETS = {"ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation2"}
_SHARED_N2V_EMBEDDING_CACHE = {}
SPECIAL_GROUPED_MATERIALIZE_MAX_EDGES = 16000000
SPECIAL_GROUPED_MATERIALIZE_MAX_BYTES = 256 * 1024 * 1024


def project_path(value):
    path = Path(value).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def set_seed(seed):
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def start_timer(device=None):
    if device is not None and torch.cuda.is_available() and (torch.device(device).type == "cuda"):
        torch.cuda.synchronize(device)
    return time.perf_counter()


def stop_timer(started, device=None):
    if device is not None and torch.cuda.is_available() and (torch.device(device).type == "cuda"):
        torch.cuda.synchronize(device)
    return round(time.perf_counter() - started, 3)


RESOURCE_PEAK_KEYS = ("cpu_peak_rss_mb", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb")


def peak_resource_usage(profiles):
    profiles = [profile for profile in profiles if profile]
    return {key: round(max((float(profile.get(key, 0.0)) for profile in profiles), default=0.0), 3) for key in RESOURCE_PEAK_KEYS}


def framework_name(value, dataset):
    value = str(value).lower()
    if value == "auto":
        return "ogb" if str(dataset).lower().startswith("ogbl-") else "pyg"
    return "ogb" if value in {"ogb", "ogbl"} else "pyg"


def num_nodes(bundle, framework):
    return int(bundle["x"].size(0)) if framework == "pyg" else int(bundle["data"].num_nodes)


def _uses_generated_heart_protocol(framework, expected_framework, mode, dataset, datasets):
    return (
        framework == expected_framework
        and str(mode).strip().lower() == "heart"
        and (str(dataset).strip().lower() in datasets)
    )


def use_generated_pyg_heart_protocol(framework, mode, dataset):
    return _uses_generated_heart_protocol(framework, "pyg", mode, dataset, HEART_BENCHMARK_PYG_DATASETS)


def edge_tensor(endpoint, candidates, direction, device):
    candidates = candidates.to(device, non_blocking=True)
    fixed = torch.full_like(candidates, int(endpoint))
    if direction == "out":
        return torch.stack([fixed, candidates])
    if direction == "in":
        return torch.stack([candidates, fixed])
    return torch.stack([torch.minimum(fixed, candidates), torch.maximum(fixed, candidates)])


def value_stats(values):
    if values.numel() == 0:
        return {"min": 0, "mean": 0.0, "max": 0}
    return {"min": int(values.min()), "mean": float(values.to(torch.float64).mean()), "max": int(values.max())}


def candidate_summary(left, right, nodes):
    return {
        "graph_nodes_per_side_before_filtering": nodes,
        "maximum_nodes_both_sides_before_filtering": 2 * nodes,
        "left_legal_candidates": value_stats(left),
        "right_legal_candidates": value_stats(right),
        "total_legal_candidates": value_stats(left + right),
    }


def grouped_negative_edges(negative_edges, positive_edges):
    positives = torch.as_tensor(positive_edges).to(dtype=torch.long)
    if positives.dim() != 2 or int(positives.size(1)) != 2:
        raise ValueError(f"Positive edges must have shape [N,2], got {tuple(positives.shape)}.")
    count = int(positives.size(0))
    if bool(getattr(negative_edges, "is_streaming_negative", False)):
        if not hasattr(negative_edges, "iter_chunks"):
            raise ValueError("This evaluator requires fixed-width streaming negative groups.")
        if int(negative_edges.dim()) != 3 or int(negative_edges.size(0)) != count or int(negative_edges.size(2)) != 2:
            raise ValueError(
                f"Streaming grouped negatives must have shape [num_positive,K,2], got {tuple(negative_edges.shape)} for {count} positives."
            )
        if int(negative_edges.size(1)) <= 0 and count:
            raise ValueError("Each positive edge must have at least one HeaRT negative.")
        return negative_edges
    negatives = torch.as_tensor(negative_edges).to(dtype=torch.long)
    if negatives.dim() == 3:
        if int(negatives.size(0)) != count or int(negatives.size(2)) != 2:
            raise ValueError(f"Grouped negatives must have shape [num_positive,K,2], got {tuple(negatives.shape)} for {count} positives.")
        grouped = negatives
    elif negatives.dim() == 2 and int(negatives.size(1)) == 2:
        if count == 0:
            if int(negatives.size(0)) != 0:
                raise ValueError("Non-empty negatives were provided for no positives.")
            grouped = negatives.reshape(0, 0, 2)
        elif int(negatives.size(0)) % count:
            raise ValueError(f"{negatives.size(0)} negative edges cannot be grouped across {count} positives.")
        else:
            grouped = negatives.reshape(count, -1, 2)
    elif negatives.dim() in {1, 2}:
        if count == 0:
            if negatives.numel():
                raise ValueError("Non-empty negative targets were provided for no positives.")
            grouped = torch.empty((0, 0, 2), dtype=torch.long)
        else:
            if negatives.numel() % count:
                raise ValueError(f"{negatives.numel()} negative targets cannot be grouped across {count} positives.")
            targets = negatives.reshape(count, -1)
            sources = positives[:, 0:1].expand_as(targets)
            grouped = torch.stack([sources, targets], dim=2)
    else:
        raise ValueError(f"HeaRT negatives must be edge pairs or citation target ids, got shape {tuple(negatives.shape)}.")
    if int(grouped.size(1)) <= 0 and count:
        raise ValueError("Each positive edge must have at least one HeaRT negative.")
    return grouped


def uses_reference_row_grouped_evaluation(model, framework, dataset):
    return (
        str(framework).strip().lower() == "pyg"
        and str(dataset).strip().lower() in HEART_BENCHMARK_PYG_DATASETS
        and (int(getattr(model, "reference_evaluation_row_batch_size", 0) or 0) > 0)
    )


def _special_grouped_tensor_reason(model, *, framework, dataset):
    reasons = []
    if uses_reference_row_grouped_evaluation(model, framework, dataset):
        reasons.append("reference-row-grouped-decoder")
    return "+".join(reasons) if reasons else None


def _bounded_special_grouped_tensor(negative_edges, positive_edges, *, dataset, reason, split_name):
    if not bool(getattr(negative_edges, "is_streaming_negative", False)):
        return (negative_edges, None)
    dataset_key = str(dataset).strip().lower()
    if dataset_key not in HEART_BENCHMARK_PYG_DATASETS:
        raise RuntimeError(
            f"{reason} requires tensor grouped negatives, but {dataset} must remain streamed. No unbounded materialization is permitted."
        )
    positives = positive_edges.long().cpu()
    grouped = grouped_negative_edges(negative_edges, positives)
    rows = int(grouped.size(0))
    per_positive = int(grouped.size(1))
    edge_count = rows * per_positive
    byte_count = edge_count * 2 * torch.empty((), dtype=torch.long).element_size()
    if rows != int(positives.size(0)):
        raise ValueError(f"{split_name} streaming negative rows do not match positives.")
    if edge_count > SPECIAL_GROUPED_MATERIALIZE_MAX_EDGES or byte_count > SPECIAL_GROUPED_MATERIALIZE_MAX_BYTES:
        raise RuntimeError(
            f"Refusing to materialize {edge_count:,} {split_name} negatives ({byte_count / 1024 ** 2:.1f} MiB) for {reason}; hard limits are {SPECIAL_GROUPED_MATERIALIZE_MAX_EDGES:,} edges and {SPECIAL_GROUPED_MATERIALIZE_MAX_BYTES / 1024 ** 2:.0f} MiB."
        )
    materialize = getattr(grouped, "materialize", None)
    if not callable(materialize):
        raise TypeError(f"{reason} requires a materializable fixed-width streaming object.")
    tensor = materialize()
    tensor = torch.as_tensor(tensor, dtype=torch.long).cpu().contiguous()
    expected_shape = (rows, per_positive, 2)
    if tuple(tensor.shape) != expected_shape:
        raise ValueError(f"Materialized {split_name} negatives have shape {tuple(tensor.shape)}, expected {expected_shape}.")
    metadata = {
        "dataset": dataset_key,
        "split": str(split_name),
        "reason": str(reason),
        "shape": list(expected_shape),
        "edge_count": edge_count,
        "bytes": byte_count,
        "max_edges": SPECIAL_GROUPED_MATERIALIZE_MAX_EDGES,
        "max_bytes": SPECIAL_GROUPED_MATERIALIZE_MAX_BYTES,
        "policy": "reference-planetoid-only-hard-bounded",
    }
    return (tensor, metadata)


def materialize_special_grouped_splits(model, bundle, *, framework, dataset, requested_split, grouped):
    if not grouped:
        return None
    reason = _special_grouped_tensor_reason(model, framework=framework, dataset=dataset)
    if reason is None:
        return None
    if str(framework).strip().lower() != "pyg":
        raise RuntimeError(f"Tensor-only grouped evaluator {reason} is supported only for PyG.")
    records = bundle.setdefault("_bounded_grouped_materializations", {})
    for split_name in ("valid", "test"):
        if requested_split not in {split_name, "both"}:
            continue
        negative_key = f"{split_name}_neg"
        positive_key = f"{split_name}_pos"
        if negative_key not in bundle or positive_key not in bundle:
            raise KeyError(f"{reason} requires bundle[{positive_key!r}] and bundle[{negative_key!r}].")
        (value, metadata) = _bounded_special_grouped_tensor(
            bundle[negative_key], bundle[positive_key], dataset=dataset, reason=reason, split_name=split_name
        )
        bundle[negative_key] = value
        if metadata is not None:
            records[split_name] = metadata
    return dict(records) if records else None


class _StreamingGroupedView:
    is_streaming_negative = True

    def __init__(self, grouped):
        self.shape = tuple((int(value) for value in grouped.shape))

    def dim(self):
        return 3

    def size(self, dim=None):
        if dim is None:
            return torch.Size(self.shape)
        return self.shape[int(dim)]

    def numel(self):
        return int(self.shape[0]) * int(self.shape[1]) * 2

    def contiguous(self):
        return self


class _QueryOrientedStreamingNegatives(_StreamingGroupedView):
    def __init__(self, grouped, positive_edges):
        super().__init__(grouped)
        self._grouped = grouped
        self._positives = positive_edges.long().cpu().contiguous()

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def iter_chunks(self, pos_chunk_size=None):
        expected_start = 0
        blocks = self._grouped.iter_chunks(pos_chunk_size)
        for start, end, edge_block in blocks:
            start = int(start)
            end = int(end)
            if start != expected_start or end <= start or end > self.shape[0]:
                raise ValueError("Streaming ranked-negative chunks must cover positive rows once in contiguous order.")
            block = edge_block.long().cpu()
            expected_shape = (end - start, self.shape[1], 2)
            if tuple(block.shape) != expected_shape:
                raise ValueError(f"Streaming ranked-negative chunk has shape {tuple(block.shape)}, expected {expected_shape}.")
            positives = self._positives[start:end]
            fixed_left = block[:, :, 0] == positives[:, 0:1]
            fixed_right = block[:, :, 1] == positives[:, 1:2]
            if not bool((fixed_left ^ fixed_right).all().item()):
                raise ValueError("Ranked heuristic negatives must corrupt exactly one query endpoint.")
            oriented = block.clone()
            right_rows = oriented[fixed_right]
            oriented[fixed_right] = right_rows.flip(1)
            yield (start, end, oriented)
            expected_start = end
        if expected_start != self.shape[0]:
            raise ValueError("Streaming ranked-negative chunks ended before every positive row was covered.")


class _QueryOrientedEndpointNegatives(_StreamingGroupedView):
    is_endpoint_grouped_negative = True
    endpoint_query_oriented = True

    def __init__(self, grouped):
        super().__init__(grouped)
        self._grouped = grouped
        self.candidate_summary_prevalidated = bool(getattr(grouped, "candidate_summary_prevalidated", False))

    @property
    def candidate_summary(self):
        value = getattr(self._grouped, "candidate_summary", None)
        return value() if callable(value) else value

    def to(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError("Endpoint-grouped negatives cannot be materialized; consume iter_endpoint_group_chunks().")

    def iter_endpoint_group_chunks(self):
        yield from self._grouped.iter_endpoint_group_chunks()

    def iter_chunks(self):
        yield from self.iter_endpoint_group_chunks()


def _query_oriented_heuristic_negatives(negative_edges, positive_edges):
    positives = positive_edges.long().cpu()
    grouped = grouped_negative_edges(negative_edges, positives)
    if bool(getattr(grouped, "is_endpoint_grouped_negative", False)):
        return _QueryOrientedEndpointNegatives(grouped)
    if bool(getattr(grouped, "is_streaming_negative", False)):
        return _QueryOrientedStreamingNegatives(grouped, positives)
    fixed_left = grouped[:, :, 0] == positives[:, 0:1]
    fixed_right = grouped[:, :, 1] == positives[:, 1:2]
    if not bool((fixed_left ^ fixed_right).all().item()):
        raise ValueError("Ranked heuristic negatives must corrupt exactly one query endpoint.")
    oriented = grouped.clone()
    right_rows = oriented[fixed_right]
    oriented[fixed_right] = right_rows.flip(1)
    return oriented


def _prevalidated_grouped_candidate_summary(grouped, count, per_positive):
    if not bool(getattr(grouped, "candidate_summary_prevalidated", False)):
        return None
    raw = getattr(grouped, "candidate_summary", None)
    if callable(raw):
        raw = raw()
    if not isinstance(raw, dict):
        raise ValueError("A prevalidated streaming candidate summary must be a dictionary.")
    if int(raw.get("grouped_negatives_per_positive", -1)) != int(per_positive):
        raise ValueError("Streaming candidate summary K does not match the negative shape.")
    if raw.get("num_positive_edges") is not None and int(raw["num_positive_edges"]) != int(count):
        raise ValueError("Streaming candidate summary row count does not match positives.")
    try:
        normalized = {
            key: dict(raw[key])
            for key in (
                "fixed_left_endpoint_candidates",
                "fixed_right_endpoint_candidates",
                "other_grouped_candidates",
                "total_grouped_candidates",
            )
        }
    except (KeyError, TypeError) as exc:
        raise ValueError("Prevalidated streaming candidate statistics are incomplete.") from exc
    total = normalized["total_grouped_candidates"]
    both_sides = raw.get("both_corruption_sides_combined")
    if not isinstance(both_sides, bool):
        raise ValueError("Streaming candidate summary must declare whether both sides occur.")
    return {
        "grouped_negatives_per_positive": int(per_positive),
        "both_corruption_sides_combined": both_sides,
        **normalized,
        "total_legal_candidates": dict(total),
    }


def grouped_candidate_summary(positive_edges, grouped_negatives, row_batch_size=8192):
    positives = positive_edges.long().cpu()
    streaming = bool(getattr(grouped_negatives, "is_streaming_negative", False))
    grouped = grouped_negatives if streaming else grouped_negatives.long().cpu()
    count = int(positives.size(0))
    per_positive = int(grouped.size(1)) if grouped.dim() == 3 else 0
    if streaming:
        prevalidated = _prevalidated_grouped_candidate_summary(grouped, count, per_positive)
        if prevalidated is not None:
            return prevalidated
    fixed_left = torch.zeros(count, dtype=torch.int64)
    fixed_right = torch.zeros(count, dtype=torch.int64)
    if streaming:
        blocks = grouped.iter_chunks()
    else:
        batch_size = max(1, int(row_batch_size))
        blocks = ((start, min(start + batch_size, count), grouped[start : start + batch_size]) for start in range(0, count, batch_size))
    for start, end, block in blocks:
        block = block.long().cpu()
        fixed_left[start:end] = (block[:, :, 0] == positives[start:end, 0:1]).sum(dim=1, dtype=torch.int64)
        fixed_right[start:end] = (block[:, :, 1] == positives[start:end, 1:2]).sum(dim=1, dtype=torch.int64)
    total = torch.full((count,), per_positive, dtype=torch.int64)
    other = (total - fixed_left - fixed_right).clamp_min(0)
    total_stats = value_stats(total)
    both_sides = bool(count and bool((fixed_left > 0).all()) and bool((fixed_right > 0).all()))
    return {
        "grouped_negatives_per_positive": per_positive,
        "both_corruption_sides_combined": both_sides,
        "fixed_left_endpoint_candidates": value_stats(fixed_left),
        "fixed_right_endpoint_candidates": value_stats(fixed_right),
        "other_grouped_candidates": value_stats(other),
        "total_grouped_candidates": total_stats,
        "total_legal_candidates": total_stats,
    }


def endpoint_groups(pos_edges, framework):
    groups = defaultdict(list)
    for row, (left, right) in enumerate(pos_edges.long().cpu().tolist()):
        groups[left, "out"].append((row, right, 0))
        groups[right, "in"].append((row, left, 1))
    return groups


def checkpoint_number(path):
    match = CHECKPOINT_RE.search(path.name)
    return int(match.group(1)) if match else 10**9


def load_checkpoint(path):
    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, dict) or "model_state_dict" not in checkpoint:
        raise ValueError(f"Unsupported checkpoint: {path}")
    return checkpoint


def discover_checkpoints(args, checkpoint_mode=None):
    requested_values = [int(run) for run in args.runs or []]
    if len(requested_values) != len(set(requested_values)):
        raise ValueError(f"--runs contains duplicate run ids: {requested_values}.")
    if args.checkpoint:
        paths = [project_path(path) for path in args.checkpoint]
    else:
        if not args.mode or not args.dataset or (not args.model):
            raise ValueError("--mode, --dataset and --model are required without --checkpoint.")
        folder = project_path(args.checkpoint_root) / str(checkpoint_mode or args.mode) / args.dataset / args.model
        requested = set(requested_values)
        paths = [
            path
            for path in folder.glob("model_checkpoint*")
            if path.is_file() and CHECKPOINT_RE.fullmatch(path.name) and (not requested or checkpoint_number(path) in requested)
        ]
    paths = sorted(paths, key=lambda path: (checkpoint_number(path), str(path)))
    if not paths:
        raise FileNotFoundError("No model_checkpointN files were found.")
    run_numbers = [checkpoint_number(path) for path in paths]
    if len(run_numbers) != len(set(run_numbers)):
        raise ValueError(f"Duplicate checkpoint run numbers were selected: {run_numbers}.")
    requested = set(requested_values)
    if requested and set(run_numbers) != requested:
        missing = sorted(requested - set(run_numbers))
        unexpected = sorted(set(run_numbers) - requested)
        raise FileNotFoundError(f"The selected checkpoint set does not exactly match --runs: missing={missing}, unexpected={unexpected}.")
    missing_paths = [str(path) for path in paths if not path.is_file()]
    if missing_paths:
        raise FileNotFoundError("Selected checkpoint files do not exist: " + ", ".join(missing_paths))
    return paths


def _checkpoint_construction_fingerprint(checkpoint):
    config = dict(checkpoint.get("model_config") or {})
    state = checkpoint.get("model_state_dict") or {}
    layout = [[str(key), list(value.shape), str(value.dtype)] for (key, value) in sorted(state.items()) if torch.is_tensor(value)]
    value = json.dumps({"model_config": config, "state_layout": layout}, sort_keys=True, default=str)
    return hashlib.sha256(value.encode()).hexdigest()


def preflight_checkpoint_set(paths, *, framework, dataset, model_name, mode, requested_runs=None, expected_data_seed=None):
    expected = (framework_name(framework, dataset), str(dataset), normalized_model_name(model_name), str(mode))
    requested = set((int(run) for run in requested_runs or []))
    seen_runs, fingerprints, data_seeds = set(), set(), set()
    required = {
        "framework", "dataset", "model", "mode", "run", "seed", "epoch", "timed_out",
        "checkpoint_type", "best_validation_metric", "model_config", "model_state_dict",
    }
    for path in paths:
        checkpoint = load_checkpoint(path)
        missing = sorted(required.difference(checkpoint))
        if missing:
            raise ValueError(f"Checkpoint {path} is missing required metadata: {missing}.")
        if str(checkpoint["framework"]).strip().lower() not in {"pyg", "ogb", "ogbl"}:
            raise ValueError(f"Checkpoint {path} has unsupported framework metadata {checkpoint['framework']!r}.")
        actual = (
            framework_name(checkpoint["framework"], checkpoint["dataset"]), str(checkpoint["dataset"]),
            normalized_model_name(checkpoint["model"]), str(checkpoint["mode"]),
        )
        if actual != expected:
            raise ValueError(f"Checkpoint {path} identity {actual!r} does not match expected {expected!r}.")
        try:
            run, epoch = int(checkpoint["run"]), int(checkpoint["epoch"])
            int(checkpoint["seed"])
            data_seed = int((checkpoint.get("arguments") or {})["seed"])
            best_validation = float(checkpoint["best_validation_metric"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Checkpoint {path} has invalid run, epoch, seed, data seed, or validation metric.") from exc
        except KeyError as exc:
            raise ValueError(f"Checkpoint {path} is missing arguments.seed (the fixed data split/panel seed).") from exc
        if run <= 0 or epoch <= 0 or run != checkpoint_number(path) or run in seen_runs:
            raise ValueError(f"Checkpoint {path} has an invalid or duplicate run/epoch identity.")
        if checkpoint["timed_out"] is not False:
            raise RuntimeError(f"Checkpoint {path} is marked timed_out or has non-boolean timeout metadata and cannot be evaluated.")
        if str(checkpoint["checkpoint_type"]) != "best_validation_model_state":
            raise RuntimeError(f"Checkpoint {path} is not a best_validation_model_state checkpoint.")
        if not np.isfinite(best_validation):
            raise ValueError(f"Checkpoint {path} has a non-finite best validation metric.")
        config, state = checkpoint["model_config"], checkpoint["model_state_dict"]
        if not isinstance(config, dict) or not config:
            raise ValueError(f"Checkpoint {path} has no resolved model_config.")
        if not isinstance(state, dict) or not state:
            raise ValueError(f"Checkpoint {path} has no model state.")
        if any(not isinstance(key, str) or not torch.is_tensor(value) for key, value in state.items()):
            raise ValueError(f"Checkpoint {path} model_state_dict must contain only string-to-tensor entries.")
        seen_runs.add(run)
        data_seeds.add(data_seed)
        fingerprints.add(_checkpoint_construction_fingerprint(checkpoint))
    if requested and seen_runs != requested:
        raise FileNotFoundError(f"Preflighted runs differ from --runs: expected={sorted(requested)}, actual={sorted(seen_runs)}.")
    if len(fingerprints) != 1:
        raise ValueError("Selected checkpoints do not share one model construction and state-layout fingerprint.")
    if len(data_seeds) != 1:
        raise ValueError(f"Selected checkpoints do not share one fixed data split/panel seed: {sorted(data_seeds)}.")
    if expected_data_seed is not None and data_seeds != {int(expected_data_seed)}:
        raise ValueError(f"Checkpoint data seed {sorted(data_seeds)} differs from requested seed {int(expected_data_seed)}.")
    return sorted(seen_runs)


def metadata(explicit, checkpoints, key):
    if explicit is not None:
        return str(explicit)
    values = {str(checkpoint[key]) for checkpoint in checkpoints if checkpoint.get(key) is not None}
    if len(values) != 1:
        raise ValueError(f"Could not infer {key} from checkpoints: {sorted(values)}")
    return values.pop()


def resolve_checkpoint_model_construction(checkpoint, framework, dataset, reported_model_name):
    del checkpoint, framework, dataset
    return str(reported_model_name)


def checkpoint_provenance(checkpoint, model_name, *, framework=None, dataset=None):
    del framework, dataset
    config = dict(checkpoint.get("model_config") or {})
    implementation = config.get("model_implementation") or checkpoint.get("model_implementation")
    return {
        "checkpoint_type": str(checkpoint.get("checkpoint_type") or "unspecified"),
        "model_implementation": str(implementation or normalized_model_name(model_name)),
        "compatibility_fingerprint": None,
        "upstream_checkpoint_compatible": None,
    }


def saved_cap(checkpoint, framework, dataset):
    arguments = checkpoint.get("arguments") or {}
    if arguments.get("eval_cap") is not None:
        return int(arguments["eval_cap"])
    mode = str(arguments.get("mode", checkpoint.get("mode", ""))).lower()
    if framework == "pyg":
        return 500
    if mode in HEART_MODES and framework == "ogb":
        from ogbl.protocol import resolve_ogbl_eval_cap

        return resolve_ogbl_eval_cap(None, mode, dataset)
    return 500


def resolve_cap(value, checkpoint=None, framework=None, dataset=None):
    value = str(value).strip().lower()
    if value == "checkpoint":
        return saved_cap(checkpoint, framework, dataset) if checkpoint is not None else 500
    if value in {"all", "full", "entire", "0"}:
        return 0
    value = int(value)
    if value < 0:
        raise ValueError("--eval-cap cannot be negative.")
    return value


def resolve_ranked_selector_cap(value, dataset):
    del dataset
    normalized = str(value).strip().lower().replace("_", "-")
    if normalized in {"checkpoint", "all", "full", "entire", "0"}:
        return 0
    try:
        numeric = int(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Ranked selector modes use --test-positive-cap for test-query selection; unsupported auxiliary --eval-cap value {value!r}."
        ) from exc
    if numeric < 0:
        raise ValueError("--eval-cap cannot be negative.")
    if numeric != 0:
        raise ValueError(
            "Ranked selector modes use --test-positive-cap for test-query selection; their auxiliary --eval-cap must be checkpoint/all/0."
        )
    return 0


def ranked_selector_positive_scope(dataset, test_positive_cap):
    """Describe the deterministic test-query panel used by ranked modes."""
    del dataset
    cap = int(test_positive_cap)
    if cap < 0:
        raise ValueError("--test-positive-cap cannot be negative.")
    return "complete-test-positive-split" if cap == 0 else f"deterministic-test-positive-cap-{cap}"


class SavedNode2Vec(torch.nn.Module):

    def __init__(self, weight):
        super().__init__()
        self.decode_is_dedup_safe = True
        self.decode_is_symmetric = True
        self.embedding = torch.nn.Parameter(weight.detach().clone(), requires_grad=False)

    def embed(self, data):
        return self.embedding

    def decode(self, z, edge_index):
        return (z[edge_index[0]] * z[edge_index[1]]).sum(dim=-1)


def extract_n2v_weight(state_dict):
    for key in ["encoder.node2vec.embedding.weight", "node2vec.embedding.weight", "embedding.weight"]:
        if torch.is_tensor(state_dict.get(key)):
            return state_dict[key]
    values = [value for (key, value) in state_dict.items() if key.endswith("node2vec.embedding.weight")]
    if len(values) == 1:
        return values[0]
    raise KeyError("Node2Vec embedding table was not found in the checkpoint.")


def _n2v_tensor_sha256(tensor):
    tensor = tensor.detach().to(device="cpu").contiguous()
    byte_view = memoryview(tensor.numpy()).cast("B")
    digest = hashlib.sha256()
    chunk_bytes = 64 * 1024**2
    for start in range(0, len(byte_view), chunk_bytes):
        digest.update(byte_view[start : start + chunk_bytes])
    return digest.hexdigest()


def _load_shared_n2v_embedding(checkpoint, config):
    arguments = dict(checkpoint.get("arguments") or {})
    saved = (
        checkpoint.get("n2v_embedding_path")
        or config.get("n2v_embedding_path")
        or arguments.get("reference_embedding_path_resolved")
        or arguments.get("reference_embedding_path")
        or arguments.get("paper_embedding_path_resolved")
        or arguments.get("paper_embedding_path")
    )
    if not saved:
        raise KeyError("Reference Node2Vec checkpoint does not name its shared embedding artifact.")
    saved_path = Path(str(saved)).expanduser()
    candidates = [saved_path]
    if not saved_path.is_absolute():
        candidates.append(PROJECT_ROOT / saved_path)
    configured_cache = os.environ.get("N2V_EMBEDDING_CACHE_DIR")
    if configured_cache:
        candidates.append(Path(configured_cache).expanduser() / saved_path.name)
    candidates.append(Path("/ephemeral/ubuntu/LinkPrediction/n2v_embeddings") / saved_path.name)
    root = project_path(arguments.get("root", "dataset"))
    candidates.append(root / saved_path.name)
    candidates = list(dict.fromkeys(candidates))
    embedding_path = next((path for path in candidates if path.is_file()), None)
    if embedding_path is None:
        checked = ", ".join((str(path) for path in candidates))
        raise FileNotFoundError(f"Shared reference Node2Vec embedding was not found. Checked: {checked}")
    cache_key = str(embedding_path.resolve())
    cached = _SHARED_N2V_EMBEDDING_CACHE.get(cache_key)
    if cached is None:
        try:
            artifact = torch.load(embedding_path, map_location="cpu", weights_only=False)
        except TypeError:
            artifact = torch.load(embedding_path, map_location="cpu")
        artifact_recipe = None
        artifact_checksum = None
        embedding = artifact
        if isinstance(artifact, dict):
            artifact_recipe = artifact.get("recipe")
            artifact_checksum = artifact.get("embedding_sha256")
            embedding = artifact.get("entity_embedding", artifact.get("embedding"))
        if not torch.is_tensor(embedding) or embedding.dim() != 2:
            raise ValueError(f"Expected a 2-D tensor in {embedding_path}, got {type(embedding).__name__}.")
        embedding = embedding.detach().float().contiguous()
        actual_sha256 = _n2v_tensor_sha256(embedding)
        if artifact_checksum and str(artifact_checksum) != actual_sha256:
            raise ValueError(
                f"Shared reference Node2Vec cache checksum mismatch: artifact declares {artifact_checksum}, but {embedding_path} contains {actual_sha256}."
            )
        _SHARED_N2V_EMBEDDING_CACHE[cache_key] = (embedding, actual_sha256, artifact_recipe)
    else:
        (embedding, actual_sha256, artifact_recipe) = cached
    expected_sha256 = config.get("embedding_sha256")
    if expected_sha256 and str(expected_sha256) != actual_sha256:
        raise ValueError(
            f"Shared reference Node2Vec embedding checksum mismatch: checkpoint expects {expected_sha256}, but {embedding_path} is {actual_sha256}."
        )
    expected_recipe_digest = config.get("embedding_recipe_digest")
    if expected_recipe_digest:
        if not isinstance(artifact_recipe, dict):
            raise ValueError(
                f"This reference Node2Vec checkpoint requires a recipe-keyed embedding artifact, but {embedding_path} has no recipe."
            )
        encoded_recipe = json.dumps(artifact_recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")
        actual_recipe_digest = hashlib.sha256(encoded_recipe).hexdigest()
        if str(expected_recipe_digest) != actual_recipe_digest:
            raise ValueError(
                f"Shared reference Node2Vec recipe mismatch: checkpoint expects {expected_recipe_digest}, but {embedding_path} has {actual_recipe_digest}."
            )
    return (embedding.float(), embedding_path)


class _ReferenceN2VEvaluationAdapter(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.decode_is_symmetric = True
        self.decode_is_dedup_safe = True
        self.reference_evaluation_transform = getattr(model, "reference_evaluation_transform", "identity")
        self.reference_evaluation_row_batch_size = int(getattr(model, "reference_evaluation_row_batch_size", 0))
        self.reference_evaluation_negative_layout = getattr(model, "reference_evaluation_negative_layout", "grouped")

    def embed(self, data=None):
        return self.model.embed(data)

    def decode(self, z, edge_index):
        (src, dst) = edge_index
        return self.model.predictor(z[src], z[dst]).view(-1)


def build_model(checkpoint, framework, dataset, model_name, bundle, device):
    state_dict = checkpoint["model_state_dict"]
    normalized = str(model_name).lower().replace("-", "").replace("_", "")
    config = dict(checkpoint.get("model_config") or {})
    arguments = dict(checkpoint.get("arguments") or {})
    if normalized in {"n2v", "node2vec"}:
        protocol = (
            str(checkpoint.get("n2v_protocol") or config.get("protocol") or arguments.get("n2v_protocol_effective") or "legacy-direct")
            .strip()
            .lower()
        )
        if protocol in {"reference", "paper"}:
            from model.node2vec_model import ReferenceN2VLink

            if framework == "ogb":
                data = bundle["data"]
                raw_x = getattr(data, "x", None)
                num_nodes = int(data.num_nodes)
            elif framework == "pyg":
                raw_x = bundle.get("x")
                num_nodes = int(raw_x.size(0)) if torch.is_tensor(raw_x) else 0
            else:
                raise ValueError(f"Unsupported framework for reference Node2Vec: {framework!r}.")
            feature_composition = (
                str(
                    config.get("feature_composition")
                    or arguments.get("reference_feature_composition")
                    or arguments.get("paper_feature_composition")
                    or "raw-plus-node2vec"
                )
                .strip()
                .lower()
            )
            if feature_composition not in {"node2vec-only", "raw-plus-node2vec"}:
                raise ValueError(f"Unsupported reference Node2Vec feature composition: {feature_composition!r}.")
            if feature_composition == "raw-plus-node2vec" and raw_x is None:
                raise ValueError("Reference Node2Vec raw-plus-node2vec reconstruction requires raw node features.")
            expected_num_nodes = config.get("num_nodes")
            if expected_num_nodes is not None and int(expected_num_nodes) != num_nodes:
                raise ValueError(f"Reference Node2Vec node-count mismatch: checkpoint expects {expected_num_nodes}, bundle has {num_nodes}.")
            expected_raw_dim = config.get("raw_feature_dim")
            if feature_composition == "node2vec-only":
                if expected_raw_dim is not None and int(expected_raw_dim) != 0:
                    raise ValueError(
                        f"Reference Node2Vec node2vec-only checkpoint must declare raw_feature_dim=0, but declares {expected_raw_dim}."
                    )
            elif expected_raw_dim is not None and int(raw_x.size(1)) != int(expected_raw_dim):
                raise ValueError(
                    f"Reference Node2Vec raw feature width mismatch: checkpoint expects {expected_raw_dim}, bundle has {raw_x.size(1)}."
                )
            expected_feature_sha256 = config.get("base_feature_sha256")
            if feature_composition == "raw-plus-node2vec" and expected_feature_sha256:
                actual_feature_sha256 = _n2v_tensor_sha256(raw_x.to(dtype=torch.float32))
                if str(expected_feature_sha256) != actual_feature_sha256:
                    raise ValueError(
                        f"Reference Node2Vec base feature checksum mismatch: checkpoint expects {expected_feature_sha256}, but the evaluation bundle has {actual_feature_sha256}."
                    )
            (n2v_embedding, embedding_path) = _load_shared_n2v_embedding(checkpoint, config)
            expected_embedding_dim = config.get("n2v_embedding_dim")
            if expected_embedding_dim is not None and int(n2v_embedding.size(1)) != int(expected_embedding_dim):
                raise ValueError(
                    f"Reference Node2Vec embedding width mismatch: checkpoint expects {expected_embedding_dim}, artifact has {n2v_embedding.size(1)}."
                )
            if int(n2v_embedding.size(0)) != num_nodes:
                raise ValueError(f"Shared Node2Vec embedding has {n2v_embedding.size(0)} rows but {dataset} has {num_nodes} nodes.")
            n2v_device = n2v_embedding.to(device=device, dtype=torch.float32, non_blocking=True)
            raw_device = None
            if feature_composition == "node2vec-only":
                features = n2v_device
            else:
                raw_device = raw_x.to(device=device, dtype=torch.float32, non_blocking=True)
                features = torch.cat([raw_device, n2v_device], dim=-1)
            expected_input_channels = config.get("input_channels")
            if expected_input_channels is not None and int(features.size(1)) != int(expected_input_channels):
                raise ValueError(
                    f"Reference Node2Vec reconstructed feature width mismatch: checkpoint expects {expected_input_channels}, rebuilt features have {features.size(1)}."
                )
            model = ReferenceN2VLink(
                input_channels=int(config.get("input_channels", features.size(1))),
                hidden_channels=int(config.get("hidden_channels", 128)),
                num_layers=int(config.get("num_layers", 3)),
                predictor_layers=int(config.get("predictor_layers", 3)),
                dropout=float(config.get("dropout", 0.0)),
                node_encode_batch_size=int(config.get("node_encode_batch_size", 262144)),
            ).to(device)
            model.load_state_dict(state_dict)
            model.set_node_features(features)
            model.n2v_embedding_path = str(embedding_path)
            if raw_device is not None:
                del raw_device
            del n2v_device, n2v_embedding
            adapter = _ReferenceN2VEvaluationAdapter(model).to(device).eval()
            if int(adapter.reference_evaluation_row_batch_size) <= 0:
                adapter.reference_evaluation_row_batch_size = int(config.get("batch_size") or 1024)
            return adapter
        return SavedNode2Vec(extract_n2v_weight(state_dict)).to(device).eval()
    from model.pairwise_models import get_model

    data = bundle["x"] if framework == "pyg" else bundle["data"]
    params = {
        **config,
        "in_channels": int(data.size(1) if framework == "pyg" else data.x.size(-1)),
        "num_nodes": int(data.size(0) if framework == "pyg" else data.num_nodes),
        "dataset_name": dataset,
        "evaluation_mode": str(checkpoint.get("mode", arguments.get("mode", "all"))),
        "train_samples_per_epoch": int(config.get("train_samples_per_epoch", arguments.get("train_samples_per_epoch", 0) or 0)),
        "stage1_train_samples_per_epoch": int(
            config.get("stage1_train_samples_per_epoch", arguments.get("train_samples_per_epoch", 0) or 0)
        ),
    }
    if framework == "ogb":
        params["train_edge_index"] = getattr(data, "edge_index", None)
        fixed_feature_preprocessing = is_aggregated_mlp(normalized) and str(dataset).lower() == "ogbl-ddi"
        params["use_node_emb"] = str(dataset).lower() == "ogbl-ddi" and (not fixed_feature_preprocessing)
        if arguments.get("model_decode_batch_size") is not None:
            params["decode_batch_size"] = int(arguments["model_decode_batch_size"])
    model = get_model(model_name, params)
    compact_loader = getattr(model, "load_checkpoint_state_dict", None)
    if callable(compact_loader):
        compact_loader(state_dict, strict=True)
    else:
        model.load_state_dict(state_dict, strict=True)
    if hasattr(model, "configure_epoch"):
        epoch = max(1, int(checkpoint.get("epoch", 1)))
        total_epochs = int(arguments.get("epochs") or config.get("epochs") or 500)
        model.configure_epoch(epoch, max(epoch, total_epochs))
    return model.to(device).eval()


def resolve_checkpoint_aggregated_recipe(checkpoint, model_name):
    config = dict(checkpoint.get("model_config") or {})
    arguments = dict(checkpoint.get("arguments") or {})
    method = aggregated_mlp_method(model_name)
    if method is None:
        raise ValueError(f"{model_name} is not a fixed-preprocessing MLP method.")
    method_label = "PPR" if method == "ppr" else method
    preprocessing_key = f"{method}_preprocessing"
    recipe = config.get(preprocessing_key)
    if recipe is None:
        if method == "ppr":
            legacy_alpha = config.get("alpha", arguments.get("alpha"))
            legacy_detail = f" (saved alpha={legacy_alpha})" if legacy_alpha is not None else ""
            raise ValueError(
                f"This is a legacy residual-blend PPR checkpoint{legacy_detail}. The current PPR method concatenates normalized X and A_norm X, so its MLP input width is different. Retrain PPR before full-graph evaluation."
            )
        raise ValueError(
            f"{method} checkpoint is missing its explicit preprocessing recipe at model_config[{preprocessing_key!r}]. Retrain {method} before full-graph evaluation."
        )
    recipe = str(recipe)
    expected_recipe = aggregated_mlp_recipe(method)
    if recipe != expected_recipe:
        raise ValueError(f"Unsupported {method_label} preprocessing recipe '{recipe}'; expected '{expected_recipe}'.")
    base_dim = int(config.get(f"{method}_base_feature_dim", 0))
    output_dim = int(config.get(f"{method}_output_feature_dim", 0))
    if base_dim <= 0 or output_dim != 2 * base_dim:
        raise ValueError(
            f"{method_label} checkpoint has invalid feature dimensions: base={base_dim}, output={output_dim}; expected output=2*base."
        )
    return (recipe, base_dim, output_dim)


def prepare_model_features(checkpoint, framework, dataset, model_name, bundle, device):
    if not is_aggregated_mlp(model_name):
        return
    method = aggregated_mlp_method(model_name)
    method_label = "PPR" if method == "ppr" else method
    (recipe, base_dim, output_dim) = resolve_checkpoint_aggregated_recipe(checkpoint, model_name)
    config = dict(checkpoint.get("model_config") or {})
    featureless_dim = int(config.get(f"{method}_feature_dim", config.get("emb_size", 0))) if str(dataset).lower() == "ogbl-ddi" else None
    featureless_seed = int(config.get(f"{method}_feature_seed", 0))
    if featureless_dim is not None and featureless_dim != base_dim:
        raise ValueError(
            f"{method_label} checkpoint DDI sketch width does not match its saved base feature width: sketch={featureless_dim}, base={base_dim}."
        )
    preprocessing_key = (recipe, base_dim, output_dim, featureless_dim, featureless_seed)
    bundle_cache_key = f"_{method}_preprocessing_key"
    if bundle.get(bundle_cache_key) == preprocessing_key:
        return
    if framework == "pyg":
        training_graph = bundle.get("adj", bundle["train_pos"])
        if device.type == "cuda":
            training_graph = training_graph.to(device)
            if "adj" in bundle:
                bundle["adj"] = training_graph
        bundle["x"] = preprocess_aggregated_mlp(
            model_name,
            dataset,
            bundle["x"].to(device, non_blocking=True),
            training_graph,
            featureless_dim=featureless_dim,
            featureless_seed=featureless_seed,
        )
    else:
        data = bundle["data"]
        if getattr(data, "x", None) is not None and data.x.dtype != torch.float:
            data.x = data.x.float()
        training_graph = getattr(data, "adj_t", None)
        if training_graph is None:
            training_graph = bundle.get("adj", bundle["train_pos"])
        if device.type == "cuda":
            training_graph = training_graph.to(device)
            if getattr(data, "adj_t", None) is not None:
                data.adj_t = training_graph
        data.x = preprocess_aggregated_mlp(
            model_name,
            dataset,
            data.x.to(device, non_blocking=True) if data.x is not None else None,
            training_graph,
            featureless_dim=featureless_dim,
            featureless_seed=featureless_seed,
        )
        if "x" in bundle:
            bundle["x"] = data.x
    actual_output_dim = int(bundle["x"].size(-1) if framework == "pyg" else bundle["data"].x.size(-1))
    if actual_output_dim != output_dim:
        raise ValueError(
            f"Reconstructed {method_label} feature width does not match the checkpoint: got {actual_output_dim}, expected {output_dim}."
        )
    bundle[bundle_cache_key] = preprocessing_key


def checkpoint_batch_size(checkpoint, model, requested):
    arguments = checkpoint.get("arguments") or {}
    config = checkpoint.get("model_config") or {}
    value = int(requested)
    if value <= 0:
        value = int(arguments.get("eval_batch_size") or arguments.get("edge_batch_size") or config.get("eval_batch_size") or 65536)
    return max(1, value)


def pyg_embedding(model, bundle, device):
    adj = bundle["adj"].to(device) if device.type == "cuda" else bundle["adj"]
    x = bundle["x"].to(device, non_blocking=True)
    prepare_fixed_graph = getattr(model, "prepare_fixed_graph", None)
    if callable(prepare_fixed_graph):
        prepare_fixed_graph(bundle["train_pos"], int(x.size(0)))
    data = SimpleNamespace(
        x=x,
        adj_t=adj,
        edge_index=None,
        train_pos=bundle["train_pos"],
        num_nodes=int(x.size(0)),
        csr_rowptr=bundle["csr_train_rowptr"],
        csr_col=bundle["csr_train_col"],
    )
    return model.embed(data)


def ogb_embeddings(model, bundle, dataset, device, batch_size, split):
    from ogbl.train_eval import _ensure_test_embedding, move_graph_data_to_device, prepare_ogbl_evaluation

    data = bundle["data"]
    if getattr(data, "x", None) is not None and data.x.dtype != torch.float:
        data.x = data.x.float()
    data = move_graph_data_to_device(data, device)
    bundle["data"] = data
    eval_edges = {"pos_train_edge": bundle["train_pos"], "pos_valid_edge": bundle["valid_pos"]}
    valid_input_pos = bundle.get("valid_input_pos")
    if valid_input_pos is None:
        valid_input_pos = getattr(data, "valid_input_pos", None)
    if torch.is_tensor(valid_input_pos):
        eval_edges["valid_input_pos"] = valid_input_pos
    context = prepare_ogbl_evaluation(
        model=model, data=data, eval_edges=eval_edges, dataset_name=dataset, device=device, batch_size=batch_size, test_only=split == "test"
    )
    valid = context["z_train"] if split in {"valid", "both"} else None
    test = _ensure_test_embedding(context) if split in {"test", "both"} else None
    return (valid, test, context)


def rank_metrics(ranks):
    ranks = ranks.to(torch.float64).cpu()
    if ranks.numel() == 0:
        return {"MRR": 0.0, **{f"Hits@{k}": 0.0 for k in HITS_K}}
    return {"MRR": float((1.0 / ranks).mean()), **{f"Hits@{k}": float((ranks <= k).to(torch.float64).mean()) for k in HITS_K}}


def rank_hit_counts(ranks):
    ranks = ranks.to(torch.float64).cpu()
    return {f"Hits@{k}": int((ranks <= k).sum().item()) for k in HITS_K}


def _grouped_split_result(positives, grouped, ranks, auc, candidate_label):
    count = int(positives.size(0))
    candidates_per_positive = int(grouped.size(1))
    metrics = rank_metrics(ranks)
    result = {
        "num_positive_edges": count,
        "metrics": metrics,
        "hit_counts": rank_hit_counts(ranks),
        "hit_rate_resolution": 1.0 / count if count else None,
        "candidate_counts": grouped_candidate_summary(positives, grouped),
        "negative_group_shape": [count, candidates_per_positive, 2],
        "rank_definition": "0.5 * (#grouped negative >= positive + #grouped negative > positive) + 1",
    }
    if auc is not None:
        metrics["AUC"] = float(auc)
        result["auc_definition"] = (
            f"binary AUC over all split positives and the flattened grouped {candidate_label} negatives; ties receive 0.5"
        )
        result["num_auc_negative_predictions"] = count * candidates_per_positive
    return result


def _auc_wins(sorted_positive, scores):
    lower = torch.searchsorted(sorted_positive, scores, right=False)
    upper = torch.searchsorted(sorted_positive, scores, right=True)
    return sorted_positive.numel() - upper.to(torch.float64) + 0.5 * (upper - lower).to(torch.float64)


@torch.inference_mode()
def _evaluate_endpoint_grouped_split(
    positives,
    grouped,
    scoring_grouped,
    positive_scores,
    score_negative_edges,
    edge_batch_size,
    quiet,
    description,
    candidate_label,
    endpoint_score_reuse_safe,
    compute_auc,
):
    if endpoint_score_reuse_safe is not True:
        raise RuntimeError(
            "Endpoint-grouped checkpoint evaluation reuses one decoded score for every occurrence of an identical directed edge, but this checkpoint model does not declare decode_is_dedup_safe=True. Refusing to materialize or silently change its score semantics."
        )
    scoring_base = getattr(scoring_grouped, "_grouped", scoring_grouped)
    if not bool(getattr(scoring_base, "is_endpoint_grouped_negative", False)):
        raise ValueError("Endpoint-grouped candidates require an endpoint-grouped scoring view.")
    if scoring_base is not grouped:
        source_digest = getattr(grouped, "manifest_sha256", None)
        scoring_digest = getattr(scoring_base, "manifest_sha256", None)
        if source_digest is None or source_digest != scoring_digest:
            raise ValueError("Endpoint scoring view is not bound to the evaluated candidate artifact.")
    count = int(positives.size(0))
    candidates_per_positive = int(grouped.size(1))
    work_device = positive_scores.device
    sorted_positive = torch.sort(positive_scores).values if compute_auc else None
    ge_total = torch.zeros(count, dtype=torch.int32, device=work_device)
    gt_total = torch.zeros_like(ge_total)
    candidate_counts = torch.zeros_like(ge_total)
    auc_numerator = torch.zeros((), dtype=torch.float64, device=work_device) if compute_auc else None
    batch_size = max(1, int(edge_batch_size))
    query_oriented = bool(getattr(scoring_grouped, "endpoint_query_oriented", False))
    decoded_union_edges = 0
    logical_occurrences = 0
    shard_count = len(getattr(scoring_base, "shards", ())) or None
    iterator = tqdm(scoring_grouped.iter_endpoint_group_chunks(), total=shard_count, desc=description, unit="shard", disable=quiet)
    for chunk in iterator:
        union_lengths = chunk.union_rowptr[1:] - chunk.union_rowptr[:-1]
        union_endpoints = torch.repeat_interleave(chunk.endpoints, union_lengths)
        if int(union_endpoints.numel()) != int(chunk.union_nodes.numel()):
            raise ValueError("Endpoint-grouped union pointers do not match union nodes.")
        if int(chunk.side) == 0 or query_oriented:
            union_edges = torch.stack([union_endpoints, chunk.union_nodes], dim=1)
        else:
            union_edges = torch.stack([chunk.union_nodes, union_endpoints], dim=1)
        score_parts = []
        for start in range(0, int(union_edges.size(0)), batch_size):
            end = min(start + batch_size, int(union_edges.size(0)))
            values = score_negative_edges(union_edges[start:end]).detach().view(-1)
            if not values.is_floating_point():
                values = values.float()
            values = values.to(device=work_device, dtype=positive_scores.dtype)
            if int(values.numel()) != end - start:
                raise RuntimeError(f"Expected {end - start} endpoint-union scores, got {values.numel()}.")
            score_parts.append(values)
        union_scores = torch.cat(score_parts) if score_parts else torch.empty(0, dtype=positive_scores.dtype, device=work_device)
        decoded_union_edges += int(union_scores.numel())
        local_indices = chunk.candidate_local_indices.to(device=work_device, dtype=torch.long)
        occurrence_group = chunk.occurrence_endpoint_index.to(device=work_device, dtype=torch.long)
        union_rowptr = chunk.union_rowptr.to(device=work_device, dtype=torch.long)
        global_indices = local_indices + union_rowptr[occurrence_group].view(-1, 1)
        occurrence_scores = union_scores[global_indices]
        row_ids = chunk.occurrence_row_ids.to(device=work_device, dtype=torch.long)
        logical_occurrences += int(occurrence_scores.numel())
        thresholds = positive_scores[row_ids].view(-1, 1)
        rank_counts = torch.stack([(occurrence_scores >= thresholds).sum(dim=1), (occurrence_scores > thresholds).sum(dim=1)], dim=0).to(
            torch.int32
        )
        ge_total.index_add_(0, row_ids, rank_counts[0])
        gt_total.index_add_(0, row_ids, rank_counts[1])
        candidate_counts.index_add_(
            0, row_ids, torch.full((int(row_ids.numel()),), int(grouped.negatives_per_side), dtype=torch.int32, device=work_device)
        )
        if compute_auc:
            multiplicity = chunk.union_occurrence_multiplicity.to(device=work_device, dtype=torch.float64)
            auc_numerator += (_auc_wins(sorted_positive, union_scores) * multiplicity).sum()
    if not bool(candidate_counts.eq(candidates_per_positive).all().item()):
        raise ValueError(
            "Endpoint-grouped checkpoint evaluation did not consume exactly one complete left and right side for every positive row."
        )
    total_negatives = count * candidates_per_positive
    if logical_occurrences != total_negatives:
        raise ValueError("Endpoint-grouped logical occurrence count does not match its shape.")
    ranks = 0.5 * (ge_total.float() + gt_total.float()) + 1.0
    auc_denominator = count * total_negatives
    auc = float((auc_numerator / auc_denominator).item()) if compute_auc and auc_denominator else (0.0 if compute_auc else None)
    result = _grouped_split_result(positives, grouped, ranks, auc, candidate_label)
    result["endpoint_grouped_reuse"] = {
        "decoded_union_edges": decoded_union_edges,
        "logical_candidate_occurrences": logical_occurrences,
        "decode_reuse_ratio": logical_occurrences / decoded_union_edges if decoded_union_edges else 0.0,
    }
    return result


@torch.inference_mode()
def evaluate_grouped_split(
    pos_edges,
    negative_edges,
    positive_scores,
    score_negative_edges,
    edge_batch_size,
    quiet,
    description,
    candidate_label="HeaRT",
    negative_edges_for_scoring=None,
    endpoint_score_reuse_safe=False,
    compute_auc=True,
):
    positives = pos_edges.long().cpu()
    grouped = grouped_negative_edges(negative_edges, positives)
    count = int(positives.size(0))
    candidates_per_positive = int(grouped.size(1))
    positive_scores = positive_scores.detach().view(-1)
    if not positive_scores.is_floating_point():
        positive_scores = positive_scores.float()
    if int(positive_scores.numel()) != count:
        raise RuntimeError(f"Expected {count} positive scores, got {positive_scores.numel()}.")
    if count == 0:
        return _grouped_split_result(positives, grouped, torch.empty(0), 0.0 if compute_auc else None, candidate_label)
    if negative_edges_for_scoring is None:
        scoring_grouped = grouped
    else:
        scoring_grouped = grouped_negative_edges(negative_edges_for_scoring, positives)
        if tuple(scoring_grouped.shape) != tuple(grouped.shape):
            raise ValueError("Scoring-oriented grouped negatives changed candidate shape.")
    if bool(getattr(grouped, "is_endpoint_grouped_negative", False)):
        return _evaluate_endpoint_grouped_split(
            positives,
            grouped,
            scoring_grouped,
            positive_scores,
            score_negative_edges,
            edge_batch_size,
            quiet,
            description,
            candidate_label,
            endpoint_score_reuse_safe,
            compute_auc,
        )
    work_device = positive_scores.device
    sorted_positive = torch.sort(positive_scores).values if compute_auc else None
    ge_total = torch.zeros(count, dtype=torch.int64, device=work_device)
    gt_total = torch.zeros_like(ge_total)
    auc_numerator = torch.zeros((), dtype=torch.float64, device=work_device) if compute_auc else None
    streaming_scoring = bool(getattr(scoring_grouped, "is_streaming_negative", False))
    total_negatives = count * candidates_per_positive
    batch_size = max(1, int(edge_batch_size))
    if streaming_scoring:

        def negative_batches():
            for row_start, row_end, edge_block in scoring_grouped.iter_chunks():
                flat_block = edge_block.reshape(-1, 2)
                global_start = int(row_start) * candidates_per_positive
                for local_start in range(0, int(flat_block.size(0)), batch_size):
                    local_end = min(local_start + batch_size, int(flat_block.size(0)))
                    yield (global_start + local_start, global_start + local_end, flat_block[local_start:local_end])

    else:
        flat_negatives = scoring_grouped.reshape(-1, 2)

        def negative_batches():
            for start in range(0, total_negatives, batch_size):
                end = min(start + batch_size, total_negatives)
                yield (start, end, flat_negatives[start:end])

    iterator = tqdm(
        negative_batches(),
        total=None if streaming_scoring else (total_negatives + batch_size - 1) // batch_size,
        desc=description,
        unit="batch",
        disable=quiet,
    )
    for start, end, edge_rows in iterator:
        scores = score_negative_edges(edge_rows).detach().view(-1)
        if not scores.is_floating_point():
            scores = scores.float()
        scores = scores.to(device=work_device, dtype=positive_scores.dtype)
        if int(scores.numel()) != end - start:
            raise RuntimeError(f"Expected {end - start} negative scores, got {scores.numel()}.")
        rows = torch.arange(start, end, device=work_device, dtype=torch.long)
        rows = torch.div(rows, candidates_per_positive, rounding_mode="floor")
        thresholds = positive_scores[rows]
        ge_total.index_add_(0, rows, (scores >= thresholds).to(dtype=torch.int64))
        gt_total.index_add_(0, rows, (scores > thresholds).to(dtype=torch.int64))
        if compute_auc:
            auc_numerator += _auc_wins(sorted_positive, scores).sum()
    ranks = 0.5 * (ge_total.float() + gt_total.float()) + 1.0
    auc_denominator = count * total_negatives
    auc = float((auc_numerator / auc_denominator).item()) if compute_auc and auc_denominator else (0.0 if compute_auc else None)
    return _grouped_split_result(positives, grouped, ranks, auc, candidate_label)


@torch.inference_mode()
def evaluate_split(
    pos_edges,
    rowptr,
    col,
    framework,
    score_all_nodes,
    nodes,
    filter_existing,
    comparison_batch_size,
    quiet,
    description,
    positive_scores=None,
    compute_auc=True,
):
    pos_edges = pos_edges.long().cpu().contiguous()
    if isinstance(rowptr, dict) != isinstance(col, dict):
        raise TypeError("Direction-aware full-graph filters require both rowptr and col to be dictionaries.")
    if isinstance(rowptr, dict):
        if set(rowptr) != set(col):
            raise ValueError("Direction-aware rowptr/col filters must expose the same keys.")
        rowptr = {direction: value.long().cpu() for (direction, value) in rowptr.items()}
        col = {direction: value.long().cpu() for (direction, value) in col.items()}

        def filter_csr(direction):
            if direction not in rowptr:
                raise KeyError(f"No full-graph filter CSR is available for {direction!r}.")
            return (rowptr[direction], col[direction])

    else:
        rowptr = rowptr.long().cpu()
        col = col.long().cpu()

        def filter_csr(direction):
            del direction
            return (rowptr, col)

    count = int(pos_edges.size(0))
    if count == 0:
        empty = torch.empty(0, dtype=torch.long)
        metrics = rank_metrics(empty)
        if compute_auc and (positive_scores is not None or bool(getattr(score_all_nodes, "is_retrieval_rerank", False))):
            metrics["AUC"] = 0.0
        return {
            "num_positive_edges": 0,
            "metrics": metrics,
            "hit_counts": rank_hit_counts(empty),
            "hit_rate_resolution": None,
            "candidate_counts": candidate_summary(empty, empty, nodes),
        }
    sorted_positive = None
    auc_numerator = None
    auc_negative_count = None
    retrieval_rerank = bool(getattr(score_all_nodes, "is_retrieval_rerank", False))
    query_auc_sum = 0.0
    query_auc_count = 0
    retrieval_hits = 0
    retrieval_queries = 0
    if positive_scores is not None:
        positive_scores = positive_scores.detach().view(-1)
        if not positive_scores.is_floating_point():
            positive_scores = positive_scores.float()
        if int(positive_scores.numel()) != count:
            raise RuntimeError(f"Expected {count} positive scores, got {positive_scores.numel()}.")
        if compute_auc:
            sorted_positive = torch.sort(positive_scores).values
            auc_numerator = torch.zeros((), dtype=torch.float64, device=sorted_positive.device)
    accumulator_device = torch.device("cpu") if retrieval_rerank else positive_scores.device
    ge_total = torch.zeros(count, dtype=torch.int64, device=accumulator_device)
    gt_total = torch.zeros_like(ge_total)
    left_counts = torch.zeros_like(ge_total)
    right_counts = torch.zeros_like(ge_total)
    if sorted_positive is not None:
        auc_negative_count = torch.zeros((), dtype=torch.int64, device=accumulator_device)
    groups = endpoint_groups(pos_edges, framework)
    query_score_many = getattr(score_all_nodes, "score_queries", None)

    def scored_groups():
        entries = list(groups.items())
        if callable(query_score_many):
            for key, items in entries:
                yield (key, items, None)
            return
        score_many = getattr(score_all_nodes, "score_many", None)
        if not callable(score_many):
            for (endpoint, direction), items in entries:
                (direction_rowptr, direction_col) = filter_csr(direction)
                filter_start = int(direction_rowptr[endpoint])
                filter_end = int(direction_rowptr[endpoint + 1])
                excluded_nodes = direction_col[filter_start:filter_end] if filter_existing else None
                yield ((endpoint, direction), items, score_all_nodes(endpoint, direction, excluded_nodes=excluded_nodes).view(-1))
            return
        endpoint_batch_size = max(1, int(getattr(score_all_nodes, "score_many_batch_size", 1)))
        for start in range(0, len(entries), endpoint_batch_size):
            block = entries[start : start + endpoint_batch_size]
            endpoints = torch.tensor([key[0] for (key, _items) in block], dtype=torch.long)
            directions = [key[1] for (key, _items) in block]
            score_matrix = score_many(endpoints, directions)
            if tuple(score_matrix.shape) != (int(nodes), len(block)):
                raise RuntimeError(
                    f"Expected batched full-graph scores with shape {(int(nodes), len(block))}, got {tuple(score_matrix.shape)}."
                )
            for column, (key, items) in enumerate(block):
                yield (key, items, score_matrix[:, column].contiguous())

    iterator = tqdm(scored_groups(), total=len(groups), desc=description, unit="endpoint", disable=quiet)
    for (endpoint, direction), items, scores in iterator:
        (direction_rowptr, direction_col) = filter_csr(direction)
        rows = torch.tensor([item[0] for item in items], dtype=torch.long, device=accumulator_device)
        counterparts = torch.tensor([item[1] for item in items], dtype=torch.long, device=accumulator_device)
        sides = torch.tensor([item[2] for item in items], dtype=torch.long, device=accumulator_device)
        filter_start = int(direction_rowptr[endpoint])
        filter_end = int(direction_rowptr[endpoint + 1])
        endpoint_cpu = torch.tensor([int(endpoint)], dtype=torch.long)
        if filter_existing and filter_end > filter_start:
            excluded_cpu = torch.unique(torch.cat([endpoint_cpu, direction_col[filter_start:filter_end].to(dtype=torch.long)]), sorted=True)
        else:
            excluded_cpu = endpoint_cpu
        if callable(query_score_many):
            excluded_positions = torch.searchsorted(excluded_cpu, counterparts.cpu(), right=False)
            positive_in_candidates = (excluded_positions >= excluded_cpu.numel()) | (
                excluded_cpu[excluded_positions.clamp_max(excluded_cpu.numel() - 1)] != counterparts.cpu()
            )
            valid_candidate_count = int(nodes) - int(excluded_cpu.numel())
            negative_counts = (valid_candidate_count - positive_in_candidates.to(torch.int64)).to(accumulator_device)
            left_mask = sides == 0
            left_counts[rows[left_mask]] = negative_counts[left_mask]
            right_counts[rows[~left_mask]] = negative_counts[~left_mask]
            ge = torch.zeros(len(items), dtype=torch.int64, device=accumulator_device)
            gt = torch.zeros_like(ge)
            query_batch = min(
                max(1, int(comparison_batch_size)), max(1, int(getattr(score_all_nodes, "score_queries_batch_size", comparison_batch_size)))
            )
            excluded_neighbors = direction_col[filter_start:filter_end] if filter_existing else None
            for start in range(0, len(items), query_batch):
                end = min(start + query_batch, len(items))
                query_nodes = counterparts[start:end]
                score_matrix = query_score_many(endpoint, direction, query_nodes, excluded_nodes=excluded_neighbors)
                expected_shape = (int(nodes), end - start)
                if tuple(score_matrix.shape) != expected_shape:
                    raise RuntimeError(
                        f"Expected query-aware full-graph scores with shape {expected_shape}, got {tuple(score_matrix.shape)}."
                    )
                score_device = score_matrix.device
                query_device = query_nodes.to(device=score_device, dtype=torch.long, non_blocking=True)
                columns = torch.arange(end - start, device=score_device, dtype=torch.long)
                thresholds = score_matrix[query_device, columns]
                excluded_device = excluded_cpu.to(device=score_device, non_blocking=True)
                excluded_scores = score_matrix[excluded_device]
                ge_block = (score_matrix >= thresholds[None, :]).sum(dim=0, dtype=torch.int64)
                gt_block = (score_matrix > thresholds[None, :]).sum(dim=0, dtype=torch.int64)
                ge_block -= (excluded_scores >= thresholds[None, :]).sum(dim=0, dtype=torch.int64)
                gt_block -= (excluded_scores > thresholds[None, :]).sum(dim=0, dtype=torch.int64)
                ge_block -= positive_in_candidates[start:end].to(device=score_device, dtype=torch.int64)
                ge[start:end] = ge_block.clamp_min(0).to(accumulator_device)
                gt[start:end] = gt_block.to(accumulator_device)
                query_hits = getattr(score_all_nodes, "last_query_retrieval_hits", torch.empty(0, dtype=torch.bool)).view(-1)
                if int(query_hits.numel()) != end - start:
                    raise RuntimeError("Query-aware cascade did not report one retrieval indicator per positive.")
                retrieval_hits += int(query_hits.sum().item())
                retrieval_queries += end - start
                if compute_auc:
                    negative_device = negative_counts[start:end].to(device=score_device, dtype=torch.float64)
                    valid_auc = negative_device > 0
                    if bool(valid_auc.any()):
                        ge_float = ge_block.to(torch.float64)
                        gt_float = gt_block.to(torch.float64)
                        tied_negative = (ge_float - gt_float).clamp_min(0.0)
                        wins = negative_device - gt_float - 0.5 * tied_negative
                        query_auc_sum += float((wins[valid_auc] / negative_device[valid_auc]).sum().item())
                        query_auc_count += int(valid_auc.sum().item())
            ge_total.index_add_(0, rows, ge)
            gt_total.index_add_(0, rows, gt)
            continue
        if int(scores.numel()) != int(nodes):
            raise RuntimeError(f"Expected {nodes} full-graph scores, got {scores.numel()}.")
        score_device = scores.device
        counterparts_device = counterparts.to(score_device, non_blocking=True)
        thresholds = scores[counterparts_device]
        if retrieval_rerank:
            retrieved = getattr(score_all_nodes, "last_retrieved_nodes", torch.empty(0, dtype=torch.long, device=score_device)).to(
                score_device
            )
            retrieval_hits += int(torch.isin(counterparts_device, retrieved).sum().item())
            retrieval_queries += int(counterparts_device.numel())
        excluded_device = excluded_cpu.to(device=score_device, non_blocking=True)
        excluded_positions = torch.searchsorted(excluded_device, counterparts_device, right=False)
        positive_in_candidates = (excluded_positions >= excluded_device.numel()) | (
            excluded_device[excluded_positions.clamp_max(excluded_device.numel() - 1)] != counterparts_device
        )
        excluded_scores = scores[excluded_device]
        valid_candidate_count = int(nodes) - int(excluded_device.numel())
        negative_counts = (valid_candidate_count - positive_in_candidates.to(torch.int64)).to(accumulator_device)
        if sorted_positive is not None:
            auc_scores = scores.to(device=sorted_positive.device, dtype=sorted_positive.dtype)
            contributions = _auc_wins(sorted_positive, auc_scores)
            excluded_auc_scores = excluded_scores.to(device=sorted_positive.device, dtype=sorted_positive.dtype)
            excluded_contributions = _auc_wins(sorted_positive, excluded_auc_scores)
            auc_numerator += len(items) * (contributions.sum() - excluded_contributions.sum())
            included = counterparts_device[positive_in_candidates]
            included_scores = scores[included].to(device=sorted_positive.device, dtype=sorted_positive.dtype)
            auc_numerator -= _auc_wins(sorted_positive, included_scores).sum()
            auc_negative_count.add_(negative_counts.sum())
        left_mask = sides == 0
        left_counts[rows[left_mask]] = negative_counts[left_mask]
        right_counts[rows[~left_mask]] = negative_counts[~left_mask]
        ge = torch.zeros(len(items), dtype=torch.int64, device=score_device)
        gt = torch.zeros_like(ge)
        query_batch = max(1, int(comparison_batch_size))
        for start in range(0, len(items), query_batch):
            end = min(start + query_batch, len(items))
            block = thresholds[start:end]
            ge[start:end] = (scores[:, None] >= block[None, :]).sum(dim=0, dtype=torch.int64)
            gt[start:end] = (scores[:, None] > block[None, :]).sum(dim=0, dtype=torch.int64)
            ge[start:end] -= (excluded_scores[:, None] >= block[None, :]).sum(dim=0, dtype=torch.int64)
            gt[start:end] -= (excluded_scores[:, None] > block[None, :]).sum(dim=0, dtype=torch.int64)
        ge -= positive_in_candidates.to(torch.int64)
        ge_total.index_add_(0, rows, ge.clamp_min(0).to(accumulator_device))
        gt_total.index_add_(0, rows, gt.to(accumulator_device))
        if retrieval_rerank and compute_auc:
            negative_device = negative_counts.to(device=score_device, dtype=torch.float64)
            valid_auc = negative_device > 0
            if bool(valid_auc.any()):
                ge_float = ge.to(torch.float64)
                gt_float = gt.to(torch.float64)
                tied_negative = (ge_float - gt_float).clamp_min(0.0)
                wins = negative_device - gt_float - 0.5 * tied_negative
                query_auc_sum += float((wins[valid_auc] / negative_device[valid_auc]).sum().item())
                query_auc_count += int(valid_auc.sum().item())
    ranks = 0.5 * (ge_total.float() + gt_total.float()) + 1.0
    definition = "0.5 * (#negative >= positive + #negative > positive) + 1"
    metrics = rank_metrics(ranks)
    if retrieval_rerank and compute_auc:
        metrics["AUC"] = query_auc_sum / query_auc_count if query_auc_count else 0.0
    elif sorted_positive is not None:
        negative_count = int(auc_negative_count.item())
        denominator = int(count) * negative_count
        metrics["AUC"] = float((auc_numerator / denominator).item()) if denominator else 0.0
    result = {
        "num_positive_edges": count,
        "metrics": metrics,
        "hit_counts": rank_hit_counts(ranks),
        "hit_rate_resolution": 1.0 / count,
        "candidate_counts": candidate_summary(left_counts, right_counts, nodes),
        "rank_definition": definition,
    }
    if retrieval_rerank:
        if compute_auc:
            result["auc_definition"] = "mean query-conditioned AUC over every legal full-graph negative; ties receive 0.5"
        result["retrieval"] = {
            "protocol": "factorized_retrieval_then_llama_rerank",
            "filtered_query_positive_handling": "restore_only_the_current_query_before_topk_reranking",
            "retrieve_k": int(getattr(score_all_nodes, "retrieve_k", 0)),
            "positive_retrieval_hits": retrieval_hits,
            "positive_queries": retrieval_queries,
            "positive_recall_at_k": retrieval_hits / retrieval_queries if retrieval_queries else 0.0,
            "missed_positive_handling": "retains_retriever_rank",
        }
    return result


def model_scorer(model, z, device, chunk_size, *, filter_existing):
    nodes = int(z.size(0))
    decoder = getattr(model, "decoder", None)
    dot_product = isinstance(model, SavedNode2Vec) or (decoder is not None and decoder.__class__.__name__ == "DotProductDecoder")
    chunk_size = max(1, int(chunk_size))
    cascade = getattr(model, "score_full_graph_nodes", None)
    max_score_elements = 64000000
    preserve_dtype = bool(getattr(model, "reference_evaluation_preserve_dtype", False))
    strict_lpformer = (
        str(getattr(model, "implementation_name", "")).strip().lower() == "lpformer-optimized-adaptation"
        or model.__class__.__name__ == "LPFormerPredictor"
    )
    if strict_lpformer:
        if bool(model.training):
            raise RuntimeError(
                "LPFormer full-graph evaluation received a model in training mode; narrow training-path fallback is disabled."
            )
        uses_wide = getattr(model, "_uses_wide_evaluation_blocks", None)
        if not callable(uses_wide) or not bool(uses_wide()):
            raise RuntimeError(
                "LPFormer full-graph evaluation has no validated wide decode path for this dataset; narrow fallback is disabled."
            )
        configured_decode_batch = int(model.evaluation_decode_batch_size)
        if configured_decode_batch < 65536:
            raise RuntimeError(
                f"LPFormer full-graph evaluation has an invalid strict decode contract: evaluation_decode_batch_size={configured_decode_batch}, minimum=65536."
            )
        required_chunk = min(nodes, configured_decode_batch)
        if chunk_size < required_chunk:
            raise ValueError(
                f"LPFormer full-graph node chunk is smaller than its strict evaluation decode contract: node_chunk_size={chunk_size}, required_at_least={required_chunk}. Automatic slow fallback is disabled."
            )

    def evaluation_scores(values):
        shape = tuple(values.shape)
        values = values.reshape(-1)
        if not preserve_dtype or not values.is_floating_point():
            values = values.float()
        return _apply_reference_evaluation_transform(model, values).reshape(shape)

    def exact_dot_many(endpoint_index):
        endpoint_index = endpoint_index.to(device=device, dtype=torch.long, non_blocking=True)
        endpoint_count = int(endpoint_index.numel())
        if endpoint_count == 0:
            return torch.empty((nodes, 0), dtype=z.dtype if preserve_dtype else torch.float32, device=device)
        feature_count = max(1, int(z.size(1)))
        if torch.device(device).type == "cuda":
            total_memory = int(torch.cuda.get_device_properties(device).total_memory)
            temporary_bytes = min(1 * 1024**3, max(64 * 1024**2, int(total_memory * 0.03)))
        else:
            temporary_bytes = 256 * 1024**2
        temporary_elements = max(1, temporary_bytes // max(1, int(z.element_size())))
        row_chunk = max(1, temporary_elements // (endpoint_count * feature_count))
        fixed = z[endpoint_index]
        output = torch.empty((nodes, endpoint_count), dtype=z.dtype if preserve_dtype else torch.float32, device=device)
        for start in range(0, nodes, row_chunk):
            end = min(start + row_chunk, nodes)
            products = z[start:end, None, :] * fixed[None, :, :]
            values = products.reshape(-1, feature_count).sum(dim=-1).view(end - start, endpoint_count)
            if not preserve_dtype:
                values = values.to(torch.float32)
            output[start:end] = values
        return evaluation_scores(output).reshape(nodes, endpoint_count)

    def score(endpoint, direction, *, excluded_nodes=None):
        endpoint = int(endpoint)
        if callable(cascade):
            values = cascade(z, endpoint, direction, filter_existing=bool(filter_existing), excluded_nodes=excluded_nodes)
            values = evaluation_scores(values)
            topk_getter = getattr(model, "last_full_graph_topk", None)
            score.last_retrieved_nodes = (
                topk_getter().detach() if callable(topk_getter) else torch.empty(0, dtype=torch.long, device=device)
            )
            return values
        if dot_product:
            return exact_dot_many(torch.tensor([endpoint], dtype=torch.long))[:, 0]
        out = None
        for start in range(0, nodes, chunk_size):
            end = min(start + chunk_size, nodes)
            candidates = torch.arange(start, end, device=device, dtype=torch.long)
            edges = edge_tensor(endpoint, candidates, direction, device)
            values = evaluation_scores(model.decode(z, edges))
            if out is None:
                out = torch.empty(nodes, dtype=values.dtype, device=values.device)
            elif preserve_dtype and (values.dtype != out.dtype or values.device != out.device):
                raise RuntimeError("A dtype-preserving reference evaluator returned inconsistent score dtypes or devices across chunks.")
            out[start:end] = values.to(device=out.device, dtype=out.dtype)
        if out is not None:
            return out
        return torch.empty(0, dtype=z.dtype if preserve_dtype else torch.float32, device=z.device)

    if callable(cascade):
        cascade_queries = getattr(model, "score_full_graph_queries", None)
        if not callable(cascade_queries):
            raise RuntimeError(
                "The full-graph cascade checkpoint does not expose the batched score_full_graph_queries interface. Per-query fallback is disabled because it can turn a bounded evaluation into an unbounded slow run."
            )

        def score_queries(endpoint, direction, query_nodes, *, excluded_nodes=None):
            query_nodes = torch.as_tensor(query_nodes, device=device, dtype=torch.long).view(-1)
            values = cascade_queries(
                z, int(endpoint), direction, query_nodes, filter_existing=bool(filter_existing), excluded_nodes=excluded_nodes
            )
            hit_getter = getattr(model, "last_full_graph_query_hits", None)
            score.last_query_retrieval_hits = (
                hit_getter().detach() if callable(hit_getter) else torch.zeros(query_nodes.numel(), dtype=torch.bool, device=device)
            )
            values = evaluation_scores(values)
            expected = (nodes, int(query_nodes.numel()))
            if tuple(values.shape) != expected:
                raise RuntimeError(f"Expected query-aware cascade scores with shape {expected}, got {tuple(values.shape)}.")
            return values

        score.score_queries = score_queries
        score.score_queries_batch_size = max(1, min(32, 8000000 // max(1, nodes)))
    if dot_product and (not callable(cascade)):

        def score_many(endpoints, directions):
            del directions
            return exact_dot_many(endpoints)

        score.score_many = score_many
        score.score_many_batch_size = max(1, min(64, max_score_elements // max(1, nodes)))
        score.score_backend = "exact_elementwise_dot_reduction"
    elif callable(cascade):
        score.score_backend = "factorized_retrieval_then_rerank"
    else:
        score.score_backend = "checkpoint_model_decode"
    score.is_retrieval_rerank = callable(cascade)
    score.last_retrieved_nodes = torch.empty(0, dtype=torch.long, device=device)
    score.last_query_retrieval_hits = torch.empty(0, dtype=torch.bool, device=device)
    score.retrieve_k = int(getattr(model, "full_graph_retrieve_k", 0))
    return score


@torch.inference_mode()
def score_positive_edges(model, z, pos_edges, framework, device, batch_size, canonicalize_pyg=True):
    edges = pos_edges.to(device=device, dtype=torch.long)
    preserve_dtype = bool(getattr(model, "reference_evaluation_preserve_dtype", False))
    scores = []
    for start in range(0, int(edges.size(0)), max(1, int(batch_size))):
        chunk = edges[start : start + max(1, int(batch_size))]
        if framework == "pyg" and canonicalize_pyg:
            chunk = torch.stack([torch.minimum(chunk[:, 0], chunk[:, 1]), torch.maximum(chunk[:, 0], chunk[:, 1])], dim=1)
        values = model.decode(z, chunk.t().contiguous()).view(-1)
        if not preserve_dtype or not values.is_floating_point():
            values = values.float()
        scores.append(_apply_reference_evaluation_transform(model, values))
    return torch.cat(scores) if scores else torch.empty(0, dtype=torch.float64 if preserve_dtype else torch.float32, device=device)


def model_edge_scorer(model, z, device):
    preserve_dtype = bool(getattr(model, "reference_evaluation_preserve_dtype", False))

    def score(edges):
        chunk = edges.to(device=device, dtype=torch.long, non_blocking=True)
        values = model.decode(z, chunk.t().contiguous()).view(-1)
        if not preserve_dtype or not values.is_floating_point():
            values = values.float()
        return _apply_reference_evaluation_transform(model, values)

    return score


def _cached_edge_scorer(scores, *, backend="ordered-score-row-grouped-cache"):
    offset = 0
    scores = scores.detach().view(-1)

    def score(edges):
        nonlocal offset
        count = int(edges.size(0))
        end = offset + count
        if end > int(scores.numel()):
            raise RuntimeError("Ordered grouped-score cache was exhausted")
        output = scores[offset:end]
        offset = end
        return output

    score.score_backend = str(backend)
    return score


def release_model(model, context=None):
    if context is not None:
        from ogbl.train_eval import release_ogbl_evaluation

        release_ogbl_evaluation(context)
    for name in ["clear_decode_cache", "clear_runtime_cache"]:
        function = getattr(model, name, None)
        if callable(function):
            function()


def make_evaluation_decode_strict(model):
    from model.pairwise_models import FastAdvancedPredictor

    if getattr(getattr(model, "decode", None), "__func__", None) is not FastAdvancedPredictor.decode:
        return
    block_size = int(model.decode_batch_size)
    if block_size <= 0:
        raise ValueError("decode_batch_size must be positive.")

    def decode(self, z, edges):
        if edges.size(0) != 2:
            edges = edges.t().contiguous()
        edges = edges.to(device=z.device, dtype=torch.long, non_blocking=True)
        if edges.size(1) == 0:
            return z.new_empty((0,))
        outputs = [self._decode_block(z, edges[:, start : start + block_size]) for start in range(0, edges.size(1), block_size)]
        return torch.cat(outputs, dim=0)

    model.decode = MethodType(decode, model)


def parse_args(evaluator):
    """Parse common arguments plus one evaluator's own arguments."""
    parser = argparse.ArgumentParser(description=f"Run the independent {evaluator.POLICY} evaluator.")
    add = parser.add_argument
    add("--framework", choices=["auto", "pyg", "ogb", "ogbl"], default="auto")
    add("--mode")
    add("--dataset")
    add("--model")
    add("--heuristic", choices=["all", "cn", "aa", "ra", "shortest_path", "katz"])
    add("--seed", type=int, default=0)
    add("--source-batch-size", type=int)
    add("--shortest-path-cutoff", type=int)
    add("--katz-beta", type=float)
    add("--katz-max-length", type=int)
    add("--checkpoint", action="append", default=[])
    add("--checkpoint-root", default="checkpoints")
    add("--runs", nargs="+", type=int)
    add("--split", choices=["test"], default="test")
    add("--candidate-policy", choices=[evaluator.POLICY], default=evaluator.POLICY)
    if bool(getattr(evaluator, "USES_FIXED_PLANETOID_INPUTS", False)):
        add(
            "--planetoid-input-root",
            help="Optional root containing fixed Planetoid positive splits and gnn_feature; negatives are never loaded from it.",
        )
    add("--eval-cap", "--positive-cap", dest="eval_cap", default="checkpoint")
    add("--test-positive-cap", type=int, default=100000)
    add("--root", default="dataset")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--edge-batch-size", type=int, default=0)
    add("--node-chunk-size", type=int, default=0)
    add("--comparison-batch-size", type=int, default=256)
    add("--compute-auc", choices=["yes", "no"], default="yes")
    add("--output")
    add("--no-save", action="store_true")
    add("--quiet", action="store_true")
    evaluator.add_evaluator_arguments(parser)
    return parser.parse_args()


def load_bundle(framework, dataset, root, seed, eval_cap):
    if framework == "pyg":
        from pyg.heart_generation import _prepare_ranked_base
        from pyg.prepare_data import resolve_pyg_eval_cap

        eval_cap = resolve_pyg_eval_cap(eval_cap, "heart", dataset)
        bundle = _prepare_ranked_base(dataset, (0.85, 0.05, 0.1), seed, str(root), eval_cap, default_eval_cap=0)
        bundle["mode"] = "all"
        return bundle
    from ogbl import prepare_data
    from ogbl.protocol import resolve_ogbl_eval_cap

    eval_cap = resolve_ogbl_eval_cap(eval_cap, "heart", dataset)
    if str(dataset).lower() == "ogbl-citation2":
        bundle = prepare_data._load_citation2_base(dataset, str(root), int(eval_cap), int(seed))
    else:
        bundle = prepare_data._load_noncitation_base(dataset, str(root), int(eval_cap), int(seed))
    prepare_data._restore_complete_test_split(bundle, dataset)
    return bundle


def load_fixed_planetoid_positive_base(dataset, root, seed, eval_cap, *, planetoid_input_root=None):
    from pyg.planetoid_inputs import load_fixed_planetoid_positive_split, missing_fixed_planetoid_positive_split_message
    from pyg.heart_generation import _prepare_ranked_base

    positive_split = load_fixed_planetoid_positive_split(dataset, root=str(root), input_root=planetoid_input_root)
    if positive_split is None:
        raise FileNotFoundError(
            missing_fixed_planetoid_positive_split_message(
                dataset, root=str(root), input_root=planetoid_input_root
            )
        )
    bundle = _prepare_ranked_base(
        dataset, (0.85, 0.05, 0.1), int(seed), str(root), int(eval_cap), default_eval_cap=0, positive_split=positive_split
    )
    bundle["all_test_pos"] = positive_split["test_pos"].contiguous()
    bundle.update(
        {
            "mode": "heart",
            "negative_candidate_source": "generated-locally",
            "paired_heart_positive_scope": True,
        }
    )
    return bundle


def load_ranked_selector_pyg_base(dataset, root, seed):
    from pyg.heart_generation import _prepare_ranked_base

    dataset_key = str(dataset).strip().lower()
    positive_split = None
    if dataset_key in HEART_BENCHMARK_PYG_DATASETS:
        from pyg.planetoid_inputs import load_fixed_planetoid_positive_split

        positive_split = load_fixed_planetoid_positive_split(dataset, root=str(root))
        if positive_split is None:
            raise FileNotFoundError(
                f"Ranked {dataset} selector evaluation requires the fixed train_pos.txt, valid_pos.txt, test_pos.txt, and gnn_feature files under the dataset root {root}. Grouped HeaRT negative files are not required."
            )
    bundle = _prepare_ranked_base(
        dataset, (0.85, 0.05, 0.1), int(seed), str(root), 100000, default_eval_cap=0, positive_split=positive_split
    )
    bundle["mode"] = "ranked-selector"
    bundle["ranked_positive_split_source"] = (
        "fixed-planetoid-benchmark-positive-txt" if positive_split is not None else "seeded-pyg-edge-split"
    )
    bundle["ranked_feature_source"] = "fixed-planetoid-gnn-feature" if positive_split is not None else "raw-pyg-dataset-x"
    return bundle


def load_ranked_selector_bundle(framework, dataset, root, seed):
    if framework == "pyg":
        return load_ranked_selector_pyg_base(dataset, root, seed)
    bundle = load_bundle(framework, dataset, root, seed, 100000)
    bundle["mode"] = "ranked-selector"
    bundle["ranked_positive_split_source"] = "complete-native-ogb-split"
    bundle["ranked_feature_source"] = "native-ogb-selector-input"
    return bundle


def heart_bundle_metadata(bundle):
    return heart_candidate_metadata(bundle)


def ensure_complete_ranked_positive_splits(bundle, *, framework, dataset, root, seed):
    if framework == "ogb":
        if not isinstance(bundle.get("split_edge"), dict):
            raise ValueError("OGB ranked-selector evaluation requires the complete split_edge payload.")
        return
    if torch.is_tensor(bundle.get("all_valid_pos")) and torch.is_tensor(bundle.get("all_test_pos")):
        bundle.setdefault("concat_full_valid_pos", bundle["all_valid_pos"])
        bundle.setdefault("concat_full_test_pos", bundle["all_test_pos"])
        return
    from pyg.data_core import _load_dataset, _load_or_create_split

    data = _load_dataset(dataset, str(root))
    (_train_uv, valid_uv, test_uv, _rowptr, _col) = _load_or_create_split(data, dataset, (0.85, 0.05, 0.1), int(seed), str(root))
    all_valid_pos = valid_uv.t().to(torch.long).contiguous()
    all_test_pos = test_uv.t().to(torch.long).contiguous()
    bundle["all_valid_pos"] = all_valid_pos
    bundle["all_test_pos"] = all_test_pos
    bundle["concat_full_valid_pos"] = all_valid_pos
    bundle["concat_full_test_pos"] = all_test_pos


def install_complete_test_positive_scope(bundle, *, framework, dataset, root, seed, candidate_policy, test_positive_cap=0):
    policy = str(candidate_policy).strip().lower()
    if not policy:
        raise ValueError("candidate_policy is required for complete test-positive installation.")
    selected_test = bundle.get("test_pos")
    selected_test_rows = int(selected_test.size(0)) if torch.is_tensor(selected_test) else None
    validation = bundle.get("valid_pos")
    validation_rows = int(validation.size(0)) if torch.is_tensor(validation) else None
    effective_validation_cap = bundle.get("effective_validation_cap", bundle.get("effective_eval_cap"))
    ensure_complete_ranked_positive_splits(bundle, framework=framework, dataset=dataset, root=root, seed=seed)
    complete_test = bundle.get("all_test_pos")
    if not torch.is_tensor(complete_test):
        raise ValueError(f"{policy} evaluation requires the complete test-positive split.")
    complete_test = complete_test.to(torch.long).cpu().contiguous()
    if complete_test.dim() != 2 or int(complete_test.size(1)) != 2:
        raise ValueError(f"Complete test positives must have shape [N,2], got {tuple(complete_test.shape)}.")
    requested_cap = int(test_positive_cap)
    if requested_cap < 0:
        raise ValueError("--test-positive-cap cannot be negative.")
    complete_rows = int(complete_test.size(0))
    if requested_cap > 0 and complete_rows > requested_cap:
        panel_seed = int(seed) + 101
        generator = torch.Generator(device="cpu")
        generator.manual_seed(panel_seed)
        panel_indices = torch.randperm(complete_rows, generator=generator)[:requested_cap].contiguous()
        selected_test = complete_test.index_select(0, panel_indices).contiguous()
        panel_recipe = "torch-randperm-seed-plus-101-prefix-v1"
    else:
        panel_seed = None
        panel_indices = torch.arange(complete_rows, dtype=torch.long)
        selected_test = complete_test
        panel_recipe = "complete-ordered-test-split-v1"

    def tensor_sha256(tensor):
        value = tensor.detach().to("cpu").contiguous()
        digest = hashlib.sha256()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        if value.numel():
            byte_view = memoryview(value.reshape(-1).view(torch.uint8).numpy()).cast("B")
            block_size = 64 * 1024 * 1024
            for start in range(0, len(byte_view), block_size):
                digest.update(byte_view[start : start + block_size])
        return digest.hexdigest()

    bundle["all_test_pos"] = complete_test
    bundle["test_pos"] = selected_test
    bundle.pop("test_neg", None)
    bundle["effective_test_cap"] = requested_cap
    positive_query_scope = ranked_selector_positive_scope(dataset, requested_cap)
    metadata = {
        "policy": positive_query_scope,
        "candidate_policy": policy,
        "cap": "all" if requested_cap == 0 else requested_cap,
        "requested_cap": requested_cap,
        "test_positive_rows": int(selected_test.size(0)),
        "complete_test_positive_rows": complete_rows,
        "loader_selected_test_positive_rows": selected_test_rows,
        "panel_recipe": panel_recipe,
        "panel_seed": panel_seed,
        "panel_index_sha256": tensor_sha256(panel_indices),
        "selected_test_positive_sha256": tensor_sha256(selected_test),
        "complete_test_positive_sha256": tensor_sha256(complete_test),
        "validation_positive_rows": validation_rows,
        "effective_validation_cap": effective_validation_cap,
        "validation_modified": False,
        "source": bundle.get(
            "ranked_positive_split_source",
            bundle.get(
                "heart_positive_split_source",
                "complete-native-ogb-split" if str(framework).strip().lower() == "ogb" else "complete-seeded-pyg-split",
            ),
        ),
    }
    bundle["test_positive_scope"] = metadata
    return dict(metadata)


def complete_test_positive_result_metadata(bundle, *, auxiliary_loader_eval_cap=None):
    scope = bundle.get("test_positive_scope")
    if not isinstance(scope, dict):
        raise ValueError("Complete test-positive scope metadata is missing.")
    metadata = {"positive_query_scope": str(scope["policy"]), "positive_eval_cap": scope["cap"], "test_positive_scope": dict(scope)}
    if auxiliary_loader_eval_cap is not None:
        metadata["auxiliary_loader_eval_cap"] = int(auxiliary_loader_eval_cap)
    return metadata


def ensure_pyg_full_known_positive_filter(bundle, *, dataset, root, seed):
    required_filter_keys = ("csr_valid_known_rowptr", "csr_valid_known_col", "csr_test_known_rowptr", "csr_test_known_col")
    metadata_key = "full_graph_known_positive_filter"
    if all((torch.is_tensor(bundle.get(key)) for key in required_filter_keys)):
        return dict(bundle[metadata_key])
    ensure_complete_ranked_positive_splits(bundle, framework="pyg", dataset=dataset, root=root, seed=seed)
    nodes = num_nodes(bundle, "pyg")
    split_edges = {"train": bundle["train_pos"], "uncapped_valid": bundle["all_valid_pos"], "uncapped_test": bundle["all_test_pos"]}
    normalized = []
    for split_name, edges in split_edges.items():
        edges = torch.as_tensor(edges, dtype=torch.long).cpu().contiguous()
        if edges.dim() != 2 or int(edges.size(1)) != 2:
            raise ValueError(f"{split_name} positives must have shape [N,2], got {tuple(edges.shape)}.")
        if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= nodes):
            raise ValueError(f"{split_name} positives contain node ids outside [0, {nodes}).")
        normalized.append(edges)
    from torch_sparse import SparseTensor

    def build_filter(parts):
        row = torch.cat([edges[:, 0] for edges in parts] + [edges[:, 1] for edges in parts])
        col = torch.cat([edges[:, 1] for edges in parts] + [edges[:, 0] for edges in parts])
        adjacency = SparseTensor(row=row, col=col, sparse_sizes=(nodes, nodes)).coalesce()
        (rowptr, columns, _value) = adjacency.csr()
        return (rowptr.to(torch.long).cpu().contiguous(), columns.to(torch.long).cpu().contiguous())

    valid_parts = normalized[:1]
    test_parts = normalized[:2]
    policy = "fixed_split_observed_positive_filter"
    positive_scope = "validation=train;test=train+uncapped_valid"
    protocol_revision = str(bundle.get("heart_selection", "pyg-generated-fixed-split-mask-hard-heart"))
    (valid_rowptr, valid_col) = build_filter(valid_parts)
    (test_rowptr, test_col) = build_filter(test_parts)
    bundle["csr_valid_known_rowptr"] = valid_rowptr
    bundle["csr_valid_known_col"] = valid_col
    bundle["csr_test_known_rowptr"] = test_rowptr
    bundle["csr_test_known_col"] = test_col
    bundle["csr_all_known_rowptr"] = test_rowptr
    bundle["csr_all_known_col"] = test_col
    metadata = {
        "policy": policy,
        "protocol_revision": protocol_revision,
        "positive_scope": positive_scope,
        "applies_to_splits": ["valid", "test"],
        "directionality": "undirected",
        "query_positive_handling": "decoded_as_rank_threshold_excluded_from_negative_candidates",
        "positive_split_source": bundle.get("heart_positive_split_source", "seeded-pyg-edge-split"),
        "train_positive_rows": int(normalized[0].size(0)),
        "uncapped_valid_positive_rows": int(normalized[1].size(0)),
        "uncapped_test_positive_rows": int(normalized[2].size(0)),
        "valid_filtered_directed_edges": int(valid_col.numel()),
        "test_filtered_directed_edges": int(test_col.numel()),
        "csr_cached_in_bundle": True,
    }
    bundle[metadata_key] = metadata
    return dict(metadata)


def ensure_ogb_full_known_positive_filter(bundle, *, dataset):
    metadata_key = "full_graph_known_positive_filter"
    normalized_dataset = str(dataset).strip().lower()
    required_keys = (
        "heart_valid_out_rowptr",
        "heart_valid_out_col",
        "heart_valid_in_rowptr",
        "heart_valid_in_col",
        "heart_test_out_rowptr",
        "heart_test_out_col",
        "heart_test_in_rowptr",
        "heart_test_in_col",
    )
    if all((torch.is_tensor(bundle.get(key)) for key in required_keys)) and metadata_key in bundle:
        return dict(bundle[metadata_key])
    required_positive_keys = ("train_pos", "all_valid_pos", "all_test_pos")
    missing = [key for key in required_positive_keys if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise ValueError("OGB exhaustive evaluation requires the loader's uncapped positive tensors. Missing: " + ", ".join(missing))
    from ogbl.data_core import _ensure_heart_eligibility_filters

    _ensure_heart_eligibility_filters(bundle, normalized_dataset)
    if normalized_dataset == "ogbl-citation2":
        from torch_sparse import SparseTensor

        nodes = int(bundle["num_nodes"])
        test_filter = SparseTensor(
            rowptr=bundle["heart_test_out_rowptr"],
            col=bundle["heart_test_out_col"],
            sparse_sizes=(nodes, nodes),
            is_sorted=False,
        ).coalesce()
        (test_rowptr, test_col, _) = test_filter.csr()
        test_rowptr = test_rowptr.cpu().contiguous()
        test_col = test_col.cpu().contiguous()
        for role in ("out", "in"):
            bundle[f"heart_test_{role}_rowptr"] = test_rowptr
            bundle[f"heart_test_{role}_col"] = test_col
    missing = [key for key in required_keys if not torch.is_tensor(bundle.get(key))]
    if missing:
        raise RuntimeError("OGB HeaRT eligibility construction did not publish: " + ", ".join(missing))
    policy = str(bundle["heart_eligibility_policy"])
    directionality = str(bundle["heart_eligibility_orientation"])
    temporal_collab = normalized_dataset == "ogbl-collab"
    positive_scope = (
        "validation=train;test=train+uncapped_valid"
        if temporal_collab
        else (
            "validation=train;test=train+evaluated-valid"
            if normalized_dataset == "ogbl-ppa"
            else "validation=train;test=train+uncapped_valid"
        )
    )
    score_graph_policy = (
        "valid=train-weighted;test=train-weighted+uncapped-valid-unit-weight" if temporal_collab else "valid=train;test=train"
    )
    filter_views = {
        split_name: {
            "out": "source_to_legal_target_filter" if normalized_dataset == "ogbl-citation2" else "shared_undirected_filter",
            "in": "target_to_legal_source_filter" if normalized_dataset == "ogbl-citation2" else "shared_undirected_filter",
        }
        for split_name in ("valid", "test")
    }
    metadata = {
        "policy": policy,
        "protocol_revision": str(bundle.get("heart_candidate_protocol", "ogb-generated-filtered-hard-heart")),
        "positive_scope": positive_scope,
        "applies_to_splits": ["valid", "test"],
        "directionality": directionality,
        "filter_views": filter_views,
        "query_positive_handling": "decoded_as_rank_threshold_excluded_from_negative_candidates",
        "self_loop_candidate_handling": "excluded",
        "score_graph_policy": score_graph_policy,
        "positive_split_source": "ogb_loader_uncapped_positive_tensors",
        "train_positive_rows": int(bundle["train_pos"].size(0)),
        "uncapped_valid_positive_rows": int(bundle["all_valid_pos"].size(0)),
        "uncapped_test_positive_rows": int(bundle["all_test_pos"].size(0)),
        "eligibility_shared_with_generated_heart": True,
        "valid_out_filter_nnz": int(bundle["heart_valid_out_col"].numel()),
        "valid_in_filter_nnz": int(bundle["heart_valid_in_col"].numel()),
        "test_out_filter_nnz": int(bundle["heart_test_out_col"].numel()),
        "test_in_filter_nnz": int(bundle["heart_test_in_col"].numel()),
        "csr_cached_in_bundle": True,
    }
    bundle[metadata_key] = metadata
    return dict(metadata)


def full_known_positive_filter_views(bundle, framework, dataset, split_name):
    if framework == "pyg":
        prefix = f"csr_{split_name}_known"
        return (bundle[f"{prefix}_rowptr"], bundle[f"{prefix}_col"])
    del dataset
    prefix = f"heart_{split_name}"
    return (
        {"out": bundle[f"{prefix}_out_rowptr"], "in": bundle[f"{prefix}_in_rowptr"]},
        {"out": bundle[f"{prefix}_out_col"], "in": bundle[f"{prefix}_in_col"]},
    )


def configure_bundle_candidate_precision(framework, dataset, model_name, device):
    if framework != "ogb":
        return None
    return _configure_cuda_matmul_precision(device, dataset, model_name)


def _uses_collab_heart_heuristic_protocol(framework, dataset):
    return str(framework).strip().lower() == "ogb" and str(dataset).strip().lower() == "ogbl-collab"


_LEARNEDFEAT_PATH_MAX_DENSE_ELEMS = {
    # Match the specialized FullGraph workspace budgets while retaining the
    # generic targeted scorer used by LearnedFeat.
    "shortest_path": 16_000_000,
    "sp": 16_000_000,
    "katz": 64_000_000,
}


def heuristic_score_kwargs(method, framework, dataset, device, args):
    normalized_method = str(method).strip().lower()
    source_batch_size = getattr(args, "source_batch_size", None)
    edge_batch_size = int(getattr(args, "edge_batch_size", 0) or 0)
    edge_batch_size = edge_batch_size if edge_batch_size > 0 else 65536
    if _uses_collab_heart_heuristic_protocol(framework, dataset):
        from ogbl.heuristics_main import _method_kwargs

        if normalized_method in {"shortest_path", "sp"} and getattr(args, "shortest_path_cutoff", None) is not None:
            raise ValueError(
                "ogbl-collab uses the unbounded released OGB reference shortest-path protocol; --shortest-path-cutoff cannot override it."
            )
        if normalized_method == "katz":
            requested_beta = getattr(args, "katz_beta", None)
            if requested_beta is not None and abs(float(requested_beta) - 0.005) > 1e-12:
                raise ValueError("ogbl-collab uses released OGB reference Katz beta=0.005; an incompatible --katz-beta override was supplied.")
            requested_length = getattr(args, "katz_max_length", None)
            if requested_length is not None and int(requested_length) != 2:
                raise ValueError(
                    "ogbl-collab uses released OGB reference Katz max_length=2; an incompatible --katz-max-length override was supplied."
                )
        kwargs = _method_kwargs(
            normalized_method,
            device,
            edge_batch_size=edge_batch_size,
            source_batch_size=source_batch_size,
        )
    else:
        kwargs = {"device": device, "edge_batch_size": edge_batch_size}
        if source_batch_size is not None:
            kwargs["source_batch_size"] = int(source_batch_size)
        if normalized_method in {"shortest_path", "sp"}:
            cutoff = getattr(args, "shortest_path_cutoff", None)
            kwargs.update({"cutoff": 10 if cutoff is None else int(cutoff), "transform": "inv"})
        elif normalized_method == "katz":
            beta = getattr(args, "katz_beta", None)
            max_length = getattr(args, "katz_max_length", None)
            kwargs.update({"beta": 0.01 if beta is None else float(beta), "max_length": 5 if max_length is None else int(max_length)})
    if str(getattr(args, "candidate_policy", "")).strip().lower() == "learnedfeat":
        max_dense_elems = _LEARNEDFEAT_PATH_MAX_DENSE_ELEMS.get(normalized_method)
        if max_dense_elems is not None:
            kwargs["max_dense_elems"] = max_dense_elems
    return kwargs


def heuristic_protocol_metadata(framework, dataset, args):
    if _uses_collab_heart_heuristic_protocol(framework, dataset):
        from ogbl.heuristics_main import _heuristic_protocol_metadata

        metadata = dict(_heuristic_protocol_metadata("ogbl-collab"))
        metadata.update(
            {
                "graph_builder": "ogbl.heuristics_main._build_graph_cache",
                "score_kwargs_resolver": "ogbl.heuristics_main._method_kwargs",
                "candidate_policy_independent": True,
                "protocol_enforcement": "strict-no-incompatible-overrides",
            }
        )
        return metadata
    cutoff = getattr(args, "shortest_path_cutoff", None)
    beta = getattr(args, "katz_beta", None)
    max_length = getattr(args, "katz_max_length", None)
    return {
        "heuristic_protocol": "shared-evaluator-default-v1",
        "heuristic_valid_graph": "train-binary-undirected",
        "heuristic_test_graph": "train-binary-undirected",
        "cn_aa_ra_edge_values": "binary",
        "shortest_path_cutoff": 10 if cutoff is None else int(cutoff),
        "shortest_path_transform": "inverse-distance",
        "katz_beta": 0.01 if beta is None else float(beta),
        "katz_max_length": 5 if max_length is None else int(max_length),
        "candidate_policy_independent": True,
    }


def heuristic_graphs(bundle, framework, dataset, device):
    if _uses_collab_heart_heuristic_protocol(framework, dataset):
        from ogbl.heuristics_main import _build_graph_cache

        return _build_graph_cache(bundle, "ogbl-collab", device)
    from model.heuristics import build_graph_structures

    nodes = num_nodes(bundle, framework)

    def build(edges):
        (rowptr, col, deg, adj) = build_graph_structures(edges, num_nodes=nodes, make_undirected=True)
        return (rowptr, col, deg, adj.to(device) if device.type == "cuda" else adj)

    train = build(bundle["train_pos"])
    return (train, train)


def heuristic_endpoint_score_block(method, graph, endpoint_values, device, score_kwargs):
    (rowptr, col, deg, adj) = graph
    normalized_method = str(method).strip().lower()
    endpoint_values = endpoint_values.to(device="cpu", dtype=torch.long).reshape(-1)
    block_size = int(endpoint_values.numel())
    score_device = torch.device(device)
    num_graph_nodes = int(deg.numel())
    adjacency = adj.to(score_device)
    binary_adjacency = adjacency.fill_value(1.0)
    local_methods = {"cn", "common_neighbors", "aa", "adamic_adar", "adamicadar", "ra", "resource_allocation", "resourceallocation"}
    if normalized_method in local_methods:
        vectors = torch.zeros((num_graph_nodes, block_size), dtype=torch.float32, device=score_device)
        (_, _, adjacency_values) = adjacency.csr()
        degree = deg.to(device=score_device, dtype=torch.float32)
        if normalized_method in {"aa", "adamic_adar", "adamicadar"}:
            node_factor = torch.zeros_like(degree)
            mask = degree > 1
            node_factor[mask] = 1.0 / torch.log(degree[mask])
        elif normalized_method in {"ra", "resource_allocation", "resourceallocation"}:
            node_factor = torch.zeros_like(degree)
            mask = degree > 0
            node_factor[mask] = 1.0 / degree[mask]
        else:
            node_factor = None
        for column_index, endpoint_value in enumerate(endpoint_values.tolist()):
            begin = int(rowptr[endpoint_value])
            finish = int(rowptr[endpoint_value + 1])
            neighbors = col[begin:finish].to(device=score_device, dtype=torch.long, non_blocking=True)
            if adjacency_values is None:
                endpoint_factor = torch.ones(neighbors.numel(), dtype=torch.float32, device=score_device)
            else:
                endpoint_factor = adjacency_values[begin:finish].to(device=score_device, dtype=torch.float32)
            if node_factor is not None:
                endpoint_factor = endpoint_factor * node_factor[neighbors]
            vectors[neighbors, column_index] = endpoint_factor
        return adjacency.matmul(vectors).to(torch.float32)
    columns = torch.arange(block_size, dtype=torch.long, device=score_device)
    endpoint_device = endpoint_values.to(score_device, non_blocking=True)
    vectors = torch.zeros((num_graph_nodes, block_size), dtype=torch.float32, device=score_device)
    vectors[endpoint_device, columns] = 1.0
    if normalized_method in {"shortest_path", "sp"}:
        cutoff = score_kwargs.get("cutoff")
        max_distance = max(0, num_graph_nodes - 1) if cutoff is None else max(0, int(cutoff))
        unreachable_distance = score_kwargs.get("unreachable_distance")
        fill = 0.0 if unreachable_distance is None else 1.0 / float(unreachable_distance)
        scores = torch.full_like(vectors, fill)
        visited = vectors > 0
        frontier = vectors
        for distance in range(1, max_distance + 1):
            frontier = binary_adjacency.matmul(frontier)
            frontier = (frontier > 0) & ~visited
            if not bool(frontier.any()):
                break
            scores[frontier] = 1.0 / float(distance)
            visited.logical_or_(frontier)
            frontier = frontier.to(torch.float32)
        self_score = score_kwargs.get("self_score")
        if self_score is not None:
            scores[endpoint_device, columns] = float(self_score)
        return scores
    if normalized_method == "katz":
        beta = float(score_kwargs.get("beta", 0.01))
        max_length = int(score_kwargs.get("max_length", 5))
        if max_length <= 0:
            raise ValueError("Katz max_length must be positive.")
        if max_length == 2:
            length_one = binary_adjacency.matmul(vectors)
            length_two = binary_adjacency.matmul(length_one)
            accumulated = beta * length_one.to(torch.float32) + beta**2 * length_two.to(torch.float32)
        else:
            accumulated = torch.zeros_like(vectors)
            for length in range(1, max_length + 1):
                vectors = binary_adjacency.matmul(vectors)
                accumulated.add_(vectors, alpha=beta**length)
        self_score = score_kwargs.get("self_score")
        if self_score is not None:
            accumulated[endpoint_device, columns] = float(self_score)
        return accumulated
    raise ValueError(f"Heuristic {method!r} has no exact endpoint-block implementation.")


def evaluate_heuristic_split(method, graph, pos_edges, filter_rowptr, filter_col, framework, dataset, device, args, description):
    from model.heuristics import score_edges

    (rowptr, col, deg, adj) = graph
    score_kwargs = heuristic_score_kwargs(method, framework, dataset, device, args)
    normalized_method = str(method).lower()
    positive_scores = score_edges(method, rowptr, col, deg, adj, pos_edges.t().contiguous(), **score_kwargs).view(-1)
    candidates = torch.arange(int(deg.numel()), dtype=torch.long)
    score_device = torch.device(device)
    local_methods = {"cn", "common_neighbors", "aa", "adamic_adar", "adamicadar", "ra", "resource_allocation", "resourceallocation"}
    shortest_path_methods = {"shortest_path", "sp"}

    def score_endpoint_block(endpoint_values):
        return heuristic_endpoint_score_block(method, graph, endpoint_values, device, score_kwargs)

    def score_all_nodes(endpoint, direction, *, excluded_nodes=None):
        del direction, excluded_nodes
        if normalized_method in local_methods or normalized_method in shortest_path_methods or normalized_method == "katz":
            return score_endpoint_block(torch.tensor([int(endpoint)], dtype=torch.long))[:, 0]
        endpoints = torch.full_like(candidates, int(endpoint))
        query_edges = torch.stack([endpoints, candidates], dim=0)
        return score_edges(method, rowptr, col, deg, adj, query_edges, **score_kwargs).view(-1).to(torch.float32)

    if normalized_method in local_methods or normalized_method in shortest_path_methods or normalized_method == "katz":

        def score_many(endpoint_values, directions):
            del directions
            return score_endpoint_block(endpoint_values)

        score_all_nodes.score_many = score_many
        score_batch_cap = 16000000 if normalized_method in shortest_path_methods else 64000000
        score_all_nodes.score_many_batch_size = max(
            1,
            min(
                64,
                int(getattr(args, "source_batch_size", 0)) if getattr(args, "source_batch_size", None) is not None else 64,
                score_batch_cap // max(1, int(deg.numel())),
            ),
        )
    filter_existing = filter_rowptr is not None and filter_col is not None
    ranking_rowptr = filter_rowptr if filter_existing else rowptr
    ranking_col = filter_col if filter_existing else col
    result = evaluate_split(
        pos_edges,
        ranking_rowptr,
        ranking_col,
        framework,
        score_all_nodes,
        int(deg.numel()),
        filter_existing,
        max(1, int(args.comparison_batch_size)),
        args.quiet,
        description,
        positive_scores=positive_scores,
        compute_auc=args.compute_auc == "yes",
    )
    result["score_backend"] = "batched_device_one_vs_all"
    result["ranking_device"] = str(score_device)
    return result


def evaluate_grouped_heuristic_split(
    method,
    graph,
    pos_edges,
    negative_edges,
    framework,
    dataset,
    device,
    args,
    description,
    candidate_label="HeaRT",
    orient_query_endpoints=False,
):
    from model.heuristics import score_edges

    (rowptr, col, deg, adj) = graph
    score_kwargs = heuristic_score_kwargs(method, framework, dataset, device, args)

    def score(edge_rows):
        return score_edges(method, rowptr, col, deg, adj, edge_rows.t().contiguous(), **score_kwargs).view(-1)

    positive_scores = score(pos_edges)
    edge_batch_size = int(score_kwargs["edge_batch_size"])
    scoring_negatives = None
    if orient_query_endpoints:
        scoring_negatives = _query_oriented_heuristic_negatives(negative_edges, pos_edges)
    result = evaluate_grouped_split(
        pos_edges,
        negative_edges,
        positive_scores,
        score,
        edge_batch_size,
        args.quiet,
        description,
        candidate_label=candidate_label,
        negative_edges_for_scoring=scoring_negatives,
        endpoint_score_reuse_safe=True,
        compute_auc=args.compute_auc == "yes",
    )
    result["score_backend"] = "device_grouped_heuristic"
    result["ranking_device"] = str(positive_scores.device)
    return result


def format_metric_value(value, scale=1.0):
    scaled = scale * float(value)
    if scaled == 0.0:
        return "0.00000000"
    rendered = f"{scaled:.{METRIC_SIGNIFICANT_DIGITS}g}"
    (mantissa, marker, exponent) = rendered.lower().partition("e")
    if marker:
        decimal_places = METRIC_SIGNIFICANT_DIGITS - 1 - int(exponent)
        if decimal_places <= 16:
            rendered = f"{scaled:.{max(0, decimal_places)}f}"
            (mantissa, marker, exponent) = (rendered, "", "")
    significant = mantissa.replace("-", "").replace(".", "").lstrip("0")
    if len(significant) < 2:
        if "." not in mantissa:
            mantissa += "."
        mantissa += "0" * (2 - len(significant))
    return mantissa + (f"e{exponent}" if marker else "")


def print_result(label, result):
    parts = [f"{label} elapsed_sec={result['elapsed_seconds']}"]
    for split_name, split_result in result["splits"].items():
        metrics = " ".join((f"{name}={format_metric_value(value)}" for (name, value) in split_result["metrics"].items()))
        parts.append(f"{split_name}: positives={split_result['num_positive_edges']} {metrics}")
    print(" | ".join(parts), flush=True)


def aggregate(results):
    output = {}
    for split_name in sorted({name for result in results for name in result["splits"]}):
        output[split_name] = {}
        names = sorted({name for result in results if split_name in result["splits"] for name in result["splits"][split_name]["metrics"]})
        for name in names:
            values = [float(result["splits"][split_name]["metrics"][name]) for result in results if split_name in result["splits"]]
            output[split_name][name] = {
                "mean": statistics.fmean(values),
                "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
                "values": values,
            }
    return output


def save_payload(payload, args, framework, mode, dataset, name, evaluator):
    if args.no_save:
        return
    root = PROJECT_ROOT
    default_name = evaluator.RESULT_FILENAME
    if args.output:
        output = Path(args.output).expanduser()
        if not output.is_absolute():
            output = root / output
    else:
        output_root = evaluator.output_root(args, root)
        relative = Path(framework)
        if bool(getattr(evaluator, "OUTPUT_INCLUDES_CHECKPOINT_MODE", False)):
            relative /= str(mode)
        output = output_root / relative / dataset / name / default_name
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output)
    print(f"\nSaved results: {output}", flush=True)
    return output


def render_metrics(metrics):
    required = FINAL_METRICS[:-1]
    missing = [metric for metric in required if metric not in metrics]
    if missing:
        raise ValueError(f"Missing final metrics: {', '.join(missing)}")
    selected = (*required, *(("AUC",) if "AUC" in metrics else ()))
    return " ".join((f"{metric}={format_metric_value(metrics[metric], scale=100.0)}" for metric in selected))


def grouped_results_use_both_sides(results):
    flags = []
    for result in results:
        for split_result in result.get("splits", {}).values():
            candidate_counts = split_result.get("candidate_counts", {})
            if "both_corruption_sides_combined" in candidate_counts:
                flags.append(bool(candidate_counts["both_corruption_sides_combined"]))
    return bool(flags and all(flags))


def evaluate_checkpoint(
    checkpoint,
    path,
    framework,
    dataset,
    model_name,
    bundle,
    device,
    args,
    evaluator,
    provenance,
    construction_model_name,
):
    candidate_policy = evaluator.POLICY
    grouped = bool(evaluator.GROUPED)
    run_number = int(checkpoint.get("run", checkpoint_number(path)))
    seed = int(checkpoint.get("seed", 0))
    epoch = int(checkpoint.get("epoch", 0))
    timed_out_checkpoint = bool(checkpoint.get("timed_out", False))
    set_seed(seed)
    matmul_precision = _configure_cuda_matmul_precision(device, dataset, construction_model_name)
    prepare_model_features(checkpoint, framework, dataset, construction_model_name, bundle, device)
    model = build_model(checkpoint, framework, dataset, construction_model_name, bundle, device)
    make_evaluation_decode_strict(model)
    bounded_grouped_materialization = materialize_special_grouped_splits(
        model, bundle, framework=framework, dataset=dataset, requested_split="test", grouped=grouped
    )
    checkpoint.pop("model_state_dict", None)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    batch_size = checkpoint_batch_size(checkpoint, model, args.edge_batch_size)
    chunk_size = int(args.node_chunk_size) if int(args.node_chunk_size) > 0 else batch_size
    results = {}
    context = None
    evaluation_profiler = StageProfiler(device)
    evaluation_profiler.start()
    started = start_timer(device)
    try:
        if framework == "pyg":
            embedding = pyg_embedding(model, bundle, device)
        else:
            (_valid, embedding, context) = ogb_embeddings(model, bundle, dataset, device, batch_size, "test")
        split_started = start_timer(device)
        if grouped:
            if uses_reference_row_grouped_evaluation(model, framework, dataset):
                positive_scores, negative_scores = _reference_grouped_edge_scores(
                    model, embedding, bundle["test_pos"], bundle["test_neg"], batch_size
                )
                negative_scorer = _cached_edge_scorer(negative_scores, backend="ordered-score-row-grouped-cache")
            else:
                positive_scores = score_positive_edges(
                    model, embedding, bundle["test_pos"], framework, device, batch_size, canonicalize_pyg=False
                )
                negative_scorer = model_edge_scorer(model, embedding, device)
            split_result = evaluate_grouped_split(
                bundle["test_pos"], bundle["test_neg"], positive_scores, negative_scorer, batch_size, args.quiet,
                f"run {checkpoint.get('run', '?')} test {candidate_policy}", candidate_label=evaluator.CANDIDATE_LABEL,
                endpoint_score_reuse_safe=getattr(model, "decode_is_dedup_safe", None) is True,
                compute_auc=args.compute_auc == "yes",
            )
            split_result["score_backend"] = str(getattr(negative_scorer, "score_backend", "checkpoint_model_decode"))
        else:
            filter_rowptr, filter_col = full_known_positive_filter_views(bundle, framework, dataset, "test")
            scorer = model_scorer(model, embedding, device, chunk_size, filter_existing=True)
            positive_scores = None if bool(getattr(scorer, "is_retrieval_rerank", False)) else score_positive_edges(
                model, embedding, bundle["test_pos"], framework, device, batch_size, canonicalize_pyg=False
            )
            split_result = evaluate_split(
                bundle["test_pos"], filter_rowptr, filter_col, framework, scorer, num_nodes(bundle, framework), True,
                max(1, int(args.comparison_batch_size)), args.quiet, f"run {checkpoint.get('run', '?')} test",
                positive_scores=positive_scores,
                compute_auc=args.compute_auc == "yes",
            )
            split_result["known_positive_filter"] = dict(bundle["full_graph_known_positive_filter"])
            split_result["score_backend"] = str(getattr(scorer, "score_backend", "checkpoint_model_decode"))
        split_result["elapsed_seconds"] = stop_timer(split_started, device)
        results["test"] = split_result
    finally:
        release_model(model, context)
        elapsed_seconds = stop_timer(started, device)
        resource_usage = evaluation_profiler.stop()
    return {
        "checkpoint": str(path),
        "run": run_number,
        "seed": seed,
        "epoch": epoch,
        "timed_out_checkpoint": timed_out_checkpoint,
        "matmul_precision": matmul_precision,
        **provenance,
        "candidate_policy": candidate_policy,
        "edge_batch_size": batch_size,
        "model_decode_batch_size": int(model.decode_batch_size) if getattr(model, "decode_batch_size", None) is not None else None,
        "model_evaluation_decode_batch_size": (
            int(model.evaluation_decode_batch_size) if getattr(model, "evaluation_decode_batch_size", None) is not None else None
        ),
        "model_evaluation_decode_policy": getattr(model, "evaluation_decode_policy", None),
        "node_chunk_size": chunk_size,
        "elapsed_seconds": elapsed_seconds,
        "resource_usage": resource_usage,
        "splits": results,
        **({"known_positive_filter": dict(bundle["full_graph_known_positive_filter"])} if not grouped else {}),
        **(
            {"bounded_grouped_materialization": dict(bounded_grouped_materialization)}
            if bounded_grouped_materialization is not None
            else {}
        ),
        **evaluator.result_fields(bundle),
    }


def _mode_format_version(evaluator, framework, dataset, *, heuristic):
    resolver = getattr(evaluator, "format_version", None)
    return int(resolver(framework, dataset, heuristic=heuristic)) if resolver else int(evaluator.FORMAT_VERSION)


def _mode_cache_suffix(checkpoint, model_name, dataset):
    method = aggregated_mlp_method(model_name)
    if method is None:
        return ()
    recipe = resolve_checkpoint_aggregated_recipe(checkpoint, model_name)
    feature_key = None
    if str(dataset).lower() == "ogbl-ddi":
        config = dict(checkpoint.get("model_config") or {})
        feature_key = (
            int(config.get(f"{method}_feature_dim", config.get("emb_size", 0))),
            int(config.get(f"{method}_feature_seed", 0)),
        )
    return (method, recipe, feature_key)


def _load_mode_bundle(evaluator, args, context, framework, dataset, root, data_seed, cap, device, checkpoint):
    bundle = evaluator.load_bundle(
        framework,
        dataset,
        root,
        data_seed,
        cap,
        device,
        checkpoint,
        args,
        context,
    )
    evaluator.prepare_bundle(bundle)
    install_complete_test_positive_scope(
        bundle, framework=framework, dataset=dataset, root=root, seed=data_seed,
        candidate_policy=evaluator.POLICY, test_positive_cap=args.test_positive_cap,
    )
    if framework == "pyg":
        ensure_pyg_full_known_positive_filter(bundle, dataset=dataset, root=root, seed=data_seed)
    else:
        ensure_ogb_full_known_positive_filter(bundle, dataset=dataset)
    if bool(evaluator.GROUPED):
        ensure_complete_ranked_positive_splits(bundle, framework=framework, dataset=dataset, root=root, seed=data_seed)
    evaluator.install_evaluator_candidates(bundle, framework, dataset, device, args, context, data_seed, root)
    return bundle


def run_heuristics_for_mode(args, evaluator):
    if not args.dataset:
        raise ValueError("--dataset is required with --heuristic.")
    dataset = str(args.dataset)
    framework = framework_name(args.framework, dataset)
    device = torch.device(args.device)
    methods = ["cn", "aa", "ra", "shortest_path", "katz"] if args.heuristic == "all" else [args.heuristic]
    score_protocol = heuristic_protocol_metadata(framework, dataset, args)
    for method in methods:
        heuristic_score_kwargs(method, framework, dataset, device, args)
    wall_started = start_timer(device)
    context = evaluator.evaluator_context(args, dataset, framework, device)
    data_seed = evaluator.data_seed(context, int(args.seed))
    set_seed(data_seed)
    cap = evaluator.resolve_cap(args, None, framework, dataset)
    root = project_path(args.root)
    data_started = start_timer(device)
    bundle = _load_mode_bundle(evaluator, args, context, framework, dataset, root, data_seed, cap, device, None)
    data_load_seconds = stop_timer(data_started, device)
    graph_started = start_timer(device)
    _train_graph, test_graph = heuristic_graphs(bundle, framework, dataset, device)
    graph_preparation_seconds = stop_timer(graph_started, device)
    print(f"framework={framework} dataset={dataset} evaluator={evaluator.POLICY}-heuristics device={device}", flush=True)
    results = {}
    for method in methods:
        profiler = StageProfiler(device)
        profiler.start()
        started = start_timer(device)
        split_started = start_timer(device)
        if bool(evaluator.GROUPED):
            split_result = evaluate_grouped_heuristic_split(
                method, test_graph, bundle["test_pos"], bundle["test_neg"], framework, dataset, device, args,
                f"{method} test {evaluator.POLICY}", candidate_label=evaluator.CANDIDATE_LABEL,
                orient_query_endpoints=bool(evaluator.ORIENT_QUERY_ENDPOINTS),
            )
        else:
            rowptr, col = full_known_positive_filter_views(bundle, framework, dataset, "test")
            split_result = evaluate_heuristic_split(
                method, test_graph, bundle["test_pos"], rowptr, col, framework, dataset, device, args, f"{method} test"
            )
            split_result["known_positive_filter"] = dict(bundle["full_graph_known_positive_filter"])
        split_result["elapsed_seconds"] = stop_timer(split_started, device)
        result = {
            "method": method,
            "elapsed_seconds": stop_timer(started, device),
            "resource_usage": profiler.stop(),
            "splits": {"test": split_result},
            "heuristic_protocol": dict(score_protocol),
        }
        if not bool(evaluator.GROUPED):
            result["known_positive_filter"] = dict(bundle["full_graph_known_positive_filter"])
        results[method] = result
        print_result(f"heuristic={method}", result)
    both_sides = grouped_results_use_both_sides(results.values()) if bool(evaluator.GROUPED) else None
    selector_metadata = evaluator.selector_metadata(args, context, data_seed)
    payload = {
        "format_version": _mode_format_version(evaluator, framework, dataset, heuristic=True),
        "torch_cpu_threads": int(args.torch_cpu_threads),
        "evaluation": evaluator.EVALUATION_PROTOCOL,
        "evaluator": "heuristics",
        "framework": framework,
        "mode": str(args.mode or "all"),
        "dataset": dataset,
        "seed": data_seed,
        "split": args.split,
        "candidate_policy": evaluator.POLICY,
        "compute_auc": args.compute_auc == "yes",
        "heuristic_protocol": dict(score_protocol),
        **complete_test_positive_result_metadata(
            bundle, auxiliary_loader_eval_cap=cap if not evaluator.GROUPED else None
        ),
        **({f"{evaluator.POLICY}_selector": selector_metadata} if selector_metadata else {}),
        **evaluator.result_fields(bundle),
        "candidate_pool": evaluator.candidate_pool(bundle, context, both_sides),
        "timing": {
            "selector_checkpoint_load_seconds": float(context.get("load_seconds", 0.0)),
            "data_load_seconds": data_load_seconds,
            "graph_preparation_seconds": graph_preparation_seconds,
            "evaluation_seconds": round(sum(result["elapsed_seconds"] for result in results.values()), 3),
            "total_wall_seconds": stop_timer(wall_started, device),
        },
        "resource_usage": {"scope": "evaluation_only", **peak_resource_usage(result.get("resource_usage") for result in results.values())},
        "methods": results,
    }
    save_payload(
        payload,
        args,
        framework,
        str(args.mode or "all"),
        dataset,
        args.heuristic if args.heuristic != "all" else "heuristics",
        evaluator,
    )
    return 0


def run_checkpoints_for_mode(args, evaluator):
    device = torch.device(args.device)
    wall_started = start_timer(device)
    discovery_mode = getattr(evaluator, "TARGET_CHECKPOINT_MODE", None) or args.mode
    paths = discover_checkpoints(args, discovery_mode)
    first = None
    checkpoint_load_seconds = 0.0
    if args.dataset is None or args.model is None or args.mode is None:
        started = start_timer(device)
        first = load_checkpoint(paths[0])
        checkpoint_load_seconds += stop_timer(started, device)
    metadata_checkpoints = [first] if first is not None else []
    dataset = metadata(args.dataset, metadata_checkpoints, "dataset")
    model_name = metadata(args.model, metadata_checkpoints, "model")
    requested_mode = metadata(args.mode, metadata_checkpoints, "mode")
    checkpoint_mode = str(getattr(evaluator, "TARGET_CHECKPOINT_MODE", None) or requested_mode)
    framework = framework_name(args.framework, dataset)
    started = start_timer(device)
    preflight_checkpoint_set(
        paths,
        framework=framework,
        dataset=dataset,
        model_name=model_name,
        mode=checkpoint_mode,
        requested_runs=args.runs,
    )
    checkpoint_load_seconds += stop_timer(started, device)
    candidate_precision = configure_bundle_candidate_precision(framework, dataset, model_name, device)
    context = evaluator.evaluator_context(args, dataset, framework, device)
    checkpoint_load_seconds += float(context.get("load_seconds", 0.0))
    root = project_path(args.root)
    bundles = {}
    results = []
    data_load_seconds = 0.0
    print(f"framework={framework} dataset={dataset} model={model_name} evaluator={evaluator.POLICY} device={device}", flush=True)
    for index, path in enumerate(paths):
        if index == 0 and first is not None:
            checkpoint, first = first, None
        else:
            started = start_timer(device)
            checkpoint = load_checkpoint(path)
            checkpoint_load_seconds += stop_timer(started, device)
        construction_model = resolve_checkpoint_model_construction(checkpoint, framework, dataset, model_name)
        provenance = checkpoint_provenance(checkpoint, model_name, framework=framework, dataset=dataset)
        target_seed = int((checkpoint.get("arguments") or {}).get("seed", checkpoint.get("seed", 42)))
        data_seed = evaluator.data_seed(context, target_seed)
        cap = evaluator.resolve_cap(args, checkpoint, framework, dataset)
        key = tuple(evaluator.cache_key(args, context, data_seed, cap, checkpoint))
        key += _mode_cache_suffix(checkpoint, model_name, dataset)
        key += ("test-positive-cap", int(args.test_positive_cap))
        if key not in bundles:
            started = start_timer(device)
            bundles[key] = _load_mode_bundle(
                evaluator,
                args,
                context,
                framework,
                dataset,
                root,
                data_seed,
                cap,
                device,
                checkpoint,
            )
            data_load_seconds += stop_timer(started, device)
        else:
            evaluator.install_evaluator_candidates(
                bundles[key], framework, dataset, device, args, context, data_seed, root
            )
        result = evaluate_checkpoint(
            checkpoint,
            path,
            framework,
            dataset,
            model_name,
            bundles[key],
            device,
            args,
            evaluator,
            provenance,
            construction_model,
        )
        result["data_seed"] = data_seed
        result.update(complete_test_positive_result_metadata(
            bundles[key], auxiliary_loader_eval_cap=cap if not evaluator.GROUPED else None
        ))
        if "ranked_positive_split_source" in bundles[key]:
            result["ranked_positive_split_source"] = bundles[key]["ranked_positive_split_source"]
            result["ranked_feature_source"] = bundles[key].get("ranked_feature_source", "unspecified")
        if candidate_precision is not None:
            result["candidate_generation_matmul_precision"] = candidate_precision
        results.append(result)
        print_result(f"checkpoint_run={result['run']} seed={result['seed']} epoch={result['epoch']}", result)
        del checkpoint
        gc.collect()
    result_bundle = next(iter(bundles.values()))
    bundles.clear()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    both_sides = grouped_results_use_both_sides(results) if evaluator.GROUPED else None
    selector_metadata = evaluator.selector_metadata(args, context, results[0]["data_seed"])
    payload = {
        "format_version": _mode_format_version(evaluator, framework, dataset, heuristic=False),
        "torch_cpu_threads": int(args.torch_cpu_threads),
        "evaluation": evaluator.EVALUATION_PROTOCOL,
        "framework": framework,
        "checkpoint_mode": checkpoint_mode,
        "dataset": dataset,
        "model": model_name,
        "model_implementations": sorted({result["model_implementation"] for result in results}),
        "checkpoint_types": sorted({result["checkpoint_type"] for result in results}),
        "split": args.split,
        "candidate_policy": evaluator.POLICY,
        "compute_auc": args.compute_auc == "yes",
        "positive_query_scope": results[0]["positive_query_scope"],
        "positive_eval_cap": results[0]["positive_eval_cap"],
        "test_positive_scope": dict(results[0]["test_positive_scope"]),
        **({f"{evaluator.POLICY}_selector": selector_metadata} if selector_metadata else {}),
        "candidate_pool": evaluator.candidate_pool(result_bundle, context, both_sides),
        "timing": {
            "checkpoint_load_seconds": round(checkpoint_load_seconds, 3),
            "data_load_seconds": round(data_load_seconds, 3),
            "evaluation_seconds": round(sum(result["elapsed_seconds"] for result in results), 3),
            "total_wall_seconds": stop_timer(wall_started, device),
        },
        "resource_usage": {"scope": "evaluation_only", **peak_resource_usage(result.get("resource_usage") for result in results)},
        "runs": results,
        "aggregate": aggregate(results),
    }
    save_payload(payload, args, framework, checkpoint_mode, dataset, model_name, evaluator)
    means = {name: values["mean"] for name, values in payload["aggregate"]["test"].items()}
    print(f"aggregate test: {render_metrics(means)}", flush=True)
    return 0


def main_for_mode(evaluator):
    """Run one concrete evaluator. This helper never chooses or imports a mode."""
    args = parse_args(evaluator)
    args.torch_cpu_threads = configure_torch_cpu_threads()
    try:
        device = torch.device(args.device)
        if device.type == "cuda" and (not torch.cuda.is_available()):
            raise RuntimeError("CUDA was requested but is not available.")
        if args.heuristic:
            _configure_cuda_matmul_precision(device, args.dataset, args.heuristic)
        return run_heuristics_for_mode(args, evaluator) if args.heuristic else run_checkpoints_for_mode(args, evaluator)
    except Exception as error:
        print(f"ERROR: {error}", file=sys.stderr, flush=True)
        return 1
