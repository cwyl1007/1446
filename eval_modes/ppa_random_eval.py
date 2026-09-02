from __future__ import annotations
import argparse
import gc
import json
import os
import statistics
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, Optional, Tuple
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.profiling import configure_torch_cpu_threads
from pyg.main import _configure_cuda_matmul_precision
from eval_modes.evaluator_helpers import (
    HITS_K,
    _query_oriented_heuristic_negatives,
    build_model,
    checkpoint_batch_size,
    discover_checkpoints as _discover_checkpoints,
    evaluate_grouped_split,
    load_checkpoint,
    make_evaluation_decode_strict,
    model_edge_scorer,
    normalized_model_name,
    ogb_embeddings,
    preflight_checkpoint_set,
    prepare_model_features,
    project_path,
    release_model,
    resolve_checkpoint_model_construction,
    score_positive_edges,
)

DATASET = "ogbl-ppa"
FRAMEWORK = "ogb"
CHECKPOINT_MODE = "heart"
NEGATIVES_PER_POSITIVE = 500
SAMPLER_PROTOCOL = "ogbl-ppa-uniform-pooled-legal-corruptions-v1"
HEURISTIC_METHODS = (
    ("common_neighbors", "cn"),
    ("adamic_adar", "aa"),
    ("resource_allocation", "ra"),
    ("shortest_path", "shortest_path"),
    ("katz", "katz"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate PPA checkpoints on random legal negatives.")
    add = parser.add_argument
    add("--model", required=True)
    add("--checkpoint", action="append", default=[])
    add("--checkpoint-root", default="checkpoints")
    add("--runs", nargs="+", type=int)
    add("--root", default="dataset")
    add("--data-seed", type=int, default=0)
    add("--negative-seed", type=int, default=3001)
    add("--test-cap", type=int, default=100000)
    add("--legality", choices=("observed-history", "full-union"), default="observed-history")
    add("--negatives", type=int, default=NEGATIVES_PER_POSITIVE)
    add("--row-batch-size", type=int, default=512)
    add("--edge-batch-size", type=int, default=0)
    add("--source-batch-size", type=int)
    add("--compute-auc", choices=("yes", "no"), default="yes")
    add("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    add("--output")
    add("--no-save", action="store_true")
    add("--quiet", action="store_true")
    return parser.parse_args()


def _canonical_edge_ids(edge_parts: Tuple[torch.Tensor, ...], num_nodes: int) -> torch.Tensor:
    parts = []
    for value in edge_parts:
        edges = value.detach().to(device="cpu", dtype=torch.long).contiguous()
        if edges.dim() != 2 or int(edges.size(1)) != 2:
            raise ValueError(f"Positive edges must have shape [N,2], got {tuple(edges.shape)}.")
        if edges.numel():
            edges = edges[edges[:, 0] != edges[:, 1]]
            lo = torch.minimum(edges[:, 0], edges[:, 1])
            hi = torch.maximum(edges[:, 0], edges[:, 1])
            parts.append(lo * int(num_nodes) + hi)
    if not parts:
        return torch.empty(0, dtype=torch.long)
    return torch.unique(torch.cat(parts), sorted=True).contiguous()


@dataclass
class PpaUniformPooledNegatives:
    pos_edges: torch.Tensor
    forbidden_canonical_ids: torch.Tensor
    num_nodes: int
    negatives_per_positive: int = NEGATIVES_PER_POSITIVE
    base_seed: int = 3001
    row_batch_size: int = 512
    max_attempts: int = 64
    is_streaming_negative: bool = field(default=True, init=False)
    candidate_summary_prevalidated: bool = field(default=True, init=False)
    _last_summary: Optional[Dict[str, object]] = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.pos_edges = self.pos_edges.detach().to(device="cpu", dtype=torch.long).contiguous()
        self.forbidden_canonical_ids = self.forbidden_canonical_ids.detach().to(device="cpu", dtype=torch.long).contiguous()
        self.num_nodes = int(self.num_nodes)
        self.negatives_per_positive = int(self.negatives_per_positive)
        self.base_seed = int(self.base_seed)
        self.row_batch_size = int(self.row_batch_size)
        self.max_attempts = int(self.max_attempts)
        if self.pos_edges.dim() != 2 or int(self.pos_edges.size(1)) != 2:
            raise ValueError("pos_edges must have shape [N,2].")
        if self.num_nodes <= 1:
            raise ValueError("num_nodes must be greater than one.")
        if self.negatives_per_positive != NEGATIVES_PER_POSITIVE:
            raise ValueError(f"This protocol requires exactly {NEGATIVES_PER_POSITIVE} negatives per positive.")
        if self.row_batch_size <= 0 or self.max_attempts <= 0:
            raise ValueError("row_batch_size and max_attempts must be positive.")
        if self.forbidden_canonical_ids.numel() and (
            not bool((self.forbidden_canonical_ids[1:] > self.forbidden_canonical_ids[:-1]).all())
        ):
            raise ValueError("forbidden_canonical_ids must be strictly increasing.")

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (int(self.pos_edges.size(0)), self.negatives_per_positive, 2)

    def dim(self) -> int:
        return 3

    def size(self, dim: Optional[int] = None):
        if dim is None:
            return torch.Size(self.shape)
        return self.shape[int(dim)]

    def numel(self) -> int:
        (rows, count, width) = self.shape
        return rows * count * width

    def to(self, *args, **kwargs):
        del args, kwargs
        return self

    def contiguous(self):
        return self

    def _positive_membership(self, edge_ids: torch.Tensor) -> torch.Tensor:
        forbidden = self.forbidden_canonical_ids
        if forbidden.numel() == 0:
            return torch.zeros_like(edge_ids, dtype=torch.bool)
        flat = edge_ids.reshape(-1).contiguous()
        positions = torch.searchsorted(forbidden, flat)
        in_bounds = positions < int(forbidden.numel())
        found = torch.zeros_like(flat, dtype=torch.bool)
        if bool(in_bounds.any()):
            bounded = positions[in_bounds]
            found[in_bounds] = forbidden[bounded] == flat[in_bounds]
        return found.view_as(edge_ids)

    def _invalid_rows(self, codes: torch.Tensor, positives: torch.Tensor) -> torch.Tensor:
        left_side = codes < self.num_nodes
        candidate = torch.where(left_side, codes, codes - self.num_nodes)
        source = positives[:, 0:1]
        target = positives[:, 1:2]
        fixed = torch.where(left_side, source, target)
        counterpart = torch.where(left_side, target, source)
        self_loop = candidate == fixed
        query_edge = candidate == counterpart
        lo = torch.minimum(fixed, candidate)
        hi = torch.maximum(fixed, candidate)
        known_positive = self._positive_membership(lo * self.num_nodes + hi)
        sorted_codes = torch.sort(codes, dim=1).values
        duplicates = (sorted_codes[:, 1:] == sorted_codes[:, :-1]).any(dim=1)
        return (self_loop | query_edge | known_positive).any(dim=1) | duplicates

    def _sample_codes(self, positives: torch.Tensor, row_start: int) -> torch.Tensor:
        rows = int(positives.size(0))
        count = self.negatives_per_positive
        generator = torch.Generator(device="cpu").manual_seed(self.base_seed + int(row_start))
        codes = torch.empty((rows, count), dtype=torch.long)
        pending = torch.arange(rows, dtype=torch.long)
        for _ in range(self.max_attempts):
            if pending.numel() == 0:
                break
            draws = torch.randint(0, 2 * self.num_nodes, (int(pending.numel()), count), generator=generator, dtype=torch.long)
            codes[pending] = draws
            pending = pending[self._invalid_rows(draws, positives[pending])]
        if pending.numel():
            raise RuntimeError(
                f"Could not draw a complete legal, duplicate-free PPA random candidate group for {int(pending.numel())} rows after {self.max_attempts} attempts."
            )
        return codes

    def iter_chunks(self, pos_chunk_size: Optional[int] = None) -> Iterator[Tuple[int, int, torch.Tensor]]:
        count = self.negatives_per_positive
        chunk_size = self.row_batch_size if pos_chunk_size is None else int(pos_chunk_size)
        if chunk_size <= 0:
            raise ValueError("pos_chunk_size must be positive when provided.")
        left_min = count
        left_max = 0
        left_total = 0
        rows_seen = 0
        self._last_summary = None
        for start in range(0, int(self.pos_edges.size(0)), chunk_size):
            end = min(start + chunk_size, int(self.pos_edges.size(0)))
            positives = self.pos_edges[start:end]
            codes = self._sample_codes(positives, start)
            left_side = codes < self.num_nodes
            candidate = torch.where(left_side, codes, codes - self.num_nodes)
            source = positives[:, 0:1].expand_as(candidate)
            target = positives[:, 1:2].expand_as(candidate)
            edges = torch.stack([torch.where(left_side, source, candidate), torch.where(left_side, candidate, target)], dim=2).contiguous()
            left_counts = left_side.sum(dim=1, dtype=torch.int64)
            left_min = min(left_min, int(left_counts.min()))
            left_max = max(left_max, int(left_counts.max()))
            left_total += int(left_counts.sum())
            rows_seen += end - start
            yield (start, end, edges)
        left_mean = float(left_total / rows_seen) if rows_seen else 0.0
        right_min = count - left_max if rows_seen else 0
        right_max = count - left_min if rows_seen else 0
        right_mean = float(count - left_mean) if rows_seen else 0.0
        total_stats = {"min": count, "mean": float(count), "max": count}
        self._last_summary = {
            "num_positive_edges": rows_seen,
            "grouped_negatives_per_positive": count,
            "both_corruption_sides_combined": bool(rows_seen and left_min > 0 and (right_min > 0)),
            "fixed_left_endpoint_candidates": {"min": left_min if rows_seen else 0, "mean": left_mean, "max": left_max if rows_seen else 0},
            "fixed_right_endpoint_candidates": {"min": right_min, "mean": right_mean, "max": right_max},
            "other_grouped_candidates": {"min": 0, "mean": 0.0, "max": 0},
            "total_grouped_candidates": total_stats,
        }

    def candidate_summary(self) -> Dict[str, object]:
        if self._last_summary is None:
            raise RuntimeError("Candidate summary requested before one complete stream pass.")
        return dict(self._last_summary)


def discover_checkpoints(args: argparse.Namespace) -> list[Path]:
    args.mode, args.dataset = CHECKPOINT_MODE, DATASET
    return _discover_checkpoints(args, checkpoint_mode=CHECKPOINT_MODE)


def load_ppa_bundle(args: argparse.Namespace) -> Dict[str, object]:
    from ogbl.prepare_data import _load_generated_ppa_base

    os.environ["OGBL_PPA_QUERY_PANEL"] = "local-seeded"
    bundle = _load_generated_ppa_base(str(project_path(args.root)), None, 100000, int(args.data_seed))
    test_pos = bundle["test_pos"].to(torch.long).cpu().contiguous()
    all_test_pos = bundle["all_test_pos"].to(torch.long).cpu().contiguous()
    if not torch.equal(test_pos, all_test_pos):
        raise RuntimeError("PPA random evaluation requires the complete ordered test split before applying its deterministic test cap.")
    test_cap = int(args.test_cap)
    if test_cap < 0:
        raise ValueError("--test-cap must be non-negative.")
    if test_cap and test_cap < int(all_test_pos.size(0)):
        test_panel_seed = int(args.data_seed) + 101
        generator = torch.Generator(device="cpu").manual_seed(test_panel_seed)
        test_indices = torch.randperm(int(all_test_pos.size(0)), generator=generator)[:test_cap]
        test_pos = all_test_pos[test_indices].contiguous()
    else:
        test_panel_seed = None
        test_pos = all_test_pos
    bundle["test_pos"] = test_pos
    bundle["random_test_panel_seed"] = test_panel_seed
    bundle.pop("test_neg", None)
    bundle.pop("valid_neg", None)
    return bundle


def legality_parts(bundle: Dict[str, object], legality: str) -> Tuple[Tuple[torch.Tensor, ...], str]:
    if legality == "observed-history":
        return (
            (bundle["train_pos"], bundle["valid_pos"]),
            "released-HeaRT-test: exclude train positives and the evaluated local-seeded 100k validation panel; also exclude self and the current query edge",
        )
    if legality == "full-union":
        return (
            (bundle["train_pos"], bundle["all_valid_pos"], bundle["all_test_pos"]),
            "strict-complete-positive-union: exclude every train, validation, and test positive; also exclude self and the current query edge",
        )
    raise ValueError(f"Unknown legality rule: {legality!r}")


def evaluate_one_checkpoint(
    checkpoint: Dict[str, object],
    path: Path,
    bundle: Dict[str, object],
    candidates: PpaUniformPooledNegatives,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    actual = tuple(
        normalized_model_name(checkpoint.get(key, ""))
        for key in ("dataset", "model", "mode")
    )
    expected = tuple(map(normalized_model_name, (DATASET, args.model, CHECKPOINT_MODE)))
    if actual != expected or checkpoint.get("timed_out") is not False:
        raise ValueError(f"Checkpoint {path} does not match the requested complete PPA run.")
    construction_model = resolve_checkpoint_model_construction(checkpoint, FRAMEWORK, DATASET, args.model)
    _configure_cuda_matmul_precision(device, DATASET, construction_model)
    prepare_model_features(checkpoint, FRAMEWORK, DATASET, construction_model, bundle, device)
    model = build_model(checkpoint, FRAMEWORK, DATASET, construction_model, bundle, device)
    make_evaluation_decode_strict(model)
    batch_size = checkpoint_batch_size(checkpoint, model, int(args.edge_batch_size))
    context = None
    started = time.perf_counter()
    try:
        (_, test_embedding, context) = ogb_embeddings(model, bundle, DATASET, device, batch_size, "test")
        positive_scores = score_positive_edges(
            model, test_embedding, bundle["test_pos"], FRAMEWORK, device, batch_size, canonicalize_pyg=False
        )
        negative_scorer = model_edge_scorer(model, test_embedding, device)
        split_result = evaluate_grouped_split(
            bundle["test_pos"],
            candidates,
            positive_scores,
            negative_scorer,
            batch_size,
            bool(args.quiet),
            f"run {checkpoint.get('run', '?')} PPA random",
            candidate_label="uniform-random-legal-PPA",
            endpoint_score_reuse_safe=getattr(model, "decode_is_dedup_safe", None) is True,
            compute_auc=args.compute_auc == "yes",
        )
    finally:
        release_model(model, context)
    elapsed = round(time.perf_counter() - started, 3)
    split_result["elapsed_seconds"] = elapsed
    return {
        "checkpoint": str(path),
        "run": int(checkpoint["run"]),
        "seed": int(checkpoint.get("seed", 0)),
        "epoch": int(checkpoint.get("epoch", 0)),
        "edge_batch_size": batch_size,
        "elapsed_seconds": elapsed,
        "splits": {"test": split_result},
    }


def build_ppa_heuristic_graph(bundle: Dict[str, object], device: torch.device) -> Tuple[object, object, torch.Tensor, object]:
    from model.heuristics import build_graph_structures

    graph = build_graph_structures(bundle["train_pos"], num_nodes=int(bundle["num_nodes"]), make_undirected=True)
    (rowptr, col, degree, adjacency) = graph
    if device.type == "cuda":
        adjacency = adjacency.to(device)
    return (rowptr, col, degree, adjacency)


def evaluate_one_heuristic(
    result_name: str,
    method: str,
    graph: Tuple[object, object, torch.Tensor, object],
    bundle: Dict[str, object],
    candidates: PpaUniformPooledNegatives,
    args: argparse.Namespace,
    device: torch.device,
) -> Dict[str, object]:
    from ogbl.heuristics_main import _score_flat

    (rowptr, col, degree, adjacency) = graph
    edge_batch_size = int(args.edge_batch_size) if int(args.edge_batch_size) > 0 else 65536

    def score(edge_rows: torch.Tensor) -> torch.Tensor:
        return _score_flat(
            method,
            rowptr,
            col,
            degree,
            adjacency,
            edge_rows,
            device,
            edge_batch_size=edge_batch_size,
            source_batch_size=args.source_batch_size,
        )

    started = time.perf_counter()
    positive_scores = score(bundle["test_pos"])
    scoring_candidates = _query_oriented_heuristic_negatives(candidates, bundle["test_pos"])
    split_result = evaluate_grouped_split(
        bundle["test_pos"],
        candidates,
        positive_scores,
        score,
        edge_batch_size,
        bool(args.quiet),
        f"{result_name} PPA random",
        candidate_label="uniform-random-legal-PPA",
        negative_edges_for_scoring=scoring_candidates,
        endpoint_score_reuse_safe=True,
        compute_auc=args.compute_auc == "yes",
    )
    elapsed = round(time.perf_counter() - started, 3)
    split_result["elapsed_seconds"] = elapsed
    split_result["score_backend"] = "released-ogb-grouped-heuristic"
    split_result["ranking_device"] = str(positive_scores.device)
    return {
        "method": method,
        "result_name": result_name,
        "edge_batch_size": edge_batch_size,
        "source_batch_size": args.source_batch_size,
        "elapsed_seconds": elapsed,
        "splits": {"test": split_result},
    }


def aggregate_results(results: list[Dict[str, object]]) -> Dict[str, object]:
    aggregate: Dict[str, object] = {"test": {}}
    metrics = ("MRR", *(f"Hits@{k}" for k in HITS_K))
    if results and "AUC" in results[0]["splits"]["test"]["metrics"]:
        metrics = (*metrics, "AUC")
    for metric in metrics:
        values = [float(result["splits"]["test"]["metrics"][metric]) for result in results]
        aggregate["test"][metric] = {
            "mean": statistics.fmean(values),
            "sample_std": statistics.stdev(values) if len(values) > 1 else 0.0,
            "values": values,
        }
    return aggregate


def print_metrics(label: str, metrics: Dict[str, float]) -> None:
    names = ("MRR", *(f"Hits@{k}" for k in HITS_K))
    if "AUC" in metrics:
        names = (*names, "AUC")
    rendered = " ".join((f"{name}={100.0 * float(metrics[name]):.6f}%" for name in names))
    print(f"{label}: {rendered}", flush=True)


def atomic_save(payload: Dict[str, object], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    os.replace(temporary, output)


def _cleanup(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> int:
    args = parse_args()
    heuristic_mode = str(args.model).strip().lower() == "heuristics"
    if int(args.negatives) != NEGATIVES_PER_POSITIVE:
        raise ValueError(f"This protocol requires --negatives {NEGATIVES_PER_POSITIVE}.")
    if int(args.row_batch_size) <= 0:
        raise ValueError("--row-batch-size must be positive.")
    if int(args.test_cap) < 0:
        raise ValueError("--test-cap must be non-negative.")
    if args.source_batch_size is not None and int(args.source_batch_size) <= 0:
        raise ValueError("--source-batch-size must be positive when provided.")
    if heuristic_mode and args.checkpoint:
        raise ValueError("--checkpoint cannot be used with --model heuristics.")
    device = torch.device(args.device)
    if device.type == "cuda" and (not torch.cuda.is_available()):
        raise RuntimeError("CUDA was requested but is unavailable.")
    torch_threads = configure_torch_cpu_threads()
    paths = [] if heuristic_mode else discover_checkpoints(args)
    if not heuristic_mode:
        preflight_checkpoint_set(
            paths,
            framework=FRAMEWORK,
            dataset=DATASET,
            model_name=args.model,
            mode=CHECKPOINT_MODE,
            requested_runs=args.runs,
            expected_data_seed=args.data_seed,
        )
    bundle = load_ppa_bundle(args)
    nodes = int(bundle["num_nodes"])
    (positive_parts, legality_description) = legality_parts(bundle, args.legality)
    forbidden = _canonical_edge_ids(positive_parts, nodes)
    candidates = PpaUniformPooledNegatives(
        pos_edges=bundle["test_pos"],
        forbidden_canonical_ids=forbidden,
        num_nodes=nodes,
        negatives_per_positive=NEGATIVES_PER_POSITIVE,
        base_seed=int(args.negative_seed),
        row_batch_size=int(args.row_batch_size),
    )
    sampler_metadata = {
        "protocol": SAMPLER_PROTOCOL,
        "positive_query_scope": (
            "complete-ordered-test-split"
            if int(bundle["test_pos"].size(0)) == int(bundle["all_test_pos"].size(0))
            else "deterministic-local-seeded-test-panel"
        ),
        "test_positive_rows": int(bundle["test_pos"].size(0)),
        "complete_test_positive_rows": int(bundle["all_test_pos"].size(0)),
        "test_cap": int(args.test_cap),
        "test_panel_seed": bundle["random_test_panel_seed"],
        "negatives_per_positive": NEGATIVES_PER_POSITIVE,
        "selection": "uniform without replacement over pooled legal candidates",
        "legality": str(args.legality),
        "legality_description": legality_description,
        "negative_seed": int(args.negative_seed),
        "row_batch_size": int(args.row_batch_size),
        "forbidden_canonical_edges": int(forbidden.numel()),
    }
    print(
        f"dataset={DATASET} model={args.model} device={device}\n"
        f"test_positive_rows={int(bundle['test_pos'].size(0))} "
        f"negatives_per_positive={NEGATIVES_PER_POSITIVE}\n"
        f"legality={legality_description}\n"
        f"compute_auc={args.compute_auc}",
        flush=True,
    )
    payload = {
        "format_version": 1,
        "evaluation": "ogbl-ppa-uniform-random-legal-grouped-negatives",
        "framework": FRAMEWORK,
        "dataset": DATASET,
        "model": "heuristics" if heuristic_mode else args.model,
        "checkpoint_mode": None if heuristic_mode else CHECKPOINT_MODE,
        "compute_auc": args.compute_auc == "yes",
        "torch_cpu_threads": int(torch_threads),
        "candidate_sampling": sampler_metadata,
    }
    if heuristic_mode:
        from ogbl.heuristics_main import _heuristic_protocol_metadata

        graph_started = time.perf_counter()
        graph = build_ppa_heuristic_graph(bundle, device)
        graph_preparation_seconds = round(time.perf_counter() - graph_started, 3)
        heuristic_results: Dict[str, object] = {}
        for result_name, method in HEURISTIC_METHODS:
            result = evaluate_one_heuristic(result_name, method, graph, bundle, candidates, args, device)
            heuristic_results[result_name] = result
            print_metrics(f"heuristic={result_name}", result["splits"]["test"]["metrics"])
            _cleanup(device)
        payload.update({
            "evaluator": "heuristics",
            "graph_preparation_seconds": graph_preparation_seconds,
            "heuristic_protocol": _heuristic_protocol_metadata(DATASET),
            "heuristics": heuristic_results,
        })
        del graph
    else:
        results = []
        for path in paths:
            checkpoint = load_checkpoint(path)
            result = evaluate_one_checkpoint(checkpoint, path, bundle, candidates, args, device)
            results.append(result)
            print_metrics(f"run={result['run']} seed={result['seed']}", result["splits"]["test"]["metrics"])
            del checkpoint
            _cleanup(device)
        aggregate = aggregate_results(results)
        print("\nAggregate mean ± sample std:", flush=True)
        for metric, values in aggregate["test"].items():
            print(f"  {metric}: {100.0 * values['mean']:.6f}% ± {100.0 * values['sample_std']:.6f}%", flush=True)
        payload.update({"runs": results, "aggregate": aggregate})
    if not args.no_save:
        output = (
            project_path(args.output)
            if args.output
            else PROJECT_ROOT
            / "results"
            / "ogbl"
            / "random"
            / DATASET
            / ("heuristics" if heuristic_mode else args.model)
            / "random_candidate_evaluation.json"
        )
        atomic_save(payload, output)
        print(f"Saved results: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
