import argparse
import torch
import time
import os
from .prepare_data import parse_pool_argument
from .planetoid_inputs import FIXED_PLANETOID_POSITIVE_SPLIT_DATASETS
from .main import (
    RUNTIME_LIMIT_SEC,
    _print_candidate_metadata,
    _read_run_data,
    _resolve_device,
    _resolve_eval_cap,
    _runtime_exceeded,
    _write_summary,
    set_seed,
)
from model.heuristics import build_graph_structures, score_edges
from utils.metrics import StreamingAUCAP, evaluate_auc, evaluate_mrr
from utils.profiling import StageProfiler, current_cpu_rss_mb, peak_cpu_rss_mb
from utils.heart_protocol import persist_heart_candidate_metadata

HITS_K = (1, 3, 5, 10, 20, 50, 100)
METRIC_NAMES = ("AUC", "MRR", *(f"Hits@{k}" for k in HITS_K))


def _format_metrics(label, scores, precision):
    return f"{label} " + " ".join(f"{name}={100 * scores[name]:.{precision}f}" for name in METRIC_NAMES if name in scores)


def make_auc_arrays(pos_pred, neg_pred):
    pos_pred = pos_pred.view(-1).detach()
    neg_pred = neg_pred.view(-1).detach().to(pos_pred.device)
    y_pred = torch.cat([pos_pred, neg_pred], dim=0)
    y_true = torch.cat(
        [
            torch.ones(pos_pred.numel(), dtype=torch.float32, device=pos_pred.device),
            torch.zeros(neg_pred.numel(), dtype=torch.float32, device=pos_pred.device),
        ],
        dim=0,
    )
    return (y_pred, y_true)


_SYMMETRIC_SOURCE_PROPAGATION_METHODS = frozenset({"shortest_path", "sp", "katz"})


def _is_streamed_grouped_negative(value):
    return bool(getattr(value, "is_streamed_grouped_negative", False))


def _score_heuristic_edges_bounded(method, rowptr, col, deg, adj, edges, kwargs, *, edge_chunk_size=262144, output_device="cpu"):
    edges = edges.to(dtype=torch.long).cpu()
    parts = []
    for start in range(0, int(edges.size(0)), int(edge_chunk_size)):
        score = score_edges(method, rowptr, col, deg, adj, edges[start : start + int(edge_chunk_size)], **kwargs).view(-1)
        parts.append(score.detach().to(device=output_device, dtype=torch.float32))
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32, device=output_device)


def _streamed_heuristic_split(method, rowptr, col, deg, adj, pos_edge, neg_edge, kwargs, *, compute_auc=True):
    positives = pos_edge.to(dtype=torch.long).cpu()
    comparison_device = torch.device(kwargs.get("device", "cpu"))
    pos_pred = _score_heuristic_edges_bounded(method, rowptr, col, deg, adj, positives, kwargs, output_device=comparison_device)
    num_rows = int(pos_pred.numel())
    ge_counts = torch.zeros(num_rows, dtype=torch.int32)
    gt_counts = torch.zeros(num_rows, dtype=torch.int32)
    auc = StreamingAUCAP(pos_pred) if compute_auc else None
    decoded_edges = 0
    logical_occurrences = 0
    symmetric_reverse_edges_reoriented = 0
    method_key = str(method).strip().lower()
    if bool(getattr(neg_edge, "is_endpoint_grouped_negative", False)):
        for chunk in neg_edge.iter_endpoint_group_chunks():
            union_lengths = chunk.union_rowptr[1:] - chunk.union_rowptr[:-1]
            endpoints = torch.repeat_interleave(chunk.endpoints, union_lengths)
            reorient_reverse = (
                int(chunk.side) == 1 and method_key in _SYMMETRIC_SOURCE_PROPAGATION_METHODS and bool(kwargs.get("is_symmetric", False))
            )
            if int(chunk.side) == 0 or reorient_reverse:
                union_edges = torch.stack([endpoints, chunk.union_nodes], dim=1)
            else:
                union_edges = torch.stack([chunk.union_nodes, endpoints], dim=1)
            if reorient_reverse:
                symmetric_reverse_edges_reoriented += int(union_edges.size(0))
            union_scores = _score_heuristic_edges_bounded(
                method, rowptr, col, deg, adj, union_edges, kwargs, output_device=comparison_device
            )
            decoded_edges += int(union_scores.numel())
            union_offsets = chunk.union_rowptr[chunk.occurrence_endpoint_index.to(dtype=torch.long)].view(-1, 1)
            global_indices = chunk.candidate_local_indices.to(device=comparison_device, non_blocking=comparison_device.type == "cuda").to(
                dtype=torch.long
            )
            global_indices.add_(union_offsets.to(device=comparison_device, dtype=torch.long, non_blocking=comparison_device.type == "cuda"))
            scores = union_scores[global_indices]
            row_ids = chunk.occurrence_row_ids
            row_ids_device = row_ids.to(device=comparison_device, dtype=torch.long)
            logical_occurrences += int(scores.numel())
            positive = pos_pred[row_ids_device].view(-1, 1)
            side_rank_counts = torch.stack([(scores >= positive).sum(dim=1), (scores > positive).sum(dim=1)], dim=0).to(torch.int32).cpu()
            ge_counts.index_add_(0, row_ids, side_rank_counts[0])
            gt_counts.index_add_(0, row_ids, side_rank_counts[1])
            if compute_auc:
                auc.update_weighted(union_scores, chunk.union_occurrence_multiplicity)
    else:
        for start, end, grouped_edges in neg_edge.iter_grouped_chunks():
            flat = grouped_edges.reshape(-1, 2)
            edge_ids = flat[:, 0] * int(neg_edge.num_nodes) + flat[:, 1]
            (unique_ids, inverse) = torch.unique(edge_ids, sorted=True, return_inverse=True)
            unique_edges = torch.stack(
                [torch.div(unique_ids, int(neg_edge.num_nodes), rounding_mode="floor"), unique_ids % int(neg_edge.num_nodes)], dim=1
            )
            unique_scores = _score_heuristic_edges_bounded(method, rowptr, col, deg, adj, unique_edges, kwargs)
            inverse_device = inverse.to(device=unique_scores.device, dtype=torch.long)
            scores = unique_scores[inverse_device].view(end - start, int(neg_edge.negatives_per_row))
            decoded_edges += int(unique_scores.numel())
            logical_occurrences += int(scores.numel())
            positive = pos_pred[start:end].view(-1, 1)
            ge_counts[start:end] = (scores >= positive).sum(dim=1).to(torch.int32).cpu()
            gt_counts[start:end] = (scores > positive).sum(dim=1).to(torch.int32).cpu()
            if compute_auc:
                auc.update_weighted(unique_scores, torch.bincount(inverse, minlength=int(unique_scores.numel())))
    rank = 0.5 * (ge_counts.to(torch.float32) + gt_counts.to(torch.float32)) + 1.0
    neg_edge.last_evaluation_reuse = {
        "decoded_union_edges": decoded_edges,
        "logical_candidate_occurrences": logical_occurrences,
        "decode_reuse_ratio": logical_occurrences / decoded_edges if decoded_edges else 0.0,
        "symmetric_reverse_edges_reoriented": symmetric_reverse_edges_reoriented,
    }
    result = {
        "MRR": round(float((1.0 / rank).mean().item()), 4),
        **{f"Hits@{k}": round(float((rank <= k).to(torch.float32).mean().item()), 4) for k in HITS_K},
    }
    if compute_auc:
        result["AUC"] = auc.compute()["AUC"]
    return result


def _method_kwargs(method, device, reference_planetoid=False):
    kwargs = {"device": device, "is_symmetric": True}
    if method in ("shortest_path", "sp"):
        if reference_planetoid:
            kwargs.update({"cutoff": None, "transform": "inv", "unreachable_distance": 999, "self_score": 999})
        else:
            kwargs.update({"cutoff": 10, "transform": "inv"})
    if method == "katz":
        if reference_planetoid:
            kwargs.update({"beta": 0.005, "max_length": 20})
        else:
            kwargs.update({"beta": 0.01, "max_length": 5})
    return kwargs


def _move_graph_to_device(graph, device):
    (rowptr, col, deg, adj) = graph
    if str(device).startswith("cuda"):
        adj = adj.to(device)
    return (rowptr, col, deg, adj)


def run_split(method, rowptr, col, deg, adj, pos_edge, neg_edge, device, *, reference_planetoid=False, compute_auc=True):
    kwargs = _method_kwargs(method, device, reference_planetoid)
    rank_device = torch.device(device)
    if _is_streamed_grouped_negative(neg_edge):
        return _streamed_heuristic_split(method, rowptr, col, deg, adj, pos_edge, neg_edge, kwargs, compute_auc=compute_auc)
    pos_pred = score_edges(method, rowptr, col, deg, adj, pos_edge, **kwargs).view(-1).to(rank_device)
    if getattr(neg_edge, "is_ragged_negative", False):
        ranks = torch.empty(pos_pred.numel(), dtype=torch.float32, device=rank_device)
        parts = [] if compute_auc else None
        for start, end, flat_edges, local_rowptr in neg_edge.iter_ragged_chunks():
            scores = score_edges(method, rowptr, col, deg, adj, flat_edges, **kwargs).view(-1).detach().to(rank_device)
            if compute_auc:
                parts.append(scores)
            lengths = (local_rowptr[1:] - local_rowptr[:-1]).to(rank_device)
            row_ids = torch.repeat_interleave(torch.arange(end - start, device=rank_device), lengths)
            repeated_positive = pos_pred[start:end].detach()[row_ids]
            optimistic = torch.zeros(end - start, dtype=torch.float32, device=rank_device)
            pessimistic = torch.zeros_like(optimistic)
            optimistic.index_add_(0, row_ids, (scores >= repeated_positive).to(torch.float32))
            pessimistic.index_add_(0, row_ids, (scores > repeated_positive).to(torch.float32))
            ranks[start:end] = 0.5 * (optimistic + pessimistic) + 1.0
        result = {
            "MRR": round(float((1.0 / ranks).mean()), 4),
            **{f"Hits@{k}": round(float((ranks <= k).to(torch.float32).mean()), 4) for k in HITS_K},
        }
        if compute_auc:
            neg_pred = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32, device=rank_device)
            (y_pred, y_true) = make_auc_arrays(pos_pred, neg_pred)
            result["AUC"] = evaluate_auc(y_pred, y_true)["AUC"]
        return result
    neg_pred = score_edges(method, rowptr, col, deg, adj, neg_edge, **kwargs).view(-1).to(rank_device)
    npos = int(pos_pred.numel())
    neg_for_mrr = neg_pred
    if npos > 0 and neg_pred.numel() % npos == 0:
        k = int(neg_pred.numel() // npos)
        if k >= 1:
            neg_for_mrr = neg_pred.view(npos, k)
    mrr_out = evaluate_mrr(None, pos_pred, neg_for_mrr)
    hits_out = {
        "Hits@1": mrr_out["mrr_hit1"],
        "Hits@3": mrr_out["mrr_hit3"],
        "Hits@5": mrr_out["mrr_hit5"],
        "Hits@10": mrr_out["mrr_hit10"],
        "Hits@20": mrr_out["mrr_hit20"],
        "Hits@50": mrr_out["mrr_hit50"],
        "Hits@100": mrr_out["mrr_hit100"],
    }
    result = {"MRR": mrr_out["MRR"], **hits_out}
    if compute_auc:
        (y_pred, y_true) = make_auc_arrays(pos_pred, neg_pred)
        result["AUC"] = evaluate_auc(y_pred, y_true)["AUC"]
    return result


def main():
    program_t0 = time.time()
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=str, required=True)
    ap.add_argument("--mode", choices=["heart", "all"], default="heart")
    ap.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    ap.add_argument("--pool", type=parse_pool_argument, default=10000)
    ap.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, choices=[500], default=500)
    ap.add_argument("--planetoid-input-root", type=str, default=None)
    ap.add_argument("--heart-backend", choices=["auto", "gpu", "dense"], default="auto")
    ap.add_argument("--heart-batch-size", type=int, default=2048)
    ap.add_argument("--heart-ppr-iters", type=int, default=None)
    ap.add_argument("--root", type=str, default="dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device")
    ap.add_argument("--heuristic", type=str, default="all", choices=["all", "cn", "aa", "ra", "shortest_path", "katz"])
    ap.add_argument("--compute-auc", choices=["yes", "no"], default="yes")
    ap.add_argument("--save-log", action="store_true", default=True)
    args = ap.parse_args()
    device = _resolve_device(args.device)
    args.device = str(device)
    set_seed(args.seed)
    args.eval_cap = _resolve_eval_cap(args.eval_cap, args.mode, args.dataset)
    timed_out = False
    log_path = None
    if args.save_log:
        log_dir = os.path.join("results", "pyg", args.mode, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "heuristics.txt")
    print(f"\nDataset={args.dataset} | mode={args.mode} | seed={args.seed} | device={device}")
    print(f"eval_cap={args.eval_cap}")
    print(f"pool={args.pool}")
    print("heart_negatives=generated-online")
    print(f"heart_negatives_total_requested={args.heart_negatives}")
    print(f"runtime_limit_sec={RUNTIME_LIMIT_SEC}")
    print(f"runtime_limit_hours={RUNTIME_LIMIT_SEC / 3600:.2f}")
    compute_auc = args.compute_auc == "yes"
    print(f"compute_auc_effective={compute_auc}")
    t_load0 = time.perf_counter()
    data = _read_run_data(args, device)
    heart_candidate_metadata = persist_heart_candidate_metadata(args, data)
    for key, value in heart_candidate_metadata.items():
        print(f"{key}={(value if value is not None else 'not-applicable')}", flush=True)
    _print_candidate_metadata(data)
    t_load = time.perf_counter() - t_load0
    print(f"Data loaded in {t_load:.2f}s")
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED during data loading: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    t_graph0 = time.perf_counter()
    num_nodes = int(data["x"].size(0))
    (rowptr, col, deg, adj) = build_graph_structures(data["train_pos"], num_nodes=num_nodes, make_undirected=True)
    (rowptr, col, deg, adj) = _move_graph_to_device((rowptr, col, deg, adj), device)
    t_graph = time.perf_counter() - t_graph0
    print(f"Graph prepared in {t_graph:.2f}s")
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED during graph prep: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    methods = ["cn", "aa", "ra", "shortest_path", "katz"] if args.heuristic == "all" else [args.heuristic]
    reference_planetoid_heuristics = (
        args.mode == "heart"
        and str(args.dataset).strip().lower()
        in FIXED_PLANETOID_POSITIVE_SPLIT_DATASETS
    )
    results_by_method = {}
    for m in methods:
        if timed_out or _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED before {m.upper()}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
        profiler = StageProfiler(device)
        profiler.start()
        val = run_split(
            m,
            rowptr,
            col,
            deg,
            adj,
            data["valid_pos"],
            data["valid_neg"],
            device,
            reference_planetoid=reference_planetoid_heuristics,
            compute_auc=compute_auc,
        )
        test = run_split(
            m,
            rowptr,
            col,
            deg,
            adj,
            data["test_pos"],
            data["test_neg"],
            device,
            reference_planetoid=reference_planetoid_heuristics,
            compute_auc=compute_auc,
        )
        eval_info = profiler.stop()
        elapsed = eval_info["sec"]
        results_by_method[m] = {
            "val": val,
            "test": test,
            "elapsed": elapsed,
            "peak_cpu": eval_info["cpu_peak_rss_mb"],
            "peak_cuda": eval_info["cuda_peak_allocated_mb"],
            "peak_cuda_reserved": eval_info["cuda_peak_reserved_mb"],
        }
        print(f"\n=== {m.upper()} ===  (time: {elapsed:.2f}s, peak_cpu_rss_mb={eval_info['cpu_peak_rss_mb']:.2f})")
        print(_format_metrics("VALID:", val, 2))
        print(_format_metrics("TEST :", test, 2))
        if _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED after {m.upper()}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
    total_wall_sec = time.time() - program_t0
    summary_lines = []

    def log(line=""):
        print(line)
        summary_lines.append(str(line))

    log("\n" + "=" * 80)
    log("Timing summary")
    log(f"runtime_limit_exceeded: {timed_out}")
    if timed_out:
        log("status: exceeded 24 hour runtime limit")
    else:
        log("status: completed within 24 hour runtime limit")
    log(f"runtime_limit_sec: {RUNTIME_LIMIT_SEC:.2f}")
    log(f"evaluation_mode: {args.mode}")
    log(f"evaluation_positive_cap: {args.eval_cap}")
    log(f"compute_auc_effective: {compute_auc}")
    for key, value in heart_candidate_metadata.items():
        log(f"{key}: {(value if value is not None else 'not-applicable')}")
    log(f"ranking_device: {device}")
    log(f"data_load_sec: {t_load:.2f}")
    log(f"graph_prep_sec: {t_graph:.2f}")
    log(f"total_wall_sec: {total_wall_sec:.2f}")
    log(f"cpu_rss_mb_current: {current_cpu_rss_mb():.2f}")
    log(f"cpu_rss_mb_peak_process: {peak_cpu_rss_mb():.2f}")
    log("evaluation_scope: validation_and_test")
    log("\n" + "=" * 80)
    log(f"Final results on {args.dataset} | mode={args.mode} | seed={args.seed}")
    if not results_by_method:
        log("No heuristic methods completed.")
        log("=" * 80)
        _write_summary(log_path, summary_lines)
        return 1
    for m in methods:
        if m not in results_by_method:
            log(f"\n[{m.upper()}]")
            log("Method did not complete.")
            continue
        val = results_by_method[m]["val"]
        test = results_by_method[m]["test"]
        elapsed = results_by_method[m]["elapsed"]
        log(f"\n[{m.upper()}]")
        log(f"evaluation_time_sec: {elapsed:.2f}")
        log(f"evaluation_peak_cpu_rss_mb: {results_by_method[m]['peak_cpu']:.2f}")
        log(f"evaluation_peak_cuda_allocated_mb: {results_by_method[m]['peak_cuda']:.2f}")
        log(f"evaluation_peak_cuda_reserved_mb: {results_by_method[m]['peak_cuda_reserved']:.2f}")
        log(f"eval_sec: {elapsed:.2f}")
        log(f"peak_cpu_rss_mb: {results_by_method[m]['peak_cpu']:.2f}")
        log(f"peak_cuda_allocated_mb: {results_by_method[m]['peak_cuda']:.2f}")
        log(_format_metrics("VALID:", val, 6))
        log(_format_metrics("TEST :", test, 6))
    log("=" * 80)
    _write_summary(log_path, summary_lines)
    incomplete = [method for method in methods if method not in results_by_method]
    if timed_out or incomplete:
        print(f"ERROR: heuristic evaluation was incomplete; timed_out={timed_out}, missing_methods={incomplete}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
