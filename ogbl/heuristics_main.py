import argparse
import os
import time
import torch
from model.heuristics import build_graph_structures, score_edges
from utils.metrics import StreamingAUCAP, evaluate_mrr
from utils.heart_protocol import heart_candidate_metadata
from .prepare_data import parse_pool_argument, read_data
from .fast_negatives import DeduplicatedGroupedNegativeEdges, is_streaming_negative_edges, prepare_ddi_grouped_eval_edges
from .protocol import (
    RUNTIME_LIMIT_SEC,
    hit_sort_key as _hit_sort_key,
    log_protocol_summary,
    ogbl_protocol_metadata,
    print_ogbl_protocol,
    resolve_ogbl_device,
    resolve_ogbl_eval_cap,
    resolve_ogbl_metric,
    runtime_exceeded as _runtime_exceeded,
    set_seed,
    should_compute_auc as _should_compute_auc,
    write_summary as _write_summary,
)
from .train_eval import _auc_ap_against_sorted_negatives, _reshape_neg_scores, find_result_key
from utils.profiling import StageProfiler, configure_torch_cpu_threads, current_cpu_rss_mb, peak_cpu_rss_mb

CITATION2_MAX_DENSE_ELEMS = 64000000


def _method_kwargs(method, device, *, edge_batch_size=65536, source_batch_size=None, max_dense_elems=4000000, deadline=None, progress_callback=None):
    kwargs = {"device": device, "edge_batch_size": max(1, int(edge_batch_size)), "max_dense_elems": max(1, int(max_dense_elems))}
    if source_batch_size is not None:
        kwargs["source_batch_size"] = max(1, int(source_batch_size))
    if deadline is not None:
        kwargs["deadline"] = float(deadline)
    if progress_callback is not None:
        kwargs["progress_callback"] = progress_callback
    if method in ("shortest_path", "sp"):
        kwargs.update(cutoff=None, transform="inv", unreachable_distance=999, self_score=999)
    if method == "katz":
        kwargs.update(beta=0.005, max_length=2, is_symmetric=True, self_score=0.0)
    return kwargs


def _score_flat(method, rowptr, col, deg, adj, edges, device, *, edge_batch_size=65536, source_batch_size=None, max_dense_elems=4000000, deadline=None, progress_callback=None):
    kwargs = _method_kwargs(
        method,
        device,
        edge_batch_size=edge_batch_size,
        source_batch_size=source_batch_size,
        max_dense_elems=max_dense_elems,
        deadline=deadline,
        progress_callback=progress_callback,
    )
    return score_edges(method, rowptr, col, deg, adj, edges.t().contiguous(), **kwargs).view(-1).detach()


def _heuristic_protocol_metadata(dataset_name):
    collab = str(dataset_name).strip().lower() == "ogbl-collab"
    return {
        "heuristic_protocol": "heart-released-source-recipe-v1",
        "heuristic_source_driver": "HeaRT_ogb/main_heuristic_ogb.py",
        "heuristic_valid_graph": "train-weighted-undirected" if collab else "train-binary-undirected",
        "heuristic_test_graph": "train-weighted+uncapped-valid-unit-weight-undirected" if collab else "train-binary-undirected",
        "cn_aa_ra_edge_values": "collaboration-counts" if collab else "binary",
        "shortest_path_cutoff": "unbounded",
        "shortest_path_transform": "inverse-distance",
        "shortest_path_unreachable_distance": 999,
        "shortest_path_self_score": 999,
        "shortest_path_remove_query_edge": False,
        "katz_variant": "katz_apro",
        "katz_beta": 0.005,
        "katz_max_length": 2,
        "katz_self_score": 0.0,
        "katz_remove_query_edge": False,
        "katz_is_symmetric": True,
    }


def _move_graph_to_device(graph, device):
    return (*graph[:3], graph[3].to(device) if str(device).startswith("cuda") else graph[3])


def _build_graph_cache(bundle, dataset_name, device):
    num_nodes = int(bundle["x"].size(0))
    if dataset_name != "ogbl-collab":
        graph = build_graph_structures(bundle["train_pos"], num_nodes=num_nodes, make_undirected=True)
        graph = _move_graph_to_device(graph, device)
        return (graph, graph)
    data = bundle["data"]
    train_edges = data.edge_index.detach().to(device="cpu", dtype=torch.long)
    train_weights = data.edge_weight.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    train_graph = build_graph_structures(train_edges, num_nodes=num_nodes, make_undirected=False, edge_weight=train_weights)
    valid_input_pos = bundle.get("valid_input_pos", bundle["valid_pos"]).detach().to(device="cpu", dtype=torch.long)
    valid_edges = valid_input_pos.t().contiguous()
    valid_edges = torch.cat([valid_edges, valid_edges.flip(0)], dim=1)
    valid_input_weight = bundle.get("valid_input_weight")
    valid_input_weight = (
        torch.ones(valid_input_pos.size(0), dtype=torch.float32)
        if valid_input_weight is None
        else valid_input_weight.detach().to(device="cpu", dtype=torch.float32).reshape(-1)
    )
    valid_weights = torch.cat([valid_input_weight, valid_input_weight])
    test_graph = build_graph_structures(
        torch.cat([train_edges, valid_edges], dim=1),
        num_nodes=num_nodes,
        make_undirected=False,
        edge_weight=torch.cat([train_weights, valid_weights]),
    )
    return (_move_graph_to_device(train_graph, device), _move_graph_to_device(test_graph, device))


def parse_args():
    parser = argparse.ArgumentParser(description="Heuristic link prediction on OGBL")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--metric", type=str, default="auto")
    parser.add_argument("--mode", choices=["heart", "all"], default="heart")
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device")
    parser.add_argument("--heuristic", type=str, default="all", choices=["all", "cn", "aa", "ra", "shortest_path", "katz"])
    parser.add_argument("--edge-batch-size", "--edge_batch_size", dest="edge_batch_size", type=int, default=65536)
    parser.add_argument("--citation2-query-batch-size", "--citation2_query_batch_size", dest="citation2_query_batch_size", type=int, default=512)
    parser.add_argument("--all-negatives", type=int, default=None)
    parser.add_argument("--pool", type=parse_pool_argument, default=10000)
    parser.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, default=500)
    parser.add_argument("--ranked-negatives-backend", choices=["auto", "official", "batched", "fast", "dense"], default="auto")
    parser.add_argument("--negative-cache-dir", type=str, default=None)
    parser.add_argument("--no-negative-cache", action="store_true")
    parser.add_argument("--compute-auc", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--save-log", action="store_true", default=True)
    return parser.parse_args()


def _ordered_metric_keys(result):
    def key_fn(name):
        s = str(name).strip().lower()
        priority = {"auc": 0, "ap": 1, "mrr": 2}
        if s in priority:
            return (priority[s], 0, s)
        if s.startswith("hits@"):
            return (3, _hit_sort_key(s), s)
        return (4, 0, s)

    return sorted(result.keys(), key=key_fn)


def _progress_reporter(label):
    last_reported = -1

    def report(completed, total):
        nonlocal last_reported
        total = max(1, int(total))
        completed = int(completed)
        interval = max(1, total // 10)
        if completed >= total or last_reported < 0 or completed - last_reported >= interval:
            print(f"{label}: {completed}/{total} completed", flush=True)
            last_reported = completed

    return report


def _orient_citation2_grouped_negatives(pos_edges, neg_edges):
    if neg_edges.dim() != 3 or neg_edges.size(-1) != 2:
        raise ValueError("Grouped negatives must have shape [queries, negatives, 2].")
    if int(pos_edges.size(0)) != int(neg_edges.size(0)):
        raise ValueError("Grouped negatives must have one row per positive edge.")
    rows, width = neg_edges.shape[:2]
    row_ids = torch.arange(rows).repeat_interleave(width)
    oriented, swapped, _ = _orient_fixed_endpoint_flat(pos_edges, neg_edges.reshape(-1, 2), row_ids, require_fixed_endpoint=True)
    return (oriented.view(rows, width, 2), swapped)


def _orient_fixed_endpoint_flat(pos_edges, flat_edges, row_ids, *, require_fixed_endpoint):
    flat = flat_edges.to(dtype=torch.long).cpu()
    positive = pos_edges.to(dtype=torch.long).cpu()[row_ids]
    fixed_source = flat[:, 0].eq(positive[:, 0])
    fixed_target = flat[:, 1].eq(positive[:, 1])
    legal = fixed_source | fixed_target
    if not bool(legal.all()):
        if require_fixed_endpoint:
            bad = int((~legal).sum().item())
            raise ValueError(f"Found {bad} grouped negatives that do not share either endpoint with their positive edge.")
        return (flat, 0, False)
    swap = ~fixed_source & fixed_target
    if not bool(swap.any()):
        return (flat, 0, True)
    oriented = flat.clone()
    left = oriented[swap, 0].clone()
    oriented[swap, 0] = oriented[swap, 1]
    oriented[swap, 1] = left
    return (oriented.contiguous(), int(swap.sum().item()), True)


def _score_citation2_grouped_split(method, rowptr, col, deg, adj, pos_edges, neg_edges, device, source_batch_size, edge_batch_size, *, deadline=None, label="Citation2 split"):
    (oriented, swapped) = _orient_citation2_grouped_negatives(pos_edges, neg_edges)
    flat_neg = oriented.view(-1, 2)
    combined = torch.cat([pos_edges.to(torch.long).cpu(), flat_neg], dim=0).contiguous()
    unique_sources = int(torch.unique(combined[:, 0]).numel())
    print(
        f"{label}: positive_edges={pos_edges.size(0)} negative_edges={flat_neg.size(0)} swapped={swapped} fixed_endpoint_sources={unique_sources}",
        flush=True,
    )
    prediction = _score_flat(
        method,
        rowptr,
        col,
        deg,
        adj,
        combined,
        device,
        edge_batch_size=edge_batch_size,
        source_batch_size=source_batch_size,
        max_dense_elems=CITATION2_MAX_DENSE_ELEMS,
        deadline=deadline,
        progress_callback=_progress_reporter(label),
    )
    num_pos = int(pos_edges.size(0))
    return (prediction[:num_pos], prediction[num_pos:].view(neg_edges.size(0), neg_edges.size(1)))


def _score_citation2_edges(method, rowptr, col, deg, adj, edges, device, source_batch_size, edge_batch_size, *, deadline=None, label="Citation2 edges"):
    return _score_flat(
        method,
        rowptr,
        col,
        deg,
        adj,
        edges,
        device,
        edge_batch_size=edge_batch_size,
        source_batch_size=source_batch_size,
        max_dense_elems=CITATION2_MAX_DENSE_ELEMS,
        deadline=deadline,
        progress_callback=_progress_reporter(label),
    )


def _score_citation2_neg(method, rowptr, col, deg, adj, pos_edges, neg_targets, device, query_batch_size, edge_batch_size, *, deadline=None, label="Citation2 target negatives"):
    num_queries = int(pos_edges.size(0))
    num_neg = int(neg_targets.size(1))
    out = torch.empty((num_queries, num_neg), dtype=torch.float32, device=torch.device(device))
    src_all = pos_edges[:, 0].to(torch.long)
    for q_start in range(0, num_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, num_queries)
        src = src_all[q_start:q_end]
        neg = neg_targets[q_start:q_end].to(torch.long)
        src_rep = src.view(-1, 1).repeat(1, num_neg).reshape(-1)
        dst_rep = neg.reshape(-1)
        edges = torch.stack([src_rep, dst_rep], dim=1)
        prediction = _score_citation2_edges(
            method, rowptr, col, deg, adj, edges, device, query_batch_size, edge_batch_size, deadline=deadline, label=label
        )
        out[q_start:q_end] = prediction.view(q_end - q_start, num_neg)
    return out


def _rank_result_from_ranks(rank):
    rank = rank.to(torch.float32)
    result = {"MRR": round((1.0 / rank).mean().item(), 4)}
    result.update({f"mrr_hit{k}": round((rank <= k).to(torch.float32).mean().item(), 4) for k in (1, 3, 5, 10, 20, 50, 100)})
    return result


def _iter_grouped_heuristic_scores(method, graph, positive_edges, negative_edges, device, edge_batch_size, *, deadline=None, citation2_reference_settings=False, source_batch_size=None, label=None):
    (rowptr, col, deg, adj) = graph
    total_rows = int(negative_edges.size(0))
    if int(positive_edges.size(0)) != total_rows:
        raise RuntimeError("Grouped negative rows must match their positive query edges.")
    progress_callback = _progress_reporter(label) if label is not None and method in ("shortest_path", "sp", "katz") else None
    if isinstance(negative_edges, DeduplicatedGroupedNegativeEdges):
        if not negative_edges.canonical_undirected:
            raise RuntimeError(
                "The DDI heuristic compact path requires canonical undirected pairs because every heuristic graph is symmetrized."
            )
        unique_prediction = _score_flat(
            method,
            rowptr,
            col,
            deg,
            adj,
            negative_edges.unique_edges,
            device,
            edge_batch_size=edge_batch_size,
            source_batch_size=source_batch_size if citation2_reference_settings else None,
            max_dense_elems=CITATION2_MAX_DENSE_ELEMS if citation2_reference_settings else 4000000,
            deadline=deadline,
            progress_callback=progress_callback,
        ).to(torch.float32)
        print(
            f"{label or 'Grouped negatives'}: original_edges={negative_edges.original_edge_count} unique_canonical_edges={negative_edges.unique_edge_count} score_fraction={negative_edges.decode_fraction:.6f}",
            flush=True,
        )
        gather_batch_size = max(int(edge_batch_size), 1048576)
        width = int(negative_edges.size(1))
        for start, end, inverse in negative_edges.iter_inverse_chunks(gather_batch_size):
            prediction = unique_prediction.index_select(
                0, inverse.reshape(-1).to(device=unique_prediction.device, dtype=torch.long, non_blocking=True)
            )
            yield (start, end, prediction.view(end - start, width), None)
        return
    if getattr(negative_edges, "is_ragged_negative", False):
        iterator = negative_edges.iter_ragged_chunks()
        require_fixed_endpoint = False
    elif is_streaming_negative_edges(negative_edges):

        def grouped_iterator():
            iter_kwargs = {}
            if getattr(negative_edges, "is_endpoint_corruption_negative", False):
                width = max(1, int(negative_edges.size(1)))
                iter_kwargs["pos_chunk_size"] = max(int(getattr(negative_edges, "pos_chunk_size", 1)), 4194304 // width)
            for start, end, chunk in negative_edges.iter_chunks(**iter_kwargs):
                width = int(chunk.size(1))
                row_ids = torch.arange(end - start).repeat_interleave(width)
                yield (start, end, chunk.reshape(-1, 2), row_ids)

        iterator = grouped_iterator()
        require_fixed_endpoint = False
    else:
        if not torch.is_tensor(negative_edges) or negative_edges.dim() != 3:
            raise ValueError("Grouped negatives must be [queries, candidates, 2] or a streaming grouped container.")
        width = max(1, int(negative_edges.size(1)))
        score_chunk = max(int(edge_batch_size), 4194304)
        row_batch = max(1, score_chunk // width)

        def tensor_iterator():
            for start in range(0, total_rows, row_batch):
                end = min(start + row_batch, total_rows)
                yield (start, end, negative_edges[start:end], None)

        iterator = tensor_iterator()
        require_fixed_endpoint = True
    swapped_total = 0
    oriented_all = True
    for start, end, flat_edges, local_rowptr_or_ids in iterator:
        dense_grouped = local_rowptr_or_ids is None
        if dense_grouped:
            (oriented_grouped, swapped) = _orient_citation2_grouped_negatives(positive_edges[start:end], flat_edges)
            row_ids = None
            oriented = oriented_grouped.reshape(-1, 2)
            oriented_chunk = True
        elif getattr(negative_edges, "is_ragged_negative", False):
            lengths = local_rowptr_or_ids[1:] - local_rowptr_or_ids[:-1]
            row_ids = torch.repeat_interleave(torch.arange(end - start), lengths)
            (oriented, swapped, oriented_chunk) = _orient_fixed_endpoint_flat(
                positive_edges[start:end], flat_edges, row_ids, require_fixed_endpoint=require_fixed_endpoint
            )
        else:
            row_ids = local_rowptr_or_ids
            (oriented, swapped, oriented_chunk) = _orient_fixed_endpoint_flat(
                positive_edges[start:end], flat_edges, row_ids, require_fixed_endpoint=require_fixed_endpoint
            )
        swapped_total += swapped
        oriented_all = oriented_all and oriented_chunk
        prediction = _score_flat(
            method,
            rowptr,
            col,
            deg,
            adj,
            oriented,
            device,
            edge_batch_size=edge_batch_size,
            source_batch_size=source_batch_size if citation2_reference_settings else None,
            max_dense_elems=CITATION2_MAX_DENSE_ELEMS if citation2_reference_settings else 4000000,
            deadline=deadline,
            progress_callback=progress_callback,
        )
        if dense_grouped:
            prediction = prediction.view(end - start, int(flat_edges.size(1)))
        yield (start, end, prediction, row_ids)
    if swapped_total:
        print(f"fixed_endpoint_orientation=enabled swapped_edges={swapped_total}", flush=True)
    elif not oriented_all:
        print("fixed_endpoint_orientation=not-applicable", flush=True)


def _stream_grouped_heuristic_metrics(method, graph, positive_edges, positive_predictions, negative_edges, device, edge_batch_size, *, compute_auc, deadline=None, citation2_reference_settings=False, source_batch_size=None, label=None):
    predictions = tuple((value.reshape(-1).to(torch.float32) for value in positive_predictions))
    work_device = predictions[0].device if predictions else torch.device(device)
    if any((value.device != work_device for value in predictions)):
        predictions = tuple((value.to(work_device) for value in predictions))
    total_rows = int(negative_edges.size(0))
    for value in predictions:
        if int(value.numel()) != total_rows:
            raise RuntimeError("Grouped negative rows must match positive predictions.")
    ranks = [torch.empty(total_rows, dtype=torch.float32, device=work_device) for _ in predictions]
    accumulators = [StreamingAUCAP(value) for value in predictions] if compute_auc else []
    for start, end, negative_prediction, row_ids in _iter_grouped_heuristic_scores(
        method,
        graph,
        positive_edges,
        negative_edges,
        device,
        edge_batch_size,
        deadline=deadline,
        citation2_reference_settings=citation2_reference_settings,
        source_batch_size=source_batch_size,
        label=label,
    ):
        negative_prediction = negative_prediction.to(device=work_device, dtype=torch.float32)
        for accumulator in accumulators:
            accumulator.update(negative_prediction.reshape(-1))
        for positive, rank in zip(predictions, ranks):
            if row_ids is None:
                repeated_positive = positive[start:end].view(-1, 1)
                optimistic = (negative_prediction >= repeated_positive).sum(dim=1).to(torch.float32)
                pessimistic = (negative_prediction > repeated_positive).sum(dim=1).to(torch.float32)
                rank[start:end] = 0.5 * (optimistic + pessimistic) + 1.0
                continue
            row_ids_device = row_ids.to(device=work_device, dtype=torch.long, non_blocking=True)
            repeated_positive = positive[start:end][row_ids_device]
            optimistic = torch.zeros(end - start, dtype=torch.float32, device=work_device)
            pessimistic = torch.zeros_like(optimistic)
            optimistic.index_add_(0, row_ids_device, (negative_prediction >= repeated_positive).to(torch.float32))
            pessimistic.index_add_(0, row_ids_device, (negative_prediction > repeated_positive).to(torch.float32))
            rank[start:end] = 0.5 * (optimistic + pessimistic) + 1.0
    rank_metrics = [_rank_result_from_ranks(rank) for rank in ranks]
    auc_metrics = [accumulator.compute() for accumulator in accumulators] if compute_auc else []
    return (rank_metrics, auc_metrics)


def _combine_grouped_metric_results(train_rank, valid_rank, test_rank, *, train_auc=None, valid_auc=None, test_auc=None):
    result = {"MRR": (train_rank["MRR"], valid_rank["MRR"], test_rank["MRR"])}
    for k in (1, 3, 5, 10, 20, 50, 100):
        key = f"mrr_hit{k}"
        result[f"Hits@{k}"] = (train_rank[key], valid_rank[key], test_rank[key])
    if train_auc is not None and valid_auc is not None and (test_auc is not None):
        result["AUC"] = (train_auc["AUC"], valid_auc["AUC"], test_auc["AUC"])
        result["AP"] = (train_auc["AP"], valid_auc["AP"], test_auc["AP"])
    return result


def _metric_score_noncitation(pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, compute_auc=True):
    t_mrr = time.time()
    mrr_train = evaluate_mrr(None, pos_train_pred, neg_valid_pred)
    mrr_valid = evaluate_mrr(None, pos_valid_pred, neg_valid_pred)
    mrr_test = evaluate_mrr(None, pos_test_pred, neg_test_pred)
    mrr_sec = time.time() - t_mrr
    result = {
        "MRR": (mrr_train["MRR"], mrr_valid["MRR"], mrr_test["MRR"]),
        "Hits@1": (mrr_train["mrr_hit1"], mrr_valid["mrr_hit1"], mrr_test["mrr_hit1"]),
        "Hits@3": (mrr_train["mrr_hit3"], mrr_valid["mrr_hit3"], mrr_test["mrr_hit3"]),
        "Hits@5": (mrr_train["mrr_hit5"], mrr_valid["mrr_hit5"], mrr_test["mrr_hit5"]),
        "Hits@10": (mrr_train["mrr_hit10"], mrr_valid["mrr_hit10"], mrr_test["mrr_hit10"]),
        "Hits@20": (mrr_train["mrr_hit20"], mrr_valid["mrr_hit20"], mrr_test["mrr_hit20"]),
        "Hits@50": (mrr_train["mrr_hit50"], mrr_valid["mrr_hit50"], mrr_test["mrr_hit50"]),
        "Hits@100": (mrr_train["mrr_hit100"], mrr_valid["mrr_hit100"], mrr_test["mrr_hit100"]),
    }
    auc_sec = 0.0
    if compute_auc:
        neg_valid_flat = neg_valid_pred.view(-1)
        neg_test_flat = neg_test_pred.view(-1)
        t_auc = time.time()
        sorted_valid_neg = torch.sort(neg_valid_flat).values
        sorted_test_neg = torch.sort(neg_test_flat).values
        auc_train = _auc_ap_against_sorted_negatives(pos_train_pred, sorted_valid_neg)
        auc_valid = _auc_ap_against_sorted_negatives(pos_valid_pred, sorted_valid_neg)
        auc_test = _auc_ap_against_sorted_negatives(pos_test_pred, sorted_test_neg)
        auc_sec = time.time() - t_auc
        result["AUC"] = (auc_train["AUC"], auc_valid["AUC"], auc_test["AUC"])
        result["AP"] = (auc_train["AP"], auc_valid["AP"], auc_test["AP"])
    return (result, mrr_sec, auc_sec)


def _metric_score_citation2(pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, compute_auc=True):
    t_mrr = time.time()
    mrr_train = evaluate_mrr(None, pos_valid_pred, neg_valid_pred)
    mrr_valid = evaluate_mrr(None, pos_valid_pred, neg_valid_pred)
    mrr_test = evaluate_mrr(None, pos_test_pred, neg_test_pred)
    mrr_sec = time.time() - t_mrr
    result = {
        "MRR": (mrr_train["MRR"], mrr_valid["MRR"], mrr_test["MRR"]),
        "Hits@1": (mrr_train["mrr_hit1"], mrr_valid["mrr_hit1"], mrr_test["mrr_hit1"]),
        "Hits@3": (mrr_train["mrr_hit3"], mrr_valid["mrr_hit3"], mrr_test["mrr_hit3"]),
        "Hits@5": (mrr_train["mrr_hit5"], mrr_valid["mrr_hit5"], mrr_test["mrr_hit5"]),
        "Hits@10": (mrr_train["mrr_hit10"], mrr_valid["mrr_hit10"], mrr_test["mrr_hit10"]),
        "Hits@20": (mrr_train["mrr_hit20"], mrr_valid["mrr_hit20"], mrr_test["mrr_hit20"]),
        "Hits@50": (mrr_train["mrr_hit50"], mrr_valid["mrr_hit50"], mrr_test["mrr_hit50"]),
        "Hits@100": (mrr_train["mrr_hit100"], mrr_valid["mrr_hit100"], mrr_test["mrr_hit100"]),
    }
    auc_sec = 0.0
    if compute_auc:
        neg_valid_flat = neg_valid_pred.view(-1)
        neg_test_flat = neg_test_pred.view(-1)
        t_auc = time.time()
        auc_valid = _auc_ap_against_sorted_negatives(pos_valid_pred, torch.sort(neg_valid_flat).values)
        auc_test = _auc_ap_against_sorted_negatives(pos_test_pred, torch.sort(neg_test_flat).values)
        auc_sec = time.time() - t_auc
        result["AUC"] = (auc_valid["AUC"], auc_valid["AUC"], auc_test["AUC"])
        result["AP"] = (auc_valid["AP"], auc_valid["AP"], auc_test["AP"])
    return (result, mrr_sec, auc_sec)


def _eval_method(dataset_name, bundle, method, device, edge_batch_size, citation2_query_batch_size, train_graph, test_graph, compute_auc=True, deadline=None):
    (tr_rowptr, tr_col, tr_deg, tr_adj) = train_graph
    (te_rowptr, te_col, te_deg, te_adj) = test_graph
    citation2_target_only = dataset_name == "ogbl-citation2" and torch.is_tensor(bundle["valid_neg"]) and (bundle["valid_neg"].dim() == 2)
    if citation2_target_only:
        pos_valid_pred = _score_citation2_edges(
            method,
            tr_rowptr,
            tr_col,
            tr_deg,
            tr_adj,
            bundle["valid_pos"],
            device,
            citation2_query_batch_size,
            edge_batch_size,
            deadline=deadline,
            label=f"{method.upper()} Citation2 valid positives",
        )
        pos_test_pred = _score_citation2_edges(
            method,
            te_rowptr,
            te_col,
            te_deg,
            te_adj,
            bundle["test_pos"],
            device,
            citation2_query_batch_size,
            edge_batch_size,
            deadline=deadline,
            label=f"{method.upper()} Citation2 test positives",
        )
        neg_valid_pred = _score_citation2_neg(
            method,
            tr_rowptr,
            tr_col,
            tr_deg,
            tr_adj,
            bundle["valid_pos"],
            bundle["valid_neg"],
            device,
            citation2_query_batch_size,
            edge_batch_size,
            deadline=deadline,
            label=f"{method.upper()} Citation2 valid negatives",
        )
        neg_test_pred = _score_citation2_neg(
            method,
            te_rowptr,
            te_col,
            te_deg,
            te_adj,
            bundle["test_pos"],
            bundle["test_neg"],
            device,
            citation2_query_batch_size,
            edge_batch_size,
            deadline=deadline,
            label=f"{method.upper()} Citation2 test negatives",
        )
        (result, mrr_sec, auc_sec) = _metric_score_citation2(
            pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, compute_auc=compute_auc
        )
        return (result, mrr_sec, auc_sec, 0.0)
    citation2_grouped = (
        dataset_name == "ogbl-citation2"
        and (is_streaming_negative_edges(bundle["valid_neg"]) or (torch.is_tensor(bundle["valid_neg"]) and bundle["valid_neg"].dim() == 3))
        and (is_streaming_negative_edges(bundle["test_neg"]) or (torch.is_tensor(bundle["test_neg"]) and bundle["test_neg"].dim() == 3))
    )
    if citation2_grouped:
        tensor_grouped = torch.is_tensor(bundle["valid_neg"]) and torch.is_tensor(bundle["test_neg"])
        if tensor_grouped and method in ("shortest_path", "sp", "katz"):
            (pos_valid_pred, neg_valid_pred) = _score_citation2_grouped_split(
                method,
                tr_rowptr,
                tr_col,
                tr_deg,
                tr_adj,
                bundle["valid_pos"],
                bundle["valid_neg"],
                device,
                citation2_query_batch_size,
                edge_batch_size,
                deadline=deadline,
                label=f"{method.upper()} Citation2 valid",
            )
            (pos_test_pred, neg_test_pred) = _score_citation2_grouped_split(
                method,
                te_rowptr,
                te_col,
                te_deg,
                te_adj,
                bundle["test_pos"],
                bundle["test_neg"],
                device,
                citation2_query_batch_size,
                edge_batch_size,
                deadline=deadline,
                label=f"{method.upper()} Citation2 test",
            )
            (result, mrr_sec, auc_sec) = _metric_score_citation2(
                pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, compute_auc=compute_auc
            )
            return (result, mrr_sec, auc_sec, 0.0)
        pos_valid_pred = _score_citation2_edges(
            method,
            tr_rowptr,
            tr_col,
            tr_deg,
            tr_adj,
            bundle["valid_pos"],
            device,
            citation2_query_batch_size,
            edge_batch_size,
            deadline=deadline,
            label=f"{method.upper()} Citation2 valid positives",
        )
        pos_test_pred = _score_citation2_edges(
            method,
            te_rowptr,
            te_col,
            te_deg,
            te_adj,
            bundle["test_pos"],
            device,
            citation2_query_batch_size,
            edge_batch_size,
            deadline=deadline,
            label=f"{method.upper()} Citation2 test positives",
        )
        t_rank = time.time()
        (valid_ranks, valid_aucs) = _stream_grouped_heuristic_metrics(
            method,
            train_graph,
            bundle["valid_pos"],
            (pos_valid_pred, pos_valid_pred),
            bundle["valid_neg"],
            device,
            edge_batch_size,
            compute_auc=compute_auc,
            deadline=deadline,
            citation2_reference_settings=True,
            source_batch_size=citation2_query_batch_size,
            label=f"{method.upper()} Citation2 valid negatives",
        )
        (test_ranks, test_aucs) = _stream_grouped_heuristic_metrics(
            method,
            test_graph,
            bundle["test_pos"],
            (pos_test_pred,),
            bundle["test_neg"],
            device,
            edge_batch_size,
            compute_auc=compute_auc,
            deadline=deadline,
            citation2_reference_settings=True,
            source_batch_size=citation2_query_batch_size,
            label=f"{method.upper()} Citation2 test negatives",
        )
        result = _combine_grouped_metric_results(
            valid_ranks[0],
            valid_ranks[1],
            test_ranks[0],
            train_auc=valid_aucs[0] if compute_auc else None,
            valid_auc=valid_aucs[1] if compute_auc else None,
            test_auc=test_aucs[0] if compute_auc else None,
        )
        return (result, 0.0, 0.0, time.time() - t_rank)
    pos_train_pred = _score_flat(method, tr_rowptr, tr_col, tr_deg, tr_adj, bundle["train_val"], device)
    pos_valid_pred = _score_flat(method, tr_rowptr, tr_col, tr_deg, tr_adj, bundle["valid_pos"], device)
    pos_test_pred = _score_flat(method, te_rowptr, te_col, te_deg, te_adj, bundle["test_pos"], device)
    grouped_valid = is_streaming_negative_edges(bundle["valid_neg"]) or (
        torch.is_tensor(bundle["valid_neg"]) and bundle["valid_neg"].dim() == 3
    )
    grouped_test = is_streaming_negative_edges(bundle["test_neg"]) or (
        torch.is_tensor(bundle["test_neg"]) and bundle["test_neg"].dim() == 3
    )
    if grouped_valid or grouped_test:
        if not (grouped_valid and grouped_test):
            raise RuntimeError("Validation and test negatives must use the same grouped evaluation layout.")
        citation2_streaming = dataset_name == "ogbl-citation2" and (
            is_streaming_negative_edges(bundle["valid_neg"]) or is_streaming_negative_edges(bundle["test_neg"])
        )
        t_rank = time.time()
        (valid_ranks, valid_aucs) = _stream_grouped_heuristic_metrics(
            method,
            train_graph,
            bundle["valid_pos"],
            (pos_train_pred, pos_valid_pred),
            bundle["valid_neg"],
            device,
            edge_batch_size,
            compute_auc=compute_auc,
            deadline=deadline,
            citation2_reference_settings=citation2_streaming,
            source_batch_size=citation2_query_batch_size if citation2_streaming else None,
            label=f"{method.upper()} valid negatives",
        )
        (test_ranks, test_aucs) = _stream_grouped_heuristic_metrics(
            method,
            test_graph,
            bundle["test_pos"],
            (pos_test_pred,),
            bundle["test_neg"],
            device,
            edge_batch_size,
            compute_auc=compute_auc,
            deadline=deadline,
            citation2_reference_settings=citation2_streaming,
            source_batch_size=citation2_query_batch_size if citation2_streaming else None,
            label=f"{method.upper()} test negatives",
        )
        result = _combine_grouped_metric_results(
            valid_ranks[0],
            valid_ranks[1],
            test_ranks[0],
            train_auc=valid_aucs[0] if compute_auc else None,
            valid_auc=valid_aucs[1] if compute_auc else None,
            test_auc=test_aucs[0] if compute_auc else None,
        )
        return (result, 0.0, 0.0, time.time() - t_rank)
    neg_valid_pred = _reshape_neg_scores(
        bundle["valid_pos"], bundle["valid_neg"], _score_flat(method, tr_rowptr, tr_col, tr_deg, tr_adj, bundle["valid_neg"], device)
    )
    neg_test_pred = _reshape_neg_scores(
        bundle["test_pos"], bundle["test_neg"], _score_flat(method, te_rowptr, te_col, te_deg, te_adj, bundle["test_neg"], device)
    )
    (result, mrr_sec, auc_sec) = _metric_score_noncitation(
        pos_train_pred, pos_valid_pred, neg_valid_pred, pos_test_pred, neg_test_pred, compute_auc=compute_auc
    )
    return (result, mrr_sec, auc_sec, 0.0)


def main():
    program_t0 = time.time()
    args = parse_args()
    cpu_threads = configure_torch_cpu_threads()
    device = resolve_ogbl_device(args.device)
    args.device = str(device)
    mode = args.mode
    args.eval_cap = resolve_ogbl_eval_cap(args.eval_cap, mode, args.dataset)
    metric_key = resolve_ogbl_metric(args.metric, args.dataset)
    protocol_metadata = ogbl_protocol_metadata(
        dataset=args.dataset,
        mode=mode,
        eval_cap=args.eval_cap,
        selection_metric=metric_key,
    )
    methods = ["cn", "aa", "ra", "shortest_path", "katz"] if args.heuristic == "all" else [args.heuristic]
    heuristic_protocol = _heuristic_protocol_metadata(args.dataset)
    timed_out = False
    log_path = None
    if args.save_log:
        log_dir = os.path.join("results", "ogbl", args.mode, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "heuristics.txt")
    print(f"Using device: {device}")
    print(f"torch_cpu_threads={cpu_threads}")
    print_ogbl_protocol(args, mode, protocol_metadata, metric_key)
    print(f"seed={args.seed}")
    for key, value in heuristic_protocol.items():
        print(f"{key}={(str(value).lower() if isinstance(value, bool) else value)}")
    if args.dataset == "ogbl-citation2":
        citation2_scope = f"up_to_{args.eval_cap}_positives_per_split" if int(args.eval_cap or 0) > 0 else "full_split"
        print(f"citation2_evaluation_scope={citation2_scope}")
    set_seed(args.seed)
    t_data = time.time()
    bundle = read_data(
        args.dataset,
        mode,
        eval_cap=args.eval_cap,
        seed=args.seed,
        root=args.root,
        heart_negatives=args.heart_negatives,
        pool=args.pool,
        all_negatives=args.all_negatives,
        ranked_backend=args.ranked_negatives_backend,
        negative_cache_dir=args.negative_cache_dir,
        cache_negatives=not args.no_negative_cache,
    )
    candidate_provenance = heart_candidate_metadata(bundle)
    heuristic_eval_edges = {"neg_valid_edge": bundle["valid_neg"], "neg_test_edge": bundle["test_neg"]}
    ddi_dedup_t0 = time.time()
    (heuristic_eval_edges, ddi_dedup_summaries) = prepare_ddi_grouped_eval_edges(
        heuristic_eval_edges, dataset_name=args.dataset, model_name="heuristic", num_nodes=int(bundle["x"].size(0)), source_bundle=bundle
    )
    ddi_dedup_prepare_sec = time.time() - ddi_dedup_t0
    data_load_sec = time.time() - t_data
    print(f"data_load_sec={data_load_sec:.2f}")
    for summary in ddi_dedup_summaries:
        print(
            f"ddi_eval_dedup key={summary['key']} original_edges={summary['original_edges']} unique_edges={summary['unique_edges']} score_fraction={summary['decode_fraction']:.6f} canonical_undirected={summary['canonical_undirected']} cpu_storage_bytes={summary['storage_nbytes']}",
            flush=True,
        )
    if ddi_dedup_summaries:
        print(f"ddi_eval_dedup_prepare_sec={ddi_dedup_prepare_sec:.2f}", flush=True)
    if bundle.get("pool_per_side") is not None:
        for key in ("pool_setting", "pool_full_graph", "pool_cap_applied", "pool_sampling", "pool_requested_per_side", "pool_requested_total"):
            print(f"{key}={bundle.get(key)}")
        print(f"pool_per_side_effective={bundle.get('pool_per_side')}")
        print(f"pool_total_effective={bundle.get('pool_total')}")
    if bundle.get("heart_candidate_universe") is not None:
        for key in ("heart_candidate_universe", "heart_candidate_graph_nodes", "heart_selection", "heart_negatives_requested_per_side", "heart_negatives_requested_total"):
            print(f"{key}={bundle.get(key)}")
        for label, key in (("heart_negatives_per_side_effective", "heart_negatives_per_side"), ("heart_negatives_total_effective", "heart_negatives_total")):
            print(f"{label}={bundle.get(key)}")
    if bundle.get("negative_cache_path"):
        print(f"negative_cache_path={bundle.get('negative_cache_path')}")
    for key, value in sorted(candidate_provenance.items()):
        print(f"{key}={value}")
    compute_auc = _should_compute_auc(args.compute_auc, mode)
    print(f"compute_auc_effective={compute_auc}")
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED after data loading: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    t_graph = time.time()
    (train_graph, test_graph) = _build_graph_cache(bundle, args.dataset, device)
    graph_prep_sec = time.time() - t_graph
    print(f"graph_prep_sec={graph_prep_sec:.2f}")
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED after graph prep: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    results_by_method = {}
    for method in methods:
        if timed_out or _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED before method {method.upper()}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
        eval_profiler = StageProfiler(device)
        eval_profiler.start()
        timeout_error = None
        try:
            (result, mrr_sec, auc_sec, grouped_pipeline_sec) = _eval_method(
                args.dataset,
                bundle,
                method,
                device,
                args.edge_batch_size,
                args.citation2_query_batch_size,
                train_graph,
                test_graph,
                compute_auc=compute_auc,
                deadline=program_t0 + RUNTIME_LIMIT_SEC,
            )
        except TimeoutError as exc:
            timeout_error = exc
        finally:
            eval_info = eval_profiler.stop()
        if timeout_error is not None:
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED during method {method.upper()}: {timeout_error}", flush=True)
            break
        eval_sec = eval_info["sec"]
        selected_key = find_result_key(result, metric_key)
        if selected_key is None:
            raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(result.keys())}")
        results_by_method[method] = {
            "result": result,
            "selected_test": float(result[selected_key][2]),
            "eval_sec": eval_sec,
            "mrr_sec": mrr_sec,
            "auc_sec": auc_sec,
            "grouped_pipeline_sec": grouped_pipeline_sec,
            "peak_cpu_rss_mb": eval_info["cpu_peak_rss_mb"],
            "peak_cuda_allocated_mb": eval_info["cuda_peak_allocated_mb"],
            "peak_cuda_reserved_mb": eval_info["cuda_peak_reserved_mb"],
        }
        parts = [
            f"{method.upper()} eval_sec={eval_sec:.2f} mrr_sec={mrr_sec:.2f} auc_sec={auc_sec:.2f} grouped_pipeline_sec={grouped_pipeline_sec:.2f}",
            f"peak_cpu_rss_mb={eval_info['cpu_peak_rss_mb']:.2f}",
            f"peak_cuda_allocated_mb={eval_info['cuda_peak_allocated_mb']:.2f}",
        ]
        for k in _ordered_metric_keys(result):
            parts.append(f"test {k}={100 * float(result[k][2]):.6f}")
        print(" | ".join(parts))
        if _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED after method {method.upper()}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
    total_wall_sec = time.time() - program_t0
    summary_lines = []

    def log(line=""):
        print(line)
        summary_lines.append(str(line))

    log("\n" + "=" * 80)
    log("Timing summary")
    log(f"torch_cpu_threads: {cpu_threads}")
    log(f"runtime_limit_exceeded: {timed_out}")
    log("status: exceeded 24 hour runtime limit" if timed_out else "status: completed within 24 hour runtime limit")
    log(f"runtime_limit_sec: {RUNTIME_LIMIT_SEC:.2f}")
    log_protocol_summary(log, args, mode, protocol_metadata, metric_key, bundle, device)
    for key, value in sorted(candidate_provenance.items()):
        log(f"{key}: {value}")
    for summary in ddi_dedup_summaries:
        prefix = str(summary["key"]).replace("neg_", "").replace("_edge", "")
        log(f"ddi_{prefix}_negative_edges_original: {summary['original_edges']}")
        log(f"ddi_{prefix}_negative_edges_unique: {summary['unique_edges']}")
        log(f"ddi_{prefix}_score_fraction: {summary['decode_fraction']:.6f}")
        log(f"ddi_{prefix}_canonical_undirected: {str(summary['canonical_undirected']).lower()}")
    if ddi_dedup_summaries:
        log(f"ddi_eval_dedup_prepare_sec: {ddi_dedup_prepare_sec:.2f}")
    log(f"data_load_sec: {data_load_sec:.2f}")
    log(f"graph_prep_sec: {graph_prep_sec:.2f}")
    log(f"total_wall_sec: {total_wall_sec:.2f}")
    log(f"cpu_rss_mb_current: {current_cpu_rss_mb():.2f}")
    log(f"cpu_rss_mb_peak_process: {peak_cpu_rss_mb():.2f}")
    log("evaluation_scope: validation_and_test")
    for key, value in heuristic_protocol.items():
        log(f"{key}: {(str(value).lower() if isinstance(value, bool) else value)}")
    if args.dataset == "ogbl-citation2":
        citation2_scope = f"selected_{int(bundle['valid_pos'].size(0))}_valid_{int(bundle['test_pos'].size(0))}_test_under_cap_{int(args.eval_cap or 0)}"
        log(f"citation2_evaluation_scope: {citation2_scope}")
    log("\n" + "=" * 80)
    log(f"Final heuristic results on {args.dataset} | mode={args.mode} | seed={args.seed}")
    for method in methods:
        if method not in results_by_method:
            log(f"\n[{method.upper()}]")
            log("Method did not complete.")
            continue
        info = results_by_method[method]
        result = info["result"]
        log(f"\n[{method.upper()}]")
        log(f"eval_sec: {info['eval_sec']:.2f}")
        log(f"mrr_sec: {info['mrr_sec']:.2f}")
        log(f"auc_sec: {info['auc_sec']:.2f}")
        log(f"grouped_scoring_and_metrics_sec: {info['grouped_pipeline_sec']:.2f}")
        log(f"peak_cpu_rss_mb: {info['peak_cpu_rss_mb']:.2f}")
        log(f"peak_cuda_allocated_mb: {info['peak_cuda_allocated_mb']:.2f}")
        log(f"peak_cuda_reserved_mb: {info['peak_cuda_reserved_mb']:.2f}")
        log(f"Test {metric_key}: {100 * info['selected_test']:.6f} (percent)")
        for name in ("AUC", "AP", "MRR"):
            if name in result:
                log(f"Test {name}: {100 * float(result[name][2]):.6f}")
        for k_name in sorted([k for k in result.keys() if isinstance(k, str) and k.startswith("Hits@")], key=_hit_sort_key):
            log(f"{k_name}: {100 * float(result[k_name][2]):.6f}")
    log("=" * 80)
    _write_summary(log_path, summary_lines)
    incomplete = [method for method in methods if method not in results_by_method]
    if timed_out or incomplete:
        print(f"ERROR: heuristic evaluation was incomplete; timed_out={timed_out}, missing_methods={incomplete}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
