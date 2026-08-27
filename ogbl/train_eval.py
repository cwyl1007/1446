import time
import torch
from torch_geometric.utils import to_undirected
from torch_sparse import SparseTensor
from utils.metrics import StreamingAUCAP, evaluate_mrr, evaluate_auc
from .fast_negatives import DeduplicatedGroupedNegativeEdges, is_streaming_negative_edges
from .training import (
    _clear_gcnconv_cache,
    _clear_model_decode_cache,
    cache_eval_edges_on_device,
    find_result_key,
    move_graph_data_to_device,
    recommended_decode_batch_size,
    recommended_train_samples_per_epoch,
    train_one_epoch_ogbl,
)


def _profile_add(profile, key, elapsed):
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(elapsed)


def _sync_if_profiled(profile, device):
    if profile is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _apply_model_evaluation_transform(model, scores):
    transform = str(getattr(model, "reference_evaluation_transform", "identity")).strip().lower()
    if transform == "sigmoid":
        return torch.sigmoid(scores.float())
    if transform in ("", "identity", "none"):
        return scores.to(torch.float32)
    raise ValueError(f"Unknown OGB evaluation transform: {transform}")


@torch.no_grad()
def _predict_edge_scores(model, z, edges, device, batch_size=65536):
    edges = edges.to(dtype=torch.long)
    n = int(edges.size(0))
    if n == 0:
        return torch.empty((0,), dtype=torch.float32, device=device)
    out = torch.empty((n,), dtype=torch.float32, device=device)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        e = edges[start:end].to(device, non_blocking=True).t().contiguous()
        scores = model.decode(z, e).view(-1)
        out[start:end] = _apply_model_evaluation_transform(model, scores)
    return out


@torch.no_grad()
def _predict_grouped_edge_scores(model, z, edges, device, batch_size=65536):
    shape = tuple(edges.shape[:-1])
    flat = edges.reshape(-1, 2)
    pred = _predict_edge_scores(model, z, flat, device, batch_size=batch_size)
    return pred.view(*shape)


@torch.no_grad()
def _predict_citation2_neg_scores(model, z, pos_edges, neg_targets, device, query_batch_size=1024, edge_batch_size=65536):
    src_all = pos_edges[:, 0]
    num_queries = int(pos_edges.size(0))
    num_neg = int(neg_targets.size(1))
    out = torch.empty((num_queries, num_neg), dtype=torch.float32, device=device)
    for q_start in range(0, num_queries, query_batch_size):
        q_end = min(q_start + query_batch_size, num_queries)
        src = src_all[q_start:q_end].to(device, non_blocking=True)
        neg = neg_targets[q_start:q_end].to(device, non_blocking=True)
        batch_queries = int(src.size(0))
        src_rep = src.view(-1, 1).repeat(1, num_neg).reshape(-1)
        dst_rep = neg.reshape(-1)
        e_srcdst = torch.stack([src_rep, dst_rep], dim=0)
        chunks = []
        for e_start in range(0, e_srcdst.size(1), edge_batch_size):
            e_end = min(e_start + edge_batch_size, e_srcdst.size(1))
            e = e_srcdst[:, e_start:e_end]
            scores = model.decode(z, e).view(-1)
            chunks.append(_apply_model_evaluation_transform(model, scores))
        out[q_start:q_end, :] = torch.cat(chunks, dim=0).view(batch_queries, num_neg)
    return out


def _reshape_neg_scores(pos_edges, neg_edges, neg_pred):
    if neg_edges.dim() == 3:
        return neg_pred.view(neg_edges.size(0), neg_edges.size(1))
    npos = int(pos_edges.size(0))
    if npos == 0:
        return neg_pred.view(0, 1)
    if neg_pred.numel() % npos != 0:
        raise RuntimeError("Negative edges are incompatible with positive edges for grouped evaluation.")
    return neg_pred.view(npos, neg_pred.numel() // npos)


def _rank_metrics_from_rank_tensors(rank):
    rank = rank.to(torch.float32)
    out = {
        "MRR": round((1.0 / rank).mean().item(), 4),
        "mrr_hit1": round((rank <= 1).to(torch.float32).mean().item(), 4),
        "mrr_hit3": round((rank <= 3).to(torch.float32).mean().item(), 4),
        "mrr_hit5": round((rank <= 5).to(torch.float32).mean().item(), 4),
        "mrr_hit10": round((rank <= 10).to(torch.float32).mean().item(), 4),
        "mrr_hit20": round((rank <= 20).to(torch.float32).mean().item(), 4),
        "mrr_hit50": round((rank <= 50).to(torch.float32).mean().item(), 4),
        "mrr_hit100": round((rank <= 100).to(torch.float32).mean().item(), 4),
    }
    return out


def _iter_grouped_negative_chunks(negative_edges, batch_size):
    if is_streaming_negative_edges(negative_edges):
        yield from negative_edges.iter_chunks()
        return
    rows = int(negative_edges.size(0))
    k = max(1, int(negative_edges.size(1)))
    row_batch = max(1, int(batch_size) // k)
    for start in range(0, rows, row_batch):
        end = min(start + row_batch, rows)
        yield (start, end, negative_edges[start:end])


def _decoder_is_endpoint_symmetric(model) -> bool:
    candidate = getattr(model, "module", model)
    return bool(getattr(candidate, "decode_is_symmetric", False))


def _decoder_supports_exact_dedup(model) -> bool:
    candidate = getattr(model, "module", model)
    return bool(getattr(candidate, "decode_is_dedup_safe", False))


@torch.no_grad()
def _iter_grouped_negative_score_chunks(model, z, negative_edges, device, batch_size):
    if isinstance(negative_edges, DeduplicatedGroupedNegativeEdges):
        if not _decoder_supports_exact_dedup(model):
            raise RuntimeError("Compact DDI negatives require a context-independent decoder with decode_is_dedup_safe=True.")
        if negative_edges.canonical_undirected and (not _decoder_is_endpoint_symmetric(model)):
            raise RuntimeError("Canonical undirected DDI negatives require a decoder with decode_is_symmetric=True.")
        unique_scores = _predict_edge_scores(model, z, negative_edges.unique_edges, device, batch_size=batch_size)
        k = int(negative_edges.size(1))
        for start, end, inverse in negative_edges.iter_inverse_chunks(batch_size):
            device_inverse = inverse.reshape(-1).to(device=device, dtype=torch.long, non_blocking=True)
            scores = unique_scores.index_select(0, device_inverse)
            yield (start, end, scores.view(end - start, k))
        return
    for start, end, neg_edges in _iter_grouped_negative_chunks(negative_edges, batch_size):
        yield (start, end, _predict_grouped_edge_scores(model, z, neg_edges, device, batch_size=batch_size))


def _ragged_rank_counts(model, z, pos_predictions, negative_edges, device, batch_size, auc_accumulators=()):
    ranks = [torch.empty(int(predictions.numel()), dtype=torch.float32, device=device) for predictions in pos_predictions]
    for start, end, flat_edges, local_rowptr in negative_edges.iter_ragged_chunks():
        lengths = (local_rowptr[1:] - local_rowptr[:-1]).to(device=device, dtype=torch.long)
        scores = _predict_edge_scores(model, z, flat_edges, device, batch_size=batch_size)
        for accumulator in auc_accumulators:
            accumulator.update(scores)
        row_ids = torch.repeat_interleave(torch.arange(end - start, device=device), lengths)
        for predictions, output in zip(pos_predictions, ranks):
            repeated_positive = predictions[start:end].to(device=device, dtype=torch.float32)[row_ids]
            optimistic = torch.zeros(end - start, dtype=torch.float32, device=device)
            pessimistic = torch.zeros_like(optimistic)
            optimistic.index_add_(0, row_ids, (scores >= repeated_positive).to(torch.float32))
            pessimistic.index_add_(0, row_ids, (scores > repeated_positive).to(torch.float32))
            output[start:end] = 0.5 * (optimistic + pessimistic) + 1.0
    return tuple(ranks)


@torch.no_grad()
def _stream_grouped_mrr(model, z, pos_predictions, negative_edges, device, batch_size=65536, compute_auc=False):
    pos_predictions = tuple(pos_predictions)
    rows = int(negative_edges.size(0))
    if any(int(predictions.numel()) != rows for predictions in pos_predictions):
        raise RuntimeError(f"Grouped negatives have {rows} rows but positive predictions do not.")
    accumulators = tuple(StreamingAUCAP(predictions) for predictions in pos_predictions) if compute_auc else ()
    if getattr(negative_edges, "is_ragged_negative", False):
        ranks = _ragged_rank_counts(model, z, pos_predictions, negative_edges, device, batch_size, accumulators)
    else:
        ranks = [torch.empty(rows, dtype=torch.float32, device=device) for _ in pos_predictions]
        for start, end, neg_pred in _iter_grouped_negative_score_chunks(model, z, negative_edges, device, batch_size):
            for accumulator in accumulators:
                accumulator.update(neg_pred)
            for predictions, output in zip(pos_predictions, ranks):
                positive = predictions[start:end].view(-1, 1).to(device=device, dtype=torch.float32)
                optimistic = (neg_pred >= positive).sum(dim=1).to(torch.float32)
                pessimistic = (neg_pred > positive).sum(dim=1).to(torch.float32)
                output[start:end] = 0.5 * (optimistic + pessimistic) + 1.0
    stats = tuple(_rank_metrics_from_rank_tensors(rank) for rank in ranks)
    return (*stats, *(accumulator.compute() for accumulator in accumulators))


def _rank_result_pair(train_stats, valid_stats, test_value=None):
    result = {"MRR": (train_stats["MRR"], valid_stats["MRR"], test_value.get("MRR") if test_value else None)}
    for k in [1, 3, 5, 10, 20, 50, 100]:
        result[f"Hits@{k}"] = (train_stats[f"mrr_hit{k}"], valid_stats[f"mrr_hit{k}"], test_value.get(f"Hits@{k}") if test_value else None)
    return result


def _single_rank_result(stats):
    result = {"MRR": stats["MRR"]}
    for k in [1, 3, 5, 10, 20, 50, 100]:
        result[f"Hits@{k}"] = stats[f"mrr_hit{k}"]
    return result


def _auc_pair(pos_pred, neg_pred):
    neg_flat = neg_pred.reshape(-1)
    device = pos_pred.device
    pred = torch.cat([pos_pred.reshape(-1), neg_flat])
    true = torch.cat(
        [torch.ones(pos_pred.numel(), dtype=torch.uint8, device=device), torch.zeros(neg_flat.numel(), dtype=torch.uint8, device=device)]
    )
    return evaluate_auc(pred, true)


def _auc_ap_against_sorted_negatives(pos_pred, sorted_negatives):
    pos = pos_pred.reshape(-1)
    neg = sorted_negatives.reshape(-1)
    n_pos = int(pos.numel())
    n_neg = int(neg.numel())
    if n_pos == 0 or n_neg == 0:
        raise ValueError("AUC/AP require both positive and negative predictions.")
    lower = torch.searchsorted(neg, pos, right=False)
    upper = torch.searchsorted(neg, pos, right=True)
    auc = (lower.to(torch.float64) + 0.5 * (upper - lower).to(torch.float64)).mean() / float(n_neg)
    pos_desc = torch.sort(pos, descending=True).values
    group_end = torch.ones(n_pos, dtype=torch.bool, device=pos.device)
    if n_pos > 1:
        group_end[:-1] = pos_desc[:-1] != pos_desc[1:]
    end_index = torch.nonzero(group_end, as_tuple=False).view(-1)
    thresholds = pos_desc[end_index]
    cumulative_tp = (end_index + 1).to(torch.float64)
    previous_tp = torch.cat([cumulative_tp.new_zeros(1), cumulative_tp[:-1]])
    negative_ge = (n_neg - torch.searchsorted(neg, thresholds, right=False)).to(torch.float64)
    precision = cumulative_tp / (cumulative_tp + negative_ge)
    ap = torch.sum((cumulative_tp - previous_tp) / float(n_pos) * precision)
    return {"AUC": round(float(auc.item()), 4), "AP": round(float(ap.item()), 4)}


def _validation_metrics_from_scores(pos_train_pred, pos_valid_pred, neg_valid_pred, compute_auc, profile=None):
    device = pos_valid_pred.device
    _sync_if_profiled(profile, device)
    t_mrr = time.time()
    mrr_train = evaluate_mrr(None, pos_train_pred, neg_valid_pred)
    mrr_valid = evaluate_mrr(None, pos_valid_pred, neg_valid_pred)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "mrr_sec", time.time() - t_mrr)
    result = _rank_result_pair(mrr_train, mrr_valid)
    if compute_auc:
        _sync_if_profiled(profile, device)
        t_auc = time.time()
        use_shared_cuda = neg_valid_pred.is_cuda and neg_valid_pred.numel() >= 1000000
        if use_shared_cuda:
            sorted_neg = torch.sort(neg_valid_pred.reshape(-1)).values
            auc_train = _auc_ap_against_sorted_negatives(pos_train_pred, sorted_neg)
            auc_valid = _auc_ap_against_sorted_negatives(pos_valid_pred, sorted_neg)
        else:
            auc_train = _auc_pair(pos_train_pred, neg_valid_pred)
            auc_valid = _auc_pair(pos_valid_pred, neg_valid_pred)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "auc_sec", time.time() - t_auc)
        result["AUC"] = (auc_train["AUC"], auc_valid["AUC"], None)
        result["AP"] = (auc_train["AP"], auc_valid["AP"], None)
    return result


def _test_metrics_from_scores(pos_test_pred, neg_test_pred, compute_auc, profile=None):
    device = pos_test_pred.device
    _sync_if_profiled(profile, device)
    t_mrr = time.time()
    mrr_test = evaluate_mrr(None, pos_test_pred, neg_test_pred)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "mrr_sec", time.time() - t_mrr)
    result = _single_rank_result(mrr_test)
    if compute_auc:
        _sync_if_profiled(profile, device)
        t_auc = time.time()
        if neg_test_pred.is_cuda and neg_test_pred.numel() >= 1000000:
            sorted_neg = torch.sort(neg_test_pred.reshape(-1)).values
            auc_test = _auc_ap_against_sorted_negatives(pos_test_pred, sorted_neg)
        else:
            auc_test = _auc_pair(pos_test_pred, neg_test_pred)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "auc_sec", time.time() - t_auc)
        result["AUC"] = auc_test["AUC"]
        result["AP"] = auc_test["AP"]
    return result


def merge_ogbl_results(validation_results, test_results):
    merged = {}
    for key, triple in validation_results.items():
        if not isinstance(triple, (tuple, list)) or len(triple) != 3:
            continue
        merged[key] = (triple[0], triple[1], test_results.get(key))
    for key, value in test_results.items():
        if key not in merged:
            merged[key] = (None, None, value)
    return merged


@torch.no_grad()
def prepare_ogbl_evaluation(model, data, eval_edges, dataset_name, device, batch_size=65536, citation2_query_batch_size=512, profile=None, test_only=False):
    model.eval()
    device = torch.device(device)
    data = move_graph_data_to_device(data, device)
    _clear_model_decode_cache(model)
    needs_distinct_test_graph = str(dataset_name) == "ogbl-collab"
    z_train = None
    if not (bool(test_only) and needs_distinct_test_graph):
        _sync_if_profiled(profile, device)
        t0 = time.time()
        z_train = model.embed(data)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "inference_sec", time.time() - t0)
    return {
        "model": model,
        "data": data,
        "eval_edges": eval_edges,
        "dataset_name": str(dataset_name),
        "device": device,
        "batch_size": max(1, int(batch_size)),
        "citation2_query_batch_size": max(1, int(citation2_query_batch_size)),
        "z_train": z_train,
        "z_test": z_train if not needs_distinct_test_graph else None,
    }


@torch.no_grad()
def _ensure_test_embedding(context, profile=None):
    if context["z_test"] is not None:
        return context["z_test"]
    model = context["model"]
    data = context["data"]
    eval_edges = context["eval_edges"]
    device = context["device"]
    valid_input_pos = eval_edges.get("valid_input_pos")
    if valid_input_pos is None:
        valid_input_pos = getattr(data, "valid_input_pos", None)
    if valid_input_pos is None:
        valid_input_pos = eval_edges["pos_valid_edge"]
    train_pos = eval_edges["pos_train_edge"]
    valid_input_pos = valid_input_pos.to(device=device, dtype=torch.long, non_blocking=True)
    valid_input_weight = eval_edges.get("valid_input_weight")
    if valid_input_weight is None:
        valid_input_weight = getattr(data, "valid_input_weight", None)
    if valid_input_weight is None:
        valid_input_weight = torch.ones(valid_input_pos.size(0), dtype=torch.float32)
    valid_input_weight = valid_input_weight.to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
    train_edge_index = getattr(data, "edge_index", None)
    train_edge_weight = getattr(data, "edge_weight", None)
    if train_edge_index is not None and train_edge_weight is not None:
        train_edge_index = train_edge_index.to(device=device, dtype=torch.long, non_blocking=True)
        train_edge_weight = train_edge_weight.to(device=device, dtype=torch.float32, non_blocking=True).reshape(-1)
        (valid_edge_index, valid_edge_weight) = to_undirected(
            valid_input_pos.t().contiguous(), valid_input_weight, num_nodes=int(data.num_nodes)
        )
        tv_edge = torch.cat([train_edge_index, valid_edge_index], dim=1)
        tv_weight = torch.cat([train_edge_weight, valid_edge_weight], dim=0)
    else:
        train_pos_device = train_pos.to(device=device, dtype=torch.long, non_blocking=True)
        input_edge = torch.cat([train_pos_device, valid_input_pos], dim=0)
        input_weight = torch.cat([torch.ones(train_pos_device.size(0), dtype=torch.float32, device=device), valid_input_weight])
        (tv_edge, tv_weight) = to_undirected(input_edge.t().contiguous(), input_weight, num_nodes=int(data.num_nodes))
    data_test = type("EvalGraph", (), {})()
    data_test.x = data.x
    data_test.num_nodes = int(data.num_nodes)
    data_test.edge_index = tv_edge
    data_test.edge_weight = tv_weight
    test_rowptr = getattr(data, "csr_tv_rowptr", None)
    test_col = getattr(data, "csr_tv_col", None)
    if test_rowptr is not None and test_col is not None:
        data_test.csr_rowptr = test_rowptr
        data_test.csr_col = test_col
    data_test.adj_t = SparseTensor.from_edge_index(tv_edge, tv_weight, (int(data.num_nodes), int(data.num_nodes)))
    _sync_if_profiled(profile, device)
    t0 = time.time()
    _clear_gcnconv_cache(model)
    _clear_model_decode_cache(model)
    context["z_test"] = model.embed(data_test)
    _clear_gcnconv_cache(model)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "inference_sec", time.time() - t0)
    return context["z_test"]


@torch.no_grad()
def evaluate_ogbl_validation(context, profile=None, compute_auc=True):
    model = context["model"]
    edges = context["eval_edges"]
    dataset_name = context["dataset_name"]
    device = context["device"]
    batch_size = context["batch_size"]
    z = context["z_train"]
    _sync_if_profiled(profile, device)
    t_inference = time.time()
    citation2_target_only = (
        dataset_name == "ogbl-citation2" and torch.is_tensor(edges["neg_valid_edge"]) and (edges["neg_valid_edge"].dim() == 2)
    )
    if citation2_target_only:
        pos_valid = _predict_edge_scores(model, z, edges["pos_valid_edge"], device, batch_size=batch_size)
        neg_valid = _predict_citation2_neg_scores(
            model,
            z,
            edges["pos_valid_edge"],
            edges["neg_valid_edge"],
            device,
            query_batch_size=context["citation2_query_batch_size"],
            edge_batch_size=batch_size,
        )
        _sync_if_profiled(profile, device)
        _profile_add(profile, "inference_sec", time.time() - t_inference)
        t_metric = time.time()
        t_mrr = time.time()
        stats = evaluate_mrr(None, pos_valid, neg_valid)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "mrr_sec", time.time() - t_mrr)
        result = _rank_result_pair(stats, stats)
        if compute_auc:
            t_auc = time.time()
            auc_valid = _auc_pair(pos_valid, neg_valid)
            _sync_if_profiled(profile, device)
            _profile_add(profile, "auc_sec", time.time() - t_auc)
            result["AUC"] = (auc_valid["AUC"], auc_valid["AUC"], None)
            result["AP"] = (auc_valid["AP"], auc_valid["AP"], None)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "testing_sec", time.time() - t_metric)
        return result
    pos_train = _predict_edge_scores(model, z, edges["train_val_edge"], device, batch_size=batch_size)
    pos_valid = _predict_edge_scores(model, z, edges["pos_valid_edge"], device, batch_size=batch_size)
    neg_edges = edges["neg_valid_edge"]
    grouped = is_streaming_negative_edges(neg_edges) or (torch.is_tensor(neg_edges) and neg_edges.dim() == 3)
    if grouped:
        _sync_if_profiled(profile, device)
        _profile_add(profile, "inference_sec", time.time() - t_inference)
        t_grouped = time.time()
        if compute_auc:
            (train_stats, valid_stats, auc_train, auc_valid) = _stream_grouped_mrr(
                model, z, (pos_train, pos_valid), neg_edges, device, batch_size=batch_size, compute_auc=True
            )
        else:
            (train_stats, valid_stats) = _stream_grouped_mrr(
                model, z, (pos_train, pos_valid), neg_edges, device, batch_size=batch_size
            )
        _sync_if_profiled(profile, device)
        grouped_sec = time.time() - t_grouped
        _profile_add(profile, "streamed_grouped_eval_sec", grouped_sec)
        _profile_add(profile, "testing_sec", grouped_sec)
        result = _rank_result_pair(train_stats, valid_stats)
        if compute_auc:
            result["AUC"] = (auc_train["AUC"], auc_valid["AUC"], None)
            result["AP"] = (auc_train["AP"], auc_valid["AP"], None)
        return result
    neg_valid = _predict_edge_scores(model, z, neg_edges, device, batch_size=batch_size)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "inference_sec", time.time() - t_inference)
    t_metric = time.time()
    result = _validation_metrics_from_scores(pos_train, pos_valid, neg_valid, compute_auc, profile=profile)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "testing_sec", time.time() - t_metric)
    return result


@torch.no_grad()
def evaluate_ogbl_test(context, profile=None, compute_auc=True):
    model = context["model"]
    edges = context["eval_edges"]
    dataset_name = context["dataset_name"]
    device = context["device"]
    batch_size = context["batch_size"]
    z = _ensure_test_embedding(context, profile=profile)
    _sync_if_profiled(profile, device)
    t_inference = time.time()
    citation2_target_only = (
        dataset_name == "ogbl-citation2" and torch.is_tensor(edges["neg_test_edge"]) and (edges["neg_test_edge"].dim() == 2)
    )
    if citation2_target_only:
        pos_test = _predict_edge_scores(model, z, edges["pos_test_edge"], device, batch_size=batch_size)
        neg_test = _predict_citation2_neg_scores(
            model,
            z,
            edges["pos_test_edge"],
            edges["neg_test_edge"],
            device,
            query_batch_size=context["citation2_query_batch_size"],
            edge_batch_size=batch_size,
        )
        _sync_if_profiled(profile, device)
        _profile_add(profile, "inference_sec", time.time() - t_inference)
        t_metric = time.time()
        t_mrr = time.time()
        mrr_test = evaluate_mrr(None, pos_test, neg_test)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "mrr_sec", time.time() - t_mrr)
        result = _single_rank_result(mrr_test)
        if compute_auc:
            t_auc = time.time()
            result.update(_auc_pair(pos_test, neg_test))
            _sync_if_profiled(profile, device)
            _profile_add(profile, "auc_sec", time.time() - t_auc)
        _sync_if_profiled(profile, device)
        _profile_add(profile, "testing_sec", time.time() - t_metric)
        return result
    pos_test = _predict_edge_scores(model, z, edges["pos_test_edge"], device, batch_size=batch_size)
    neg_edges = edges["neg_test_edge"]
    grouped = is_streaming_negative_edges(neg_edges) or (torch.is_tensor(neg_edges) and neg_edges.dim() == 3)
    if grouped:
        _sync_if_profiled(profile, device)
        _profile_add(profile, "inference_sec", time.time() - t_inference)
        t_grouped = time.time()
        if compute_auc:
            (stats, auc_test) = _stream_grouped_mrr(model, z, (pos_test,), neg_edges, device, batch_size=batch_size, compute_auc=True)
        else:
            (stats,) = _stream_grouped_mrr(model, z, (pos_test,), neg_edges, device, batch_size=batch_size)
        _sync_if_profiled(profile, device)
        grouped_sec = time.time() - t_grouped
        _profile_add(profile, "streamed_grouped_eval_sec", grouped_sec)
        _profile_add(profile, "testing_sec", grouped_sec)
        result = _single_rank_result(stats)
        if compute_auc:
            result.update(auc_test)
        return result
    neg_test = _predict_edge_scores(model, z, neg_edges, device, batch_size=batch_size)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "inference_sec", time.time() - t_inference)
    t_metric = time.time()
    result = _test_metrics_from_scores(pos_test, neg_test, compute_auc, profile=profile)
    _sync_if_profiled(profile, device)
    _profile_add(profile, "testing_sec", time.time() - t_metric)
    return result


def release_ogbl_evaluation(context):
    if context is not None:
        _clear_model_decode_cache(context.get("model"))
        context.clear()
