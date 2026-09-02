from collections import defaultdict
import fcntl
import hashlib
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import NamedTuple
import torch
from utils.metrics import StreamingAUCAP, evaluate_mrr, evaluate_auc
from .training import _make_graph_data, all_train
from .grouped_negatives import _canonical_json, _file_sha256, _fsync_directory, _unlock

K_LIST = [1, 3, 5, 10, 20, 50, 100]
HIT_KEYS = [f"mrr_hit{k}" for k in K_LIST]
_ENDPOINT_EVALUATION_OCCURRENCE_LIMIT = 65536
_ENDPOINT_EVALUATION_GROUP_LIMIT = 8192
_ENDPOINT_EVALUATION_UNION_LIMIT = 4194304
_ENDPOINT_EVALUATION_TRANSIENT_BYTE_LIMIT = 768 * 1024**2
_ENDPOINT_EVALUATION_CACHE_BYTE_LIMIT = 8 * 1024**3
_ENDPOINT_GLOBAL_SCORE_CHUNK_SIZE = 1048576
_ENDPOINT_GLOBAL_PLAN_RETAINED_BYTE_LIMIT = 6 * 1024**3
_ENDPOINT_GLOBAL_PLAN_BUILD_BYTE_LIMIT = 12 * 1024**3
_ENDPOINT_GLOBAL_PLAN_FORMAT = "endpoint-global-score-plan-v1"


def _sync_if_profiled(profile, device):
    if profile is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


def _finish_profile(profile, device, key, started):
    _sync_if_profiled(profile, device)
    if profile is not None:
        profile[key] = float(profile.get(key, 0.0)) + float(time.time() - started)


def _positive_setting(obj, name, default):
    return max(1, int(getattr(obj, name, default) or default))


def _empty_rank_stats(rank_list=None):
    if rank_list is None:
        rank_list = torch.empty(0, dtype=torch.float32)
    out = {"MRR": 0.0, "rank_list": rank_list}
    out.update({k: 0.0 for k in HIT_KEYS})
    return out


def _rank_stats_from_rank_list(rank_list):
    if rank_list.numel() == 0:
        return _empty_rank_stats(rank_list)
    rr = 1.0 / rank_list
    out = {"MRR": round(float(rr.mean().item()), 4), "rank_list": rank_list}
    out.update({f"mrr_hit{k}": round(float((rank_list <= k).float().mean().item()), 4) for k in K_LIST})
    return out


def _pack_rank_results(train_stats, val_stats, test_stats):
    out = {"MRR": (train_stats["MRR"], val_stats["MRR"], test_stats["MRR"])}
    out.update({f"Hits@{k}": (train_stats[f"mrr_hit{k}"], val_stats[f"mrr_hit{k}"], test_stats[f"mrr_hit{k}"]) for k in K_LIST})
    return out


def _single_rank_result(stats):
    result = {"MRR": stats["MRR"]}
    result.update({f"Hits@{k}": stats[f"mrr_hit{k}"] for k in K_LIST})
    return result


def _validation_only_result(stats, auc=None):
    result = {"MRR": (None, stats["MRR"], None)}
    result.update({f"Hits@{k}": (None, stats[f"mrr_hit{k}"], None) for k in K_LIST if f"mrr_hit{k}" in stats})
    if auc is not None:
        result["AUC"] = (None, auc["AUC"], None)
        result["AP"] = (None, auc["AP"], None)
    return result


def _auc_pair(pos_pred, neg_pred):
    device = pos_pred.device
    pred = torch.cat([pos_pred.view(-1), neg_pred.view(-1)])
    true = torch.cat(
        [
            torch.ones(pos_pred.numel(), dtype=torch.float32, device=device),
            torch.zeros(neg_pred.numel(), dtype=torch.float32, device=device),
        ]
    )
    return evaluate_auc(pred, true)


def _group_scores(pos_pred, neg_pred, split=None):
    num_positive = int(pos_pred.numel())
    if num_positive > 0 and neg_pred.numel() % num_positive == 0:
        return neg_pred.view(num_positive, -1)
    if split is not None:
        raise RuntimeError(f"{split} negatives are incompatible with {split.lower()} positives.")
    return neg_pred


def _test_metrics(pos_pred, neg_pred, rank_scores=None, rank_stats=None, include_auc=True):
    stats = rank_stats if rank_stats is not None else evaluate_mrr(None, pos_pred, rank_scores if rank_scores is not None else neg_pred)
    results = _single_rank_result(stats)
    if include_auc:
        results.update(_auc_pair(pos_pred, neg_pred))
    return results


def evaluate_mrr_only(pos_pred, neg_pred):
    pos = pos_pred.view(-1)
    comparison_dtype = torch.float64 if pos.dtype == torch.float64 or neg_pred.dtype == torch.float64 else torch.float32
    if neg_pred.dim() == 2:
        if neg_pred.size(0) != pos.numel():
            raise ValueError(f"Expected grouped negatives [Npos,K], got {tuple(neg_pred.shape)} for Npos={pos.numel()}.")
        positive = pos.to(comparison_dtype).view(-1, 1)
        negative = neg_pred.to(device=pos.device, dtype=comparison_dtype)
        optimistic = (negative >= positive).sum(dim=1)
        pessimistic = (negative > positive).sum(dim=1)
        rank = 0.5 * (optimistic + pessimistic) + 1
    else:
        positive = pos.detach().to(comparison_dtype)
        negative = neg_pred.detach().view(-1).to(device=positive.device, dtype=comparison_dtype)
        negative = torch.sort(negative).values
        lower = torch.searchsorted(negative, positive, right=False)
        upper = torch.searchsorted(negative, positive, right=True)
        optimistic = (negative.numel() - lower).to(torch.float32)
        pessimistic = (negative.numel() - upper).to(torch.float32)
        rank = 0.5 * (optimistic + pessimistic) + 1.0
    return {"MRR": round((1.0 / rank.to(torch.float32)).mean().item(), 4)}


def _pack_auc_results(pos_train_pred, pos_val_pred, pos_test_pred, neg_val_pred, neg_test_pred):
    auc_train = _auc_pair(pos_train_pred, neg_val_pred)
    auc_val = _auc_pair(pos_val_pred, neg_val_pred)
    auc_test = _auc_pair(pos_test_pred, neg_test_pred)
    return {"AUC": (auc_train["AUC"], auc_val["AUC"], auc_test["AUC"]), "AP": (auc_train["AP"], auc_val["AP"], auc_test["AP"])}


def _apply_reference_evaluation_transform(model, scores):
    transform = str(getattr(model, "reference_evaluation_transform", "identity")).strip().lower()
    if transform in {"", "identity", "none"}:
        return scores
    if transform == "sigmoid":
        return torch.sigmoid(scores.float())
    if transform == "sigmoid_cpu":
        original_device = scores.device
        return torch.sigmoid(scores.float().cpu()).to(original_device)
    raise ValueError(f"Unknown reference evaluation transform: {transform}")


@torch.no_grad()
def _test_edge_model(model, input_data, z, batch_size):
    device = z.device
    input_data = input_data.to(device=device, dtype=torch.long)
    if device.type == "cuda":
        if model.__class__.__name__ == "LinkPredictor":
            batch_size = max(int(batch_size), 262144)
        else:
            decode_batch_size = max(int(getattr(model, "decode_batch_size", 0) or 0), int(getattr(model, "evaluation_decode_batch_size", 0) or 0))
            batch_size = max(int(batch_size), decode_batch_size) if decode_batch_size > 0 else batch_size
    preds = []
    for start in range(0, input_data.size(0), batch_size):
        edge = input_data[start : start + batch_size].t().contiguous()
        scores = model.decode(z, edge).view(-1)
        preds.append(_apply_reference_evaluation_transform(model, scores))
    return torch.cat(preds, dim=0) if preds else torch.empty(0, dtype=torch.float32, device=device)


@torch.no_grad()
def _reference_grouped_edge_scores(model, z, positive_edges, grouped_negative_edges, fallback_batch_size):
    row_batch_size = int(getattr(model, "reference_evaluation_row_batch_size", 0) or 0)
    if row_batch_size <= 0:
        row_batch_size = max(1, int(fallback_batch_size))
    device = z.device
    positive_edges = positive_edges.to(device=device, dtype=torch.long)
    grouped_negative_edges = grouped_negative_edges.to(device=device, dtype=torch.long)
    num_positive = int(positive_edges.size(0))
    if num_positive <= 0:
        empty = torch.empty(0, dtype=torch.float32, device=device)
        return (empty, empty)
    num_negative = int(grouped_negative_edges.numel() // 2)
    negatives_per_positive = num_negative // num_positive
    grouped = grouped_negative_edges.reshape(num_positive, negatives_per_positive, 2)
    negative_layout = str(getattr(model, "reference_evaluation_negative_layout", "flat")).strip().lower()
    positive_parts = []
    negative_parts = []
    for start in range(0, num_positive, row_batch_size):
        end = min(start + row_batch_size, num_positive)
        positive_edge = positive_edges[start:end].t()
        negative_block = grouped[start:end]
        negative_edge = negative_block.permute(2, 0, 1)
        if negative_layout == "flat":
            negative_edge = negative_edge.reshape(2, -1)
        positive_score = model.decode(z, positive_edge).view(-1)
        negative_score = model.decode(z, negative_edge).view(-1)
        positive_parts.append(_apply_reference_evaluation_transform(model, positive_score))
        negative_parts.append(_apply_reference_evaluation_transform(model, negative_score))
    return (torch.cat(positive_parts), torch.cat(negative_parts))


def _ragged_scores_and_stats(model, z, positives, negatives, batch_size, flat_scores=None):
    ranks = torch.empty(positives.numel(), dtype=torch.float32, device=z.device)
    score_parts = []
    score_offset = 0
    for start, end, flat_edges, local_rowptr in negatives.iter_ragged_chunks():
        if flat_scores is None:
            scores = _test_edge_model(model, flat_edges, z, batch_size).view(-1)
            score_parts.append(scores)
        else:
            scores = flat_scores[score_offset : score_offset + flat_edges.size(0)]
            score_offset += flat_edges.size(0)
        lengths = (local_rowptr[1:] - local_rowptr[:-1]).to(device=z.device, dtype=torch.long)
        row_ids = torch.repeat_interleave(torch.arange(end - start, device=z.device), lengths)
        repeated_positive = positives[start:end][row_ids]
        optimistic = torch.zeros(end - start, dtype=torch.float32, device=z.device)
        pessimistic = torch.zeros_like(optimistic)
        optimistic.index_add_(0, row_ids, (scores >= repeated_positive).to(torch.float32))
        pessimistic.index_add_(0, row_ids, (scores > repeated_positive).to(torch.float32))
        ranks[start:end] = 0.5 * (optimistic + pessimistic) + 1.0
    if flat_scores is None:
        flat_scores = torch.cat(score_parts) if score_parts else torch.empty(0, dtype=torch.float32, device=z.device)
    return (_rank_stats_from_rank_list(ranks), flat_scores)


def _is_streamed_grouped_negative(value):
    return bool(getattr(value, "is_streamed_grouped_negative", False) or getattr(value, "is_streaming_grouped_negative", False))


def _is_ragged_negative(value):
    return bool(getattr(value, "is_ragged_negative", False))


class _EndpointEvaluationBatch(NamedTuple):
    side: int
    endpoints: torch.Tensor
    union_rowptr: torch.Tensor
    union_nodes: torch.Tensor
    occurrence_endpoint_index: torch.Tensor
    occurrence_row_ids: torch.Tensor
    candidate_local_indices: torch.Tensor
    union_occurrence_multiplicity: torch.Tensor


class _EndpointGlobalScorePlan(NamedTuple):
    canonical_undirected: bool
    num_nodes: int
    unique_edge_ids: torch.Tensor
    batch_unique_indices: tuple
    batch_union_counts: tuple
    unique_occurrence_multiplicity: torch.Tensor
    physical_union_edges: int
    logical_candidate_occurrences: int
    retained_bytes: int


def _pack_endpoint_evaluation_chunks(chunks, local_index_dtype):
    side = int(chunks[0].side)
    union_offset = 0
    group_offset = 0
    rowptr_parts = [torch.zeros(1, dtype=torch.long)]
    occurrence_endpoint_parts = []
    for chunk in chunks:
        rowptr_parts.append(chunk.union_rowptr[1:] + union_offset)
        occurrence_endpoint_parts.append(chunk.occurrence_endpoint_index + group_offset)
        union_offset += int(chunk.union_nodes.numel())
        group_offset += int(chunk.endpoints.numel())
    return _EndpointEvaluationBatch(
        side=side,
        endpoints=torch.cat([chunk.endpoints for chunk in chunks]).contiguous(),
        union_rowptr=torch.cat(rowptr_parts).contiguous(),
        union_nodes=torch.cat([chunk.union_nodes for chunk in chunks]).contiguous(),
        occurrence_endpoint_index=torch.cat(occurrence_endpoint_parts).to(dtype=torch.int32).contiguous(),
        occurrence_row_ids=torch.cat([chunk.occurrence_row_ids for chunk in chunks]).contiguous(),
        candidate_local_indices=torch.cat([chunk.candidate_local_indices for chunk in chunks]).to(dtype=local_index_dtype).contiguous(),
        union_occurrence_multiplicity=torch.cat([chunk.union_occurrence_multiplicity for chunk in chunks])
        .to(dtype=torch.int32)
        .contiguous(),
    )


def _endpoint_batch_transient_bytes(occurrences, groups, union_nodes, negatives_per_side):
    candidate_values = int(occurrences) * int(negatives_per_side)
    candidate_bytes = candidate_values * 32
    union_bytes = int(union_nodes) * 48
    metadata_bytes = int(occurrences) * 32 + int(groups) * 32
    return candidate_bytes + union_bytes + metadata_bytes


def _endpoint_compact_cache_bytes(negatives):
    shards = getattr(negatives, "shards", None)
    if not shards:
        return None
    candidates = 0
    union_nodes = 0
    groups = 0
    occurrences = 0
    for shard in shards:
        shard_occurrences = int(shard["occurrence_count"])
        local_bytes = 1 if shard["local_index_dtype"] == "uint8" else 2
        candidates += shard_occurrences * int(negatives.negatives_per_side) * local_bytes
        occurrences += shard_occurrences
        union_nodes += int(shard["union_node_count"])
        groups += int(shard["group_count"])
    return candidates + union_nodes * 12 + groups * 16 + occurrences * 12


def _endpoint_evaluation_cache_signature(negatives):
    return (
        str(getattr(negatives, "manifest_sha256", "")),
        int(negatives.num_rows),
        int(negatives.negatives_per_side),
        int(negatives.num_nodes),
        int(getattr(negatives, "occurrence_rows_per_shard", 0) or 0),
        int(getattr(negatives, "groups_per_shard", 0) or 0),
        int(getattr(negatives, "union_nodes_per_shard", 0) or 0),
        int(getattr(negatives, "endpoint_evaluation_occurrence_limit", _ENDPOINT_EVALUATION_OCCURRENCE_LIMIT)),
        int(getattr(negatives, "endpoint_evaluation_group_limit", _ENDPOINT_EVALUATION_GROUP_LIMIT)),
        int(getattr(negatives, "endpoint_evaluation_union_limit", _ENDPOINT_EVALUATION_UNION_LIMIT)),
        int(getattr(negatives, "endpoint_evaluation_transient_byte_limit", _ENDPOINT_EVALUATION_TRANSIENT_BYTE_LIMIT)),
        int(getattr(negatives, "endpoint_evaluation_cache_byte_limit", _ENDPOINT_EVALUATION_CACHE_BYTE_LIMIT)),
    )


def _iter_endpoint_evaluation_batches(negatives):
    signature = _endpoint_evaluation_cache_signature(negatives)
    cached = getattr(negatives, "_endpoint_evaluation_batch_cache_v1", None)
    if isinstance(cached, dict) and cached.get("signature") == signature:
        yield from cached["batches"]
        return
    occurrence_limit = _positive_setting(negatives, "endpoint_evaluation_occurrence_limit", _ENDPOINT_EVALUATION_OCCURRENCE_LIMIT)
    group_limit = _positive_setting(negatives, "endpoint_evaluation_group_limit", _ENDPOINT_EVALUATION_GROUP_LIMIT)
    union_limit = _positive_setting(negatives, "endpoint_evaluation_union_limit", _ENDPOINT_EVALUATION_UNION_LIMIT)
    transient_byte_limit = _positive_setting(
        negatives, "endpoint_evaluation_transient_byte_limit", _ENDPOINT_EVALUATION_TRANSIENT_BYTE_LIMIT
    )
    cache_byte_limit = max(0, int(getattr(negatives, "endpoint_evaluation_cache_byte_limit", _ENDPOINT_EVALUATION_CACHE_BYTE_LIMIT)))
    estimated_cache_bytes = _endpoint_compact_cache_bytes(negatives)
    retain_cache = cache_byte_limit > 0 and (estimated_cache_bytes is None or estimated_cache_bytes <= cache_byte_limit)
    def new_buffer():
        return {"chunks": [], "occurrences": 0, "groups": 0, "union_nodes": 0}

    buffers = defaultdict(new_buffer)
    packed_batches = [] if retain_cache else None

    def flush(key):
        state = buffers[key]
        if not state["chunks"]:
            return None
        dtype = torch.uint8 if key[1] == 8 else torch.uint16
        batch = _pack_endpoint_evaluation_chunks(state["chunks"], dtype)
        buffers[key] = new_buffer()
        if packed_batches is not None:
            packed_batches.append(batch)
        return batch

    for chunk in negatives.iter_endpoint_group_chunks():
        group_widths = chunk.union_rowptr[1:] - chunk.union_rowptr[:-1]
        max_group_width = int(group_widths.max().item())
        width_bits = 8 if max_group_width <= 256 else 16
        key = (int(chunk.side), width_bits)
        state = buffers[key]
        chunk_occurrences = int(chunk.occurrence_row_ids.numel())
        chunk_groups = int(chunk.endpoints.numel())
        chunk_union_nodes = int(chunk.union_nodes.numel())
        chunk_transient_bytes = _endpoint_batch_transient_bytes(
            chunk_occurrences, chunk_groups, chunk_union_nodes, negatives.negatives_per_side
        )
        if chunk_transient_bytes > transient_byte_limit:
            raise RuntimeError(
                f"One physical endpoint shard exceeds the configured evaluation transient-memory ceiling: estimated={chunk_transient_bytes} bytes, limit={transient_byte_limit} bytes."
            )
        exceeds_limit = state["chunks"] and (
            state["occurrences"] + chunk_occurrences > occurrence_limit
            or state["groups"] + chunk_groups > group_limit
            or state["union_nodes"] + chunk_union_nodes > union_limit
            or (
                _endpoint_batch_transient_bytes(
                    state["occurrences"] + chunk_occurrences,
                    state["groups"] + chunk_groups,
                    state["union_nodes"] + chunk_union_nodes,
                    negatives.negatives_per_side,
                )
                > transient_byte_limit
            )
        )
        if exceeds_limit:
            yield flush(key)
            state = buffers[key]
        state["chunks"].append(chunk)
        state["occurrences"] += chunk_occurrences
        state["groups"] += chunk_groups
        state["union_nodes"] += chunk_union_nodes
    for key in sorted(buffers):
        batch = flush(key)
        if batch is not None:
            yield batch
    if packed_batches is not None:
        negatives._endpoint_evaluation_batch_cache_v1 = {
            "signature": signature,
            "batches": tuple(packed_batches),
            "estimated_bytes": estimated_cache_bytes,
        }


def _decoder_explicitly_declares(model, attribute):
    candidate = getattr(model, "module", model)
    return getattr(candidate, attribute, None) is True


def _endpoint_union_edge_ids(batch, num_nodes, canonical_undirected):
    num_nodes = int(num_nodes)
    if num_nodes * num_nodes > torch.iinfo(torch.int64).max:
        raise OverflowError("Endpoint edge IDs do not fit in signed int64 for this graph.")
    union_lengths = batch.union_rowptr[1:] - batch.union_rowptr[:-1]
    union_endpoints = torch.repeat_interleave(batch.endpoints.to(dtype=torch.long), union_lengths)
    union_nodes = batch.union_nodes.to(dtype=torch.long)
    side = int(batch.side)
    if side == 0:
        (src, dst) = (union_endpoints, union_nodes)
    else:
        (src, dst) = (union_nodes, union_endpoints)
    if canonical_undirected:
        low = torch.minimum(src, dst)
        high = torch.maximum(src, dst)
        (src, dst) = (low, high)
    return (src * num_nodes + dst).contiguous()


def _endpoint_global_plan_cache_signature(negatives):
    return (
        _ENDPOINT_GLOBAL_PLAN_FORMAT,
        _endpoint_evaluation_cache_signature(negatives),
        int(getattr(negatives, "endpoint_global_plan_retained_byte_limit", _ENDPOINT_GLOBAL_PLAN_RETAINED_BYTE_LIMIT)),
        int(getattr(negatives, "endpoint_global_plan_build_byte_limit", _ENDPOINT_GLOBAL_PLAN_BUILD_BYTE_LIMIT)),
    )


def _endpoint_manifest_union_count(negatives):
    shards = getattr(negatives, "shards", None)
    if not shards:
        return None
    return sum((int(shard["union_node_count"]) for shard in shards))


def _endpoint_plan_retained_bytes(unique_edge_ids, unique_multiplicity, batch_unique_indices):
    return sum(
        int(tensor.numel()) * tensor.element_size()
        for tensor in (unique_edge_ids, unique_multiplicity, *batch_unique_indices)
    )


def _cache_endpoint_global_plan(negatives, signature, key, value):
    cached = getattr(negatives, "_endpoint_global_score_plan_cache_v1", None)
    if not isinstance(cached, dict) or cached.get("signature") != signature:
        cached = {"signature": signature, "plans": {}}
    cached["plans"][key] = value
    negatives._endpoint_global_score_plan_cache_v1 = cached


def _build_endpoint_global_score_plan_in_memory(negatives, *, canonical_undirected):
    signature = _endpoint_global_plan_cache_signature(negatives)
    key = "undirected" if canonical_undirected else "directed"
    cached = getattr(negatives, "_endpoint_global_score_plan_cache_v1", None)
    if isinstance(cached, dict) and cached.get("signature") == signature and (key in cached.get("plans", {})):
        return cached["plans"][key]
    retained_limit = max(0, int(signature[-2]))
    build_limit = max(0, int(signature[-1]))
    expected_physical = _endpoint_manifest_union_count(negatives)
    if expected_physical is not None:
        conservative_build_bytes = int(expected_physical) * 72
        conservative_retained_bytes = int(expected_physical) * 24
        if (
            build_limit <= 0
            or conservative_build_bytes > build_limit
            or retained_limit <= 0
            or (conservative_retained_bytes > retained_limit)
        ):
            _cache_endpoint_global_plan(negatives, signature, key, None)
            return None
    batches = []
    batch_union_counts = []
    id_parts = [] if expected_physical is None else None
    all_edge_ids = torch.empty(int(expected_physical), dtype=torch.long) if expected_physical is not None else None
    offset = 0
    for batch in _iter_endpoint_evaluation_batches(negatives):
        edge_ids = _endpoint_union_edge_ids(batch, negatives.num_nodes, canonical_undirected)
        count = int(edge_ids.numel())
        batches.append(batch)
        batch_union_counts.append(count)
        if all_edge_ids is None:
            id_parts.append(edge_ids)
            offset += count
            if build_limit <= 0 or offset * 72 > build_limit or retained_limit <= 0 or (offset * 24 > retained_limit):
                _cache_endpoint_global_plan(negatives, signature, key, None)
                return None
        else:
            end = offset + count
            all_edge_ids[offset:end].copy_(edge_ids)
            offset = end
    if all_edge_ids is None:
        all_edge_ids = torch.cat(id_parts) if id_parts else torch.empty(0, dtype=torch.long)
    if all_edge_ids.numel():
        (unique_edge_ids, inverse) = torch.unique(all_edge_ids, sorted=True, return_inverse=True)
        unique_edge_ids = unique_edge_ids.contiguous()
    else:
        unique_edge_ids = torch.empty(0, dtype=torch.long)
        inverse = torch.empty(0, dtype=torch.long)
    del all_edge_ids, id_parts
    inverse_dtype = torch.int32 if int(unique_edge_ids.numel()) <= torch.iinfo(torch.int32).max else torch.int64
    unique_multiplicity = torch.zeros(int(unique_edge_ids.numel()), dtype=torch.long)
    batch_unique_indices = []
    logical_candidate_occurrences = 0
    offset = 0
    for batch, count in zip(batches, batch_union_counts):
        end = offset + int(count)
        mapping_source = inverse[offset:end]
        if inverse_dtype == torch.int32:
            mapping = mapping_source.to(dtype=torch.int32).contiguous()
        else:
            mapping = mapping_source.clone().contiguous()
        multiplicity = batch.union_occurrence_multiplicity.to(dtype=torch.long).view(-1)
        if mapping.numel():
            unique_multiplicity.index_add_(0, mapping.to(dtype=torch.long), multiplicity)
        logical_candidate_occurrences += int(multiplicity.sum().item())
        batch_unique_indices.append(mapping)
        offset = end
    del inverse, batches
    retained_bytes = _endpoint_plan_retained_bytes(unique_edge_ids, unique_multiplicity, batch_unique_indices)
    if retained_bytes > retained_limit:
        _cache_endpoint_global_plan(negatives, signature, key, None)
        return None
    plan = _EndpointGlobalScorePlan(
        canonical_undirected=canonical_undirected,
        num_nodes=negatives.num_nodes,
        unique_edge_ids=unique_edge_ids,
        batch_unique_indices=tuple(batch_unique_indices),
        batch_union_counts=tuple(batch_union_counts),
        unique_occurrence_multiplicity=unique_multiplicity,
        physical_union_edges=offset,
        logical_candidate_occurrences=logical_candidate_occurrences,
        retained_bytes=retained_bytes,
    )
    _cache_endpoint_global_plan(negatives, signature, key, plan)
    return plan


def _endpoint_global_plan_sidecar_path(negatives, canonical_undirected):
    manifest_path = getattr(negatives, "manifest_path", None)
    if not manifest_path:
        return None
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        return None
    orientation = "undirected" if canonical_undirected else "directed"
    return Path(f"{manifest_path}.{_ENDPOINT_GLOBAL_PLAN_FORMAT}.{orientation}.pt")


def _endpoint_global_plan_binding(negatives, signature, canonical_undirected):
    manifest_path = Path(negatives.manifest_path).expanduser().resolve()
    return {
        "format": _ENDPOINT_GLOBAL_PLAN_FORMAT,
        "manifest_name": manifest_path.name,
        "manifest_file_sha256": _file_sha256(manifest_path),
        "manifest_sha256": str(negatives.manifest_sha256),
        "evaluation_signature_sha256": hashlib.sha256(_canonical_json(signature)).hexdigest(),
        "canonical_undirected": bool(canonical_undirected),
        "num_nodes": int(negatives.num_nodes),
        "num_rows": int(negatives.num_rows),
        "negatives_per_side": int(negatives.negatives_per_side),
        "negatives_per_row": int(negatives.negatives_per_row),
    }


def _update_digest_with_tensor(digest, name, tensor):
    tensor = torch.as_tensor(tensor).detach().cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    header = _canonical_json({"name": str(name), "dtype": str(tensor.dtype), "shape": list(tensor.shape)})
    digest.update(len(header).to_bytes(8, "little"))
    digest.update(header)
    byte_view = memoryview(tensor.numpy()).cast("B")
    block_size = 64 * 1024**2
    for start in range(0, len(byte_view), block_size):
        digest.update(byte_view[start : start + block_size])


def _endpoint_global_plan_digest(plan):
    digest = hashlib.sha256()
    metadata = _canonical_json(
        {
            "canonical_undirected": bool(plan.canonical_undirected),
            "num_nodes": int(plan.num_nodes),
            "physical_union_edges": int(plan.physical_union_edges),
            "logical_candidate_occurrences": int(plan.logical_candidate_occurrences),
            "batch_union_counts": list(plan.batch_union_counts),
        }
    )
    digest.update(metadata)
    _update_digest_with_tensor(digest, "unique_edge_ids", plan.unique_edge_ids)
    _update_digest_with_tensor(digest, "unique_occurrence_multiplicity", plan.unique_occurrence_multiplicity)
    for index, mapping in enumerate(plan.batch_unique_indices):
        _update_digest_with_tensor(digest, f"batch_unique_indices/{index}", mapping)
    return digest.hexdigest()


def _save_endpoint_global_plan_sidecar(path, binding, plan):
    payload = {
        "binding": binding,
        "plan_digest_sha256": _endpoint_global_plan_digest(plan),
        "canonical_undirected": bool(plan.canonical_undirected),
        "num_nodes": int(plan.num_nodes),
        "unique_edge_ids": plan.unique_edge_ids,
        "batch_unique_indices": list(plan.batch_unique_indices),
        "batch_union_counts": list(plan.batch_union_counts),
        "unique_occurrence_multiplicity": plan.unique_occurrence_multiplicity,
        "physical_union_edges": int(plan.physical_union_edges),
        "logical_candidate_occurrences": int(plan.logical_candidate_occurrences),
        "retained_bytes": int(plan.retained_bytes),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = tempfile.NamedTemporaryFile(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False)
    temporary_path = Path(temporary.name)
    temporary.close()
    try:
        torch.save(payload, temporary_path)
        with open(temporary_path, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        _fsync_directory(path.parent)
    finally:
        try:
            temporary_path.unlink(missing_ok=True)
        except OSError:
            pass


def _is_cpu_vector(value, dtypes, length=None):
    dtypes = dtypes if isinstance(dtypes, tuple) else (dtypes,)
    return bool(
        torch.is_tensor(value)
        and value.device.type == "cpu"
        and value.dtype in dtypes
        and value.ndim == 1
        and (length is None or int(value.numel()) == int(length))
    )


def _require_valid_sidecar(condition):
    if not condition:
        raise ValueError


def _load_endpoint_global_plan_sidecar(path, binding, negatives):
    if not path.is_file():
        return None
    try:
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        except TypeError:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        _require_valid_sidecar(isinstance(payload, dict) and payload.get("binding") == binding)
        payload_num_nodes = int(payload["num_nodes"])
        _require_valid_sidecar(payload_num_nodes == int(binding["num_nodes"]) == int(negatives.num_nodes))
        unique_edge_ids = payload["unique_edge_ids"]
        unique_multiplicity = payload["unique_occurrence_multiplicity"]
        batch_unique_indices = payload["batch_unique_indices"]
        batch_union_counts = tuple((int(value) for value in payload["batch_union_counts"]))
        _require_valid_sidecar(
            _is_cpu_vector(unique_edge_ids, torch.long)
            and _is_cpu_vector(unique_multiplicity, torch.long, unique_edge_ids.numel())
            and isinstance(batch_unique_indices, (list, tuple))
            and len(batch_unique_indices) == len(batch_union_counts)
        )
        if unique_edge_ids.numel():
            _require_valid_sidecar(0 <= int(unique_edge_ids[0]) and int(unique_edge_ids[-1]) < int(negatives.num_nodes) ** 2)
            _require_valid_sidecar(bool((unique_edge_ids[1:] > unique_edge_ids[:-1]).all()))
        physical_union_edges = int(payload["physical_union_edges"])
        _require_valid_sidecar(
            physical_union_edges == sum(batch_union_counts) == int(_endpoint_manifest_union_count(negatives) or -1)
        )
        for mapping, count in zip(batch_unique_indices, batch_union_counts):
            _require_valid_sidecar(_is_cpu_vector(mapping, (torch.int32, torch.int64), count))
            if mapping.numel():
                _require_valid_sidecar(0 <= int(mapping.min()) and int(mapping.max()) < int(unique_edge_ids.numel()))
        logical_candidate_occurrences = int(payload["logical_candidate_occurrences"])
        expected_logical = int(negatives.num_rows) * int(negatives.negatives_per_row)
        _require_valid_sidecar(
            logical_candidate_occurrences == expected_logical
            and not bool((unique_multiplicity < 0).any())
            and int(unique_multiplicity.sum()) == expected_logical
        )
        retained_bytes = _endpoint_plan_retained_bytes(unique_edge_ids, unique_multiplicity, batch_unique_indices)
        retained_limit = max(0, int(getattr(negatives, "endpoint_global_plan_retained_byte_limit", _ENDPOINT_GLOBAL_PLAN_RETAINED_BYTE_LIMIT)))
        _require_valid_sidecar(
            retained_bytes == int(payload["retained_bytes"]) and retained_bytes <= retained_limit
        )
        plan = _EndpointGlobalScorePlan(
            canonical_undirected=payload["canonical_undirected"],
            num_nodes=payload_num_nodes,
            unique_edge_ids=unique_edge_ids,
            batch_unique_indices=tuple(batch_unique_indices),
            batch_union_counts=batch_union_counts,
            unique_occurrence_multiplicity=unique_multiplicity,
            physical_union_edges=physical_union_edges,
            logical_candidate_occurrences=logical_candidate_occurrences,
            retained_bytes=retained_bytes,
        )
        _require_valid_sidecar(plan.canonical_undirected == bool(binding["canonical_undirected"]))
        _require_valid_sidecar(_endpoint_global_plan_digest(plan) == str(payload["plan_digest_sha256"]))
        return plan
    except (EOFError, KeyError, OSError, OverflowError, pickle.UnpicklingError, RuntimeError, TypeError, ValueError):
        return None


def _get_or_build_endpoint_global_score_plan(negatives, *, canonical_undirected):
    signature = _endpoint_global_plan_cache_signature(negatives)
    key = "undirected" if canonical_undirected else "directed"
    cached = getattr(negatives, "_endpoint_global_score_plan_cache_v1", None)
    if isinstance(cached, dict) and cached.get("signature") == signature and (key in cached.get("plans", {})):
        return cached["plans"][key]
    sidecar_path = _endpoint_global_plan_sidecar_path(negatives, canonical_undirected)
    if sidecar_path is None:
        return _build_endpoint_global_score_plan_in_memory(negatives, canonical_undirected=canonical_undirected)
    binding = _endpoint_global_plan_binding(negatives, signature, canonical_undirected)
    lock_path = Path(f"{sidecar_path}.lock")
    try:
        lock_handle = open(lock_path, "a+b")
    except OSError as exc:
        print(f"Endpoint global score-plan lock unavailable; using an in-memory plan only: {exc}", flush=True)
        return _build_endpoint_global_score_plan_in_memory(negatives, canonical_undirected=canonical_undirected)
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        loaded = _load_endpoint_global_plan_sidecar(sidecar_path, binding, negatives)
        if loaded is not None:
            _cache_endpoint_global_plan(negatives, signature, key, loaded)
            print(
                f"Loaded manifest-bound endpoint global score plan: {sidecar_path} ({loaded.physical_union_edges} physical -> {loaded.unique_edge_ids.numel()} unique edges)",
                flush=True,
            )
            return loaded
        plan = _build_endpoint_global_score_plan_in_memory(negatives, canonical_undirected=canonical_undirected)
        if plan is None:
            return None
        try:
            _save_endpoint_global_plan_sidecar(sidecar_path, binding, plan)
            print(
                f"Cached manifest-bound endpoint global score plan: {sidecar_path} ({plan.physical_union_edges} physical -> {plan.unique_edge_ids.numel()} unique edges)",
                flush=True,
            )
        except (OSError, RuntimeError) as exc:
            print(f"Endpoint global score plan could not be persisted; this process will still reuse it in memory: {exc}", flush=True)
        return plan
    finally:
        _unlock(lock_handle)


@torch.no_grad()
def _score_endpoint_global_plan(model, z, negatives, plan, batch_size):
    chunk_size = _positive_setting(negatives, "endpoint_global_score_chunk_size", _ENDPOINT_GLOBAL_SCORE_CHUNK_SIZE)
    unique_ids = plan.unique_edge_ids
    total = int(unique_ids.numel())
    output = None
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        ids = unique_ids[start:end]
        edges = torch.stack([torch.div(ids, plan.num_nodes, rounding_mode="floor"), ids % plan.num_nodes], dim=1)
        scores = _test_edge_model(model, edges, z, batch_size).view(-1).detach().cpu()
        if output is None:
            output = torch.empty(total, dtype=scores.dtype)
        output[start:end].copy_(scores)
    if output is None:
        output = torch.empty(0, dtype=torch.float32)
    return (output, chunk_size)


def _iter_endpoint_global_plan_batches(negatives, plan):
    batch_index = 0
    for batch in _iter_endpoint_evaluation_batches(negatives):
        if batch_index >= len(plan.batch_unique_indices):
            raise ValueError("Endpoint evaluation batches no longer match the global plan.")
        expected = int(plan.batch_union_counts[batch_index])
        if int(batch.union_nodes.numel()) != expected:
            raise ValueError("Endpoint evaluation batch shape no longer matches the global plan.")
        yield (batch, plan.batch_unique_indices[batch_index])
        batch_index += 1
    if batch_index != len(plan.batch_unique_indices):
        raise ValueError("Endpoint evaluation produced fewer batches than the global plan.")


def _update_endpoint_global_plan_auc(accumulators, unique_scores_cpu, plan, device, chunk_size):
    if not accumulators:
        return
    total = int(unique_scores_cpu.numel())
    chunk_size = int(chunk_size)
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        scores = unique_scores_cpu[start:end].to(device=device, non_blocking=device.type == "cuda")
        multiplicity = plan.unique_occurrence_multiplicity[start:end].to(
            device=device, dtype=torch.long, non_blocking=device.type == "cuda"
        )
        for accumulator in accumulators.values():
            accumulator.update_weighted(scores, multiplicity)


@torch.no_grad()
def _score_positive_rows_bounded(model, z, positive_edges, batch_size, *, row_chunk_size=65536, keep_device=False):
    positive_edges = positive_edges.to(dtype=torch.long).cpu()
    parts = []
    for start in range(0, int(positive_edges.size(0)), int(row_chunk_size)):
        scores = _test_edge_model(model, positive_edges[start : start + int(row_chunk_size)], z, batch_size).view(-1)
        scores = scores.detach()
        parts.append(scores if keep_device else scores.cpu())
    return torch.cat(parts) if parts else torch.empty(0, dtype=torch.float32, device=z.device if keep_device else torch.device("cpu"))


def _online_grouped_rank_stats(ge_counts, gt_counts, include_hits):
    total = int(ge_counts.numel())
    if total == 0:
        return _empty_rank_stats(None)
    reciprocal_sum = 0.0
    hit_totals = {k: 0 for k in K_LIST}
    chunk_size = 1048576
    for start in range(0, total, chunk_size):
        end = min(start + chunk_size, total)
        rank = 0.5 * (ge_counts[start:end].to(torch.float32) + gt_counts[start:end].to(torch.float32)) + 1.0
        reciprocal_sum += float((1.0 / rank).to(torch.float64).sum().item())
        if include_hits:
            for k in K_LIST:
                hit_totals[k] += int((rank <= k).sum().item())
    result = {"MRR": round(reciprocal_sum / total, 4), "rank_list": None}
    if include_hits:
        result.update({f"mrr_hit{k}": round(hit_totals[k] / total, 4) for k in K_LIST})
    return result


def _grouped_metric_output(names, ge_counts, gt_counts, auc_accumulators, include_auc, include_hits):
    return {
        name: {
            "rank": _online_grouped_rank_stats(ge_counts[name].cpu(), gt_counts[name].cpu(), include_hits),
            "auc": auc_accumulators[name].compute() if include_auc else None,
        }
        for name in names
    }


@torch.no_grad()
def _endpoint_grouped_metrics_from_scores(model, z, positive_scores, negatives, batch_size, *, include_auc, include_hits):
    names = tuple(positive_scores)
    num_rows = int(negatives.num_rows)
    for name in names:
        if int(positive_scores[name].numel()) != num_rows:
            raise ValueError(f"Positive score set {name!r} does not match grouped rows.")
    score_device = z.device
    positive_scores_device = {
        name: positive_scores[name].to(device=score_device, non_blocking=score_device.type == "cuda") for name in names
    }
    ge_counts = {name: torch.zeros(num_rows, dtype=torch.int32, device=score_device) for name in names}
    gt_counts = {name: torch.zeros(num_rows, dtype=torch.int32, device=score_device) for name in names}
    candidate_counts = torch.zeros(num_rows, dtype=torch.int32, device=score_device)
    auc_accumulators = {name: StreamingAUCAP(positive_scores_device[name]) for name in names} if include_auc else {}
    global_plan = (
        _get_or_build_endpoint_global_score_plan(
            negatives, canonical_undirected=_decoder_explicitly_declares(model, "decode_is_symmetric")
        )
        if _decoder_explicitly_declares(model, "decode_is_dedup_safe")
        else None
    )
    unique_scores_cpu = global_score_chunk_size = None
    if global_plan is not None:
        (unique_scores_cpu, global_score_chunk_size) = _score_endpoint_global_plan(model, z, negatives, global_plan, batch_size)
    decoded_union_edges = int(global_plan.unique_edge_ids.numel()) if global_plan is not None else 0
    logical_candidate_occurrences = 0
    evaluation_batches = 0
    batch_iterator = (
        _iter_endpoint_global_plan_batches(negatives, global_plan)
        if global_plan is not None
        else ((batch, None) for batch in _iter_endpoint_evaluation_batches(negatives))
    )
    for chunk, global_inverse in batch_iterator:
        evaluation_batches += 1
        if global_plan is None:
            union_lengths = chunk.union_rowptr[1:] - chunk.union_rowptr[:-1]
            union_endpoints = torch.repeat_interleave(chunk.endpoints, union_lengths)
            if int(chunk.side) == 0:
                union_edges = torch.stack([union_endpoints, chunk.union_nodes], dim=1)
            else:
                union_edges = torch.stack([chunk.union_nodes, union_endpoints], dim=1)
            union_scores_device = _test_edge_model(model, union_edges, z, batch_size).view(-1).detach()
            decoded_union_edges += int(union_scores_device.numel())
        else:
            union_scores_cpu = unique_scores_cpu.index_select(0, global_inverse.to(dtype=torch.long))
            union_scores_device = union_scores_cpu.to(device=score_device, non_blocking=score_device.type == "cuda")
        union_offsets_cpu = chunk.union_rowptr[chunk.occurrence_endpoint_index.to(dtype=torch.long)]
        compact_local_indices = chunk.candidate_local_indices.to(
            device=union_scores_device.device, non_blocking=union_scores_device.device.type == "cuda"
        )
        global_local_indices = compact_local_indices.to(dtype=torch.long)
        global_local_indices.add_(
            union_offsets_cpu.to(
                device=union_scores_device.device, dtype=torch.long, non_blocking=union_scores_device.device.type == "cuda"
            ).view(-1, 1)
        )
        occurrence_scores = union_scores_device[global_local_indices]
        row_ids = chunk.occurrence_row_ids.to(device=score_device, dtype=torch.long, non_blocking=score_device.type == "cuda")
        logical_candidate_occurrences += int(occurrence_scores.numel())
        side_counts = torch.full((int(row_ids.numel()),), int(negatives.negatives_per_side), dtype=torch.int32, device=score_device)
        candidate_counts.index_add_(0, row_ids, side_counts)
        selected_positives = (
            torch.stack([positive_scores_device[name][row_ids] for name in names], dim=0).to(dtype=occurrence_scores.dtype).unsqueeze(-1)
        )
        occurrence_scores_batched = occurrence_scores.unsqueeze(0)
        side_rank_counts = torch.stack(
            [(occurrence_scores_batched >= selected_positives).sum(dim=2), (occurrence_scores_batched > selected_positives).sum(dim=2)],
            dim=0,
        ).to(dtype=torch.int32)
        for name_index, name in enumerate(names):
            ge_counts[name].index_add_(0, row_ids, side_rank_counts[0, name_index])
            gt_counts[name].index_add_(0, row_ids, side_rank_counts[1, name_index])
        if include_auc and global_plan is None:
            for accumulator in auc_accumulators.values():
                accumulator.update_weighted(union_scores_device, chunk.union_occurrence_multiplicity)
    if global_plan is not None:
        if logical_candidate_occurrences != global_plan.logical_candidate_occurrences:
            raise ValueError("Endpoint global plan did not reconstruct every logical candidate occurrence.")
        if include_auc:
            _update_endpoint_global_plan_auc(auc_accumulators, unique_scores_cpu, global_plan, score_device, global_score_chunk_size)
    expected = int(negatives.negatives_per_row)
    if not bool(candidate_counts.eq(expected).all()):
        raise ValueError("Endpoint-grouped evaluation did not consume exactly one complete left and right candidate side for every positive row.")
    output = _grouped_metric_output(names, ge_counts, gt_counts, auc_accumulators, include_auc, include_hits)
    output["reuse"] = {
        "decoded_union_edges": decoded_union_edges,
        "logical_candidate_occurrences": logical_candidate_occurrences,
        "decode_reuse_ratio": logical_candidate_occurrences / decoded_union_edges if decoded_union_edges else 0.0,
        "evaluation_batches": evaluation_batches,
        "physical_shards": len(getattr(negatives, "shards", ())),
        "physical_union_edges": int(global_plan.physical_union_edges) if global_plan is not None else decoded_union_edges,
        "global_edge_plan": global_plan is not None,
        "canonical_undirected": bool(global_plan is not None and global_plan.canonical_undirected),
        "global_plan_retained_bytes": int(global_plan.retained_bytes) if global_plan is not None else 0,
    }
    return output


@torch.no_grad()
def _row_sharded_grouped_metrics_from_scores(model, z, positive_scores, negatives, batch_size, *, include_auc, include_hits):
    names = tuple(positive_scores)
    num_rows = int(negatives.num_rows)
    ge_counts = {name: torch.zeros(num_rows, dtype=torch.int32) for name in names}
    gt_counts = {name: torch.zeros(num_rows, dtype=torch.int32) for name in names}
    auc_accumulators = {name: StreamingAUCAP(positive_scores[name]) for name in names} if include_auc else {}
    decoded_unique_edges = 0
    logical_candidate_occurrences = 0
    for start, end, grouped_edges in negatives.iter_grouped_chunks():
        flat = grouped_edges.reshape(-1, 2)
        edge_ids = flat[:, 0] * int(negatives.num_nodes) + flat[:, 1]
        (unique_ids, inverse) = torch.unique(edge_ids, sorted=True, return_inverse=True)
        unique_edges = torch.stack(
            [torch.div(unique_ids, int(negatives.num_nodes), rounding_mode="floor"), unique_ids % int(negatives.num_nodes)], dim=1
        )
        unique_scores = _test_edge_model(model, unique_edges, z, batch_size).view(-1).detach().cpu()
        decoded_unique_edges += int(unique_scores.numel())
        grouped_scores = unique_scores[inverse].view(end - start, int(negatives.negatives_per_row))
        logical_candidate_occurrences += int(grouped_scores.numel())
        for name in names:
            positive = positive_scores[name][start:end].view(-1, 1)
            ge_counts[name][start:end] = (grouped_scores >= positive).sum(dim=1).to(torch.int32)
            gt_counts[name][start:end] = (grouped_scores > positive).sum(dim=1).to(torch.int32)
        if include_auc:
            multiplicity = torch.bincount(inverse, minlength=int(unique_scores.numel()))
            for accumulator in auc_accumulators.values():
                accumulator.update_weighted(unique_scores, multiplicity)
    output = _grouped_metric_output(names, ge_counts, gt_counts, auc_accumulators, include_auc, include_hits)
    output["reuse"] = {
        "decoded_union_edges": decoded_unique_edges,
        "logical_candidate_occurrences": logical_candidate_occurrences,
        "decode_reuse_ratio": logical_candidate_occurrences / decoded_unique_edges if decoded_unique_edges else 0.0,
    }
    return output


@torch.no_grad()
def _streamed_grouped_split_metrics(model, z, positive_edge_sets, negatives, batch_size, *, include_auc=True, include_hits=True):
    endpoint_grouped = bool(getattr(negatives, "is_endpoint_grouped_negative", False))
    positive_scores = {
        name: _score_positive_rows_bounded(model, z, edges, batch_size, keep_device=endpoint_grouped)
        for (name, edges) in positive_edge_sets.items()
    }
    return _streamed_grouped_metrics_from_positive_scores(
        model, z, positive_scores, negatives, batch_size, include_auc=include_auc, include_hits=include_hits
    )


@torch.no_grad()
def _streamed_grouped_metrics_from_positive_scores(model, z, positive_scores, negatives, batch_size, *, include_auc=True, include_hits=True):
    if not _decoder_explicitly_declares(model, "decode_is_dedup_safe"):
        raise RuntimeError(
            "Endpoint/streamed grouped evaluation reuses one decoded score for repeated directed edges, but this model does not explicitly declare decode_is_dedup_safe=True. Refusing a potentially batch-dependent or stochastic candidate evaluation."
        )
    endpoint_grouped = bool(getattr(negatives, "is_endpoint_grouped_negative", False))
    if endpoint_grouped:
        normalized_scores = {
            name: torch.as_tensor(scores).detach().view(-1).to(device=z.device, non_blocking=z.device.type == "cuda")
            for (name, scores) in positive_scores.items()
        }
        return _endpoint_grouped_metrics_from_scores(
            model, z, normalized_scores, negatives, batch_size, include_auc=include_auc, include_hits=include_hits
        )
    normalized_scores = {name: torch.as_tensor(scores).detach().view(-1).cpu() for (name, scores) in positive_scores.items()}
    return _row_sharded_grouped_metrics_from_scores(
        model, z, normalized_scores, negatives, batch_size, include_auc=include_auc, include_hits=include_hits
    )


def get_metric_score(
    _evaluator_hit,
    _evaluator_mrr,
    pos_train_pred,
    pos_val_pred,
    neg_val_pred,
    pos_test_pred,
    neg_test_pred,
    *,
    include_auc=True,
):
    rank_stats = [evaluate_mrr(None, pos, neg) for pos, neg in ((pos_train_pred, neg_val_pred), (pos_val_pred, neg_val_pred), (pos_test_pred, neg_test_pred))]
    result = _pack_rank_results(*rank_stats)
    if include_auc:
        result.update(_pack_auc_results(pos_train_pred, pos_val_pred, pos_test_pred, neg_val_pred.view(-1), neg_test_pred.view(-1)))
    return result


@torch.no_grad()
def evaluate_test_only_from_embedding(model, z, data, batch_size, *, include_auc=True):
    device = z.device
    mode = data.get("mode", "heart")
    test_pos_source = data["test_pos"].to(dtype=torch.long).cpu()
    test_neg = data.get("test_neg")
    if _is_streamed_grouped_negative(test_neg):
        streamed = _streamed_grouped_split_metrics(
            model, z, {"test": test_pos_source}, test_neg, batch_size, include_auc=include_auc, include_hits=True
        )
        results = _single_rank_result(streamed["test"]["rank"])
        if include_auc:
            results.update(streamed["test"]["auc"])
        test_neg.last_evaluation_reuse = dict(streamed["reuse"])
        return results
    test_pos = test_pos_source.to(device=device, dtype=torch.long)
    if test_neg is not None and not _is_ragged_negative(test_neg):
        test_neg = test_neg.to(device=device, dtype=torch.long)
    reference_row_batch = int(getattr(model, "reference_evaluation_row_batch_size", 0) or 0)
    if reference_row_batch > 0 and test_neg is not None and not _is_ragged_negative(test_neg) and mode != "all":
        (pos_pred, neg_pred_flat) = _reference_grouped_edge_scores(model, z, test_pos, test_neg, batch_size)
        return _test_metrics(
            pos_pred,
            neg_pred_flat,
            rank_scores=_group_scores(pos_pred, neg_pred_flat, "Test"),
            include_auc=include_auc,
        )
    pos_pred = _test_edge_model(model, test_pos, z, batch_size).view(-1)
    if _is_ragged_negative(test_neg):
        (test_stats, neg_pred_flat) = _ragged_scores_and_stats(model, z, pos_pred, test_neg, batch_size)
        return _test_metrics(pos_pred, neg_pred_flat, rank_stats=test_stats, include_auc=include_auc)
    neg_pred_flat = _test_edge_model(model, test_neg, z, batch_size).view(-1)
    return _test_metrics(
        pos_pred,
        neg_pred_flat,
        rank_scores=_group_scores(pos_pred, neg_pred_flat),
        include_auc=include_auc,
    )


@torch.no_grad()
def evaluate_validation_only_from_embedding(model, z, data, batch_size, profile=None, include_auc=True, include_hits=True):
    device = z.device
    valid_pos_source = data["valid_pos"].to(dtype=torch.long).cpu()
    valid_neg = data.get("valid_neg")
    if _is_streamed_grouped_negative(valid_neg):
        _sync_if_profiled(profile, device)
        started = time.time()
        streamed = _streamed_grouped_split_metrics(
            model, z, {"valid": valid_pos_source}, valid_neg, batch_size, include_auc=include_auc, include_hits=include_hits
        )
        _finish_profile(profile, device, "testing_sec", started)
        result = _validation_only_result(streamed["valid"]["rank"], streamed["valid"]["auc"])
        valid_neg.last_evaluation_reuse = dict(streamed["reuse"])
        return result
    valid_pos = valid_pos_source.to(device=device, dtype=torch.long)
    valid_neg = valid_neg.to(device=device, dtype=torch.long)
    _sync_if_profiled(profile, device)
    started = time.time()
    if int(getattr(model, "reference_evaluation_row_batch_size", 0) or 0) > 0:
        (pos_pred, neg_pred_flat) = _reference_grouped_edge_scores(model, z, valid_pos, valid_neg, batch_size)
    else:
        pos_pred = _test_edge_model(model, valid_pos, z, batch_size).view(-1)
        neg_pred_flat = _test_edge_model(model, valid_neg, z, batch_size).view(-1)
    neg_pred = _group_scores(pos_pred, neg_pred_flat, "Validation")
    stats = evaluate_mrr(None, pos_pred, neg_pred) if include_hits else evaluate_mrr_only(pos_pred, neg_pred)
    auc = _auc_pair(pos_pred, neg_pred_flat) if include_auc else None
    _finish_profile(profile, device, "testing_sec", started)
    return _validation_only_result(stats, auc)


def _embed_for_evaluation(model, data, x, profile):
    device = x.device
    adj = data["adj"].to(device) if str(device).startswith("cuda") else data["adj"]
    _sync_if_profiled(profile, device)
    started = time.time()
    z = model.embed(_make_graph_data(x, adj, data.get("csr_train_rowptr"), data.get("csr_train_col")))
    _finish_profile(profile, device, "inference_sec", started)
    return z


@torch.no_grad()
def validation_only(model, data, x, batch_size, profile=None, include_auc=True, include_hits=True):
    model.eval()
    z = _embed_for_evaluation(model, data, x, profile)
    return evaluate_validation_only_from_embedding(
        model, z, data, batch_size, profile=profile, include_auc=include_auc, include_hits=include_hits
    )


@torch.no_grad()
def test_only(model, data, x, batch_size, profile=None, *, include_auc=True):
    model.eval()
    z = _embed_for_evaluation(model, data, x, profile)
    started = time.time()
    results = evaluate_test_only_from_embedding(model, z, data, batch_size, include_auc=include_auc)
    _finish_profile(profile, x.device, "testing_sec", started)
    return results


@torch.no_grad()
def test(model, data, x, evaluator_hit, evaluator_mrr, batch_size, profile=None, return_scores=True, *, include_auc=True):
    model.eval()
    mode = data.get("mode", "heart")
    device = x.device
    adj = data["adj"].to(device) if str(device).startswith("cuda") else data["adj"]
    _sync_if_profiled(profile, device)
    t_infer = time.time()
    z = model.embed(_make_graph_data(x, adj, data.get("csr_train_rowptr"), data.get("csr_train_col")))
    train_val_source = data["train_val"].to(dtype=torch.long).cpu()
    valid_pos_source = data["valid_pos"].to(dtype=torch.long).cpu()
    test_pos_source = data["test_pos"].to(dtype=torch.long).cpu()
    valid_neg_source = data.get("valid_neg")
    test_neg_source = data.get("test_neg")
    valid_streamed = _is_streamed_grouped_negative(valid_neg_source)
    test_streamed = _is_streamed_grouped_negative(test_neg_source)
    if valid_streamed or test_streamed:
        _finish_profile(profile, device, "inference_sec", t_infer)
        t_test = time.time()
        valid_stream = _streamed_grouped_split_metrics(
            model,
            z,
            {"train": train_val_source, "valid": valid_pos_source},
            valid_neg_source,
            batch_size,
            include_auc=include_auc,
            include_hits=True,
        )
        test_stream = _streamed_grouped_split_metrics(
            model, z, {"test": test_pos_source}, test_neg_source, batch_size, include_auc=include_auc, include_hits=True
        )
        results = _pack_rank_results(valid_stream["train"]["rank"], valid_stream["valid"]["rank"], test_stream["test"]["rank"])
        if include_auc:
            results["AUC"] = (valid_stream["train"]["auc"]["AUC"], valid_stream["valid"]["auc"]["AUC"], test_stream["test"]["auc"]["AUC"])
            results["AP"] = (valid_stream["train"]["auc"]["AP"], valid_stream["valid"]["auc"]["AP"], test_stream["test"]["auc"]["AP"])
        valid_neg_source.last_evaluation_reuse = dict(valid_stream["reuse"])
        test_neg_source.last_evaluation_reuse = dict(test_stream["reuse"])
        _finish_profile(profile, device, "testing_sec", t_test)
        return (results, [])
    train_val = train_val_source.to(device, dtype=torch.long)
    valid_pos = valid_pos_source.to(device, dtype=torch.long)
    test_pos = test_pos_source.to(device, dtype=torch.long)
    valid_neg = valid_neg_source.to(device, dtype=torch.long) if valid_neg_source is not None else None
    test_neg = test_neg_source.to(device, dtype=torch.long) if test_neg_source is not None else None
    reference_grouped = (
        mode == "heart"
        and int(getattr(model, "reference_evaluation_row_batch_size", 0) or 0) > 0
        and (valid_neg is not None)
        and (test_neg is not None)
        and not _is_ragged_negative(valid_neg)
        and not _is_ragged_negative(test_neg)
    )
    pos_train_pred = _test_edge_model(model, train_val, z, batch_size).view(-1)
    if reference_grouped:
        (pos_valid_pred, neg_valid_pred_flat) = _reference_grouped_edge_scores(model, z, valid_pos, valid_neg, batch_size)
        (pos_test_pred, neg_test_pred_flat) = _reference_grouped_edge_scores(model, z, test_pos, test_neg, batch_size)
        neg_valid_pred = _group_scores(pos_valid_pred, neg_valid_pred_flat, "Validation")
        neg_test_pred = _group_scores(pos_test_pred, neg_test_pred_flat, "Test")
    else:
        pos_valid_pred = _test_edge_model(model, valid_pos, z, batch_size).view(-1)
        pos_test_pred = _test_edge_model(model, test_pos, z, batch_size).view(-1)
        if _is_ragged_negative(valid_neg) and _is_ragged_negative(test_neg):
            (valid_stats, neg_valid_flat) = _ragged_scores_and_stats(model, z, pos_valid_pred, valid_neg, batch_size)
            (train_stats, _) = _ragged_scores_and_stats(
                model, z, pos_train_pred, valid_neg, batch_size, flat_scores=neg_valid_flat
            )
            (test_stats, neg_test_flat) = _ragged_scores_and_stats(model, z, pos_test_pred, test_neg, batch_size)
            results = _pack_rank_results(train_stats, valid_stats, test_stats)
            if include_auc:
                results.update(_pack_auc_results(pos_train_pred, pos_valid_pred, pos_test_pred, neg_valid_flat, neg_test_flat))
            scores = [pos_valid_pred.cpu(), neg_valid_flat.cpu(), pos_test_pred.cpu(), neg_test_flat.cpu(), z.cpu()] if return_scores else []
            return (results, scores)
        neg_valid_pred_flat = _test_edge_model(model, valid_neg, z, batch_size).view(-1)
        neg_test_pred_flat = _test_edge_model(model, test_neg, z, batch_size).view(-1)
        neg_valid_pred = _group_scores(pos_valid_pred, neg_valid_pred_flat)
        neg_test_pred = _group_scores(pos_test_pred, neg_test_pred_flat)
    _finish_profile(profile, device, "inference_sec", t_infer)
    t_test = time.time()
    results = get_metric_score(
        evaluator_hit,
        evaluator_mrr,
        pos_train_pred,
        pos_valid_pred,
        neg_valid_pred,
        pos_test_pred,
        neg_test_pred,
        include_auc=include_auc,
    )
    _finish_profile(profile, device, "testing_sec", t_test)
    scores = [pos_valid_pred.cpu(), neg_valid_pred_flat.cpu(), pos_test_pred.cpu(), neg_test_pred_flat.cpu(), z.cpu()] if return_scores else []
    return (results, scores)
