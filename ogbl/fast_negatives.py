from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterator, List, Optional, Tuple
import torch


@dataclass
class EndpointCorruptionGroupedNegativeEdges:
    pos_edges: torch.Tensor
    candidate_nodes: torch.Tensor
    num_nodes: int
    pos_chunk_size: int = 256
    is_streaming_negative: bool = True
    is_endpoint_corruption_negative: bool = True

    def __post_init__(self) -> None:
        self.num_nodes = int(self.num_nodes)
        self.pos_chunk_size = max(1, int(self.pos_chunk_size))

    @property
    def candidates_per_side(self) -> int:
        return int(self.candidate_nodes.size(1)) // 2

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (int(self.pos_edges.size(0)), int(self.candidate_nodes.size(1)), 2)

    def dim(self) -> int:
        return 3

    def size(self, dim: Optional[int] = None):
        shape = self.shape
        return torch.Size(shape) if dim is None else shape[int(dim)]

    def numel(self) -> int:
        return int(self.pos_edges.size(0)) * int(self.candidate_nodes.size(1)) * 2

    @property
    def storage_nbytes(self) -> int:
        return int(self.pos_edges.numel()) * int(self.pos_edges.element_size()) + int(self.candidate_nodes.numel()) * int(
            self.candidate_nodes.element_size()
        )

    @property
    def device(self) -> torch.device:
        return self.pos_edges.device

    def cache_on_device(self, device, non_blocking: bool = True) -> "EndpointCorruptionGroupedNegativeEdges":
        device = torch.device(device)
        if self.device == device:
            return self
        cached = EndpointCorruptionGroupedNegativeEdges(
            pos_edges=self.pos_edges.to(device=device, dtype=torch.long, non_blocking=non_blocking),
            candidate_nodes=self.candidate_nodes.to(device=device, dtype=torch.int32, non_blocking=non_blocking),
            num_nodes=self.num_nodes,
            pos_chunk_size=self.pos_chunk_size,
        )
        return cached

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def iter_chunks(self, pos_chunk_size: Optional[int] = None) -> Iterator[Tuple[int, int, torch.Tensor]]:
        rows = int(self.pos_edges.size(0))
        per_side = self.candidates_per_side
        chunk_size = self.pos_chunk_size if pos_chunk_size is None else max(1, int(pos_chunk_size))
        for start in range(0, rows, chunk_size):
            end = min(start + chunk_size, rows)
            positives = self.pos_edges[start:end]
            compact_candidates = self.candidate_nodes[start:end]
            candidates = compact_candidates.to(torch.long)
            chunk = torch.empty((end - start, 2 * per_side, 2), dtype=torch.long, device=positives.device)
            chunk[:, :per_side, 0] = positives[:, 0:1]
            chunk[:, :per_side, 1] = candidates[:, :per_side]
            chunk[:, per_side:, 0] = candidates[:, per_side:]
            chunk[:, per_side:, 1] = positives[:, 1:2]
            yield (start, end, chunk)

    def materialize(self) -> torch.Tensor:
        out = torch.empty(self.shape, dtype=torch.long, device=self.pos_edges.device)
        for start, end, chunk in self.iter_chunks():
            out[start:end] = chunk
        return out.contiguous()

    def __repr__(self) -> str:
        return f"EndpointCorruptionGroupedNegativeEdges(shape={self.shape}, num_nodes={self.num_nodes}, candidate_dtype={self.candidate_nodes.dtype}, pos_chunk_size={self.pos_chunk_size})"


@dataclass
class RaggedGroupedNegativeEdges:
    flat_edges: torch.Tensor
    rowptr: torch.Tensor
    max_per_side: int
    side_counts: Optional[torch.Tensor] = None
    pos_chunk_size: int = 64
    is_streaming_negative: bool = True
    is_ragged_negative: bool = True

    def __post_init__(self):
        self.flat_edges = self.flat_edges.to(torch.long).cpu().contiguous()
        self.rowptr = self.rowptr.to(torch.long).cpu().contiguous()
        if self.side_counts is not None:
            self.side_counts = self.side_counts.to(torch.long).cpu().contiguous()
        if self.rowptr.dim() != 1 or self.rowptr.numel() == 0:
            raise ValueError("rowptr must be a non-empty rank-1 tensor.")
        if int(self.rowptr[0]) != 0 or int(self.rowptr[-1]) != int(self.flat_edges.size(0)):
            raise ValueError("rowptr must delimit every flat candidate edge.")

    @property
    def num_pos(self) -> int:
        return int(self.rowptr.numel() - 1)

    @property
    def shape(self) -> Tuple[int, int, int]:
        counts = self.rowptr[1:] - self.rowptr[:-1]
        maximum = int(counts.max().item()) if counts.numel() else 0
        return (self.num_pos, maximum, 2)

    def dim(self) -> int:
        return 3

    def size(self, dim: Optional[int] = None):
        shape = self.shape
        return torch.Size(shape) if dim is None else shape[int(dim)]

    def numel(self) -> int:
        return int(self.flat_edges.numel())

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def iter_ragged_chunks(self):
        for start in range(0, self.num_pos, max(1, int(self.pos_chunk_size))):
            end = min(start + max(1, int(self.pos_chunk_size)), self.num_pos)
            edge_start = int(self.rowptr[start])
            edge_end = int(self.rowptr[end])
            local_rowptr = self.rowptr[start : end + 1] - edge_start
            yield (start, end, self.flat_edges[edge_start:edge_end], local_rowptr)

    def candidate_count_stats(self):
        counts = self.rowptr[1:] - self.rowptr[:-1]
        if not counts.numel():
            return {"min": 0, "mean": 0.0, "max": 0}
        return {"min": int(counts.min().item()), "mean": float(counts.to(torch.float64).mean().item()), "max": int(counts.max().item())}


@dataclass
class DeduplicatedGroupedNegativeEdges:
    unique_edges: torch.Tensor
    inverse_indices: torch.Tensor
    canonical_undirected: bool = False
    is_streaming_negative: bool = True
    is_deduplicated_grouped_negative: bool = True

    def __post_init__(self) -> None:
        self.unique_edges = self.unique_edges.to(device="cpu", dtype=torch.long).contiguous()
        self.inverse_indices = self.inverse_indices.to(device="cpu", dtype=torch.int32).contiguous()

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (int(self.inverse_indices.size(0)), int(self.inverse_indices.size(1)), 2)

    def dim(self) -> int:
        return 3

    def size(self, dim: Optional[int] = None):
        shape = self.shape
        return torch.Size(shape) if dim is None else shape[int(dim)]

    def numel(self) -> int:
        return int(self.inverse_indices.numel()) * 2

    @property
    def original_edge_count(self) -> int:
        return int(self.inverse_indices.numel())

    @property
    def unique_edge_count(self) -> int:
        return int(self.unique_edges.size(0))

    @property
    def decode_fraction(self) -> float:
        if self.original_edge_count == 0:
            return 1.0
        return self.unique_edge_count / float(self.original_edge_count)

    @property
    def storage_nbytes(self) -> int:
        return int(self.unique_edges.numel()) * int(self.unique_edges.element_size()) + int(self.inverse_indices.numel()) * int(
            self.inverse_indices.element_size()
        )

    def to(self, *args, **kwargs):
        return self

    def contiguous(self):
        return self

    def iter_inverse_chunks(self, edge_batch_size: int) -> Iterator[Tuple[int, int, torch.Tensor]]:
        k = max(1, int(self.inverse_indices.size(1)))
        row_batch = max(1, int(edge_batch_size) // k)
        rows = int(self.inverse_indices.size(0))
        for start in range(0, rows, row_batch):
            end = min(start + row_batch, rows)
            yield (start, end, self.inverse_indices[start:end])

    def iter_chunks(self) -> Iterator[Tuple[int, int, torch.Tensor]]:
        for start, end, inverse in self.iter_inverse_chunks(1048576):
            flat = self.unique_edges.index_select(0, inverse.reshape(-1).to(torch.long))
            yield (start, end, flat.view(end - start, self.size(1), 2))

    def __repr__(self) -> str:
        return f"DeduplicatedGroupedNegativeEdges(shape={self.shape}, unique={self.unique_edge_count}, decode_fraction={self.decode_fraction:.4f}, canonical_undirected={self.canonical_undirected})"


_SYMMETRIC_DDI_MODEL_NAMES = frozenset({"gcn", "gat", "sage", "mlp", "ppr", "concat", "mf", "gae", "node2vec", "n2v", "heuristic"})
_DEDUP_SAFE_DDI_MODEL_NAMES = frozenset(set(_SYMMETRIC_DDI_MODEL_NAMES) | {"buddy", "ncn", "ncnc", "nbfnet", "peg", "lpformer", "lpf"})


def ddi_model_has_symmetric_decoder(model_name: str) -> bool:
    normalized = str(model_name).strip().lower().replace("-", "").replace("_", "")
    return normalized in _SYMMETRIC_DDI_MODEL_NAMES


def ddi_model_supports_exact_dedup(model_name: str) -> bool:
    normalized = str(model_name).strip().lower().replace("-", "").replace("_", "")
    return normalized in _DEDUP_SAFE_DDI_MODEL_NAMES


def _linear_pair_ids(edge_chunk: torch.Tensor, num_nodes: int, canonical_undirected: bool) -> torch.Tensor:
    src = edge_chunk[:, 0].to(torch.long)
    dst = edge_chunk[:, 1].to(torch.long)
    if canonical_undirected:
        lower = torch.minimum(src, dst)
        upper = torch.maximum(src, dst)
        (src, dst) = (lower, upper)
    return src * int(num_nodes) + dst


def deduplicate_grouped_negative_edges(
    edges,
    *,
    num_nodes: int,
    canonical_undirected: bool = False,
    min_edge_count: int = 1000000,
    scan_chunk_size: int = 2000000,
    max_lookup_entries: int = 50000000,
):
    if not torch.is_tensor(edges) or edges.dim() != 3 or edges.size(-1) != 2 or (edges.device.type != "cpu") or (not edges.is_contiguous()):
        return edges
    rows = int(edges.size(0))
    candidates = int(edges.size(1))
    total = rows * candidates
    num_nodes = int(num_nodes)
    universe = num_nodes * num_nodes
    if (
        total < max(1, int(min_edge_count))
        or num_nodes <= 0
        or universe > int(max_lookup_entries)
        or (total > torch.iinfo(torch.int32).max)
    ):
        return edges
    flat = edges.view(-1, 2)
    scan_chunk_size = max(1, int(scan_chunk_size))
    seen = torch.zeros(universe, dtype=torch.bool)
    for start in range(0, total, scan_chunk_size):
        end = min(start + scan_chunk_size, total)
        chunk = flat[start:end]
        if chunk.numel():
            lower = int(chunk.min().item())
            upper = int(chunk.max().item())
            if lower < 0 or upper >= num_nodes:
                raise ValueError(f"Grouped negative edge contains an endpoint outside [0, {num_nodes}): min={lower}, max={upper}.")
        ids = _linear_pair_ids(chunk, num_nodes, bool(canonical_undirected))
        seen[ids] = True
    unique_ids = torch.nonzero(seen, as_tuple=True)[0]
    unique_count = int(unique_ids.numel())
    if unique_count == 0 or unique_count * 4 >= total * 3:
        return edges
    lookup = torch.full((universe,), -1, dtype=torch.int32)
    lookup[unique_ids] = torch.arange(unique_count, dtype=torch.int32)
    inverse = torch.empty(total, dtype=torch.int32)
    for start in range(0, total, scan_chunk_size):
        end = min(start + scan_chunk_size, total)
        ids = _linear_pair_ids(flat[start:end], num_nodes, bool(canonical_undirected))
        inverse[start:end] = lookup[ids]
    unique_edges = torch.stack([torch.div(unique_ids, num_nodes, rounding_mode="floor"), unique_ids.remainder(num_nodes)], dim=1)
    return DeduplicatedGroupedNegativeEdges(
        unique_edges=unique_edges, inverse_indices=inverse.view(rows, candidates), canonical_undirected=bool(canonical_undirected)
    )


def prepare_ddi_grouped_eval_edges(
    eval_edges: Dict[str, object], *, dataset_name: str, model_name: str, num_nodes: int, source_bundle: Optional[Dict[str, object]] = None
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    out = eval_edges
    if str(dataset_name).strip().lower() != "ogbl-ddi":
        return (out, [])
    if not ddi_model_supports_exact_dedup(model_name):
        return (out, [])
    canonical = ddi_model_has_symmetric_decoder(model_name)
    summaries: List[Dict[str, object]] = []
    for key in ("neg_valid_edge", "neg_test_edge"):
        original = out.get(key)
        dedup_input = original.materialize() if isinstance(original, EndpointCorruptionGroupedNegativeEdges) else original
        compact = deduplicate_grouped_negative_edges(dedup_input, num_nodes=int(num_nodes), canonical_undirected=canonical)
        if compact is dedup_input:
            del original, dedup_input, compact
            continue
        out[key] = compact
        if source_bundle is not None:
            source_bundle["valid_neg" if key == "neg_valid_edge" else "test_neg"] = compact
        summaries.append(
            {
                "key": key,
                "original_edges": compact.original_edge_count,
                "unique_edges": compact.unique_edge_count,
                "decode_fraction": compact.decode_fraction,
                "canonical_undirected": compact.canonical_undirected,
                "storage_nbytes": compact.storage_nbytes,
            }
        )
        del original, dedup_input, compact
    return (out, summaries)


def is_streaming_negative_edges(obj) -> bool:
    return bool(getattr(obj, "is_streaming_negative", False))


def make_forbidden_edge_ids(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    if edge_index.numel() == 0:
        return torch.empty((0,), dtype=torch.long)
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    src = edge_index[0].to(torch.long).cpu()
    dst = edge_index[1].to(torch.long).cpu()
    ids = src * int(num_nodes) + dst
    return torch.unique(ids).sort().values.contiguous()


def _is_forbidden(src: torch.Tensor, dst: torch.Tensor, forbidden_edge_ids: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    bad = src == dst
    if forbidden_edge_ids is None or forbidden_edge_ids.numel() == 0:
        return bad
    cand_ids = src.to(torch.long) * int(num_nodes) + dst.to(torch.long)
    pos = torch.searchsorted(forbidden_edge_ids, cand_ids)
    in_bounds = pos < forbidden_edge_ids.numel()
    exists = torch.zeros_like(bad, dtype=torch.bool)
    if bool(in_bounds.any()):
        pos_in = pos[in_bounds]
        exists[in_bounds] = forbidden_edge_ids[pos_in] == cand_ids[in_bounds]
    return bad | exists


def sample_global_negative_edges(
    num_nodes: int, num_samples: int, forbidden_edge_ids: Optional[torch.Tensor], seed: int, oversample: float = 1.08, strict=True
) -> torch.Tensor:
    num_samples = int(num_samples)
    if num_samples <= 0:
        return torch.empty((0, 2), dtype=torch.long)
    strict_bool = bool(strict)
    forbidden = None
    if strict_bool and forbidden_edge_ids is not None:
        forbidden = forbidden_edge_ids.to(torch.long).cpu().contiguous()
    out = torch.empty((num_samples, 2), dtype=torch.long)
    filled = 0
    g = torch.Generator().manual_seed(int(seed))
    while filled < num_samples:
        need = num_samples - filled
        draw = max(need + 1024, int(need * oversample))
        cand = torch.randint(0, int(num_nodes), (draw, 2), generator=g, dtype=torch.long)
        if strict_bool:
            bad = _is_forbidden(cand[:, 0], cand[:, 1], forbidden, int(num_nodes))
        else:
            bad = cand[:, 0] == cand[:, 1]
        keep = cand[~bad]
        if keep.numel() == 0:
            oversample = min(2.0, oversample * 1.25)
            continue
        take = min(need, int(keep.size(0)))
        out[filled : filled + take] = keep[:take]
        filled += take
    return out.contiguous()
