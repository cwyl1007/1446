from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch

ENDPOINT_GROUPED_FORMAT = "pyg-endpoint-grouped-side-uint-v2"
STREAMED_GROUPED_MANIFEST_VERSION = 1
_SHA256_RE = re.compile("^[0-9a-f]{64}$")
_ENDPOINT_ACCELERATOR_VERSION = 1
_ENDPOINT_ACCELERATOR_LAYOUT = "independent-side-buffered-v1"
_ENDPOINT_ACCELERATOR_KEY = "physical_accelerator"
_ENDPOINT_VALIDATION_CONTRACT = {
    "version": 1,
    "publisher": "EndpointGroupedNegativeWriter",
    "validation": "exhaustive-before-manifest-publish",
    "union_nodes": "strictly-increasing-unique",
    "group_row_ids": "strictly-increasing-unique",
    "row_coverage": "exactly-once-per-side",
    "candidate_count": "fixed-negatives-per-side",
    "candidate_legality": "no-self-no-positive-counterpart",
    "hard_prefix": "unique-per-row",
    "fallback_suffix": "ordered-replacement-multiset",
    "union_multiplicity": "exact-logical-occurrence-count",
}


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _edge_rows(value, name):
    tensor = torch.as_tensor(value, dtype=torch.long).detach().cpu()
    _require(tensor.ndim == 2 and int(tensor.size(1)) == 2, f"{name} must have shape [N, 2], got {tuple(tensor.shape)}.")
    return tensor.contiguous()


def _normalized_sha256(value, name):
    digest = str(value or "").strip().lower()
    _require(_SHA256_RE.fullmatch(digest), f"{name} must be a lowercase SHA-256 digest.")
    return digest


def _tensor_sha256(tensor, row_chunk_size=65536):
    rows = _edge_rows(tensor, "positive rows")
    digest = hashlib.sha256()
    digest.update(str(rows.dtype).encode("ascii"))
    digest.update(np.asarray(tuple(rows.shape), dtype=np.int64).tobytes())
    for start in range(0, int(rows.size(0)), max(1, int(row_chunk_size))):
        block = rows[start : start + int(row_chunk_size)].numpy()
        digest.update(memoryview(np.ascontiguousarray(block)).cast("B"))
    return digest.hexdigest()


def _canonical_json(payload):
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _manifest_digest(payload):
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    return hashlib.sha256(_canonical_json(body)).hexdigest()


def _file_sha256(path, chunk_bytes=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _fsync_directory(path):
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _unlock(handle):
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
    finally:
        handle.close()


def _atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("wb") as handle:
            handle.write(_canonical_json(payload))
            handle.write(b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def _verify_file(path, byte_count, sha256, verified, stats):
    filename = path.name
    if filename in verified:
        current = path.stat()
        if (current.st_size, current.st_mtime_ns, current.st_ino) != stats[filename]:
            raise RuntimeError("A verified endpoint-grouped shard was replaced.")
        return
    before = path.stat()
    if before.st_size != byte_count:
        raise ValueError("Endpoint-grouped shard size changed before validation.")
    digest = _file_sha256(path)
    after = path.stat()
    before_key = (before.st_size, before.st_mtime_ns, before.st_ino)
    after_key = (after.st_size, after.st_mtime_ns, after.st_ino)
    if before_key != after_key:
        raise RuntimeError("Endpoint-grouped shard changed during validation.")
    if digest != sha256:
        raise ValueError("Endpoint-grouped shard SHA-256 mismatch.")
    verified.add(filename)
    stats[filename] = after_key


class _GroupedNegativeView:
    is_streaming_negative = True
    is_streamed_grouped_negative = True
    is_ragged_negative = False

    def dim(self):
        return 3

    @property
    def ndim(self):
        return 3

    def size(self, dim=None):
        return torch.Size(self.shape) if dim is None else self.shape[int(dim)]

    def numel(self):
        return self.num_rows * self.negatives_per_row * 2

    @property
    def num_edges(self):
        return self.num_rows * self.negatives_per_row

    def __len__(self):
        return self.num_rows

    def cpu(self):
        return self

    def contiguous(self):
        return self

    def to(self, *args, **kwargs):
        del args, kwargs
        raise RuntimeError(self._materialize_error)


@dataclass(frozen=True)
class EndpointGroupChunk:
    side: int
    endpoints: torch.Tensor
    union_rowptr: torch.Tensor
    union_nodes: torch.Tensor
    occurrence_endpoint_index: torch.Tensor
    occurrence_row_ids: torch.Tensor
    candidate_local_indices: torch.Tensor
    hard_prefix_count: torch.Tensor
    union_occurrence_multiplicity: torch.Tensor

    @property
    def num_groups(self):
        return int(self.endpoints.numel())

    @property
    def num_occurrences(self):
        return int(self.occurrence_row_ids.numel())


class EndpointGroupedNegatives(_GroupedNegativeView):
    is_endpoint_grouped_negative = True
    candidate_summary_prevalidated = True
    _materialize_error = "Endpoint-grouped negatives cannot be materialized with .to(); consume iter_endpoint_group_chunks() instead."

    def __init__(self, manifest_path, positives, *, expected_recipe_sha256=None, verify_shards="lazy"):
        self.manifest_path = str(Path(manifest_path).expanduser().resolve())
        self._manifest_dir = Path(self.manifest_path).parent
        self.positives = _edge_rows(positives, "positive rows")
        with open(self.manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        _require(isinstance(manifest, dict), "Endpoint-grouped manifest must contain an object.")
        identity, expected_manifest_sha = _endpoint_manifest_identity(manifest)
        self.storage_format = ENDPOINT_GROUPED_FORMAT
        self.manifest_sha256 = expected_manifest_sha
        self.num_rows = identity["num_rows"]
        self.negatives_per_side = identity["negatives_per_side"]
        self.negatives_per_row = identity["negatives_per_row"]
        self.num_nodes = identity["num_nodes"]
        self.split = identity["split"]
        self.positive_sha256 = identity["positive_sha256"]
        self.candidate_recipe_sha256 = identity["candidate_recipe_sha256"]
        if expected_recipe_sha256 is not None:
            _require(self.candidate_recipe_sha256 == _normalized_sha256(expected_recipe_sha256, "expected_recipe_sha256"), "Endpoint-grouped candidate recipe SHA-256 mismatch.")
        _require(self.num_rows == int(self.positives.size(0)), "Endpoint-grouped rows do not match the positive split.")
        limits = identity["shard_limits"]
        self.occurrence_rows_per_shard = limits["occurrence_rows"]
        self.groups_per_shard = limits["endpoint_groups"]
        self.union_nodes_per_shard = limits["union_nodes"]
        _require(_tensor_sha256(self.positives) == self.positive_sha256, "Endpoint-grouped positive-split SHA-256 mismatch.")
        self.shape = (self.num_rows, self.negatives_per_row, 2)
        raw_shards = manifest.get("shards")
        _require(isinstance(raw_shards, list) and raw_shards, "Endpoint-grouped manifest contains no shards.")
        self.shards = []
        totals = {0: 0, 1: 0}
        for raw in raw_shards:
            _require(isinstance(raw, Mapping), "Invalid endpoint-grouped shard entry.")
            filename = str(raw["filename"])
            _require(Path(filename).name == filename, "Endpoint-grouped shard filename is unsafe.")
            side = int(raw["side"])
            _require(side in (0, 1), "Endpoint-grouped shard side must be 0 or 1.")
            entry = {
                "filename": filename,
                "side": side,
                "group_count": int(raw["group_count"]),
                "occurrence_count": int(raw["occurrence_count"]),
                "union_node_count": int(raw["union_node_count"]),
                "byte_count": int(raw["byte_count"]),
                "sha256": _normalized_sha256(raw.get("sha256"), "shard sha256"),
                "local_index_dtype": str(raw.get("local_index_dtype", "")),
            }
            _require(entry["local_index_dtype"] in {"uint8", "uint16"}, "Endpoint-grouped shard local-index dtype is invalid.")
            _require(min(entry["group_count"], entry["occurrence_count"], entry["union_node_count"]) > 0, "Endpoint-grouped shard counts must be positive.")
            _require(
                entry["group_count"] <= self.groups_per_shard
                and entry["occurrence_count"] <= self.occurrence_rows_per_shard
                and entry["union_node_count"] <= self.union_nodes_per_shard,
                "Endpoint-grouped shard exceeds its manifest-bound limit.",
            )
            totals[side] += entry["occurrence_count"]
            self.shards.append(entry)
        _require(totals == {0: self.num_rows, 1: self.num_rows}, "Endpoint-grouped shards must cover every row exactly once per side.")
        self.shards = tuple(self.shards)
        self._verified = set()
        self._verified_stats = {}
        self._semantically_validated = set()
        self._coverage_validated = False
        verify = str(verify_shards).strip().lower()
        _require(verify in {"lazy", "eager"}, "verify_shards must be 'lazy' or 'eager'.")
        if verify == "eager":
            for shard in self.shards:
                self._verify_shard(shard)

    @property
    def candidate_summary(self):
        fixed = lambda value: {"min": int(value), "mean": float(value), "max": int(value)}
        return {
            "num_positive_edges": self.num_rows,
            "grouped_negatives_per_positive": self.negatives_per_row,
            "both_corruption_sides_combined": True,
            "fixed_left_endpoint_candidates": fixed(self.negatives_per_side),
            "fixed_right_endpoint_candidates": fixed(self.negatives_per_side),
            "other_grouped_candidates": fixed(0),
            "total_grouped_candidates": fixed(self.negatives_per_row),
        }

    def _verify_shard(self, shard):
        _verify_file(self._manifest_dir / shard["filename"], shard["byte_count"], shard["sha256"], self._verified, self._verified_stats)

    def iter_endpoint_group_chunks(self):
        seen = None if self._coverage_validated else np.zeros((2, self.num_rows), dtype=np.bool_)
        for shard in self.shards:
            self._verify_shard(shard)
            path = self._manifest_dir / shard["filename"]
            with np.load(path, allow_pickle=False) as payload:
                endpoint_ids = np.asarray(payload["endpoint_ids"], dtype=np.uint32)
                union_rowptr = np.asarray(payload["union_rowptr"], dtype=np.uint64)
                union_nodes = np.asarray(payload["union_nodes"], dtype=np.uint32)
                occurrence_endpoint_index = np.asarray(payload["occurrence_endpoint_index"], dtype=np.uint32)
                occurrence_row_ids = np.asarray(payload["occurrence_row_ids"], dtype=np.uint32)
                local_indices = np.asarray(payload["candidate_local_indices"])
                expected_local_dtype = np.dtype(shard["local_index_dtype"])
                _require(local_indices.dtype == expected_local_dtype, "Endpoint-grouped local-index payload dtype does not match its manifest.")
                hard_prefix_count = np.asarray(payload["hard_prefix_count"], dtype=np.uint16)
                union_multiplicity = np.asarray(payload["union_occurrence_multiplicity"], dtype=np.uint32)
            occurrences = shard["occurrence_count"]
            _require(tuple(local_indices.shape) == (occurrences, self.negatives_per_side), "Endpoint-grouped local-index shape is invalid.")
            _require(tuple(hard_prefix_count.shape) == (occurrences,) and not bool(np.any(hard_prefix_count > self.negatives_per_side)), "Endpoint-grouped hard-prefix boundary is invalid.")
            _require(endpoint_ids.size == shard["group_count"] and tuple(union_rowptr.shape) == (shard["group_count"] + 1,), "Endpoint-grouped group arrays are invalid.")
            _require(union_nodes.size == shard["union_node_count"], "Endpoint-grouped union-node count is invalid.")
            _require(tuple(union_multiplicity.shape) == (union_nodes.size,), "Endpoint-grouped union multiplicity shape is invalid.")
            _require(occurrence_endpoint_index.size == occurrences and occurrence_row_ids.size == occurrences, "Endpoint-grouped occurrence arrays are invalid.")
            _require(
                union_rowptr[0] == 0 and union_rowptr[-1] == union_nodes.size and not bool(np.any(union_rowptr[1:] < union_rowptr[:-1])),
                "Endpoint-grouped union row pointers are invalid.",
            )
            _require(not bool(np.any(occurrence_endpoint_index >= endpoint_ids.size)) and not bool(np.any(occurrence_row_ids >= self.num_rows)), "Endpoint-grouped occurrence index is invalid.")
            side = int(shard["side"])
            if shard["filename"] not in self._semantically_validated:
                _require(not bool(np.any(endpoint_ids >= self.num_nodes)) and not bool(np.any(union_nodes >= self.num_nodes)), "Endpoint-grouped shard contains invalid node ids.")
                group_widths = union_rowptr[1:] - union_rowptr[:-1]
                occurrence_widths = group_widths[occurrence_endpoint_index]
                _require(not bool(np.any(local_indices >= occurrence_widths[:, None])), "Endpoint-grouped candidate index exceeds its union.")
                query_column = self.positives[:, side].numpy()
                _require(bool(np.all(query_column[occurrence_row_ids] == endpoint_ids[occurrence_endpoint_index])), "Endpoint-grouped records do not match query endpoints.")
                _require(union_multiplicity.sum(dtype=np.uint64) == occurrences * self.negatives_per_side, "Endpoint-grouped union multiplicities have an invalid logical-occurrence total.")
                self._semantically_validated.add(shard["filename"])
            if seen is not None:
                _require(not bool(np.any(seen[side, occurrence_row_ids])), "Endpoint-grouped row occurs twice on one side.")
                seen[side, occurrence_row_ids] = True
            yield EndpointGroupChunk(
                side=side,
                endpoints=torch.from_numpy(endpoint_ids.astype(np.int64, copy=True)),
                union_rowptr=torch.from_numpy(union_rowptr.astype(np.int64, copy=True)),
                union_nodes=torch.from_numpy(union_nodes.astype(np.int64, copy=True)),
                occurrence_endpoint_index=torch.from_numpy(occurrence_endpoint_index.astype(np.int64, copy=True)),
                occurrence_row_ids=torch.from_numpy(occurrence_row_ids.astype(np.int64, copy=True)),
                candidate_local_indices=torch.from_numpy(local_indices.copy()),
                hard_prefix_count=torch.from_numpy(hard_prefix_count.astype(np.int64, copy=True)),
                union_occurrence_multiplicity=torch.from_numpy(union_multiplicity.astype(np.int64, copy=True)),
            )
        if seen is not None:
            _require(bool(seen.all()), "Endpoint-grouped iteration did not cover every row and side.")
            self._coverage_validated = True

    def iter_chunks(self):
        yield from self.iter_endpoint_group_chunks()


class EndpointGroupedNegativeWriter:
    def __init__(self, manifest_path, positives, *, num_nodes, split, candidate_recipe_sha256, negatives_per_side=250, occurrence_rows_per_shard=8192, groups_per_shard=512, union_nodes_per_shard=1048576):
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.positives = _edge_rows(positives, "positive rows")
        self.num_nodes = int(num_nodes)
        self.split = str(split)
        self.candidate_recipe_sha256 = _normalized_sha256(candidate_recipe_sha256, "candidate_recipe_sha256")
        self.negatives_per_side = int(negatives_per_side)
        self.occurrence_rows_per_shard = max(1, int(occurrence_rows_per_shard))
        self.groups_per_shard = max(1, int(groups_per_shard))
        self.union_nodes_per_shard = max(1, int(union_nodes_per_shard))
        _require(0 < self.negatives_per_side <= 65535, "negatives_per_side cannot use uint16 indices.")
        _require(1 < self.num_nodes <= 2**32, "num_nodes cannot use uint32 storage.")
        self.positive_sha256 = _tensor_sha256(self.positives)
        _require(self.positives.size(0) <= 2**32, "Endpoint-grouped row ids cannot use uint32 storage.")
        self._seen = np.zeros((2, int(self.positives.size(0))), dtype=np.bool_)
        self._lock_handle = None
        self._build_id = uuid.uuid4().hex
        self._buffer_side = None
        self._groups = []
        self._buffered_occurrences = 0
        self._buffered_union_nodes = 0
        self._shards = []
        self._published_files = []
        self._prior_shard_paths = []
        self._finished = False

    def __enter__(self):
        lock_path = self.manifest_path.with_suffix(self.manifest_path.suffix + ".lock")
        self._lock_handle = open(lock_path, "a+b")
        fcntl.flock(self._lock_handle.fileno(), fcntl.LOCK_EX)
        referenced_names = set()
        shard_prefix = f"{self.manifest_path.stem}."

        def is_shard_name(name):
            return bool(
                name
                and Path(name).name == name
                and name.startswith(shard_prefix)
                and ".side" in name
                and name.endswith(".npz")
            )

        if self.manifest_path.exists():
            try:
                with self.manifest_path.open("r", encoding="utf-8") as handle:
                    prior_manifest = json.load(handle)
                if prior_manifest.get("format") == ENDPOINT_GROUPED_FORMAT:
                    for shard in prior_manifest.get("shards", []):
                        filename = str(shard.get("filename", ""))
                        if is_shard_name(filename):
                            referenced_names.add(filename)
                            self._prior_shard_paths.append(self.manifest_path.parent / filename)
            except (OSError, TypeError, ValueError):
                pass
        for entry in os.scandir(self.manifest_path.parent):
            name = entry.name
            if not (entry.is_file(follow_symlinks=False) and is_shard_name(name) and name not in referenced_names):
                continue
            try:
                os.unlink(entry.path)
            except FileNotFoundError:
                pass
        return self

    def _require_open(self):
        if self._lock_handle is None:
            raise RuntimeError("EndpointGroupedNegativeWriter must be a context manager.")
        if self._finished:
            raise RuntimeError("EndpointGroupedNegativeWriter is already finished.")

    def append_endpoint_group(self, side, endpoint, row_ids, union_candidate_nodes, candidate_local_indices, hard_prefix_count):
        self._require_open()
        side = int(side)
        endpoint = int(endpoint)
        _require(side in (0, 1), "side must be 0 (left) or 1 (right).")
        _require(0 <= endpoint < self.num_nodes, "endpoint is outside the graph.")
        rows = torch.as_tensor(row_ids, dtype=torch.long).detach().cpu().view(-1)
        union = torch.as_tensor(union_candidate_nodes, dtype=torch.long).detach().cpu().view(-1)
        local = torch.as_tensor(candidate_local_indices, dtype=torch.long).detach().cpu()
        prefix = torch.as_tensor(hard_prefix_count, dtype=torch.long).detach().cpu().view(-1)
        row_count, union_count = rows.numel(), union.numel()
        _require(row_count > 0 and union_count > 0, "Endpoint groups must contain rows and candidates.")
        _require(row_count <= self.occurrence_rows_per_shard, "Endpoint group exceeds the occurrence-row shard limit; split this endpoint into disjoint row blocks.")
        _require(union_count <= self.union_nodes_per_shard, "Endpoint group exceeds the union-node shard limit; split this endpoint into disjoint row blocks.")
        _require(local.ndim == 2 and tuple(local.shape) == (row_count, self.negatives_per_side), "candidate_local_indices has the wrong shape.")
        _require(prefix.numel() == row_count and not bool((prefix < 0).any()) and not bool((prefix > self.negatives_per_side).any()), "hard_prefix_count must contain one valid boundary per row.")
        _require(union_count <= 65535, "Endpoint candidate union exceeds uint16 capacity.")
        _require(not bool((rows < 0).any()) and not bool((rows >= self.positives.size(0)).any()), "Endpoint group row id is outside the split.")
        rows_np = rows.numpy().astype(np.uint32, copy=False)
        _require(rows_np.size < 2 or not bool(np.any(rows_np[1:] <= rows_np[:-1])), "Endpoint group row ids must be strictly increasing and unique.")
        _require(bool(self.positives[rows, side].eq(endpoint).all()), "Endpoint group row does not match its query endpoint.")
        _require(not bool((union < 0).any()) and not bool((union >= self.num_nodes).any()), "Endpoint union contains an invalid node id.")
        _require(union_count < 2 or bool((union[1:] > union[:-1]).all()), "Endpoint union nodes must be strictly increasing and unique.")
        _require(not bool(union.eq(endpoint).any()), "Endpoint union cannot contain its self node.")
        _require(not bool((local < 0).any()) and not bool((local >= union_count).any()), "Endpoint local candidate index is invalid.")
        mapped = union[local]
        counterparts = self.positives[rows, 1 - side]
        _require(not bool(mapped.eq(counterparts.view(-1, 1)).any()), "Endpoint candidates cannot contain the query counterpart.")
        positions = torch.arange(self.negatives_per_side, dtype=torch.long).view(1, -1)
        in_hard_prefix = positions < prefix.view(-1, 1)
        prefix_uniqueness_values = torch.where(in_hard_prefix, local, int(union.numel()) + positions)
        sorted_prefix_values = torch.sort(prefix_uniqueness_values, dim=1).values
        _require(not bool((sorted_prefix_values[:, 1:] == sorted_prefix_values[:, :-1]).any()), "Hard-prefix candidates must be unique; duplicates are allowed only in the fallback suffix.")
        self._buffer_record(
            side,
            endpoint,
            rows_np,
            union.numpy().astype(np.uint32, copy=True),
            local.numpy().astype(np.uint16, copy=True),
            prefix.numpy().astype(np.uint16, copy=True),
        )

    def _buffer_record(self, side, endpoint, rows, union, local, prefix):
        _require(not bool(np.any(self._seen[side, rows])), "Endpoint group repeats a row on the same side.")
        projected_occurrences = self._buffered_occurrences + rows.size
        projected_union_nodes = self._buffered_union_nodes + union.size
        if self._groups and (
            self._buffer_side != side
            or projected_occurrences > self.occurrence_rows_per_shard
            or len(self._groups) + 1 > self.groups_per_shard
            or projected_union_nodes > self.union_nodes_per_shard
        ):
            self._flush()
        self._buffer_side = side
        self._seen[side, rows] = True
        self._groups.append((endpoint, rows, union, local, prefix))
        self._buffered_occurrences += rows.size
        self._buffered_union_nodes += union.size
        if (
            self._buffered_occurrences >= self.occurrence_rows_per_shard
            or len(self._groups) >= self.groups_per_shard
            or self._buffered_union_nodes >= self.union_nodes_per_shard
        ):
            self._flush()

    def _flush(self):
        if not self._groups:
            return
        side = int(self._buffer_side)
        endpoint_ids = np.asarray([group[0] for group in self._groups], dtype=np.uint32)
        union_lengths = np.asarray([group[2].size for group in self._groups], dtype=np.uint64)
        union_rowptr = np.concatenate([np.zeros(1, dtype=np.uint64), np.cumsum(union_lengths, dtype=np.uint64)])
        union_nodes = np.concatenate([group[2] for group in self._groups])
        occurrence_endpoint_index = np.concatenate(
            [np.full(group[1].size, index, dtype=np.uint32) for (index, group) in enumerate(self._groups)]
        )
        occurrence_row_ids = np.concatenate([group[1] for group in self._groups])
        local_indices = np.concatenate([group[3] for group in self._groups], axis=0)
        hard_prefix_count = np.concatenate([group[4] for group in self._groups])
        local_index_dtype = "uint8" if int(union_lengths.max()) <= 256 else "uint16"
        local_indices_stored = local_indices.astype(np.dtype(local_index_dtype), copy=False)
        global_indices = local_indices.astype(np.uint64, copy=False) + union_rowptr[occurrence_endpoint_index, None]
        union_multiplicity_u64 = np.bincount(
            global_indices.reshape(-1).astype(np.int64, copy=False), minlength=int(union_nodes.size)
        ).astype(np.uint64, copy=False)
        if int(union_multiplicity_u64.max()) > np.iinfo(np.uint32).max:
            raise ValueError("Endpoint candidate multiplicity exceeds uint32 capacity; split this endpoint occurrence group before writing.")
        union_multiplicity = union_multiplicity_u64.astype(np.uint32, copy=False)
        shard_index = len(self._shards)
        filename = f"{self.manifest_path.stem}.{self._build_id}.side{side}.{shard_index:06d}.npz"
        final_path = self.manifest_path.parent / filename
        temporary = final_path.with_name(f".{filename}.tmp")
        try:
            with temporary.open("wb") as handle:
                np.savez(
                    handle,
                    endpoint_ids=endpoint_ids,
                    union_rowptr=union_rowptr,
                    union_nodes=union_nodes,
                    occurrence_endpoint_index=occurrence_endpoint_index,
                    occurrence_row_ids=occurrence_row_ids,
                    candidate_local_indices=local_indices_stored,
                    hard_prefix_count=hard_prefix_count,
                    union_occurrence_multiplicity=union_multiplicity,
                )
                handle.flush()
                os.fsync(handle.fileno())
            sha256 = _file_sha256(temporary)
            byte_count = int(temporary.stat().st_size)
            os.replace(temporary, final_path)
        finally:
            if temporary.exists():
                temporary.unlink()
        entry = {
            "filename": filename,
            "side": side,
            "group_count": int(endpoint_ids.size),
            "occurrence_count": int(occurrence_row_ids.size),
            "union_node_count": int(union_nodes.size),
            "byte_count": byte_count,
            "sha256": sha256,
            "local_index_dtype": local_index_dtype,
        }
        self._published_files.append(final_path)
        self._shards.append(entry)
        self._groups = []
        self._buffered_occurrences = 0
        self._buffered_union_nodes = 0
        self._buffer_side = None

    def finish(self, *, verify_shards="lazy"):
        self._require_open()
        self._flush()
        if not bool(self._seen.all()):
            missing = np.argwhere(~self._seen)
            (side, row) = (int(value) for value in missing[0])
            raise ValueError(f"Endpoint-grouped writer is missing side={side}, row={row}.")
        manifest = {
            "manifest_version": STREAMED_GROUPED_MANIFEST_VERSION,
            "format": ENDPOINT_GROUPED_FORMAT,
            "split": self.split,
            "num_rows": int(self.positives.size(0)),
            "negatives_per_row": 2 * self.negatives_per_side,
            "negatives_per_side": self.negatives_per_side,
            "num_nodes": self.num_nodes,
            "positive_sha256": self.positive_sha256,
            "candidate_recipe_sha256": self.candidate_recipe_sha256,
            "validation_contract": dict(_ENDPOINT_VALIDATION_CONTRACT),
            "shard_limits": {
                "occurrence_rows": self.occurrence_rows_per_shard,
                "endpoint_groups": self.groups_per_shard,
                "union_nodes": self.union_nodes_per_shard,
            },
            "shards": list(self._shards),
        }
        manifest["manifest_sha256"] = _manifest_digest(manifest)
        _atomic_json(self.manifest_path, manifest)
        self._finished = True
        current_names = {entry["filename"] for entry in self._shards}
        for path in self._prior_shard_paths:
            if path.name in current_names:
                continue
            try:
                path.unlink()
            except OSError:
                pass
        return EndpointGroupedNegatives(
            self.manifest_path, self.positives, expected_recipe_sha256=self.candidate_recipe_sha256, verify_shards=verify_shards
        )

    def __exit__(self, exc_type, exc, traceback):
        del exc, traceback
        if exc_type is not None or not self._finished:
            for path in self._published_files:
                try:
                    path.unlink()
                except FileNotFoundError:
                    pass
        if self._lock_handle is not None:
            _unlock(self._lock_handle)
            self._lock_handle = None
        return False


class IndependentSideBufferedEndpointGroupedNegativeWriter(EndpointGroupedNegativeWriter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._side_pending = {0: [], 1: []}
        self._side_pending_occurrences = {0: 0, 1: 0}
        self._side_pending_union_nodes = {0: 0, 1: 0}

    @staticmethod
    def _owned_long_tensor(value, *, flatten=False):
        tensor = torch.as_tensor(value, dtype=torch.long).detach().cpu()
        return (tensor.view(-1) if flatten else tensor).clone()

    def _drain_side(self, side):
        side = int(side)
        pending = self._side_pending[side]
        if not pending:
            return
        self._side_pending[side] = []
        self._side_pending_occurrences[side] = 0
        self._side_pending_union_nodes[side] = 0
        for prevalidated, record in pending:
            if prevalidated:
                self._buffer_record(*record)
            else:
                super().append_endpoint_group(*record)

    def _queue_side_record(self, side, record, *, prevalidated):
        row_count, union_count = len(record[2]), len(record[3])
        pending = self._side_pending[side]
        projected_occurrences = self._side_pending_occurrences[side] + row_count
        projected_union_nodes = self._side_pending_union_nodes[side] + union_count
        if pending and (
            projected_occurrences > self.occurrence_rows_per_shard
            or len(pending) + 1 > self.groups_per_shard
            or projected_union_nodes > self.union_nodes_per_shard
        ):
            self._drain_side(side)
            pending = self._side_pending[side]
        pending.append((bool(prevalidated), record))
        self._side_pending_occurrences[side] += row_count
        self._side_pending_union_nodes[side] += union_count
        if (
            self._side_pending_occurrences[side] >= self.occurrence_rows_per_shard
            or len(pending) >= self.groups_per_shard
            or self._side_pending_union_nodes[side] >= self.union_nodes_per_shard
        ):
            self._drain_side(side)

    def append_endpoint_group(self, side, endpoint, row_ids, union_candidate_nodes, candidate_local_indices, hard_prefix_count):
        self._require_open()
        side = int(side)
        _require(side in (0, 1), "side must be 0 (left) or 1 (right).")
        rows = self._owned_long_tensor(row_ids, flatten=True)
        union = self._owned_long_tensor(union_candidate_nodes, flatten=True)
        local = self._owned_long_tensor(candidate_local_indices)
        prefix = self._owned_long_tensor(hard_prefix_count, flatten=True)
        self._queue_side_record(side, (side, int(endpoint), rows, union, local, prefix), prevalidated=False)

    def append_prevalidated_endpoint_group(self, side, endpoint, row_ids, union_candidate_nodes, candidate_local_indices, hard_prefix_count):
        self._require_open()
        side = int(side)
        endpoint = int(endpoint)
        rows = np.asarray(row_ids, dtype=np.uint32).reshape(-1).copy()
        union = np.asarray(union_candidate_nodes, dtype=np.uint32).reshape(-1).copy()
        local = np.asarray(candidate_local_indices, dtype=np.uint16).copy()
        prefix = np.asarray(hard_prefix_count, dtype=np.uint16).reshape(-1).copy()
        self._queue_side_record(side, (side, endpoint, rows, union, local, prefix), prevalidated=True)

    def finish(self, *, verify_shards="lazy"):
        self._require_open()
        self._drain_side(0)
        self._drain_side(1)
        return super().finish(verify_shards=verify_shards)


def _endpoint_accelerator_path(manifest_path):
    canonical_path = Path(manifest_path).expanduser().resolve()
    return canonical_path.with_name(f"{canonical_path.name}.accelerator-v{_ENDPOINT_ACCELERATOR_VERSION}.json")


def _read_manifest_with_file_sha256(manifest_path):
    path = Path(manifest_path).expanduser().resolve()
    raw = path.read_bytes()
    payload = json.loads(raw.decode("utf-8"))
    _require(isinstance(payload, dict), "Endpoint-grouped manifest must contain an object.")
    return (payload, hashlib.sha256(raw).hexdigest())


def _endpoint_manifest_identity(payload):
    _require(int(payload.get("manifest_version", -1)) == STREAMED_GROUPED_MANIFEST_VERSION and payload.get("format") == ENDPOINT_GROUPED_FORMAT, "Unsupported endpoint-grouped manifest format.")
    manifest_sha256 = _normalized_sha256(payload.get("manifest_sha256"), "manifest_sha256")
    _require(_manifest_digest(payload) == manifest_sha256, "Endpoint-grouped manifest SHA-256 mismatch.")
    _require(payload.get("validation_contract") == _ENDPOINT_VALIDATION_CONTRACT, "Endpoint-grouped artifact lacks the required validation contract.")
    limits = payload.get("shard_limits")
    _require(isinstance(limits, Mapping), "Endpoint-grouped shard limits are missing.")
    negatives_per_side = int(payload.get("negatives_per_side", 0))
    negatives_per_row = int(payload.get("negatives_per_row", 0))
    _require(0 < negatives_per_side <= 65535 and negatives_per_row == 2 * negatives_per_side, "Endpoint-grouped negative widths are invalid.")
    identity = {
        "manifest_version": STREAMED_GROUPED_MANIFEST_VERSION,
        "format": ENDPOINT_GROUPED_FORMAT,
        "split": str(payload.get("split", "")),
        "num_rows": int(payload.get("num_rows", -1)),
        "negatives_per_row": negatives_per_row,
        "negatives_per_side": negatives_per_side,
        "num_nodes": int(payload.get("num_nodes", -1)),
        "positive_sha256": _normalized_sha256(payload.get("positive_sha256"), "positive_sha256"),
        "candidate_recipe_sha256": _normalized_sha256(payload.get("candidate_recipe_sha256"), "candidate_recipe_sha256"),
        "validation_contract": dict(_ENDPOINT_VALIDATION_CONTRACT),
        "shard_limits": {
            "occurrence_rows": int(limits.get("occurrence_rows", 0)),
            "endpoint_groups": int(limits.get("endpoint_groups", 0)),
            "union_nodes": int(limits.get("union_nodes", 0)),
        },
    }
    _require(0 < identity["num_rows"] <= 2**32, "Endpoint-grouped row ids cannot use uint32 storage.")
    _require(1 < identity["num_nodes"] <= 2**32, "Endpoint-grouped node ids cannot use uint32 storage.")
    _require(min(identity["shard_limits"].values()) > 0, "Endpoint-grouped shard limits are invalid.")
    return (identity, manifest_sha256)


def _bound_endpoint_accelerator_matches(canonical_path, canonical_payload, canonical_file_sha256, accelerator_payload):
    try:
        (canonical_identity, canonical_manifest_sha256) = _endpoint_manifest_identity(canonical_payload)
        (accelerator_identity, _) = _endpoint_manifest_identity(accelerator_payload)
        binding = accelerator_payload.get(_ENDPOINT_ACCELERATOR_KEY)
        if not isinstance(binding, Mapping):
            return False
        return (
            accelerator_identity == canonical_identity
            and int(binding.get("version", -1)) == _ENDPOINT_ACCELERATOR_VERSION
            and (str(binding.get("layout", "")) == _ENDPOINT_ACCELERATOR_LAYOUT)
            and (str(binding.get("source_manifest_name", "")) == Path(canonical_path).name)
            and (_normalized_sha256(binding.get("source_manifest_file_sha256"), "source_manifest_file_sha256") == canonical_file_sha256)
            and (_normalized_sha256(binding.get("source_manifest_sha256"), "source_manifest_sha256") == canonical_manifest_sha256)
            and bool(_normalized_sha256(binding.get("logical_groups_sha256"), "logical_groups_sha256"))
        )
    except (KeyError, TypeError, ValueError):
        return False


def _load_bound_endpoint_accelerator(canonical_path, canonical_payload, canonical_file_sha256, positives, expected_recipe_sha256, verify_shards):
    accelerator_path = _endpoint_accelerator_path(canonical_path)
    if not accelerator_path.exists():
        return None
    try:
        accelerator_payload, _ = _read_manifest_with_file_sha256(accelerator_path)
        if not _bound_endpoint_accelerator_matches(canonical_path, canonical_payload, canonical_file_sha256, accelerator_payload):
            return None
        accelerated = EndpointGroupedNegatives(
            accelerator_path, positives, expected_recipe_sha256=expected_recipe_sha256, verify_shards=verify_shards
        )
        return accelerator_payload, accelerated
    except (OSError, TypeError, ValueError):
        return None


def _endpoint_compaction_lower_bound(payload):
    limits = payload.get("shard_limits")
    shards = payload.get("shards")
    if not isinstance(limits, Mapping) or not isinstance(shards, list):
        return 0
    bounds = tuple(int(limits.get(name, 0)) for name in ("occurrence_rows", "endpoint_groups", "union_nodes"))
    if min(bounds) <= 0:
        return 0
    totals = [[0, 0, 0], [0, 0, 0]]
    try:
        for shard in shards:
            side = int(shard["side"])
            if side not in (0, 1):
                return 0
            for index, name in enumerate(("occurrence_count", "group_count", "union_node_count")):
                totals[side][index] += int(shard[name])
    except (KeyError, TypeError, ValueError):
        return 0
    return sum(max((total + limit - 1) // limit for total, limit in zip(totals[side], bounds)) for side in (0, 1))


def _endpoint_manifest_is_fragmented(payload):
    shards = payload.get("shards")
    lower_bound = _endpoint_compaction_lower_bound(payload)
    return isinstance(shards, list) and lower_bound > 0 and len(shards) > max(4096, 4 * lower_bound)


def _iter_endpoint_group_records(chunk):
    endpoint_ids = chunk.endpoints.detach().cpu().numpy()
    union_rowptr = chunk.union_rowptr.detach().cpu().numpy()
    union_nodes = chunk.union_nodes.detach().cpu().numpy()
    occurrence_group = chunk.occurrence_endpoint_index.detach().cpu().numpy()
    occurrence_rows = chunk.occurrence_row_ids.detach().cpu().numpy()
    local_indices = chunk.candidate_local_indices.detach().cpu().numpy()
    hard_prefix = chunk.hard_prefix_count.detach().cpu().numpy()
    position = 0
    for group_index, endpoint in enumerate(endpoint_ids):
        start = position
        while position < occurrence_group.size and int(occurrence_group[position]) == group_index:
            position += 1
        if position == start:
            raise ValueError("Endpoint-grouped shard contains an empty group.")
        union_start = int(union_rowptr[group_index])
        union_end = int(union_rowptr[group_index + 1])
        yield (
            int(chunk.side),
            int(endpoint),
            occurrence_rows[start:position],
            union_nodes[union_start:union_end],
            local_indices[start:position],
            hard_prefix[start:position],
        )
    if position != int(occurrence_group.size):
        raise ValueError("Endpoint-grouped occurrence records are not grouped contiguously.")


def _update_endpoint_group_digest(digest, record):
    (side, endpoint, rows, union, local, prefix) = record
    digest.update(b"endpoint-logical-group-v1\x00")
    digest.update(np.asarray([int(side)], dtype=np.uint8).tobytes())
    arrays = (
        (b"endpoint", np.asarray([endpoint], dtype="<u4")),
        (b"rows", np.asarray(rows, dtype="<u4")),
        (b"union", np.asarray(union, dtype="<u4")),
        (b"local", np.asarray(local, dtype="<u2")),
        (b"prefix", np.asarray(prefix, dtype="<u2")),
    )
    for label, array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(label)
        digest.update(np.asarray(contiguous.shape, dtype="<u8").tobytes())
        digest.update(memoryview(contiguous).cast("B"))


def _endpoint_logical_digest(digests, group_counts):
    combined = hashlib.sha256()
    combined.update(b"endpoint-logical-groups-root-v1\x00")
    for side in (0, 1):
        combined.update(np.asarray([side], dtype=np.uint8).tobytes())
        combined.update(np.asarray([group_counts[side]], dtype="<u8").tobytes())
        combined.update(digests[side].digest())
    return combined.hexdigest()


def _logical_digest_for_endpoint_object(candidate_object):
    digests = {0: hashlib.sha256(), 1: hashlib.sha256()}
    group_counts = {0: 0, 1: 0}
    for chunk in candidate_object.iter_endpoint_group_chunks():
        for record in _iter_endpoint_group_records(chunk):
            side = int(record[0])
            _update_endpoint_group_digest(digests[side], record)
            group_counts[side] += 1
    return _endpoint_logical_digest(digests, group_counts)


def _delete_manifest_shards(manifest_path, payload):
    directory = Path(manifest_path).expanduser().resolve().parent
    removed = 0
    for shard in payload.get("shards", []):
        filename = str(shard.get("filename", ""))
        if not filename or Path(filename).name != filename:
            continue
        try:
            (directory / filename).unlink()
            removed += 1
        except OSError:
            pass
    if removed:
        _fsync_directory(directory)
    return removed


def ensure_endpoint_grouped_accelerator(manifest_path, positives, *, expected_recipe_sha256=None, verify_shards="lazy", remove_source_shards=False):
    canonical_path = Path(manifest_path).expanduser().resolve()
    accelerator_path = _endpoint_accelerator_path(canonical_path)
    lock_path = canonical_path.with_suffix(canonical_path.suffix + ".lock")
    lock_handle = open(lock_path, "a+b")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        (canonical_payload, canonical_file_sha256) = _read_manifest_with_file_sha256(canonical_path)
        (canonical_identity, canonical_manifest_sha256) = _endpoint_manifest_identity(canonical_payload)
        loaded = _load_bound_endpoint_accelerator(
            canonical_path, canonical_payload, canonical_file_sha256, positives, expected_recipe_sha256, verify_shards
        )
        if loaded is not None:
            accelerator_payload, accelerated = loaded
            try:
                if remove_source_shards:
                    actual_logical_sha256 = _logical_digest_for_endpoint_object(accelerated)
                    expected_logical_sha256 = _normalized_sha256(
                        accelerator_payload[_ENDPOINT_ACCELERATOR_KEY].get("logical_groups_sha256"), "logical_groups_sha256"
                    )
                    if actual_logical_sha256 != expected_logical_sha256:
                        raise ValueError("Endpoint accelerator logical digest is invalid.")
                    _delete_manifest_shards(canonical_path, canonical_payload)
                return accelerated
            except (OSError, TypeError, ValueError):
                pass
        source = EndpointGroupedNegatives(canonical_path, positives, expected_recipe_sha256=expected_recipe_sha256, verify_shards="lazy")
        source_digests = {0: hashlib.sha256(), 1: hashlib.sha256()}
        source_group_counts = {0: 0, 1: 0}
        print(
            f"Compacting endpoint-grouped candidate cache without regenerating candidates: {canonical_path} ({len(source.shards)} shards)",
            flush=True,
        )
        with IndependentSideBufferedEndpointGroupedNegativeWriter(
            accelerator_path,
            positives,
            num_nodes=canonical_identity["num_nodes"],
            split=canonical_identity["split"],
            candidate_recipe_sha256=canonical_identity["candidate_recipe_sha256"],
            negatives_per_side=canonical_identity["negatives_per_side"],
            occurrence_rows_per_shard=canonical_identity["shard_limits"]["occurrence_rows"],
            groups_per_shard=canonical_identity["shard_limits"]["endpoint_groups"],
            union_nodes_per_shard=canonical_identity["shard_limits"]["union_nodes"],
        ) as writer:
            shard_count = 0
            for chunk in source.iter_endpoint_group_chunks():
                shard_count += 1
                for record in _iter_endpoint_group_records(chunk):
                    side = int(record[0])
                    _update_endpoint_group_digest(source_digests[side], record)
                    source_group_counts[side] += 1
                    writer.append_prevalidated_endpoint_group(*record)
                if shard_count % 25000 == 0:
                    print(f"Endpoint cache compaction verified/read {shard_count}/{len(source.shards)} source shards.", flush=True)
            compact_object = writer.finish(verify_shards="lazy")
        source_logical_sha256 = _endpoint_logical_digest(source_digests, source_group_counts)
        compact_logical_sha256 = _logical_digest_for_endpoint_object(compact_object)
        if compact_logical_sha256 != source_logical_sha256:
            raise ValueError("Compacted endpoint cache changed the logical candidate records.")
        (accelerator_payload, _) = _read_manifest_with_file_sha256(accelerator_path)
        accelerator_payload[_ENDPOINT_ACCELERATOR_KEY] = {
            "version": _ENDPOINT_ACCELERATOR_VERSION,
            "layout": _ENDPOINT_ACCELERATOR_LAYOUT,
            "source_manifest_name": canonical_path.name,
            "source_manifest_file_sha256": canonical_file_sha256,
            "source_manifest_sha256": canonical_manifest_sha256,
            "logical_groups_sha256": source_logical_sha256,
        }
        accelerator_payload["manifest_sha256"] = _manifest_digest(accelerator_payload)
        _atomic_json(accelerator_path, accelerator_payload)
        if not _bound_endpoint_accelerator_matches(canonical_path, canonical_payload, canonical_file_sha256, accelerator_payload):
            raise ValueError("Compacted endpoint cache binding is invalid.")
        accelerated = EndpointGroupedNegatives(
            accelerator_path, positives, expected_recipe_sha256=expected_recipe_sha256, verify_shards=verify_shards
        )
        removed = 0
        if remove_source_shards:
            removed = _delete_manifest_shards(canonical_path, canonical_payload)
        print(
            f"Endpoint cache compaction complete: {len(source.shards)} source shards -> {len(accelerated.shards)} compact shards; removed {removed} redundant source shards.",
            flush=True,
        )
        return accelerated
    finally:
        _unlock(lock_handle)


def load_streamed_grouped_negatives(manifest_path, positives, *, expected_recipe_sha256=None, verify_shards="lazy"):
    canonical_path = Path(manifest_path).expanduser().resolve()
    (canonical_payload, canonical_file_sha256) = _read_manifest_with_file_sha256(canonical_path)
    loaded = _load_bound_endpoint_accelerator(
        canonical_path, canonical_payload, canonical_file_sha256, positives, expected_recipe_sha256, verify_shards
    )
    if loaded is not None:
        return loaded[1]
    if _endpoint_manifest_is_fragmented(canonical_payload):
        try:
            return ensure_endpoint_grouped_accelerator(
                canonical_path,
                positives,
                expected_recipe_sha256=expected_recipe_sha256,
                verify_shards=verify_shards,
                remove_source_shards=False,
            )
        except Exception as exc:
            print(f"Endpoint cache compaction was unavailable; using the canonical shards: {exc}", flush=True)
    return EndpointGroupedNegatives(canonical_path, positives, expected_recipe_sha256=expected_recipe_sha256, verify_shards=verify_shards)
