import filecmp
import hashlib
import os
import shutil
import tempfile
import torch
from torch_geometric.utils import to_undirected, coalesce
from torch_sparse import SparseTensor
from ogb.linkproppred import PygLinkPropPredDataset

_NEGATIVE_CACHE_VERSION = 12
_CITATION2_COMPACT_SPLIT_CACHE_VERSION = 2
_CITATION2_COMPACT_SPLIT_SAMPLE_BYTES = 64 * 1024
_HEART_PPR_EPS = {
    "ogbl-collab": 1e-05,
    "ogbl-ddi": 1e-05,
    "ogbl-ppa": 5e-06,
    "ogbl-citation2": 5e-05,
    "cora": 1e-07,
    "citeseer": 1e-07,
    "pubmed": 1e-05,
}


def _heart_ppr_eps(data_name):
    return float(_HEART_PPR_EPS.get(str(data_name).lower(), 5e-05))


_FULL_GRAPH_POOL_VALUES = {"all", "full", "graph", "entire", "full-graph", "full_graph", "entire-graph", "entire_graph"}


def parse_pool_argument(pool):
    value = str(pool).strip().lower()
    if value in _FULL_GRAPH_POOL_VALUES:
        return "all"
    try:
        requested_pool = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("pool must be a positive integer or 'all'.") from exc
    if requested_pool <= 0:
        raise ValueError("pool must be a positive integer or 'all'.")
    return requested_pool


def _resolve_pool_request(pool, num_nodes):
    if pool == "all":
        return (int(num_nodes), "all", True)
    return (int(pool), str(int(pool)), False)


def _has_ogb_payload(path):
    return os.path.isfile(os.path.join(path, "processed", "geometric_data_processed.pt")) and os.path.isdir(os.path.join(path, "split"))


def _merge_directory_tree(source, destination):
    source = os.path.abspath(source)
    destination = os.path.abspath(destination)
    if source == destination or not os.path.lexists(source):
        return
    os.makedirs(destination, exist_ok=True)
    for name in os.listdir(source):
        source_path = os.path.join(source, name)
        destination_path = os.path.join(destination, name)
        if not os.path.lexists(destination_path):
            shutil.move(source_path, destination_path)
            continue
        if os.path.isdir(source_path) and os.path.isdir(destination_path):
            _merge_directory_tree(source_path, destination_path)
            continue
        if os.path.isfile(source_path) and os.path.isfile(destination_path):
            if filecmp.cmp(source_path, destination_path, shallow=False):
                os.remove(source_path)
                continue
        raise FileExistsError(
            f"Cannot consolidate OGB dataset directories because both paths contain different entries named {name!r}: {source_path} and {destination_path}."
        )
    if os.path.isdir(source) and (not os.listdir(source)):
        os.rmdir(source)


def _ogb_meta_for_directory(data_name, directory):
    import pandas as pd
    import ogb.linkproppred.dataset_pyg as dataset_pyg_module

    master_path = os.path.join(os.path.dirname(dataset_pyg_module.__file__), "master.csv")
    master = pd.read_csv(master_path, index_col=0, keep_default_na=False)
    meta = master[data_name].to_dict()
    meta["dir_path"] = os.path.abspath(directory)
    return meta


def _canonical_ppa_directory(root):
    root = os.path.abspath(os.path.expanduser(str(root)))
    canonical = os.path.join(root, "ogbl-ppa")
    legacy_dirs = (os.path.join(root, "ogbl_ppa"), os.path.join(root, "ogbl_ppa_pyg"))
    for legacy in legacy_dirs:
        if not os.path.lexists(legacy):
            continue
        print(f"Consolidating legacy OGB directory {legacy} into {canonical}.", flush=True)
        _merge_directory_tree(legacy, canonical)
    return canonical


def _load_ppa_dataset_from_dashed_directory(root):
    root = os.path.abspath(os.path.expanduser(str(root)))
    os.makedirs(root, exist_ok=True)
    canonical = _canonical_ppa_directory(root)
    if _has_ogb_payload(canonical):
        dataset = PygLinkPropPredDataset(name="ogbl-ppa", meta_dict=_ogb_meta_for_directory("ogbl-ppa", canonical))
        return dataset
    staging_root = tempfile.mkdtemp(prefix=".ogbl-ppa-download-", dir=root)
    try:
        staged = PygLinkPropPredDataset(name="ogbl-ppa", root=staging_root)
        _ = staged[0]
        _ = staged.get_edge_split()
        _merge_directory_tree(staged.root, canonical)
    finally:
        shutil.rmtree(staging_root, ignore_errors=True)
    if not _has_ogb_payload(canonical):
        raise RuntimeError(f"OGB download completed without a usable dataset at {canonical}.")
    return PygLinkPropPredDataset(name="ogbl-ppa", meta_dict=_ogb_meta_for_directory("ogbl-ppa", canonical))


def _load_dataset(data_name, root):
    if data_name == "ogbl-ppa":
        dataset = _load_ppa_dataset_from_dashed_directory(root)
    else:
        dataset = PygLinkPropPredDataset(name=data_name, root=root)
    data = dataset[0]
    split_edge = dataset.get_edge_split()
    for split_name in ("valid", "test"):
        split_payload = split_edge.get(split_name)
        if isinstance(split_payload, dict):
            split_payload.pop("edge_neg", None)
            split_payload.pop("target_node_neg", None)
    num_nodes = int(data.num_nodes)
    if getattr(data, "x", None) is None:
        if data_name == "ogbl-ddi":
            data.x = torch.zeros((num_nodes, 1), dtype=torch.float)
        else:
            data.x = torch.ones((num_nodes, 1), dtype=torch.float)
    elif data.x.dtype != torch.float:
        data.x = data.x.to(torch.float)
    return (data, split_edge, _ogb_source_artifact_identity(dataset))


def _citation2_split_source_paths(dataset):
    split_type = str(dataset.meta_info["split"])
    split_dir = os.path.join(dataset.root, "split", split_type)
    combined = os.path.join(split_dir, "split_dict.pt")
    if os.path.isfile(combined):
        return (combined,)
    paths = tuple((os.path.join(split_dir, f"{name}.pt") for name in ("train", "valid", "test")))
    missing = [path for path in paths if not os.path.isfile(path)]
    if missing:
        raise FileNotFoundError("Citation2 split artifacts are missing: " + ", ".join(missing))
    return paths


def _ogb_source_artifact_paths(dataset):
    paths = set(_citation2_split_source_paths(dataset))
    processed_paths = getattr(dataset, "processed_paths", ())
    for path in processed_paths:
        if os.path.isfile(path):
            paths.add(os.path.abspath(path))
    fallback = os.path.join(dataset.root, "processed", "geometric_data_processed.pt")
    if os.path.isfile(fallback):
        paths.add(os.path.abspath(fallback))
    if not paths:
        raise FileNotFoundError(f"No immutable OGB source artifacts were found under {dataset.root}.")
    return tuple(sorted(paths))


def _artifact_identity(paths, root, method, include_root=False):
    entries = []
    digest = hashlib.sha256()
    for path in paths:
        stat = os.stat(path, follow_symlinks=True)
        relative = os.path.relpath(path, root)
        entry = {
            "path": relative,
            "device": int(stat.st_dev),
            "inode": int(stat.st_ino),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
            "ctime_ns": int(stat.st_ctime_ns),
            "sample_sha256": _sampled_file_sha256(path, _CITATION2_COMPACT_SPLIT_SAMPLE_BYTES),
        }
        entries.append(entry)
        fields = (entry[key] for key in ("path", "device", "inode", "size", "mtime_ns", "ctime_ns", "sample_sha256"))
        digest.update(("\x00".join(map(str, fields)) + "\n").encode("utf-8"))
    identity = {"method": method, "sample_bytes": int(_CITATION2_COMPACT_SPLIT_SAMPLE_BYTES), "files": entries, "sha256": digest.hexdigest()}
    if include_root:
        identity["dataset_root"] = os.path.abspath(root)
        identity = {"method": identity.pop("method"), "dataset_root": identity.pop("dataset_root"), **identity}
    return identity


def _ogb_source_artifact_identity(dataset):
    method = "immutable-ogb-files-dev-inode-size-mtime-ctime+three-sample-sha256-v1"
    return _artifact_identity(_ogb_source_artifact_paths(dataset), dataset.root, method, include_root=True)


def _sampled_file_sha256(path, sample_bytes):
    size = int(os.path.getsize(path))
    width = max(1, int(sample_bytes))
    offsets = {0, max(0, size - width), max(0, (size - width) // 2)}
    digest = hashlib.sha256()
    digest.update(f"size={size};".encode("utf-8"))
    with open(path, "rb") as handle:
        for offset in sorted(offsets):
            handle.seek(offset)
            block = handle.read(min(width, max(0, size - offset)))
            digest.update(f"offset={offset};bytes={len(block)};".encode("utf-8"))
            digest.update(block)
    return digest.hexdigest()


def _citation2_raw_split_fingerprint(dataset):
    return _artifact_identity(
        _citation2_split_source_paths(dataset), dataset.root, "dev-inode-size-mtime-ctime-three-sample-sha256-v2"
    )


def _citation2_compact_split_cache_path(dataset):
    cache_dir = os.path.join(dataset.root, "heart_cache")
    os.makedirs(cache_dir, exist_ok=True)
    return os.path.join(cache_dir, f"ogbl-citation2_generated_compact_split_v{_CITATION2_COMPACT_SPLIT_CACHE_VERSION}.pt")


def _compact_citation2_split(split_edge):
    compact = {}
    for split_name in ("train", "valid", "test"):
        compact[split_name] = {
            field: split_edge[split_name][field].detach().to(device="cpu", dtype=torch.long).contiguous()
            for field in ("source_node", "target_node")
        }
    return _validate_compact_citation2_split(compact)


def _validate_compact_citation2_split(split_edge):
    if not isinstance(split_edge, dict):
        raise TypeError("Compact Citation2 split must be a mapping.")
    for split_name in ("train", "valid", "test"):
        split = split_edge.get(split_name)
        if not isinstance(split, dict):
            raise TypeError(f"Compact Citation2 split lacks {split_name!r}.")
        source = split.get("source_node")
        target = split.get("target_node")
        for field, value in (("source_node", source), ("target_node", target)):
            if not torch.is_tensor(value):
                raise TypeError(f"Compact Citation2 {split_name}.{field} must be a tensor.")
            if value.dtype != torch.long or value.device.type != "cpu":
                raise TypeError(f"Compact Citation2 {split_name}.{field} must be CPU int64.")
            if value.dim() != 1 or not value.is_contiguous():
                raise ValueError(f"Compact Citation2 {split_name}.{field} must be contiguous 1-D.")
        if source.numel() != target.numel():
            raise ValueError(f"Compact Citation2 {split_name} endpoints are misaligned.")
        if split_name == "train" and source.numel() == 0:
            raise ValueError("Compact Citation2 training split is empty.")
    return split_edge


def _load_compact_citation2_payload(path):
    try:
        return (torch.load(path, map_location="cpu", mmap=True, weights_only=True), True)
    except TypeError:
        try:
            return (torch.load(path, map_location="cpu", weights_only=True), False)
        except TypeError:
            return (torch.load(path, map_location="cpu"), False)


def _try_load_compact_citation2_split(path, raw_fingerprint):
    if not os.path.isfile(path):
        return None
    try:
        (payload, memory_mapped) = _load_compact_citation2_payload(path)
        metadata = payload.get("metadata") if isinstance(payload, dict) else None
        expected = {
            "format_version": int(_CITATION2_COMPACT_SPLIT_CACHE_VERSION),
            "dataset": "ogbl-citation2",
            "retained_fields": ["source_node", "target_node"],
            "excluded_fields": ["target_node_neg"],
            "raw_split_fingerprint": raw_fingerprint,
        }
        if not isinstance(metadata, dict):
            raise ValueError("missing metadata")
        mismatches = [key for (key, value) in expected.items() if metadata.get(key) != value]
        if mismatches:
            raise ValueError("metadata mismatch: " + ", ".join(sorted(mismatches)))
        split_edge = _validate_compact_citation2_split(payload.get("split_edge"))
        storage = "memory-mapped" if memory_mapped else "resident"
        print(f"Loaded {storage} compact Citation2 split cache: {path}", flush=True)
        return split_edge
    except Exception as exc:
        print(f"Ignoring unreadable compact Citation2 split cache {path}: {exc}", flush=True)
        return None


def _fsync_file(path):
    with open(path, "rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path):
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def _save_compact_citation2_split(path, split_edge, raw_fingerprint):
    directory = os.path.dirname(os.path.abspath(path))
    (descriptor, temporary_path) = tempfile.mkstemp(prefix=os.path.basename(path) + ".tmp.", dir=directory)
    os.close(descriptor)
    try:
        torch.save(
            {
                "metadata": {
                    "format_version": int(_CITATION2_COMPACT_SPLIT_CACHE_VERSION),
                    "dataset": "ogbl-citation2",
                    "retained_fields": ["source_node", "target_node"],
                    "excluded_fields": ["target_node_neg"],
                    "raw_split_fingerprint": raw_fingerprint,
                },
                "split_edge": split_edge,
            },
            temporary_path,
        )
        _fsync_file(temporary_path)
        os.replace(temporary_path, path)
        _fsync_directory(directory)
        print(f"Saved compact Citation2 split cache: {path}", flush=True)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


def _load_or_build_compact_citation2_split(dataset):
    cache_path = _citation2_compact_split_cache_path(dataset)
    raw_fingerprint = _citation2_raw_split_fingerprint(dataset)
    cached = _try_load_compact_citation2_split(cache_path, raw_fingerprint)
    if cached is not None:
        return cached
    lock_path = cache_path + ".lock"
    with open(lock_path, "a+b") as lock_file:
        try:
            import fcntl

            print(f"Acquiring compact Citation2 split-cache lock: {lock_path}", flush=True)
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError) as exc:
            print(f"WARNING: compact Citation2 split-cache locking unavailable: {exc}", flush=True)
        try:
            raw_fingerprint = _citation2_raw_split_fingerprint(dataset)
            cached = _try_load_compact_citation2_split(cache_path, raw_fingerprint)
            if cached is not None:
                return cached
            print("Building compact Citation2 split cache; this first build loads the stock OGB split once.", flush=True)
            split_edge = _compact_citation2_split(dataset.get_edge_split())
            fingerprint_after = _citation2_raw_split_fingerprint(dataset)
            if fingerprint_after != raw_fingerprint:
                raise RuntimeError(
                    "Citation2 source split changed while its compact cache was being built; refusing to publish a mixed artifact."
                )
            _save_compact_citation2_split(cache_path, split_edge, raw_fingerprint)
            return split_edge
        finally:
            try:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except (ImportError, OSError):
                pass


def _pp_unique_undirected_uv(edge_index, num_nodes):
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long)
    if edge_index.dim() == 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t().contiguous()
    (row, col) = edge_index
    keep = row != col
    (row, col) = (row[keep], col[keep])
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    (uv, _) = coalesce(torch.stack([lo, hi], dim=0), None, num_nodes, num_nodes)
    return uv.contiguous()


def _undirected_filter_csr(edge_parts, num_nodes, *, base_adj=None):
    parts = [
        value.detach().to(device="cpu", dtype=torch.long).contiguous() for value in edge_parts if torch.is_tensor(value) and value.numel()
    ]
    if parts:
        extra_pos = torch.cat(parts, dim=0)
        extra_uv = _pp_unique_undirected_uv(extra_pos, int(num_nodes))
        extra_edges = torch.cat([extra_uv, extra_uv.flip(0)], dim=1)
        extra_adj = SparseTensor.from_edge_index(
            extra_edges, torch.ones(extra_edges.size(1), dtype=torch.float32), (int(num_nodes), int(num_nodes))
        ).coalesce()
        eligibility_adj = extra_adj if base_adj is None else base_adj + extra_adj
    elif base_adj is not None:
        eligibility_adj = base_adj
    else:
        eligibility_adj = SparseTensor(
            row=torch.empty(0, dtype=torch.long), col=torch.empty(0, dtype=torch.long), sparse_sizes=(int(num_nodes), int(num_nodes))
        )
    (rowptr, col, _) = eligibility_adj.csr()
    return (rowptr.cpu(), col.cpu())


def _ensure_heart_eligibility_filters(out, data_name):
    required = (
        "heart_valid_out_rowptr",
        "heart_valid_out_col",
        "heart_valid_in_rowptr",
        "heart_valid_in_col",
        "heart_test_out_rowptr",
        "heart_test_out_col",
        "heart_test_in_rowptr",
        "heart_test_in_col",
    )
    if all((key in out for key in required)):
        return tuple((out[key] for key in required))
    name = str(data_name).lower()
    train_rowptr = out["csr_train_rowptr"].cpu()
    train_col = out["csr_train_col"].cpu()
    test_rowptr = out["csr_tv_rowptr"].cpu()
    test_col = out["csr_tv_col"].cpu()
    valid_values = (train_rowptr, train_col, train_rowptr, train_col)
    test_values = (test_rowptr, test_col, test_rowptr, test_col)
    if name == "ogbl-citation2":
        policy = "released-shared-row-view"
        orientation = "directed-shared-source-rows"
    elif name == "ogbl-collab":
        policy = "released-observed-history-filter"
        orientation = "undirected-temporal"
    else:
        policy = "released-observed-graph"
        orientation = "undirected-canonical"
    for prefix, values in (("valid", valid_values), ("test", test_values)):
        for suffix, value in zip(("out_rowptr", "out_col", "in_rowptr", "in_col"), values):
            out[f"heart_{prefix}_{suffix}"] = value
    out["heart_eligibility_policy"] = policy
    out["heart_eligibility_orientation"] = orientation
    out["heart_eligibility_scope"] = "released-observed-graph-plus-query"
    return tuple((out[key] for key in required))


def _sample_positive_rows(pos, cap, seed):
    if cap and pos.size(0) > cap:
        generator = torch.Generator().manual_seed(seed)
        index = torch.randperm(pos.size(0), generator=generator)[:cap]
        pos = pos[index]
    return pos.contiguous()


def _sample_train_val(train_pos, n, seed):
    if train_pos.size(0) == 0 or n <= 0:
        return train_pos[:0].clone().contiguous()
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(train_pos.size(0), generator=g)[: min(n, train_pos.size(0))]
    return train_pos[idx].contiguous()


def _build_adj(train_uv, num_nodes):
    train_edge_index = torch.cat([train_uv, train_uv.flip(0)], dim=1)
    adj = SparseTensor.from_edge_index(train_edge_index, torch.ones(train_edge_index.size(1), dtype=torch.float), (num_nodes, num_nodes))
    return (train_edge_index, adj)


def _build_collab_model_adj(train_pos, train_split, num_nodes):
    edge_index = torch.cat([train_pos.t().contiguous(), train_pos.flip(1).t().contiguous()], dim=1)
    edge_weight = train_split.get("weight")
    if edge_weight is None:
        edge_weight = torch.ones(train_pos.size(0), dtype=torch.float32)
    else:
        edge_weight = edge_weight.detach().to(dtype=torch.float32).reshape(-1)
    edge_weight = torch.cat([edge_weight, edge_weight], dim=0).contiguous()
    adj = SparseTensor.from_edge_index(edge_index, edge_weight, (int(num_nodes), int(num_nodes)))
    return (edge_index.contiguous(), edge_weight, adj)


def _load_noncitation_base(data_name, root, eval_cap, seed, defer_train_val=False):
    (data, split_edge, source_artifact_identity) = _load_dataset(data_name, root)
    num_nodes = int(data.num_nodes)
    x = data.x
    print(f"the number of nodes in {data_name} is: ", num_nodes)
    train_pos = split_edge["train"]["edge"].to(torch.long).contiguous()
    all_valid_pos = split_edge["valid"]["edge"].to(torch.long).contiguous()
    valid_pos = all_valid_pos
    all_test_pos = split_edge["test"]["edge"].to(torch.long).contiguous()
    test_pos = all_test_pos
    cap = int(eval_cap) if eval_cap else 0
    valid_pos = _sample_positive_rows(valid_pos, cap, seed + 100)
    test_pos = _sample_positive_rows(test_pos, cap, seed + 101)
    train_uv = _pp_unique_undirected_uv(train_pos, num_nodes)
    (train_edge_index, adj) = _build_adj(train_uv, num_nodes)
    if data_name == "ogbl-collab":
        (model_edge_index, model_edge_weight, model_adj) = _build_collab_model_adj(train_pos, split_edge["train"], num_nodes)
        data.edge_index = model_edge_index
        data.edge_weight = model_edge_weight
        data.adj_t = model_adj
    else:
        data.edge_index = train_edge_index.contiguous()
        data.adj_t = adj
    (tr_rowptr, tr_col, _) = adj.csr()
    if data_name == "ogbl-collab":
        valid_input_pos = all_valid_pos
        valid_input_weight = torch.ones(valid_input_pos.size(0), dtype=torch.float32)
        heart_valid_filter_pos = valid_input_pos
        heart_train_ra_edge_index = model_edge_index
        heart_valid_ra_edge_index = to_undirected(heart_valid_filter_pos.t().contiguous(), num_nodes=num_nodes)
        heart_tv_ra_edge_index = torch.cat([heart_train_ra_edge_index, heart_valid_ra_edge_index], dim=1)
        heart_train_ra_degree = model_adj.sum(dim=0).to_dense().to(torch.float32)
        heart_tv_ra_degree = heart_train_ra_degree.clone()
        heart_tv_ra_degree.index_add_(0, heart_valid_ra_edge_index[1], torch.ones(heart_valid_ra_edge_index.size(1), dtype=torch.float32))
        data.valid_input_pos = valid_input_pos
        data.valid_input_weight = valid_input_weight
    else:
        valid_input_pos = valid_pos
        valid_input_weight = None
        heart_valid_filter_pos = valid_pos if data_name == "ogbl-ppa" else all_valid_pos
        heart_train_ra_degree = None
        heart_tv_ra_degree = None
        heart_train_ra_edge_index = None
        heart_tv_ra_edge_index = None
    heart_valid_filter_uv = _pp_unique_undirected_uv(heart_valid_filter_pos, num_nodes)
    tv_uv = torch.cat([train_uv, heart_valid_filter_uv], dim=1).contiguous()
    if data_name == "ogbl-collab":
        (tv_uv, _) = coalesce(tv_uv, None, num_nodes, num_nodes)
    valid_filter_edge_index = torch.cat([heart_valid_filter_uv, heart_valid_filter_uv.flip(0)], dim=1)
    valid_filter_adj = SparseTensor.from_edge_index(
        valid_filter_edge_index, torch.ones(valid_filter_edge_index.size(1), dtype=torch.float), (num_nodes, num_nodes)
    )
    tv_adj = adj + valid_filter_adj
    (tv_rowptr, tv_col, _) = tv_adj.csr()
    data.csr_rowptr = tr_rowptr
    data.csr_col = tr_col
    data.csr_tv_rowptr = tv_rowptr
    data.csr_tv_col = tv_col
    train_val = train_pos[:0] if defer_train_val else _sample_train_val(train_pos, valid_pos.size(0), seed + 3)
    return {
        "data": data,
        "split_edge": split_edge,
        "adj": adj,
        "train_pos": train_pos,
        "train_val": train_val,
        "valid_pos": valid_pos,
        "all_valid_pos": all_valid_pos,
        "valid_input_pos": valid_input_pos,
        "valid_input_weight": valid_input_weight,
        "valid_neg": None,
        "test_pos": test_pos,
        "all_test_pos": all_test_pos,
        "test_neg": None,
        "x": x,
        "num_nodes": num_nodes,
        "effective_eval_cap": cap,
        "effective_eval_seed": int(seed) if cap else 0,
        "heart_source_artifact_identity": source_artifact_identity,
        "train_uv": train_uv,
        "tv_uv": tv_uv,
        "csr_train_rowptr": tr_rowptr,
        "csr_train_col": tr_col,
        "csr_tv_rowptr": tv_rowptr,
        "csr_tv_col": tv_col,
        "heart_train_ra_degree": heart_train_ra_degree,
        "heart_tv_ra_degree": heart_tv_ra_degree,
        "heart_train_ra_edge_index": heart_train_ra_edge_index,
        "heart_tv_ra_edge_index": heart_tv_ra_edge_index,
    }


def _load_citation2_base(data_name, root, eval_cap, seed, defer_train_val=False):
    dataset = PygLinkPropPredDataset(name=data_name, root=root)
    data = dataset[0]
    split_edge = _load_or_build_compact_citation2_split(dataset)
    source_artifact_identity = _ogb_source_artifact_identity(dataset)
    compact_num_nodes = int(data.num_nodes)
    if getattr(data, "x", None) is None:
        data.x = torch.ones((compact_num_nodes, 1), dtype=torch.float)
    elif data.x.dtype != torch.float:
        data.x = data.x.to(torch.float)
    num_nodes = int(data.num_nodes)
    x = data.x
    print(f"the number of nodes in {data_name} is: ", num_nodes)
    source_train_edge_index = data.edge_index.detach().to(torch.long)
    train_pos = torch.stack([split_edge["train"]["source_node"], split_edge["train"]["target_node"]], dim=1).to(torch.long)
    all_valid_pos = torch.stack([split_edge["valid"]["source_node"], split_edge["valid"]["target_node"]], dim=1).to(torch.long)
    valid_pos = all_valid_pos
    all_test_pos = torch.stack([split_edge["test"]["source_node"], split_edge["test"]["target_node"]], dim=1).to(torch.long)
    test_pos = all_test_pos
    cap = int(eval_cap) if eval_cap else 0
    valid_pos = _sample_positive_rows(valid_pos, cap, seed + 100)
    test_pos = _sample_positive_rows(test_pos, cap, seed + 101)
    train_edge_index = to_undirected(train_pos.t().contiguous(), num_nodes=num_nodes)
    adj = SparseTensor.from_edge_index(train_edge_index, torch.ones(train_edge_index.size(1), dtype=torch.float), (num_nodes, num_nodes))
    data.edge_index = train_edge_index.contiguous()
    data.adj_t = adj
    source_ra_adj = SparseTensor.from_edge_index(
        source_train_edge_index.flip(0), torch.ones(source_train_edge_index.size(1), dtype=torch.float32), (num_nodes, num_nodes)
    )
    (source_tr_rowptr, source_tr_col, _) = source_ra_adj.csr()
    valid_filter_edge_index = all_valid_pos.t().contiguous()
    source_tv_edge_index = torch.cat([source_train_edge_index, valid_filter_edge_index, valid_filter_edge_index.flip(0)], dim=1)
    source_tv_adj = SparseTensor.from_edge_index(
        source_tv_edge_index, torch.ones(source_tv_edge_index.size(1), dtype=torch.float32), (num_nodes, num_nodes)
    )
    (source_tv_rowptr, source_tv_col, _) = source_tv_adj.csr()
    train_val = train_pos[:0] if defer_train_val else _sample_train_val(train_pos, valid_pos.size(0), seed + 3)
    return {
        "data": data,
        "split_edge": split_edge,
        "adj": adj,
        "train_pos": train_pos.contiguous(),
        "train_val": train_val,
        "valid_pos": valid_pos.contiguous(),
        "all_valid_pos": all_valid_pos.contiguous(),
        "valid_neg": None,
        "test_pos": test_pos.contiguous(),
        "all_test_pos": all_test_pos.contiguous(),
        "test_neg": None,
        "x": x,
        "num_nodes": num_nodes,
        "effective_eval_cap": cap,
        "effective_eval_seed": int(seed) if cap else 0,
        "heart_source_artifact_identity": source_artifact_identity,
        "train_uv": source_train_edge_index,
        "tv_uv": source_train_edge_index,
        "csr_train_rowptr": source_tr_rowptr,
        "csr_train_col": source_tr_col,
        "csr_tv_rowptr": source_tv_rowptr,
        "csr_tv_col": source_tv_col,
        "heart_directed": True,
        "heart_train_ra_degree": None,
        "heart_tv_ra_degree": None,
        "heart_train_ra_edge_index": None,
        "heart_tv_ra_edge_index": None,
    }
