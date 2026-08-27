from contextlib import contextmanager
from functools import lru_cache
import hashlib
import os
from urllib.error import URLError
import uuid
import torch
from torch_geometric.datasets import Amazon, FacebookPagePage, GitHub, Planetoid, Reddit, WikipediaNetwork
from torch_geometric.utils import coalesce
from torch_sparse import SparseTensor

_CACHE_VERSION = 5
_SPLIT_CACHE_DIGEST_VERSION = 1
_RAW_GRAPH_IDENTITY_SAMPLES = 4093
_HEART_POOL_CACHE_VERSION = 23
_HEART_TOPK_TIE_SEED = 42
_HEART_PPR_EPS = {"cora": 1e-07, "citeseer": 1e-07, "pubmed": 1e-05, "reddit": 1e-07}

# All PyG dataset selection lives in this table.
PYG_DATASET_LOADERS = {
    "cora": lambda root: Planetoid(root=root, name="cora")[0],
    "citeseer": lambda root: Planetoid(root=root, name="citeseer")[0],
    "pubmed": lambda root: Planetoid(root=root, name="pubmed")[0],
    "amazon-c": lambda root: Amazon(root=os.path.join(root, "amazon-computers"), name="Computers")[0],
    "amazon-p": lambda root: Amazon(root=os.path.join(root, "amazon-photo"), name="Photo")[0],
    "wiki-chameleon": lambda root: WikipediaNetwork(root=os.path.join(root, "wiki-chameleon"), name="chameleon")[0],
    "wiki-squirrel": lambda root: WikipediaNetwork(root=os.path.join(root, "wiki-squirrel"), name="squirrel")[0],
    "reddit": lambda root: Reddit(root=os.path.join(root, "reddit"))[0],
    "github": lambda root: GitHub(root=os.path.join(root, "github"))[0],
    "facebook": lambda root: FacebookPagePage(root=os.path.join(root, "facebook"))[0],
}
SUPPORTED_PYG_DATASETS = tuple(PYG_DATASET_LOADERS)


def _heart_ppr_eps(data_name):
    return float(_HEART_PPR_EPS.get(str(data_name).lower(), 5e-05))


def _andersen_ppr_for_node(u, rowptr, col, degree, alpha=0.15, eps=5e-05):
    u = int(u)
    alpha_eps = float(alpha) * float(eps)
    p = {u: 0.0}
    residual = {u: float(alpha)}
    stack = [u]
    queued = {u}
    while stack:
        node = stack.pop()
        queued.discard(node)
        value = float(residual.get(node, 0.0))
        p[node] = p.get(node, 0.0) + value
        residual[node] = 0.0
        node_degree = int(degree[node])
        if node_degree <= 0:
            continue
        increment = (1.0 - float(alpha)) * value / node_degree
        (start, end) = (int(rowptr[node]), int(rowptr[node + 1]))
        for neighbor in col[start:end].tolist():
            neighbor = int(neighbor)
            new_value = float(residual.get(neighbor, 0.0)) + increment
            residual[neighbor] = new_value
            if new_value >= alpha_eps * int(degree[neighbor]) and neighbor not in queued:
                stack.append(neighbor)
                queued.add(neighbor)
    ids = torch.tensor(list(p), dtype=torch.long)
    values = torch.tensor([p[int(node)] for node in ids.tolist()])
    keep = values > 0
    return (ids[keep], values[keep])


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
    parsed = parse_pool_argument(pool)
    if parsed == "all":
        requested_pool = int(num_nodes)
        return (requested_pool, "all", True)
    return (int(parsed), str(int(parsed)), False)


def _load_dataset(data_name, root):
    name = str(data_name).strip().lower()
    try:
        loader = PYG_DATASET_LOADERS[name]
    except KeyError:
        supported = ", ".join(SUPPORTED_PYG_DATASETS)
        raise ValueError(f"Unsupported PyG dataset {data_name!r}. Supported datasets: {supported}.") from None
    return loader(os.fspath(root))


def _torch_load(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _atomic_save(payload, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        torch.save(payload, tmp)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


@contextmanager
def _exclusive_cache_build(path):
    if not path:
        yield
        return
    lock_path = path + ".lock"
    os.makedirs(os.path.dirname(lock_path), exist_ok=True)
    lock_file = open(lock_path, "a+b")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except (ImportError, OSError) as exc:
            print(f"WARNING: cache locking unavailable for {lock_path}: {exc}", flush=True)
        yield
    finally:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except (ImportError, OSError):
            pass
        lock_file.close()


def _cache_dir(root):
    path = os.path.join(root, "lp_cache")
    os.makedirs(path, exist_ok=True)
    return path


def _split_cache_file_version(root, data_name, split, seed, version):
    split_tag = "_".join((str(int(round(v * 10000))) for v in split))
    return os.path.join(_cache_dir(root), f"{data_name.lower()}_split_v{int(version)}_{split_tag}_seed{int(seed)}.pt")


def _split_cache_file(root, data_name, split, seed):
    return _split_cache_file_version(root, data_name, split, seed, _CACHE_VERSION)


def _full_integer_tensor_sha256(tensor, *, chunk_values=1000000):
    value = torch.as_tensor(tensor).detach().cpu().contiguous().view(-1)
    digest = hashlib.sha256()
    digest.update(f"integer-tensor-v1;shape={tuple(tensor.shape)};numel={value.numel()};".encode("utf-8"))
    step = max(1, int(chunk_values))
    for start in range(0, int(value.numel()), step):
        chunk = value[start : start + step].to(torch.int64)
        digest.update(chunk.numpy().tobytes(order="C"))
    return digest.hexdigest()


def _split_tensor_sha256(train_uv, valid_uv, test_uv):
    digest = hashlib.sha256()
    digest.update(b"pyg-positive-split-sha256-v1;")
    for name, value in (("train", train_uv), ("valid", valid_uv), ("test", test_uv)):
        digest.update(f"{name}:".encode("ascii"))
        digest.update(_full_integer_tensor_sha256(value).encode("ascii"))
        digest.update(b";")
    return digest.hexdigest()


def _canonical_known_graph_sha256(all_uv, num_nodes):
    value = torch.as_tensor(all_uv, dtype=torch.long).cpu()
    lo = torch.minimum(value[0], value[1])
    hi = torch.maximum(value[0], value[1])
    edge_ids = lo * int(num_nodes) + hi
    edge_ids = torch.sort(edge_ids).values
    return _full_integer_tensor_sha256(edge_ids)


def _partitioned_known_graph_sha256(train_uv, valid_uv, test_uv, split_tensor_sha256):
    digest = hashlib.sha256()
    digest.update(b"full-partitioned-canonical-uv-sha256-v1;")
    digest.update(str(split_tensor_sha256).encode("ascii"))
    for name, value in (("train", train_uv), ("valid", valid_uv), ("test", test_uv)):
        digest.update(f";{name}_edges={int(torch.as_tensor(value).size(1))}".encode("ascii"))
    return digest.hexdigest()


def _bounded_raw_graph_identity(edge_index, num_nodes):
    edges = torch.as_tensor(edge_index).detach().cpu()
    count = int(edges.size(1))
    sample_count = min(int(_RAW_GRAPH_IDENTITY_SAMPLES), count)
    if sample_count:
        if sample_count == 1:
            positions = torch.zeros(1, dtype=torch.long)
        else:
            positions = torch.arange(sample_count, dtype=torch.long) * (count - 1) // (sample_count - 1)
        left = edges[0, positions].to(torch.long)
        right = edges[1, positions].to(torch.long)
        sampled = torch.stack([torch.minimum(left, right), torch.maximum(left, right)], dim=0)
    else:
        sampled = torch.empty((2, 0), dtype=torch.long)
    digest = hashlib.sha256()
    digest.update(f"bounded-canonical-raw-edge-sample-v1;nodes={int(num_nodes)};edges={count};samples={sample_count};".encode("utf-8"))
    digest.update(_full_integer_tensor_sha256(sampled).encode("ascii"))
    return digest.hexdigest()


def _heart_pool_cache_file(root, data_name, split, seed, eval_cap, pool, draw, backend, ppr_iters, full_graph):
    scope = "full" if full_graph else f"random{int(pool)}"
    split_tag = "_".join((str(int(round(v * 10000))) for v in split))
    return os.path.join(
        _cache_dir(root),
        f"{data_name.lower()}_heart_pool_v{_HEART_POOL_CACHE_VERSION}_split{split_tag}_seed{int(seed)}_cap{int(eval_cap or 0)}_{scope}_draw{int(draw)}_{backend}_ppr{int(ppr_iters)}.pt",
    )


def _pp_unique_undirected_uv(edge_index, num_nodes, fast_symmetric=False):
    (row, col) = edge_index
    mask = row != col
    row = row[mask]
    col = col[mask]
    if fast_symmetric:
        keep = row < col
        return torch.stack([row[keep], col[keep]], dim=0).contiguous()
    lo = torch.minimum(row, col)
    hi = torch.maximum(row, col)
    uv = torch.stack([lo, hi], dim=0)
    (uv, _) = coalesce(uv, None, num_nodes, num_nodes)
    return uv.contiguous()


def _pp_split_uv(all_uv, split=(0.7, 0.15, 0.15), seed=0):
    e = all_uv.size(1)
    perm = None
    if torch.cuda.is_available() and e >= 2000000:
        try:
            device = torch.device("cuda")
            g = torch.Generator(device=device).manual_seed(int(seed))
            print(f"Creating random edge split on {device} for {e} undirected edges", flush=True)
            perm = torch.randperm(e, generator=g, device=device).cpu()
            torch.cuda.empty_cache()
        except RuntimeError as exc:
            print(f"CUDA split generation failed ({exc}); falling back to CPU", flush=True)
            torch.cuda.empty_cache()
    if perm is None:
        g = torch.Generator().manual_seed(seed)
        perm = torch.randperm(e, generator=g)
    n_valid = int(split[1] * e)
    n_test = int(split[2] * e)
    valid_uv = all_uv[:, perm[:n_valid]].contiguous()
    test_uv = all_uv[:, perm[n_valid : n_valid + n_test]].contiguous()
    train_uv = all_uv[:, perm[n_valid + n_test :]].contiguous()
    del perm
    return (train_uv, valid_uv, test_uv)


def _try_load_split_cache(path, num_nodes, raw_edges, raw_graph_identity, *, return_metadata=False):
    if os.path.exists(path):
        try:
            payload = _torch_load(path)
            required = (
                "train_uv",
                "valid_uv",
                "test_uv",
                "train_rowptr",
                "train_col",
                "split_tensor_sha256",
                "known_graph_sha256",
                "known_graph_sha256_method",
                "raw_graph_identity",
            )
            if (
                int(payload.get("split_cache_digest_version", -1)) == int(_SPLIT_CACHE_DIGEST_VERSION)
                and int(payload.get("num_nodes", -1)) == num_nodes
                and (int(payload.get("raw_edges", -1)) == raw_edges)
                and (payload.get("raw_graph_identity") == raw_graph_identity)
                and all((key in payload for key in required))
            ):
                print(f"Loading cached edge split and CSR adjacency: {path}", flush=True)
                values = (
                    payload["train_uv"].to(torch.long),
                    payload["valid_uv"].to(torch.long),
                    payload["test_uv"].to(torch.long),
                    payload["train_rowptr"].to(torch.long),
                    payload["train_col"].to(torch.long),
                )
                if return_metadata:
                    return values + (
                        {
                            "split_cache_digest_version": int(payload["split_cache_digest_version"]),
                            "split_tensor_sha256": str(payload["split_tensor_sha256"]),
                            "known_graph_sha256": str(payload["known_graph_sha256"]),
                            "known_graph_sha256_method": str(payload["known_graph_sha256_method"]),
                            "raw_graph_identity": str(payload["raw_graph_identity"]),
                            "raw_graph_identity_method": str(
                                payload.get("raw_graph_identity_method", "bounded-canonical-raw-edge-sample-v1")
                            ),
                        },
                    )
                return values
        except Exception as exc:
            print(f"Ignoring invalid split cache {path}: {exc}", flush=True)
    return None


def _try_upgrade_split_cache(legacy_path, target_path, num_nodes, raw_edges, raw_graph_identity):
    if not os.path.exists(legacy_path):
        return False
    try:
        payload = _torch_load(legacy_path)
        required = ("train_uv", "valid_uv", "test_uv", "train_rowptr", "train_col")
        if (
            int(payload.get("num_nodes", -1)) != int(num_nodes)
            or int(payload.get("raw_edges", -1)) != int(raw_edges)
            or (not all((key in payload for key in required)))
        ):
            return False
        split_values = []
        for name in ("train_uv", "valid_uv", "test_uv"):
            value = torch.as_tensor(payload[name]).to(torch.long)
            if value.dim() != 2 or int(value.size(0)) != 2:
                return False
            if value.numel() and (int(value.min()) < 0 or int(value.max()) >= int(num_nodes) or bool((value[0] >= value[1]).any())):
                return False
            split_values.append(value)
        (train_uv, valid_uv, test_uv) = split_values
        train_rowptr = torch.as_tensor(payload["train_rowptr"])
        train_col = torch.as_tensor(payload["train_col"])
        if (
            tuple(train_rowptr.shape) != (int(num_nodes) + 1,)
            or int(train_col.numel()) != 2 * int(train_uv.size(1))
            or int(train_rowptr[0]) != 0
            or (int(train_rowptr[-1]) != int(train_col.numel()))
            or (train_col.numel() and (int(train_col.min()) < 0 or int(train_col.max()) >= int(num_nodes)))
        ):
            return False
        split_tensor_sha256 = _split_tensor_sha256(train_uv, valid_uv, test_uv)
        known_graph_sha256 = _partitioned_known_graph_sha256(train_uv, valid_uv, test_uv, split_tensor_sha256)
        payload.update(
            {
                "split_cache_digest_version": int(_SPLIT_CACHE_DIGEST_VERSION),
                "raw_graph_identity": raw_graph_identity,
                "raw_graph_identity_method": "bounded-canonical-raw-edge-sample-v1-probabilistic",
                "split_tensor_sha256": split_tensor_sha256,
                "known_graph_sha256": known_graph_sha256,
                "known_graph_sha256_method": "full-partitioned-canonical-uv-sha256-v1",
            }
        )
        _atomic_save(payload, target_path)
        print(f"Upgraded reusable edge split with full split/known-graph digests: {legacy_path} -> {target_path}", flush=True)
        return True
    except Exception as exc:
        print(f"Ignoring non-upgradeable split cache {legacy_path}: {exc}", flush=True)
        return False


def _load_or_create_split(d, data_name, split, seed, root, *, return_metadata=False):
    num_nodes = int(d.num_nodes)
    raw_edges = int(d.edge_index.size(1))
    path = _split_cache_file(root, data_name, split, seed)
    raw_graph_identity = _bounded_raw_graph_identity(d.edge_index, num_nodes)
    cached = _try_load_split_cache(path, num_nodes, raw_edges, raw_graph_identity, return_metadata=return_metadata)
    if cached is not None:
        return cached
    with _exclusive_cache_build(path):
        cached = _try_load_split_cache(path, num_nodes, raw_edges, raw_graph_identity, return_metadata=return_metadata)
        if cached is not None:
            return cached
        legacy_path = _split_cache_file_version(root, data_name, split, seed, 5)
        if _try_upgrade_split_cache(legacy_path, path, num_nodes, raw_edges, raw_graph_identity):
            cached = _try_load_split_cache(path, num_nodes, raw_edges, raw_graph_identity, return_metadata=return_metadata)
            if cached is not None:
                return cached
        fast_symmetric = data_name.lower() == "reddit"
        all_uv = _pp_unique_undirected_uv(d.edge_index, num_nodes, fast_symmetric=fast_symmetric)
        known_graph_sha256 = _canonical_known_graph_sha256(all_uv, num_nodes)
        known_graph_sha256_method = "full-sorted-canonical-undirected-edge-sha256-v1"
        (train_uv, valid_uv, test_uv) = _pp_split_uv(all_uv, split=split, seed=seed)
        split_tensor_sha256 = _split_tensor_sha256(train_uv, valid_uv, test_uv)
        train_edge_index = torch.cat([train_uv, train_uv.flip(0)], dim=1)
        train_adj = SparseTensor.from_edge_index(
            train_edge_index, torch.ones(train_edge_index.size(1), dtype=torch.float32), (num_nodes, num_nodes)
        )
        (train_rowptr, train_col, _) = train_adj.csr()
        storage_dtype = torch.int32 if num_nodes < 2**31 else torch.int64
        payload = {
            "split_cache_digest_version": int(_SPLIT_CACHE_DIGEST_VERSION),
            "num_nodes": num_nodes,
            "raw_edges": raw_edges,
            "raw_graph_identity": raw_graph_identity,
            "raw_graph_identity_method": "bounded-canonical-raw-edge-sample-v1-probabilistic",
            "split_tensor_sha256": split_tensor_sha256,
            "known_graph_sha256": known_graph_sha256,
            "known_graph_sha256_method": known_graph_sha256_method,
            "train_uv": train_uv.to(storage_dtype),
            "valid_uv": valid_uv.to(storage_dtype),
            "test_uv": test_uv.to(storage_dtype),
            "train_rowptr": train_rowptr,
            "train_col": train_col.to(storage_dtype),
        }
        try:
            print(f"Saving reusable edge split and CSR adjacency: {path}", flush=True)
            _atomic_save(payload, path)
        except Exception as exc:
            print(f"WARNING: could not save split cache: {exc}", flush=True)
        values = (train_uv, valid_uv, test_uv, train_rowptr, train_col)
        if return_metadata:
            return values + (
                {
                    "split_cache_digest_version": int(_SPLIT_CACHE_DIGEST_VERSION),
                    "split_tensor_sha256": split_tensor_sha256,
                    "known_graph_sha256": known_graph_sha256,
                    "known_graph_sha256_method": known_graph_sha256_method,
                    "raw_graph_identity": raw_graph_identity,
                    "raw_graph_identity_method": "bounded-canonical-raw-edge-sample-v1-probabilistic",
                },
            )
        return values


def _sample_indices(size, count, seed):
    size = int(size)
    count = min(int(count), size)
    if count >= size:
        return torch.arange(size, dtype=torch.long)
    g = torch.Generator().manual_seed(int(seed))
    if count > 100000 or count * 10 > size:
        return torch.randperm(size, generator=g)[:count]
    values = []
    seen = set()
    while len(values) < count:
        draw = torch.randint(0, size, (max(64, 2 * (count - len(values))),), generator=g)
        for value in draw.tolist():
            if value not in seen:
                seen.add(value)
                values.append(value)
                if len(values) == count:
                    break
    return torch.tensor(values, dtype=torch.long)


@lru_cache(maxsize=16)
def _heart_seeded_tie_priority(num_nodes, seed=_HEART_TOPK_TIE_SEED):
    num_nodes = int(num_nodes)
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    permutation = torch.randperm(num_nodes, generator=generator)
    priority = torch.empty(num_nodes, dtype=torch.long)
    priority[permutation] = torch.arange(num_nodes, dtype=torch.long)
    return priority


def _sample_rows(rows, count, seed):
    if count >= rows.size(0):
        return rows
    return rows[_sample_indices(rows.size(0), count, seed)].contiguous()


def _make_adj(train_uv, num_nodes, rowptr=None, col=None):
    if rowptr is not None and col is not None:
        value = torch.ones(col.numel(), dtype=torch.float32)
        return SparseTensor(rowptr=rowptr, col=col, value=value, sparse_sizes=(num_nodes, num_nodes), is_sorted=True)
    edge_index = torch.cat([train_uv, train_uv.flip(0)], dim=1)
    value = torch.ones(edge_index.size(1), dtype=torch.float32)
    return SparseTensor.from_edge_index(edge_index, value, (num_nodes, num_nodes))
