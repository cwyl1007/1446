import argparse
import fcntl
import gc
import hashlib
import os
import shutil
import tempfile
import time
from contextlib import contextmanager
import torch
import torch_geometric
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.node2vec_model import DotProductPredictor, Node2VecEncoder, ReferenceN2VLink
from utils.heart_protocol import persist_heart_candidate_metadata
from .prepare_data import load_ogbl_splits, parse_pool_argument, read_data
from .fast_negatives import prepare_ddi_grouped_eval_edges
from .data_core import _fsync_directory, _fsync_file
from .protocol import (
    RUNTIME_LIMIT_SEC,
    bind_protocol_metadata,
    log_aggregate_results,
    log_protocol_summary,
    log_run_statistics,
    ogbl_protocol_metadata,
    print_ogbl_protocol,
    resolve_ogbl_device,
    resolve_ogbl_eval_cap,
    resolve_ogbl_metric,
    runtime_exceeded as _runtime_exceeded,
    save_model_checkpoint,
    set_seed as _set_seed,
    should_compute_auc as _should_compute_auc,
    snapshot_state_dict_cpu as _snapshot_state_dict_cpu,
    write_summary as _write_summary,
)
from .train_eval import (
    evaluate_ogbl_test,
    evaluate_ogbl_validation,
    find_result_key,
    merge_ogbl_results,
    prepare_ogbl_evaluation,
    release_ogbl_evaluation,
)
from .training import cache_eval_edges_on_device
from utils.profiling import StageProfiler, configure_torch_cpu_threads, current_cpu_rss_mb, empty_stage_info, peak_cpu_rss_mb

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFERENCE_N2V_BASE = {"hidden_channels": 256, "source_epochs": 9999, "source_eval_steps": 1, "source_patience": 100}
_REFERENCE_N2V_RECIPES = {
    "ogbl-collab": {**_REFERENCE_N2V_BASE, "feature_composition": "raw-plus-node2vec", "learning_rate": 0.001},
    "ogbl-ddi": {**_REFERENCE_N2V_BASE, "feature_composition": "node2vec-only", "learning_rate": 0.01},
    "ogbl-ppa": {**_REFERENCE_N2V_BASE, "feature_composition": "raw-plus-node2vec", "learning_rate": 0.001},
    "ogbl-citation2": {
        **_REFERENCE_N2V_BASE, "feature_composition": "raw-plus-node2vec", "hidden_channels": 128,
        "learning_rate": 0.001, "source_epochs": 100, "source_patience": 20,
    },
}


def set_seed(seed):
    _set_seed(seed, deterministic_cudnn=True)


class N2VLink(torch.nn.Module):

    def __init__(self, encoder, predictor):
        super().__init__()
        self.decode_is_symmetric = True
        self.decode_is_dedup_safe = True
        self.encoder = encoder
        self.predictor = predictor

    @torch.no_grad()
    def embed(self, data):
        return self.encoder()

    def decode(self, z, edge_index):
        return self.predictor(z, edge_index)


def train_one_epoch(encoder, optimizer, device, batch_size, max_batches=0, prefetch_batches=0, show_batch_progress=False, progress_desc="Node2Vec walk batches"):
    encoder.train()
    total = torch.zeros((), device=device)
    n = 0
    max_batches = 0 if max_batches is None else int(max_batches)
    loader = encoder.loader(
        batch_size=batch_size,
        shuffle=True,
        prefetch_batches=prefetch_batches,
        pin_memory=device.type == "cuda" and int(prefetch_batches) > 0,
    )
    iterator = iter(loader)
    total_batches = len(loader)
    if max_batches > 0:
        total_batches = min(total_batches, max_batches)
    batch_progress = None
    if show_batch_progress:
        batch_progress = tqdm(total=total_batches, desc=progress_desc, leave=False, dynamic_ncols=True, mininterval=1.0)
    try:
        for batch_id, (pos_rw, neg_rw) in enumerate(iterator):
            if max_batches > 0 and batch_id >= max_batches:
                break
            pos_rw = pos_rw.to(device, non_blocking=True)
            neg_rw = neg_rw.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = encoder.rw_loss(pos_rw, neg_rw)
            loss.backward()
            optimizer.step()
            total += loss.detach()
            n += 1
            if batch_progress is not None:
                batch_progress.update(1)
    finally:
        if batch_progress is not None:
            batch_progress.close()
        close = getattr(iterator, "close", None)
        if callable(close):
            close()
    return float((total / max(n, 1)).item())


def parse_args():
    parser = argparse.ArgumentParser(description="Node2Vec link prediction on OGBL")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--metric", type=str, default="auto")
    parser.add_argument("--mode", choices=["heart", "all"], default="heart")
    parser.add_argument("--root", type=str, default="dataset")
    parser.add_argument("--n2v-protocol", choices=["auto", "reference", "legacy-direct"], default="auto")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device")
    parser.add_argument("--emb-dim", "--emb_dim", dest="emb_dim", type=int, default=128)
    parser.add_argument("--walk-length", "--walk_length", dest="walk_length", type=int, default=20)
    parser.add_argument("--context-size", "--context_size", dest="context_size", type=int, default=10)
    parser.add_argument("--walks-per-node", "--walks_per_node", dest="walks_per_node", type=int, default=10)
    parser.add_argument("--num-neg-samples", "--num_neg_samples", dest="num_neg_samples", type=int, default=1)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rw-batch-size", "--walk_batch_size", dest="rw_batch_size", type=int, default=128)
    parser.add_argument("--rw-prefetch-batches", type=int, default=4)
    parser.add_argument("--rw-progress", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--eval-batch-size", "--edge_batch_size", dest="eval_batch_size", type=int, default=65536)
    parser.add_argument("--cache-eval-edges", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--reference-eval-edge-batch-size", type=int, default=2097152)
    parser.add_argument("--eval-steps", "--eval_steps", dest="eval_steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--citation2-query-batch-size", type=int, default=512)
    parser.add_argument("--base-seed", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("--num-runs", "--runs", dest="num_runs", type=int, default=None)
    parser.add_argument("--max-walk-batches", type=int, default=None)
    parser.add_argument("--all-negatives", type=int, default=None)
    parser.add_argument("--pool", type=parse_pool_argument, default=10000)
    parser.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, default=500)
    parser.add_argument("--ranked-negatives-backend", choices=["auto", "official", "batched", "fast", "dense"], default="auto")
    parser.add_argument("--negative-cache-dir", type=str, default=None)
    parser.add_argument("--no-negative-cache", action="store_true")
    parser.add_argument("--compute-auc", choices=["auto", "yes", "no"], default="auto")
    parser.add_argument("--save-gnn-feature", action="store_true", default=False)
    parser.add_argument("--save-log", action="store_true", default=True)
    parser.add_argument("--reference-pretrain-seed", type=int, default=999)
    parser.add_argument("--reference-pretrain-epochs", type=int, default=2)
    parser.add_argument("--reference-pretrain-walk-length", type=int, default=40)
    parser.add_argument("--reference-pretrain-context-size", type=int, default=20)
    parser.add_argument("--reference-pretrain-walks-per-node", type=int, default=10)
    parser.add_argument("--reference-pretrain-batch-size", type=int, default=256)
    parser.add_argument("--reference-pretrain-workers", type=int, default=4)
    parser.add_argument("--reference-pretrain-lr", type=float, default=0.01)
    parser.add_argument("--reference-lr", type=float, default=None)
    parser.add_argument("--reference-batch-size", type=int, default=65536)
    parser.add_argument("--reference-hidden-channels", type=int, default=None)
    parser.add_argument("--reference-num-layers", type=int, default=3)
    parser.add_argument("--reference-predictor-layers", type=int, default=3)
    parser.add_argument("--reference-dropout", type=float, default=0.0)
    parser.add_argument("--reference-node-encode-batch-size", type=int, default=262144)
    parser.add_argument("--reference-embedding-path", type=str, default=None)
    return parser.parse_args()


def _resolve_n2v_protocol(option, dataset):
    option = str(option or "auto").strip().lower()
    if option == "auto":
        dataset = str(dataset).strip().lower()
        return "reference" if dataset in _REFERENCE_N2V_RECIPES else "legacy-direct"
    return option


def _resolve_reference_recipe(args):
    dataset = str(args.dataset).strip().lower()
    try:
        recipe = _REFERENCE_N2V_RECIPES[dataset]
    except KeyError as exc:
        raise ValueError("The reference Node2Vec protocol is defined only for the four released OGB benchmark datasets.") from exc
    if args.reference_hidden_channels is None:
        args.reference_hidden_channels = int(recipe["hidden_channels"])
    if args.reference_lr is None:
        args.reference_lr = float(recipe["learning_rate"])
    args.reference_feature_composition = str(recipe["feature_composition"])
    args.reference_recipe_source = f"HeaRT/scripts/hyperparameters/HeaRT_ogb/{dataset}.sh"
    return recipe


def _show_rw_progress(option, dataset):
    option = str(option or "auto").strip().lower()
    if option == "auto":
        return str(dataset).strip().lower() == "ogbl-citation2"
    return option == "yes"


def _resolve_run_defaults(args, protocol="legacy-direct"):
    defaults = {"epochs": 500 if protocol == "reference" else 300, "num_runs": 5, "eval_steps": 5, "patience": 10}
    for name, value in defaults.items():
        if getattr(args, name) is None:
            setattr(args, name, value)
    if getattr(args, "max_walk_batches", None) is None:
        args.max_walk_batches = 0
    args.max_walk_batches = max(0, int(args.max_walk_batches))
    return args


def _reference_model_protocol_faithful(args):
    recipe = _REFERENCE_N2V_RECIPES[str(args.dataset).strip().lower()]
    expected = (
        (int(args.emb_dim), 128),
        (int(args.reference_pretrain_seed), 999),
        (int(args.reference_pretrain_epochs), 2),
        (int(args.reference_pretrain_walk_length), 40),
        (int(args.reference_pretrain_context_size), 20),
        (int(args.reference_pretrain_walks_per_node), 10),
        (int(args.reference_pretrain_batch_size), 256),
        (int(args.reference_pretrain_workers), 4),
        (float(args.reference_pretrain_lr), 0.01),
        (float(args.reference_lr), float(recipe["learning_rate"])),
        (int(args.reference_batch_size), 65536),
        (int(args.reference_hidden_channels), int(recipe["hidden_channels"])),
        (int(args.reference_num_layers), 3),
        (int(args.reference_predictor_layers), 3),
        (float(args.reference_dropout), 0.0),
    )
    return all((actual == wanted for (actual, wanted) in expected))


def _reference_training_schedule_faithful(args):
    recipe = _REFERENCE_N2V_RECIPES[str(args.dataset).strip().lower()]
    return (
        int(args.epochs) == int(recipe["source_epochs"])
        and int(args.eval_steps) == int(recipe["source_eval_steps"])
        and (int(args.patience) == int(recipe["source_patience"]))
    )


def _build_eval_edges(bundle, dataset_name, mode):
    out = {
        "pos_train_edge": bundle["train_pos"],
        "train_val_edge": bundle["train_val"],
        "pos_valid_edge": bundle["valid_pos"],
        "neg_valid_edge": bundle["valid_neg"],
        "pos_test_edge": bundle["test_pos"],
        "neg_test_edge": bundle["test_neg"],
        "dataset_name": dataset_name,
        "mode": mode,
    }
    for key in ("adj", "adj_test", "csr_train_rowptr", "csr_train_col", "csr_tv_rowptr", "csr_tv_col", "all_filter_existing"):
        if key in bundle:
            out[key] = bundle[key]
    return out


def _checkpoint_dir(mode, dataset):
    return os.path.join(PROJECT_ROOT, "checkpoints", str(mode), str(dataset), "n2v")


def _available_bytes_for(path):
    probe = os.path.abspath(path)
    while not os.path.exists(probe):
        parent = os.path.dirname(probe)
        if parent == probe:
            break
        probe = parent
    return int(shutil.disk_usage(probe).free)


def _planned_atomic_checkpoint_bytes(paths, estimated_file_bytes):
    size = max(1, int(estimated_file_bytes))
    permanently_added = 0
    peak = 0
    for path in paths:
        peak = max(peak, permanently_added + size)
        if not os.path.exists(path):
            permanently_added += size
    return peak


def _require_disk_headroom(path, required_bytes, purpose):
    reserve = max(64 * 1024**2, int(required_bytes * 0.01))
    free = _available_bytes_for(path)
    required = int(required_bytes) + reserve
    print(f"{purpose}_disk_free_bytes={free} {purpose}_disk_required_bytes={required}", flush=True)
    if free < required:
        raise RuntimeError(
            f"Insufficient disk space for {purpose}: need at least {required / 1024 ** 3:.2f} GiB free for atomic output, but only {free / 1024 ** 3:.2f} GiB is available. No checkpoint was changed."
        )


def _preflight_legacy_checkpoints(args, mode, num_nodes):
    checkpoint_dir = _checkpoint_dir(mode, args.dataset)
    paths = [os.path.join(checkpoint_dir, f"model_checkpoint{args.seed + i + 1}") for i in range(int(args.num_runs))]
    parameter_bytes = int(num_nodes) * int(args.emb_dim) * 4
    existing_sizes = [os.path.getsize(path) for path in paths if os.path.isfile(path)]
    estimated = max([parameter_bytes + 1024**2] + existing_sizes)
    required = _planned_atomic_checkpoint_bytes(paths, estimated)
    _require_disk_headroom(checkpoint_dir, required, "node2vec_checkpoint_preflight")


def _reference_embedding_path(args):
    if args.reference_embedding_path:
        return os.path.abspath(os.path.expanduser(args.reference_embedding_path))
    filename = f"{args.dataset}-n2v-embedding.pt"
    cache_dir = os.environ.get("N2V_EMBEDDING_CACHE_DIR")
    if cache_dir:
        return os.path.abspath(os.path.join(os.path.expanduser(cache_dir), filename))
    cache_dir = "/ephemeral/ubuntu/LinkPrediction/n2v_embeddings"
    if os.path.isdir(cache_dir):
        return os.path.join(cache_dir, filename)
    return os.path.abspath(os.path.join(args.root, filename))


def _load_tensor_cpu(path):
    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        value = torch.load(path, map_location="cpu")
    if isinstance(value, dict):
        value = value.get("entity_embedding", value.get("embedding"))
    if not torch.is_tensor(value):
        raise TypeError(f"Expected a tensor Node2Vec embedding at {path}")
    return value.detach().to(dtype=torch.float32, device="cpu")


def _tensor_sha256(tensor):
    tensor = tensor.detach().to(device="cpu").contiguous()
    byte_view = memoryview(tensor.numpy()).cast("B")
    digest = hashlib.sha256()
    chunk_bytes = 64 * 1024**2
    for start in range(0, len(byte_view), chunk_bytes):
        digest.update(byte_view[start : start + chunk_bytes])
    return digest.hexdigest()


def _citation2_source_pretrain_edges(split_edge):
    train = split_edge["train"]
    source = train["source_node"]
    target = train["target_node"]
    source = source.to(device="cpu", dtype=torch.long).reshape(-1)
    target = target.to(device="cpu", dtype=torch.long).reshape(-1)
    return torch.stack([source, target], dim=0).contiguous()


def _reference_pretrain_edges(args, data, split_edge):
    dataset = str(args.dataset).strip().lower()
    if dataset == "ogbl-citation2":
        edge_index = _citation2_source_pretrain_edges(split_edge)
        semantics = "raw-directed-ogb-edge-index"
    else:
        edge_index = data.edge_index.detach().to(device="cpu", dtype=torch.long).contiguous()
        semantics = "prepared-undirected-ogb-train-edge-index"
    args.reference_pretrain_graph_semantics = semantics
    return edge_index


def _reference_embedding_expected_metadata(args, directed_edge_index):
    edge_sha256 = _tensor_sha256(directed_edge_index)
    return {
        "protocol": "HeaRT-reference-two-stage-node2vec",
        "dataset": str(args.dataset),
        "seed": int(args.reference_pretrain_seed),
        "embedding_dim": int(args.emb_dim),
        "walk_length": int(args.reference_pretrain_walk_length),
        "context_size": int(args.reference_pretrain_context_size),
        "walks_per_node": int(args.reference_pretrain_walks_per_node),
        "num_negative_samples": 1,
        "p": 1.0,
        "q": 1.0,
        "batch_size": int(args.reference_pretrain_batch_size),
        "workers": int(args.reference_pretrain_workers),
        "learning_rate": float(args.reference_pretrain_lr),
        "epochs": int(args.reference_pretrain_epochs),
        "graph_semantics": str(args.reference_pretrain_graph_semantics),
        "directed_edge_count": int(directed_edge_index.size(1)),
        "directed_edge_index_sha256": edge_sha256,
        "pretrain_edge_index_sha256": edge_sha256,
        "torch_version": str(torch.__version__),
        "torch_geometric_version": str(torch_geometric.__version__),
        "cuda_version": str(torch.version.cuda),
    }


def _load_metadata_file(metadata_path):
    try:
        metadata = torch.load(metadata_path, map_location="cpu", weights_only=False)
    except TypeError:
        metadata = torch.load(metadata_path, map_location="cpu")
    if not isinstance(metadata, dict):
        raise TypeError(f"Invalid Node2Vec metadata at {metadata_path}")
    if metadata.get("protocol") == "HeaRT-official-two-stage-node2vec":
        metadata = {**metadata, "protocol": "HeaRT-reference-two-stage-node2vec"}
    return metadata


def _validate_reference_metadata(path, metadata, expected):
    if metadata is None:
        raise ValueError(f"Shared Node2Vec embedding at {path} has no valid metadata")
    checked = (
        "protocol", "dataset", "seed", "embedding_dim", "walk_length", "context_size", "walks_per_node", "num_negative_samples",
        "p", "q", "batch_size", "workers", "learning_rate", "epochs", "graph_semantics", "directed_edge_count",
        "directed_edge_index_sha256",
    )
    mismatches = [
        f"{key}: saved={metadata.get(key)!r}, expected={expected[key]!r}" for key in checked if metadata.get(key) != expected[key]
    ]
    if mismatches:
        raise ValueError(f"Shared Node2Vec embedding metadata at {path}.meta.pt does not match this reference run: " + "; ".join(mismatches))
    checksum = metadata.get("embedding_sha256")
    if not isinstance(checksum, str) or len(checksum) != 64 or any((character not in "0123456789abcdef" for character in checksum)):
        raise ValueError(f"Shared Node2Vec metadata at {path}.meta.pt has no valid embedding_sha256 checksum")


@contextmanager
def _reference_embedding_lock(path):
    artifact_dir = os.path.dirname(path) or "."
    os.makedirs(artifact_dir, exist_ok=True)
    lock_path = path + ".lock"
    with open(lock_path, "a+b") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _unique_staging_path(final_path):
    artifact_dir = os.path.dirname(final_path) or "."
    prefix = f".{os.path.basename(final_path)}.tmp."
    (descriptor, staging_path) = tempfile.mkstemp(prefix=prefix, dir=artifact_dir)
    os.close(descriptor)
    return staging_path


def _publish_reference_embedding(path, embedding, metadata, expected_metadata):
    artifact_dir = os.path.dirname(path) or "."
    metadata_path = path + ".meta.pt"
    tensor_staging_path = _unique_staging_path(path)
    metadata_staging_path = _unique_staging_path(metadata_path)
    try:
        torch.save(embedding, tensor_staging_path)
        _fsync_file(tensor_staging_path)
        torch.save(metadata, metadata_staging_path)
        _fsync_file(metadata_staging_path)
        staged_metadata = _load_metadata_file(metadata_staging_path)
        _validate_reference_metadata(path, staged_metadata, expected_metadata)
        os.replace(tensor_staging_path, path)
        _fsync_directory(artifact_dir)
        os.replace(metadata_staging_path, metadata_path)
        _fsync_directory(artifact_dir)
    finally:
        for staging_path in (tensor_staging_path, metadata_staging_path):
            if os.path.exists(staging_path):
                os.remove(staging_path)


def _load_or_pretrain_reference_embedding(args, data, device, directed_edge_index):
    path = _reference_embedding_path(args)
    with _reference_embedding_lock(path):
        return _load_or_pretrain_reference_embedding_under_lock(args, data, device, directed_edge_index, path)


def _load_or_pretrain_reference_embedding_under_lock(args, data, device, directed_edge_index, path):
    expected_shape = (int(data.num_nodes), int(args.emb_dim))
    expected_metadata = _reference_embedding_expected_metadata(args, directed_edge_index)
    artifact_exists = os.path.exists(path)
    metadata_path = path + ".meta.pt"
    metadata_exists = os.path.exists(metadata_path)
    if artifact_exists != metadata_exists:
        missing_path = metadata_path if artifact_exists else path
        raise RuntimeError(
            f"Shared Node2Vec cache contains an incomplete artifact pair; missing {missing_path}. Refusing to accept or overwrite it."
        )
    if artifact_exists:
        if not os.path.isfile(path) or not os.path.isfile(metadata_path):
            raise ValueError(f"Shared Node2Vec cache paths must both be regular files: {path}, {metadata_path}")
        embedding = _load_tensor_cpu(path)
        if tuple(embedding.shape) != expected_shape:
            raise ValueError(f"Node2Vec embedding at {path} has shape {tuple(embedding.shape)}; expected {expected_shape}.")
        metadata = _load_metadata_file(metadata_path)
        _validate_reference_metadata(path, metadata, expected_metadata)
        embedding_sha256 = _tensor_sha256(embedding)
        if metadata["embedding_sha256"] != embedding_sha256:
            raise ValueError(f"Shared Node2Vec embedding checksum does not match {path}.meta.pt")
        resolved_metadata = dict(expected_metadata)
        resolved_metadata.update(metadata)
        resolved_metadata["embedding_sha256"] = embedding_sha256
        resolved_metadata["artifact_provenance_verified"] = True
        print(f"Loaded shared reference Node2Vec embedding: {path}", flush=True)
        return (embedding, path, 0.0, resolved_metadata, empty_stage_info())
    estimated_bytes = expected_shape[0] * expected_shape[1] * 4 + 1024**2
    _require_disk_headroom(os.path.dirname(path), estimated_bytes, "node2vec_embedding_preflight")
    set_seed(int(args.reference_pretrain_seed))
    print(
        f"Pretraining reference Node2Vec embedding (seed={args.reference_pretrain_seed},"
        f" epochs={args.reference_pretrain_epochs}, walk_length={args.reference_pretrain_walk_length},"
        f" context_size={args.reference_pretrain_context_size}, walks_per_node={args.reference_pretrain_walks_per_node},"
        f" batch_size={args.reference_pretrain_batch_size}, workers={args.reference_pretrain_workers})",
        flush=True,
    )
    pretrain_profiler = StageProfiler(device)
    pretrain_profiler.start()
    try:
        encoder = Node2VecEncoder(
            edge_index=directed_edge_index,
            num_nodes=int(data.num_nodes),
            emb_dim=int(args.emb_dim),
            walk_length=int(args.reference_pretrain_walk_length),
            context_size=int(args.reference_pretrain_context_size),
            walks_per_node=int(args.reference_pretrain_walks_per_node),
            num_negative_samples=1,
            p=1.0,
            q=1.0,
            sparse=True,
        ).to(device)
        loader = encoder.node2vec.loader(
            batch_size=int(args.reference_pretrain_batch_size), shuffle=True, num_workers=max(0, int(args.reference_pretrain_workers))
        )
        optimizer = torch.optim.SparseAdam(encoder.parameters(), lr=float(args.reference_pretrain_lr))
        started = time.time()
        encoder.train()
        for epoch in range(1, int(args.reference_pretrain_epochs) + 1):
            progress = tqdm(
                loader,
                total=len(loader),
                desc=f"Reference Node2Vec pretrain {epoch}/{args.reference_pretrain_epochs}",
                leave=False,
                dynamic_ncols=True,
                mininterval=1.0,
            )
            for batch_index, (pos_rw, neg_rw) in enumerate(progress, start=1):
                optimizer.zero_grad()
                loss = encoder.rw_loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()
                if batch_index % 100 == 0 or batch_index == len(loader):
                    progress.set_postfix(loss=f"{float(loss.detach().item()):.4f}")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        optimization_elapsed = time.time() - started
        embedding = encoder().detach().cpu()
    except BaseException:
        pretrain_profiler.stop()
        raise
    pretrain_resource = pretrain_profiler.stop()
    elapsed = float(pretrain_resource["sec"])
    metadata = dict(expected_metadata)
    metadata["embedding_sha256"] = _tensor_sha256(embedding)
    metadata["artifact_provenance_verified"] = True
    metadata["pretrain_optimization_seconds"] = float(optimization_elapsed)
    metadata["pretrain_total_seconds"] = float(elapsed)
    _publish_reference_embedding(path, embedding, metadata, expected_metadata)
    del optimizer, encoder
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print(f"Saved shared reference Node2Vec embedding: {path} (pretrain_sec={elapsed:.2f})", flush=True)
    return (embedding, path, elapsed, metadata, pretrain_resource)


def _train_reference_predictor_epoch(model, optimizer, features, train_pos, num_nodes, device, batch_size, show_batch_progress):
    model.train()
    loader = DataLoader(range(int(train_pos.size(0))), batch_size=max(1, int(batch_size)), shuffle=True)
    iterator = loader
    if show_batch_progress:
        iterator = tqdm(loader, total=len(loader), desc="Reference N2V predictor batches", leave=False, dynamic_ncols=True, mininterval=1.0)
    total_loss = 0.0
    total_examples = 0
    for perm in iterator:
        optimizer.zero_grad()
        pos_edge = train_pos[perm].to(device=device, dtype=torch.long, non_blocking=True).t().contiguous()
        pos_out = model.decode(features, pos_edge)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_edge = torch.randint(0, int(num_nodes), pos_edge.size(), dtype=torch.long, device=device)
        neg_out = model.decode(features, neg_edge)
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.encoder.parameters(), 1.0)
        torch.nn.utils.clip_grad_norm_(model.predictor.parameters(), 1.0)
        optimizer.step()
        count = int(pos_edge.size(1))
        total_loss += float(loss.detach().item()) * count
        total_examples += count
    return total_loss / max(1, total_examples)


def _evaluate(model, data, eval_edges, args, device, batch_size, compute_auc, *, test_only=False):
    profile = {}
    profiler = StageProfiler(device)
    context = None
    profiler.start()
    try:
        context = prepare_ogbl_evaluation(
            model=model,
            data=data,
            eval_edges=eval_edges,
            dataset_name=args.dataset,
            device=device,
            batch_size=batch_size,
            citation2_query_batch_size=args.citation2_query_batch_size,
            profile=profile,
            test_only=test_only,
        )
        evaluate = evaluate_ogbl_test if test_only else evaluate_ogbl_validation
        results = evaluate(context, compute_auc=compute_auc, profile=profile)
    finally:
        release_ogbl_evaluation(context)
        info = profiler.stop()
    return results, profile, info


def main():
    program_t0 = time.time()
    args = parse_args()
    cpu_threads = configure_torch_cpu_threads()
    device = resolve_ogbl_device(args.device)
    args.device = str(device)
    mode = args.mode
    protocol = _resolve_n2v_protocol(args.n2v_protocol, args.dataset)
    args.n2v_protocol_effective = protocol
    if protocol == "reference":
        _resolve_reference_recipe(args)
    if protocol == "reference" and float(args.reference_dropout) != 0.0:
        raise ValueError("The exact endpoint-only reference implementation requires --reference-dropout 0.")
    args.eval_cap = resolve_ogbl_eval_cap(args.eval_cap, mode, args.dataset)
    args = _resolve_run_defaults(args, protocol=protocol)
    metric_key = resolve_ogbl_metric(args.metric, args.dataset)
    protocol_metadata = ogbl_protocol_metadata(
        dataset=args.dataset,
        mode=mode,
        eval_cap=args.eval_cap,
        selection_metric=metric_key,
    )
    bind_protocol_metadata(args, protocol_metadata, metric_key)
    timed_out = False
    log_path = None
    if args.save_log:
        log_dir = os.path.join("results", "ogbl", args.mode, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "n2v.txt")
    print(f"Using device: {device}")
    print(f"torch_cpu_threads={cpu_threads}")
    print(f"n2v_protocol_requested={args.n2v_protocol}")
    print(f"n2v_protocol_effective={protocol}")
    print_ogbl_protocol(args, mode, protocol_metadata, metric_key)
    print(f"max_walk_batches={args.max_walk_batches}")
    print(f"rw_prefetch_batches={args.rw_prefetch_batches}")
    show_rw_progress = _show_rw_progress(args.rw_progress, args.dataset)
    print(f"rw_progress_effective={show_rw_progress}")
    set_seed(args.seed)
    t_data = time.time()
    mode_bundle = read_data(
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
    heart_candidate_provenance = persist_heart_candidate_metadata(args, mode_bundle)
    if args.heart_source_resolved is None:
        args.heart_source_resolved = mode_bundle.get("heart_source", "generated-online")
    data = mode_bundle.get("data")
    if data is None:
        (data, _split_edge, _base_eval_edges) = load_ogbl_splits(name=args.dataset, root=args.root)
    if getattr(data, "x", None) is not None and data.x.dtype != torch.float:
        data.x = data.x.to(torch.float)
    eval_edges_base = _build_eval_edges(mode_bundle, args.dataset, mode)
    ddi_dedup_t0 = time.time()
    (eval_edges_base, ddi_dedup_summaries) = prepare_ddi_grouped_eval_edges(
        eval_edges_base, dataset_name=args.dataset, model_name="node2vec", num_nodes=int(data.num_nodes), source_bundle=mode_bundle
    )
    ddi_dedup_prepare_sec = time.time() - ddi_dedup_t0
    data_load_sec = time.time() - t_data
    t_device = time.time()
    (eval_edges_base, eval_edges_cached, eval_edge_bytes) = cache_eval_edges_on_device(
        eval_edges_base, device, option=args.cache_eval_edges
    )
    device_prepare_sec = time.time() - t_device
    print(f"data_load_sec={data_load_sec:.2f}", flush=True)
    print(f"device_prepare_sec={device_prepare_sec:.2f}", flush=True)
    print(f"cache_eval_edges={args.cache_eval_edges}", flush=True)
    print(f"eval_edges_cached_on_device={eval_edges_cached}", flush=True)
    print(f"eval_edge_tensor_bytes={eval_edge_bytes}", flush=True)
    for summary in ddi_dedup_summaries:
        print(
            f"ddi_eval_dedup key={summary['key']} original_edges={summary['original_edges']} unique_edges={summary['unique_edges']} decode_fraction={summary['decode_fraction']:.6f} canonical_undirected={summary['canonical_undirected']} cpu_storage_bytes={summary['storage_nbytes']}",
            flush=True,
        )
    if ddi_dedup_summaries:
        print(f"ddi_eval_dedup_prepare_sec={ddi_dedup_prepare_sec:.2f}", flush=True)
    if mode_bundle.get("pool_per_side") is not None:
        for key in ("pool_setting", "pool_full_graph", "pool_cap_applied", "pool_sampling", "pool_requested_per_side", "pool_requested_total"):
            print(f"{key}={mode_bundle.get(key)}", flush=True)
        print(f"pool_per_side_effective={mode_bundle.get('pool_per_side')}", flush=True)
        print(f"pool_total_effective={mode_bundle.get('pool_total')}", flush=True)
    if mode_bundle.get("heart_candidate_universe") is not None:
        for key in ("heart_candidate_universe", "heart_candidate_graph_nodes", "heart_selection", "heart_negatives_requested_per_side", "heart_negatives_requested_total"):
            print(f"{key}={mode_bundle.get(key)}", flush=True)
        for label, key in (("heart_negatives_per_side_effective", "heart_negatives_per_side"), ("heart_negatives_total_effective", "heart_negatives_total")):
            print(f"{label}={mode_bundle.get(key)}", flush=True)
    if mode_bundle.get("negative_cache_path"):
        print(f"negative_cache_path={mode_bundle.get('negative_cache_path')}", flush=True)
    compute_auc = _should_compute_auc(args.compute_auc, mode)
    print(f"compute_auc_effective={compute_auc}", flush=True)
    selection_compute_auc = compute_auc and metric_key.lower() in {"auc", "ap"}
    print(f"selection_compute_auc_effective={selection_compute_auc}", flush=True)
    reference_algorithm_recipe_faithful = protocol == "reference" and _reference_model_protocol_faithful(args)
    reference_training_schedule_faithful = protocol == "reference" and _reference_training_schedule_faithful(args)
    reference_evaluation_scope_full = False
    reference_seed_scope_full = protocol == "reference" and int(args.seed) == 0 and (int(args.num_runs) == 10)
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED during data loading: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    edge_index = data.edge_index.to(torch.long).contiguous() if protocol != "reference" else None
    cached_sampler_state = None
    reference_features = None
    reference_pretrain_sec = 0.0
    reference_embedding_metadata = {}
    reference_pretrain_resource = empty_stage_info()
    args.reference_embedding_path_resolved = None
    if protocol == "reference":
        if args.reference_feature_composition == "raw-plus-node2vec" and getattr(data, "x", None) is None:
            raise ValueError(
                f"The released {args.dataset} Node2Vec baseline requires raw node features so they can be concatenated with Node2Vec embeddings."
            )
        directed_pretrain_edge_index = _reference_pretrain_edges(args, data, mode_bundle["split_edge"])
        if (
            directed_pretrain_edge_index.ndim != 2
            or directed_pretrain_edge_index.size(0) != 2
            or directed_pretrain_edge_index.numel() == 0
            or int(directed_pretrain_edge_index.min()) < 0
            or int(directed_pretrain_edge_index.max()) >= int(data.num_nodes)
        ):
            raise ValueError("Reference Node2Vec pretraining edges must be a nonempty [2, E] tensor within the dataset node range.")
        print(
            f"reference_pretrain_graph={args.reference_pretrain_graph_semantics} directed_edges={directed_pretrain_edge_index.size(1)}", flush=True
        )
        (reference_embedding, reference_embedding_path, reference_pretrain_sec, reference_embedding_metadata, reference_pretrain_resource) = (
            _load_or_pretrain_reference_embedding(args, data, device, directed_pretrain_edge_index)
        )
        del directed_pretrain_edge_index
        args.reference_embedding_path_resolved = reference_embedding_path
        n2v_features = reference_embedding.to(device=device, dtype=torch.float32, non_blocking=True)
        if args.reference_feature_composition == "node2vec-only":
            reference_features = n2v_features
        else:
            reference_features = torch.cat([data.x.to(device=device, dtype=torch.float32, non_blocking=True), n2v_features], dim=-1)
        reference_input_channels = int(reference_features.size(1))
        del reference_embedding, n2v_features
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        print(
            f"reference_feature_table_shape={tuple(reference_features.shape)} reference_feature_composition={args.reference_feature_composition} reference_embedding_path={reference_embedding_path}",
            flush=True,
        )
    else:
        _preflight_legacy_checkpoints(args, mode, int(data.num_nodes))
    reference_artifact_provenance_verified = bool(reference_embedding_metadata.get("artifact_provenance_verified", False))
    reference_model_protocol_faithful = reference_algorithm_recipe_faithful and reference_artifact_provenance_verified
    reference_full_reproduction_scope = (
        reference_model_protocol_faithful and reference_training_schedule_faithful and reference_evaluation_scope_full and reference_seed_scope_full
    )
    if protocol == "reference":
        print(f"reference_algorithm_recipe_faithful={str(reference_algorithm_recipe_faithful).lower()}", flush=True)
        print(f"reference_artifact_provenance_verified={str(reference_artifact_provenance_verified).lower()}", flush=True)
        print(f"reference_training_schedule_faithful={str(reference_training_schedule_faithful).lower()}", flush=True)
        print(f"reference_model_protocol_faithful={str(reference_model_protocol_faithful).lower()}", flush=True)
        print(f"reference_evaluation_scope_full={str(reference_evaluation_scope_full).lower()}", flush=True)
        print(f"reference_seed_scope_full={str(reference_seed_scope_full).lower()}", flush=True)
        if not reference_full_reproduction_scope:
            print(
                "WARNING: locally generated evaluation candidates are not an artifact-exact reference reproduction. Results record model, schedule, query, and candidate scope separately.",
                flush=True,
            )
    eval_batch_size = max(1, int(args.reference_eval_edge_batch_size)) if protocol == "reference" else args.eval_batch_size
    print(f"evaluation_edge_batch_size_effective={eval_batch_size}", flush=True)
    test_selected_metrics, test_aucs, test_aps, test_mrrs = [], [], [], []
    test_hits_any = {}
    timing_keys = ("train", "test", "eval", "inference", "testing", "mrr", "auc")
    resource_keys = tuple(
        f"{stage}_{metric}"
        for stage in ("train", "eval", "test")
        for metric in ("peak_cpu_rss_mb", "peak_cuda_allocated_mb", "peak_cuda_reserved_mb")
    )
    run_timings = {key: [] for key in timing_keys}
    run_resources = {key: [] for key in resource_keys}
    total_timings = dict.fromkeys(timing_keys, 0.0)
    run_resource_records = []
    for run_idx in range(args.num_runs):
        if timed_out or _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED before run {run_idx + 1}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
        run_seed = args.seed + run_idx
        set_seed(run_seed)
        eval_edges = dict(eval_edges_base)
        pos_train = eval_edges["pos_train_edge"]
        pos_valid = eval_edges["pos_valid_edge"]
        g = torch.Generator().manual_seed(run_seed + 3)
        idx = torch.randperm(pos_train.size(0), generator=g)[: pos_valid.size(0)]
        if idx.device != pos_train.device:
            idx = idx.to(device=pos_train.device, non_blocking=True)
        train_val_edge = pos_train[idx]
        if train_val_edge.device != pos_valid.device:
            train_val_edge = train_val_edge.to(device=pos_valid.device, non_blocking=True)
        eval_edges["train_val_edge"] = train_val_edge
        setup_t0 = time.time()
        if protocol == "reference":
            cpu_rng_state = torch.random.get_rng_state()
            model = ReferenceN2VLink(
                input_channels=reference_input_channels,
                hidden_channels=int(args.reference_hidden_channels),
                num_layers=int(args.reference_num_layers),
                predictor_layers=int(args.reference_predictor_layers),
                dropout=float(args.reference_dropout),
                node_encode_batch_size=int(args.reference_node_encode_batch_size),
            ).to(device)
            torch.random.set_rng_state(cpu_rng_state)
            model.reset_parameters()
            model.set_node_features(reference_features)
            encoder = model.encoder
            optimizer = torch.optim.Adam(model.parameters(), lr=float(args.reference_lr), weight_decay=0.0)
            sampler_setup = "shared-reference-embedding"
        else:
            encoder = Node2VecEncoder(
                edge_index=edge_index.to(device) if cached_sampler_state is None else None,
                num_nodes=int(data.num_nodes),
                emb_dim=args.emb_dim,
                walk_length=args.walk_length,
                context_size=args.context_size,
                walks_per_node=args.walks_per_node,
                num_negative_samples=args.num_neg_samples,
                p=args.p,
                q=args.q,
                sparse=True,
                sampler_state=cached_sampler_state,
            ).to(device)
            if cached_sampler_state is None:
                cached_sampler_state = encoder.sampler_state()
                sampler_setup = "built"
            else:
                sampler_setup = "reused"
            model = N2VLink(encoder, DotProductPredictor().to(device)).to(device)
            optimizer = torch.optim.SparseAdam(encoder.parameters(), lr=args.lr)
        print(f"run={run_idx + 1} node2vec_setup_sec={time.time() - setup_t0:.2f} sampler={sampler_setup}", flush=True)
        last_epoch = 0
        best_val_selected = float("-inf")
        best_test_selected_for_run = None
        best_test_other_metrics = {}
        metric_label = metric_key
        patience_counter = 0
        best_state_dict = None
        best_epoch = None
        run_timing = dict.fromkeys(timing_keys, 0.0)
        run_resource = dict.fromkeys(resource_keys, 0.0)
        run_test_completed = False
        pbar = tqdm(range(1, args.epochs + 1), desc=f"Run {run_idx + 1}/{args.num_runs} (seed={run_seed})", leave=False)
        for epoch in pbar:
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED before epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            train_profiler = StageProfiler(device)
            train_profiler.start()
            if protocol == "reference":
                loss = _train_reference_predictor_epoch(
                    model=model,
                    optimizer=optimizer,
                    features=reference_features,
                    train_pos=pos_train,
                    num_nodes=int(data.num_nodes),
                    device=device,
                    batch_size=int(args.reference_batch_size),
                    show_batch_progress=show_rw_progress,
                )
            else:
                loss = train_one_epoch(
                    encoder,
                    optimizer,
                    device,
                    args.rw_batch_size,
                    max_batches=args.max_walk_batches,
                    prefetch_batches=max(0, int(args.rw_prefetch_batches)),
                    show_batch_progress=show_rw_progress,
                    progress_desc=f"Run {run_idx + 1} epoch {epoch} walk batches",
                )
            last_epoch = epoch
            train_info = train_profiler.stop()
            epoch_train_sec = train_info["sec"]
            run_timing["train"] += epoch_train_sec
            total_timings["train"] += epoch_train_sec
            for metric in ("cpu_peak_rss_mb", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb"):
                key = f"train_peak_{metric.replace('_peak', '')}"
                run_resource[key] = max(run_resource[key], train_info[metric])
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after train epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                pbar.set_postfix(loss=f"{loss:.6f}")
                break
            if epoch % args.eval_steps != 0 and epoch != args.epochs:
                pbar.set_postfix(loss=f"{loss:.6f}")
                continue
            results_rank, eval_profile, eval_info = _evaluate(
                model, data, eval_edges, args, device, eval_batch_size, selection_compute_auc
            )
            stats = {key: float(eval_profile.get(key, 0.0)) for key in ("mrr_sec", "auc_sec")}
            epoch_eval_sec = eval_info["sec"]
            epoch_inference_sec = float(eval_profile.get("inference_sec", max(0.0, epoch_eval_sec - stats["mrr_sec"] - stats["auc_sec"])))
            epoch_testing_sec = float(eval_profile.get("testing_sec", stats["mrr_sec"] + stats["auc_sec"]))
            epoch_timings = {
                "eval": epoch_eval_sec,
                "inference": epoch_inference_sec,
                "testing": epoch_testing_sec,
                "mrr": stats["mrr_sec"],
                "auc": stats["auc_sec"],
            }
            for key, value in epoch_timings.items():
                run_timing[key] += value
                total_timings[key] += value
            for metric in ("cpu_peak_rss_mb", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb"):
                key = f"eval_peak_{metric.replace('_peak', '')}"
                run_resource[key] = max(run_resource[key], eval_info[metric])
            selected_key = find_result_key(results_rank, metric_key)
            if selected_key is None:
                raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(results_rank.keys())}")
            (_train_selected, val_selected, test_selected) = results_rank[selected_key]
            metric_label = selected_key
            if val_selected is not None and float(val_selected) > best_val_selected:
                best_val_selected = float(val_selected)
                best_test_selected_for_run = None if test_selected is None else float(test_selected)
                best_test_other_metrics = dict(results_rank)
                best_state_dict = _snapshot_state_dict_cpu(model.state_dict())
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            pbar.set_postfix(
                loss=f"{loss:.6f}",
                eval=f"{epoch_eval_sec:.2f}s",
                infer=f"{epoch_inference_sec:.2f}s",
                test=f"{epoch_testing_sec:.2f}s",
                mrr=f"{stats['mrr_sec']:.2f}s",
                auc=f"{stats['auc_sec']:.2f}s",
            )
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after eval epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            if patience_counter >= args.patience:
                break
        if best_state_dict is not None and (not timed_out):
            model.load_state_dict(best_state_dict)
            best_state_dict = None
            model.zero_grad(set_to_none=True)
            optimizer = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            final_test_results, final_test_profile, final_test_info = _evaluate(
                model, data, eval_edges, args, device, eval_batch_size, compute_auc, test_only=True
            )
            run_timing["test"] = float(final_test_info["sec"])
            for metric in ("cpu_peak_rss_mb", "cuda_peak_allocated_mb", "cuda_peak_reserved_mb"):
                run_resource[f"test_peak_{metric.replace('_peak', '')}"] = float(final_test_info[metric])
            run_test_completed = True
            total_timings["test"] += run_timing["test"]
            final_timings = {
                "inference": float(final_test_profile.get("inference_sec", 0.0)),
                "mrr": float(final_test_profile.get("mrr_sec", 0.0)),
                "auc": float(final_test_profile.get("auc_sec", 0.0)),
            }
            final_timings["testing"] = float(
                final_test_profile.get("testing_sec", final_timings["mrr"] + final_timings["auc"])
            )
            for key, value in final_timings.items():
                run_timing[key] += value
                total_timings[key] += value
            best_test_other_metrics = merge_ogbl_results(best_test_other_metrics, final_test_results)
            final_selected_key = find_result_key(best_test_other_metrics, metric_key)
            if final_selected_key is not None:
                (_, _, final_test_selected) = best_test_other_metrics[final_selected_key]
                if final_test_selected is not None:
                    best_test_selected_for_run = float(final_test_selected)
                    metric_label = final_selected_key
        for key in timing_keys:
            run_timings[key].append(run_timing[key])
        for key in resource_keys:
            run_resources[key].append(run_resource[key])
        run_resource_records.append(
            {
                "run": run_idx + 1,
                "seed": run_seed,
                "train_time_sec": run_timing["train"],
                "test_time_sec": run_timing["test"],
                **{key: run_resource[key] for key in resource_keys if key.startswith(("train_", "test_"))},
                "test_completed": run_test_completed,
            }
        )
        if best_test_selected_for_run is not None:
            test_selected_metrics.append(best_test_selected_for_run)
            for key, values in (("AUC", test_aucs), ("AP", test_aps), ("MRR", test_mrrs)):
                if key in best_test_other_metrics:
                    values.append(float(best_test_other_metrics[key][2]))
            for k, triple in best_test_other_metrics.items():
                if isinstance(k, str) and (k.startswith("Hits@") or k.startswith("mrr_hit")):
                    (_, _, t) = triple
                    test_hits_any.setdefault(k, []).append(float(t))
            tqdm.write(f"\n[RUN {run_idx + 1}] Best Val {metric_label}: {100 * best_val_selected:.6f}")
            tqdm.write(f"[RUN {run_idx + 1}] Test  {metric_label}: {100 * best_test_selected_for_run:.6f}")
        else:
            tqdm.write(f"\n[RUN {run_idx + 1}] No valid selected metric computed; skipping recording for this run.")
        tqdm.write(
            f"[RUN {run_idx + 1}] train_sec={run_timing['train']:.2f} test_sec={run_timing['test']:.2f}"
            f" eval_sec={run_timing['eval']:.2f} inference_sec={run_timing['inference']:.2f}"
            f" testing_sec={run_timing['testing']:.2f} mrr_sec={run_timing['mrr']:.2f} auc_sec={run_timing['auc']:.2f}"
            f" train_peak_cpu_rss_mb={run_resource['train_peak_cpu_rss_mb']:.2f}"
            f" test_peak_cpu_rss_mb={run_resource['test_peak_cpu_rss_mb']:.2f}"
            f" train_peak_cuda_allocated_mb={run_resource['train_peak_cuda_allocated_mb']:.2f}"
            f" test_peak_cuda_allocated_mb={run_resource['test_peak_cuda_allocated_mb']:.2f}"
        )
        checkpoint_state_dict = best_state_dict if best_state_dict is not None else _snapshot_state_dict_cpu(model.state_dict())
        checkpoint_epoch = last_epoch if best_epoch is None else best_epoch
        checkpoint_type = "final_model_state" if best_epoch is None else "best_validation_model_state"
        if protocol == "reference":
            model_config = {
                "protocol": "reference",
                "num_nodes": int(data.num_nodes),
                "raw_feature_dim": int(data.x.size(1)) if args.reference_feature_composition == "raw-plus-node2vec" else 0,
                "feature_composition": args.reference_feature_composition,
                "n2v_embedding_dim": int(args.emb_dim),
                "input_channels": int(reference_input_channels),
                "hidden_channels": int(args.reference_hidden_channels),
                "num_layers": int(args.reference_num_layers),
                "predictor_layers": int(args.reference_predictor_layers),
                "dropout": float(args.reference_dropout),
                "node_encode_batch_size": int(args.reference_node_encode_batch_size),
                "evaluation_edge_batch_size": int(eval_batch_size),
                "model_selection_evaluation": "validation-only",
                "final_test_evaluation": "once-from-restored-best-state",
                "n2v_embedding_path": args.reference_embedding_path_resolved,
                "embedding_sha256": reference_embedding_metadata.get("embedding_sha256"),
                "directed_edge_index_sha256": reference_embedding_metadata.get("directed_edge_index_sha256"),
                "pretrain_edge_index_sha256": reference_embedding_metadata.get("pretrain_edge_index_sha256"),
                "graph_semantics": reference_embedding_metadata.get("graph_semantics"),
                "torch_version": reference_embedding_metadata.get("torch_version"),
                "torch_geometric_version": reference_embedding_metadata.get("torch_geometric_version"),
                "cuda_version": reference_embedding_metadata.get("cuda_version"),
                "pretrain_seed": int(args.reference_pretrain_seed),
                "pretrain_epochs": int(args.reference_pretrain_epochs),
                "pretrain_walk_length": int(args.reference_pretrain_walk_length),
                "pretrain_context_size": int(args.reference_pretrain_context_size),
                "pretrain_walks_per_node": int(args.reference_pretrain_walks_per_node),
                "pretrain_batch_size": int(args.reference_pretrain_batch_size),
                "pretrain_workers": int(args.reference_pretrain_workers),
                "pretrain_learning_rate": float(args.reference_pretrain_lr),
                "learning_rate": float(args.reference_lr),
                "batch_size": int(args.reference_batch_size),
                "config_source": args.reference_recipe_source,
                "training_schedule": {"epochs": int(args.epochs), "eval_steps": int(args.eval_steps), "patience": int(args.patience)},
                "reference_training_schedule_faithful": bool(reference_training_schedule_faithful),
                "reference_model_protocol_faithful": bool(reference_model_protocol_faithful),
                "reference_algorithm_recipe_faithful": bool(reference_algorithm_recipe_faithful),
                "reference_artifact_provenance_verified": bool(reference_artifact_provenance_verified),
                "reference_evaluation_scope_full": bool(reference_evaluation_scope_full),
                "reference_seed_scope_full": bool(reference_seed_scope_full),
                "reference_full_reproduction_scope": bool(reference_full_reproduction_scope),
            }
        else:
            model_config = {
                "protocol": "legacy-direct",
                "num_nodes": int(data.num_nodes),
                "emb_dim": args.emb_dim,
                "walk_length": args.walk_length,
                "context_size": args.context_size,
                "walks_per_node": args.walks_per_node,
                "num_negative_samples": args.num_neg_samples,
                "p": args.p,
                "q": args.q,
                "sparse": True,
            }
        checkpoint_path = save_model_checkpoint(
            PROJECT_ROOT,
            checkpoint_state_dict,
            framework="ogbl",
            mode=mode,
            dataset=args.dataset,
            model_name="n2v",
            run_number=run_seed + 1,
            seed=run_seed,
            epoch=checkpoint_epoch,
            timed_out=timed_out,
            metric_name=metric_label,
            best_val=best_val_selected,
            best_test=best_test_selected_for_run,
            args=args,
            model_config=model_config,
            checkpoint_type=checkpoint_type,
            extra_metadata={
                "n2v_protocol": getattr(args, "n2v_protocol_effective", "legacy-direct"),
                "n2v_embedding_path": getattr(args, "reference_embedding_path_resolved", None),
            },
        )
        tqdm.write(f"[RUN {run_idx + 1}] Saved checkpoint: {checkpoint_path}")
        if args.save_gnn_feature and run_idx == args.num_runs - 1 and (not timed_out):
            model.eval()
            with torch.no_grad():
                z = model.embed(data).detach().cpu() if protocol == "reference" else encoder().detach().cpu()
            out_dir = os.path.join(args.root, args.dataset)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"entity_embedding": z}, os.path.join(out_dir, "gnn_feature"))
        del checkpoint_state_dict
        del optimizer, model, encoder
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if timed_out:
            print(f"Stopping remaining runs because runtime exceeded 24 hours. elapsed_sec={time.time() - program_t0:.2f}", flush=True)
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
    log(f"status: {'exceeded' if timed_out else 'completed within'} 24 hour runtime limit")
    log(f"runtime_limit_sec: {RUNTIME_LIMIT_SEC:.2f}")
    for key, value in (
        ("n2v_protocol_requested", args.n2v_protocol), ("n2v_protocol_effective", protocol),
        ("reference_embedding_path", args.reference_embedding_path_resolved or "not-applicable"),
        ("reference_embedding_sha256", reference_embedding_metadata.get("embedding_sha256", "not-applicable")),
        ("reference_directed_edge_index_sha256", reference_embedding_metadata.get("directed_edge_index_sha256", "not-applicable")),
        ("reference_pretrain_edge_index_sha256", reference_embedding_metadata.get("pretrain_edge_index_sha256", "not-applicable")),
        ("reference_feature_composition", getattr(args, "reference_feature_composition", "not-applicable")),
    ):
        log(f"{key}: {value}")
    log(f"reference_embedding_pretrain_sec: {reference_pretrain_sec:.2f}")
    completed_run_count = len(run_timings["train"])
    reference_pretrain_amortized_sec = reference_pretrain_sec / completed_run_count if completed_run_count else reference_pretrain_sec
    log(f"reference_embedding_pretrain_amortized_sec_per_run: {reference_pretrain_amortized_sec:.2f}")
    log(f"evaluation_edge_batch_size: {eval_batch_size}")
    log(f"cache_eval_edges: {args.cache_eval_edges}")
    log(f"eval_edges_cached_on_device: {eval_edges_cached}")
    log(f"eval_edge_tensor_bytes: {eval_edge_bytes}")
    log(f"device_prepare_sec: {device_prepare_sec:.2f}")
    log(f"reference_model_selection_evaluation: {('validation-only' if protocol == 'reference' else 'validation-and-test')}")
    for key, value in (
        ("reference_algorithm_recipe_faithful", reference_algorithm_recipe_faithful), ("reference_training_schedule_faithful", reference_training_schedule_faithful),
        ("reference_artifact_provenance_verified", reference_artifact_provenance_verified), ("reference_model_protocol_faithful", reference_model_protocol_faithful),
        ("reference_evaluation_scope_full", reference_evaluation_scope_full), ("reference_seed_scope_full", reference_seed_scope_full),
        ("reference_full_reproduction_scope", reference_full_reproduction_scope),
    ):
        log(f"{key}: {str(value).lower()}")
    log_protocol_summary(log, args, mode, protocol_metadata, metric_key, mode_bundle, device)
    for key, value in sorted(heart_candidate_provenance.items()):
        if value is not None and key != "heart_source":
            log(f"{key}: {value}")
    for summary in ddi_dedup_summaries:
        prefix = str(summary["key"]).replace("neg_", "").replace("_edge", "")
        log(f"ddi_{prefix}_negative_edges_original: {summary['original_edges']}")
        log(f"ddi_{prefix}_negative_edges_unique: {summary['unique_edges']}")
        log(f"ddi_{prefix}_decode_fraction: {summary['decode_fraction']:.6f}")
        log(f"ddi_{prefix}_canonical_undirected: {str(summary['canonical_undirected']).lower()}")
    if ddi_dedup_summaries:
        log(f"ddi_eval_dedup_prepare_sec: {ddi_dedup_prepare_sec:.2f}")
    log(f"data_load_sec: {data_load_sec:.2f}")
    log(f"train_total_sec: {total_timings['train']:.2f}")
    log(f"train_total_including_shared_pretrain_sec: {total_timings['train'] + reference_pretrain_sec:.2f}")
    for key in timing_keys[1:]:
        log(f"{key}_total_sec: {total_timings[key]:.2f}")
    log(f"total_wall_sec: {total_wall_sec:.2f}")
    log(f"cpu_rss_mb_current: {current_cpu_rss_mb():.2f}")
    log(f"cpu_rss_mb_peak_process: {peak_cpu_rss_mb():.2f}")
    log(f"shared_pretrain_peak_cpu_rss_mb: {reference_pretrain_resource['cpu_peak_rss_mb']:.2f}")
    log(f"shared_pretrain_peak_cuda_allocated_mb: {reference_pretrain_resource['cuda_peak_allocated_mb']:.2f}")
    log(f"shared_pretrain_peak_cuda_reserved_mb: {reference_pretrain_resource['cuda_peak_reserved_mb']:.2f}")
    log_run_statistics(log, run_timings, run_resources, run_resource_records)
    for metric, shared_key in (
        ("cpu_rss_mb", "cpu_peak_rss_mb"),
        ("cuda_allocated_mb", "cuda_peak_allocated_mb"),
        ("cuda_reserved_mb", "cuda_peak_reserved_mb"),
    ):
        values = run_resources[f"train_peak_{metric}"]
        log(f"train_peak_{metric}_including_shared_pretrain_max: {max([reference_pretrain_resource[shared_key], *values]):.2f}")
    log_aggregate_results(log, args.dataset, args.seed, metric_key, test_selected_metrics, test_aucs, test_aps, test_mrrs, test_hits_any)
    _write_summary(log_path, summary_lines)


if __name__ == "__main__":
    main()
