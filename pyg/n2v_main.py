import argparse
import gc
import hashlib
import json
import os
import random
import time
import numpy as np
import torch
import torch_geometric
from ogb.linkproppred import Evaluator
from torch.utils.data import DataLoader
from tqdm import tqdm
from model.node2vec_model import DotProductPredictor, Node2VecEncoder, ReferenceN2VLink
from .train_eval import (
    _finish_profile,
    _sync_if_profiled,
    evaluate_validation_only_from_embedding,
    evaluate_test_only_from_embedding,
    test,
    test_only,
    validation_only,
)
from .prepare_data import parse_pool_argument
from .main import (
    RUNTIME_LIMIT_SEC,
    _find_result_key,
    _merge_test_metrics,
    _print_candidate_metadata,
    _read_run_data,
    _record_test_metrics,
    _resolve_device,
    _resolve_eval_cap,
    _run_resource_profile,
    _runtime_exceeded,
    _save_model_checkpoint,
    _snapshot_state_dict_cpu,
    _write_timing_summary,
)
from utils.profiling import StageProfiler, current_cpu_rss_mb, peak_cpu_rss_mb
from .training import _clip_grad_norm_legacy
from .data_core import _atomic_save, _exclusive_cache_build
from utils.heart_protocol import persist_heart_candidate_metadata

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REFERENCE_CONFIG_FIELDS = ("hidden_channels", "num_layers", "predictor_layers", "dropout", "learning_rate", "weight_decay", "batch_size")
PLANETOID_REFERENCE_CONFIG = {
    name: dict(zip(_REFERENCE_CONFIG_FIELDS, values))
    for name, values in {
        "cora": (128, 1, 3, 0.1, 0.01, 1e-07, 1024),
        "citeseer": (256, 1, 3, 0.1, 0.01, 0.0, 512),
        "pubmed": (256, 3, 3, 0.1, 0.001, 1e-07, 1024),
    }.items()
}
PLANETOID_N2V_PRETRAIN = {
    "seed": 999, "embedding_dim": 128, "epochs": 70, "walk_length": 20, "context_size": 10,
    "walks_per_node": 10, "num_negative_samples": 1, "p": 1.0, "q": 1.0,
    "batch_size": 256, "workers": 4, "learning_rate": 0.01,
}


def _node2vec_state_dict(encoder, predictor):
    state_dict = {f"encoder.{key}": value for (key, value) in encoder.state_dict().items()}
    state_dict.update({f"predictor.{key}": value for (key, value) in predictor.state_dict().items()})
    return state_dict


def _load_node2vec_state_dict(encoder, predictor, state_dict):
    encoder.load_state_dict({key[len("encoder.") :]: value for (key, value) in state_dict.items() if key.startswith("encoder.")})
    predictor.load_state_dict({key[len("predictor.") :]: value for (key, value) in state_dict.items() if key.startswith("predictor.")})


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _configure_planetoid_n2v_reference_math(device):
    if torch.device(device).type != "cuda":
        return "cpu-default"
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    return "full-fp32-planetoid-reference"


def _reference_adam(parameters, *, learning_rate, weight_decay):
    return torch.optim.Adam(parameters, lr=float(learning_rate), weight_decay=float(weight_decay), foreach=False, fused=False)


def _normalize_planetoid_name(dataset):
    return str(dataset).strip().lower().replace("-", "")


def _resolve_n2v_protocol(option, dataset):
    option = str(option or "auto").strip().lower()
    if option == "auto":
        return "reference" if _normalize_planetoid_name(dataset) in PLANETOID_REFERENCE_CONFIG else "legacy-direct"
    if option == "reference" and _normalize_planetoid_name(dataset) not in PLANETOID_REFERENCE_CONFIG:
        raise ValueError("The reference Node2Vec protocol is defined only for Cora, CiteSeer, and PubMed. Use --n2v-protocol legacy-direct for extension datasets.")
    return option


def _resolve_run_defaults(args, protocol):
    if protocol == "reference":
        if args.epochs is None:
            args.epochs = 500
        elif int(args.epochs) > 500:
            print(f"WARNING: capping reference Node2Vec downstream epochs from {args.epochs} to 500.", flush=True)
            args.epochs = 500
        args.epochs = max(1, int(args.epochs))
        if args.patience is not None and int(args.patience) != 10:
            print(f"WARNING: reference Node2Vec requires patience=10; ignoring requested patience={args.patience}.", flush=True)
        args.patience = 10
        if args.eval_steps is None:
            args.eval_steps = 5
    else:
        if args.epochs is None:
            args.epochs = 300
        if args.patience is None:
            args.patience = 10
        if args.eval_steps is None:
            args.eval_steps = 5
    return args


def _tensor_sha256(tensor):
    tensor = tensor.detach().to(device="cpu").contiguous()
    byte_view = memoryview(tensor.numpy()).cast("B")
    digest = hashlib.sha256()
    chunk_bytes = 64 * 1024**2
    for start in range(0, len(byte_view), chunk_bytes):
        digest.update(byte_view[start : start + chunk_bytes])
    return digest.hexdigest()


def _planetoid_undirected_train_edge(train_pos, num_nodes):
    train_pos = train_pos.detach().to(device="cpu", dtype=torch.long)
    if train_pos.ndim != 2 or train_pos.size(1) != 2 or train_pos.numel() == 0:
        raise ValueError("Planetoid Node2Vec pretraining requires nonempty [E, 2] train_pos.")
    if int(train_pos.min()) < 0 or int(train_pos.max()) >= int(num_nodes):
        raise ValueError("Planetoid train_pos contains a node ID outside the dataset range.")
    directed = train_pos.t().contiguous()
    return torch.cat([directed, directed[[1, 0]]], dim=1).contiguous()


def _reference_embedding_recipe(dataset, edge_index, num_nodes):
    recipe = {
        "cache_format": "planetoid-reference-node2vec-v1",
        "protocol": "HeaRT-generated-two-stage-node2vec",
        "dataset": _normalize_planetoid_name(dataset),
        "num_nodes": int(num_nodes),
        "graph_semantics": "undirected-train-positive-edge-index",
        "edge_count": int(edge_index.size(1)),
        "edge_index_sha256": _tensor_sha256(edge_index),
        "torch_version": str(torch.__version__),
        "torch_geometric_version": str(torch_geometric.__version__),
        "cuda_version": str(torch.version.cuda),
    }
    recipe.update(PLANETOID_N2V_PRETRAIN)
    return recipe


def _reference_embedding_recipe_digest(recipe):
    return hashlib.sha256(json.dumps(recipe, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def _reference_embedding_cache_path(args, recipe):
    cache_dir = getattr(args, "n2v_cache_dir", None)
    if cache_dir:
        cache_dir = os.path.abspath(os.path.expanduser(cache_dir))
    else:
        cache_dir = os.path.join(os.path.abspath(args.root), _normalize_planetoid_name(args.dataset), "n2v_cache")
    digest = _reference_embedding_recipe_digest(recipe)
    return os.path.abspath(os.path.join(cache_dir, f"planetoid_n2v_v1_{digest}.pt"))


def _load_valid_reference_embedding(path, recipe):
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise TypeError("cache payload is not a dictionary")
    stored_recipe = payload.get("recipe")
    identity_keys = set(recipe) - {"cache_format", "protocol"}
    if not isinstance(stored_recipe, dict) or {key: stored_recipe.get(key) for key in identity_keys} != {
        key: recipe[key] for key in identity_keys
    }:
        raise ValueError("cache recipe does not match this graph and pretraining recipe")
    embedding = payload.get("embedding")
    expected_shape = (int(recipe["num_nodes"]), int(recipe["embedding_dim"]))
    if not torch.is_tensor(embedding) or tuple(embedding.shape) != expected_shape:
        raise ValueError(f"cache embedding shape is invalid; expected {expected_shape}")
    embedding = embedding.detach().to(dtype=torch.float32, device="cpu").contiguous()
    checksum = _tensor_sha256(embedding)
    if payload.get("embedding_sha256") != checksum:
        raise ValueError("cache embedding checksum does not match")
    metadata = dict(payload)
    metadata.pop("embedding", None)
    metadata["embedding_sha256"] = checksum
    return (embedding, metadata)


def _pretrain_planetoid_embedding(edge_index, num_nodes, device):
    config = PLANETOID_N2V_PRETRAIN
    set_seed(int(config["seed"]))
    encoder = Node2VecEncoder(
        edge_index=edge_index,
        num_nodes=int(num_nodes),
        emb_dim=int(config["embedding_dim"]),
        walk_length=int(config["walk_length"]),
        context_size=int(config["context_size"]),
        walks_per_node=int(config["walks_per_node"]),
        num_negative_samples=int(config["num_negative_samples"]),
        p=float(config["p"]),
        q=float(config["q"]),
        sparse=True,
    ).to(device)
    loader = encoder.node2vec.loader(batch_size=int(config["batch_size"]), shuffle=True, num_workers=int(config["workers"]))
    optimizer = torch.optim.SparseAdam(encoder.parameters(), lr=float(config["learning_rate"]))
    encoder.train()
    for epoch in range(1, int(config["epochs"]) + 1):
        progress = tqdm(
            loader,
            total=len(loader),
            desc=f"Reference Node2Vec pretrain {epoch}/{config['epochs']}",
            leave=False,
            dynamic_ncols=True,
            mininterval=1.0,
        )
        for pos_rw, neg_rw in progress:
            optimizer.zero_grad()
            loss = encoder.rw_loss(pos_rw.to(device), neg_rw.to(device))
            loss.backward()
            optimizer.step()
        progress.close()
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    embedding = encoder().detach().to(device="cpu", dtype=torch.float32).contiguous()
    del optimizer, encoder
    gc.collect()
    if torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    return embedding


def _load_or_pretrain_planetoid_embedding(args, data, device):
    num_nodes = int(data["x"].size(0))
    edge_index = _planetoid_undirected_train_edge(data["train_pos"], num_nodes)
    recipe = _reference_embedding_recipe(args.dataset, edge_index, num_nodes)
    path = _reference_embedding_cache_path(args, recipe)
    cache_dir = os.path.dirname(path)
    existing_paths = [path]
    if os.path.isdir(cache_dir):
        existing_paths.extend(
            os.path.join(cache_dir, name)
            for name in sorted(os.listdir(cache_dir))
            if name.startswith("planetoid_n2v_v1_") and name.endswith(".pt") and os.path.join(cache_dir, name) != path
        )

    def load_existing(candidate_path):
        (embedding, metadata) = _load_valid_reference_embedding(candidate_path, recipe)
        print(f"Loaded shared reference Node2Vec embedding: {candidate_path}", flush=True)
        return (embedding, candidate_path, 0.0, metadata)

    for candidate_path in existing_paths:
        if not os.path.isfile(candidate_path):
            continue
        try:
            return load_existing(candidate_path)
        except Exception as exc:
            print(f"WARNING: ignoring incompatible Node2Vec cache {candidate_path}: {exc}", flush=True)
    print(f"Acquiring Node2Vec embedding cache lock: {path}.lock", flush=True)
    with _exclusive_cache_build(path):
        if os.path.isfile(path):
            try:
                return load_existing(path)
            except Exception as exc:
                print(f"WARNING: rebuilding invalid Node2Vec cache {path}: {exc}", flush=True)
        print(
            "Pretraining shared HeaRT Planetoid Node2Vec embedding (seed=999, epochs=70, walk=20, context=10, walks/node=10, batch=256, lr=0.01)",
            flush=True,
        )
        started = time.time()
        embedding = _pretrain_planetoid_embedding(edge_index, num_nodes, device)
        elapsed = time.time() - started
        checksum = _tensor_sha256(embedding)
        payload = {"embedding": embedding, "recipe": recipe, "embedding_sha256": checksum, "pretrain_seconds": float(elapsed)}
        _atomic_save(payload, path)
        metadata = dict(payload)
        metadata.pop("embedding")
        print(f"Saved shared reference Node2Vec embedding: {path} (pretrain_sec={elapsed:.2f})", flush=True)
        return (embedding, path, float(elapsed), metadata)


def train_one_epoch_node2vec(encoder, optimizer, device, batch_size):
    encoder.train()
    encoder.to(device)
    loader = encoder.loader(batch_size=batch_size, shuffle=True)
    sum_loss = 0.0
    for pos_rw, neg_rw in loader:
        pos_rw = pos_rw.to(device)
        neg_rw = neg_rw.to(device)
        optimizer.zero_grad(set_to_none=True)
        loss = encoder.rw_loss(pos_rw, neg_rw)
        loss.backward()
        optimizer.step()
        sum_loss += float(loss.item())
    return sum_loss / max(len(loader), 1)


class _N2VEdgeModel(torch.nn.Module):

    def __init__(self, predictor, encoder=None):
        super().__init__()
        self.decode_is_symmetric = bool(getattr(predictor, "decode_is_symmetric", False))
        self.decode_is_dedup_safe = True
        self.predictor = predictor
        self.encoder = encoder
        self.decode_batch_size = 262144

    def embed(self, data=None):
        del data
        return self.encoder()

    def decode(self, z, edge_index):
        return torch.sigmoid(self.predictor(z, edge_index).view(-1))


class _ReferenceN2VEdgeModel(torch.nn.Module):

    def __init__(self, predictor, row_batch_size=1024):
        super().__init__()
        self.decode_is_symmetric = True
        self.decode_is_dedup_safe = True
        self.reference_evaluation_row_batch_size = max(1, int(row_batch_size))
        self.reference_evaluation_negative_layout = "grouped"
        self.predictor = predictor

    def decode(self, z, edge_index):
        (src, dst) = edge_index
        return self.predictor(z[src], z[dst]).view(-1)


def _train_planetoid_reference_predictor_epoch(model, optimizer, features, train_pos, num_nodes, device, batch_size=1024):
    model.train()
    loader = DataLoader(range(int(train_pos.size(0))), batch_size=max(1, int(batch_size)), shuffle=True)
    total_loss = 0.0
    total_examples = 0
    for perm in loader:
        optimizer.zero_grad()
        z = model.encoder(features)
        pos_edge = train_pos[perm].to(device=device, dtype=torch.long, non_blocking=True).t().contiguous()
        pos_out = model.predictor(z[pos_edge[0]], z[pos_edge[1]]).view(-1)
        pos_loss = -torch.log(pos_out + 1e-15).mean()
        neg_edge = torch.randint(0, int(num_nodes), pos_edge.size(), dtype=torch.long, device=device)
        neg_out = model.predictor(z[neg_edge[0]], z[neg_edge[1]]).view(-1)
        neg_loss = -torch.log(1.0 - neg_out + 1e-15).mean()
        loss = pos_loss + neg_loss
        loss.backward()
        _clip_grad_norm_legacy(model.encoder.parameters(), 1.0)
        _clip_grad_norm_legacy(model.predictor.parameters(), 1.0)
        optimizer.step()
        count = int(pos_edge.size(1))
        total_loss += float(loss.detach().item()) * count
        total_examples += count
    return total_loss / max(1, total_examples)


def _reference_evaluation_context(model, device, batch_size, profile):
    model.eval()
    _sync_if_profiled(profile, device)
    started = time.time()
    z = model.embed().to(device)
    _finish_profile(profile, device, "inference_sec", started)
    edge_model = _ReferenceN2VEdgeModel(model.predictor, getattr(model, "reference_evaluation_row_batch_size", batch_size)).to(device)
    return edge_model, z


@torch.no_grad()
def _planetoid_reference_validation_only(model, data_dict, device, batch_size, profile=None, include_auc=True, include_hits=True):
    edge_model, z = _reference_evaluation_context(model, device, batch_size, profile)
    return evaluate_validation_only_from_embedding(
        edge_model, z, data_dict, batch_size, profile=profile, include_auc=include_auc, include_hits=include_hits
    )


@torch.no_grad()
def _planetoid_reference_test_only(model, data_dict, device, batch_size, profile=None, *, include_auc=True):
    edge_model, z = _reference_evaluation_context(model, device, batch_size, profile)
    started = time.time()
    results = evaluate_test_only_from_embedding(edge_model, z, data_dict, batch_size, include_auc=include_auc)
    _finish_profile(profile, device, "testing_sec", started)
    return results


def parse_args():
    parser = argparse.ArgumentParser(description="Node2Vec link prediction")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--n2v-protocol", choices=["auto", "reference", "legacy-direct"], default="auto")
    parser.add_argument("--n2v-cache-dir")
    parser.add_argument("--metric", type=str, default="mrr")
    parser.add_argument("--compute-auc", choices=["yes", "no"], default="yes")
    parser.add_argument("--mode", choices=["heart", "all"], default="heart")
    parser.add_argument("--eval-cap", "--eval_cap", dest="eval_cap", type=int, default=None)
    parser.add_argument("--pool", type=parse_pool_argument, default=10000)
    parser.add_argument("--heart-negatives", "--heart_negatives", dest="heart_negatives", type=int, choices=[500], default=500)
    parser.add_argument(
        "--planetoid-input-root",
        type=str,
        default=None,
        help="Fixed Planetoid positive-split and gnn_feature root.",
    )
    parser.add_argument("--heart-backend", choices=["auto", "gpu", "dense"], default="auto")
    parser.add_argument("--heart-batch-size", type=int, default=2048)
    parser.add_argument("--heart-ppr-iters", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--device")
    parser.add_argument("--emb-dim", type=int, default=128)
    parser.add_argument("--walk-length", type=int, default=20)
    parser.add_argument("--context-size", type=int, default=10)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--num-neg-samples", type=int, default=1)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--rw-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=1024)
    parser.add_argument("--eval-steps", type=int, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--base-seed", "--seed", dest="seed", type=int, default=0)
    parser.add_argument("--num-runs", type=int, default=5)
    parser.add_argument("--save-gnn-feature", action="store_true", default=False)
    parser.add_argument("--save-log", action="store_true", default=True)
    return parser.parse_args()


def main():
    program_t0 = time.time()
    args = parse_args()
    device = _resolve_device(args.device)
    args.device = str(device)
    protocol = _resolve_n2v_protocol(args.n2v_protocol, args.dataset)
    args.n2v_protocol_effective = protocol
    _resolve_run_defaults(args, protocol)
    reference_protocol = protocol == "reference"
    reference_config = PLANETOID_REFERENCE_CONFIG[_normalize_planetoid_name(args.dataset)] if reference_protocol else None
    reference_matmul_precision = _configure_planetoid_n2v_reference_math(device) if reference_protocol else "legacy-default"
    args.eval_cap = _resolve_eval_cap(args.eval_cap, args.mode, args.dataset)
    metric_key = args.metric.strip()
    compute_auc = args.compute_auc == "yes"
    selection_requires_auc = metric_key.lower() in {"auc", "ap"}
    if selection_requires_auc and not compute_auc:
        raise ValueError("--metric AUC/AP requires --compute-auc yes.")
    timed_out = False
    log_path = None
    if args.save_log:
        log_dir = os.path.join("results", "pyg", args.mode, args.dataset)
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "n2v.txt")
    for line in (
        f"Using device: {device}", f"matmul_precision={reference_matmul_precision}", f"n2v_protocol_requested={args.n2v_protocol}",
        f"n2v_protocol_effective={protocol}", f"Selection/reporting metric: {args.metric}", f"Evaluation mode: {args.mode}",
        f"compute_auc_effective={compute_auc}",
        f"eval_cap={args.eval_cap}", f"pool={args.pool}", "heart_negatives=generated-online",
        f"heart_negatives_total_requested={args.heart_negatives}", f"runtime_limit_sec={RUNTIME_LIMIT_SEC}",
        f"runtime_limit_hours={RUNTIME_LIMIT_SEC / 3600:.2f}",
    ):
        print(line, flush=True)
    t_data = time.time()
    data = _read_run_data(args, device)
    heart_candidate_metadata = persist_heart_candidate_metadata(args, data)
    heart_validation_only = args.mode == "heart"
    selection_validation_only = reference_protocol or heart_validation_only
    selection_requires_hits = metric_key.lower() != "mrr"
    for key, value in heart_candidate_metadata.items():
        print(f"{key}={(value if value is not None else 'not-applicable')}", flush=True)
    print(
        "model_selection_evaluation=" + ("validation-final-test-only" if selection_validation_only else "validation-and-test"), flush=True
    )
    _print_candidate_metadata(data)
    data_load_sec = time.time() - t_data
    print(f"data_load_sec={data_load_sec:.2f}", flush=True)
    if _runtime_exceeded(program_t0):
        timed_out = True
        print(f"RUNTIME_LIMIT_EXCEEDED during data loading: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
    x = data["x"].to(device)
    num_nodes = x.size(0)
    reference_features = None
    reference_embedding = None
    reference_embedding_path = None
    reference_embedding_metadata = {}
    reference_pretrain_sec = 0.0
    reference_base_feature_sha256 = None
    reference_eval_batch_size = (
        max(int(args.eval_batch_size), 262144)
        if reference_protocol and device.type == "cuda"
        else max(int(args.eval_batch_size), 65536) if reference_protocol else int(args.eval_batch_size)
    )
    if reference_protocol:
        reference_base_feature_sha256 = _tensor_sha256(data["x"].to(dtype=torch.float32))
        (reference_embedding, reference_embedding_path, reference_pretrain_sec, reference_embedding_metadata) = _load_or_pretrain_planetoid_embedding(
            args, data, device
        )
        reference_features = torch.cat(
            [x.to(dtype=torch.float32), reference_embedding.to(device=device, dtype=torch.float32, non_blocking=True)], dim=-1
        ).contiguous()
        args.reference_embedding_path_resolved = reference_embedding_path
        args.reference_embedding_sha256 = reference_embedding_metadata.get("embedding_sha256")
        for line in (
            f"reference_n2v_embedding_path={reference_embedding_path}", f"reference_n2v_embedding_sha256={reference_embedding_metadata.get('embedding_sha256')}",
            f"reference_downstream_feature_shape={tuple(reference_features.shape)}", f"reference_base_feature_sha256={reference_base_feature_sha256}",
            f"reference_shared_pretrain_sec={reference_pretrain_sec:.2f}", f"reference_evaluation_edge_batch_size={reference_eval_batch_size}",
            f"reference_downstream_batch_size={reference_config['batch_size']}",
        ):
            print(line, flush=True)
    evaluator_hit = Evaluator(name="ogbl-collab")
    evaluator_mrr = Evaluator(name="ogbl-citation2")
    test_selected_metrics = []
    test_aucs = []
    test_aps = []
    test_mrrs = []
    test_hits_any = {}
    run_train_secs = []
    run_eval_secs = []
    run_test_secs = []
    run_inference_secs = []
    run_testing_secs = []
    run_resource_profiles = []
    total_train_sec = 0.0
    total_eval_sec = 0.0
    total_test_sec = 0.0
    total_inference_sec = 0.0
    total_testing_sec = 0.0
    train_peak_cpu = eval_peak_cpu = test_peak_cpu = 0.0
    train_peak_cuda = eval_peak_cuda = test_peak_cuda = 0.0
    train_peak_cuda_reserved = eval_peak_cuda_reserved = 0.0
    test_peak_cuda_reserved = 0.0
    for run_idx in range(args.num_runs):
        if timed_out or _runtime_exceeded(program_t0):
            timed_out = True
            print(f"RUNTIME_LIMIT_EXCEEDED before run {run_idx + 1}: elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
        seed = args.seed + run_idx
        set_seed(seed)
        if reference_protocol:
            cpu_rng_state = torch.random.get_rng_state()
            model = ReferenceN2VLink(
                input_channels=int(reference_features.size(1)),
                hidden_channels=int(reference_config["hidden_channels"]),
                num_layers=int(reference_config["num_layers"]),
                predictor_layers=int(reference_config["predictor_layers"]),
                dropout=float(reference_config["dropout"]),
                node_encode_batch_size=262144,
            ).to(device)
            model.reference_evaluation_row_batch_size = int(reference_config["batch_size"])
            model.reference_evaluation_negative_layout = "grouped"
            torch.random.set_rng_state(cpu_rng_state)
            model.reset_parameters()
            model.set_node_features(reference_features)
            encoder = model.encoder
            predictor = model.predictor
            optimizer = _reference_adam(
                model.parameters(), learning_rate=reference_config["learning_rate"], weight_decay=reference_config["weight_decay"]
            )
        else:
            train_edge = data["train_pos"].t().contiguous()
            train_edge = torch.cat([train_edge, train_edge[[1, 0]]], dim=1).to(device)
            encoder = Node2VecEncoder(
                edge_index=train_edge,
                num_nodes=num_nodes,
                emb_dim=args.emb_dim,
                walk_length=args.walk_length,
                context_size=args.context_size,
                walks_per_node=args.walks_per_node,
                num_negative_samples=args.num_neg_samples,
                p=args.p,
                q=args.q,
                sparse=True,
            ).to(device)
            predictor = DotProductPredictor().to(device)
            model = _N2VEdgeModel(predictor, encoder).to(device)
            optimizer = torch.optim.SparseAdam(encoder.parameters(), lr=args.lr)
        last_epoch = 0
        best_val_selected = float("-inf")
        best_test_selected_for_run = None
        best_test_other_metrics = {}
        metric_label = metric_key
        patience_counter = 0
        best_state_dict = None
        best_epoch = None
        run_train_sec = 0.0
        run_eval_sec = 0.0
        run_inference_sec = 0.0
        run_testing_sec = 0.0
        run_train_peak_cpu = 0.0
        run_eval_peak_cpu = 0.0
        run_train_peak_cuda = 0.0
        run_eval_peak_cuda = 0.0
        run_train_peak_cuda_reserved = 0.0
        run_eval_peak_cuda_reserved = 0.0
        run_test_sec = 0.0
        run_test_peak_cpu = 0.0
        run_test_peak_cuda = 0.0
        run_test_peak_cuda_reserved = 0.0
        run_test_completed = False
        pbar = tqdm(
            range(1, args.epochs + 1),
            desc=(
                f"Reference N2V downstream Run {run_idx + 1}/{args.num_runs} (seed={seed})"
                if reference_protocol
                else f"Node2Vec Run {run_idx + 1}/{args.num_runs} (seed={seed})"
            ),
            leave=False,
            dynamic_ncols=True,
            mininterval=0.2,
        )
        for epoch in pbar:
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED before epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            train_profiler = StageProfiler(device)
            train_profiler.start()
            if reference_protocol:
                train_loss = _train_planetoid_reference_predictor_epoch(
                    model, optimizer, reference_features, data["train_pos"], num_nodes, device, batch_size=int(reference_config["batch_size"])
                )
            else:
                train_loss = train_one_epoch_node2vec(encoder, optimizer, device, args.rw_batch_size)
            last_epoch = epoch
            train_info = train_profiler.stop()
            epoch_train_sec = train_info["sec"]
            run_train_peak_cpu = max(run_train_peak_cpu, train_info["cpu_peak_rss_mb"])
            run_train_peak_cuda = max(run_train_peak_cuda, train_info["cuda_peak_allocated_mb"])
            run_train_peak_cuda_reserved = max(run_train_peak_cuda_reserved, train_info["cuda_peak_reserved_mb"])
            train_peak_cpu = max(train_peak_cpu, train_info["cpu_peak_rss_mb"])
            train_peak_cuda = max(train_peak_cuda, train_info["cuda_peak_allocated_mb"])
            train_peak_cuda_reserved = max(train_peak_cuda_reserved, train_info["cuda_peak_reserved_mb"])
            run_train_sec += epoch_train_sec
            total_train_sec += epoch_train_sec
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after train epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                pbar.set_postfix(loss=f"{train_loss:.4f}")
                break
            if epoch % args.eval_steps != 0 and epoch != args.epochs:
                pbar.set_postfix(loss=f"{train_loss:.4f}")
                continue
            eval_profile = {}
            eval_profiler = StageProfiler(device)
            eval_profiler.start()
            if reference_protocol:
                results_rank = _planetoid_reference_validation_only(
                    model,
                    data,
                    device,
                    reference_eval_batch_size,
                    profile=eval_profile,
                    include_auc=selection_requires_auc,
                    include_hits=selection_requires_hits,
                )
            elif heart_validation_only:
                results_rank = validation_only(
                    model, data, x, args.eval_batch_size, profile=eval_profile,
                    include_auc=selection_requires_auc, include_hits=selection_requires_hits
                )
            else:
                (results_rank, _) = test(
                    model,
                    data,
                    x,
                    evaluator_hit,
                    evaluator_mrr,
                    args.eval_batch_size,
                    profile=eval_profile,
                    include_auc=compute_auc,
                )
            eval_info = eval_profiler.stop()
            epoch_eval_sec = eval_info["sec"]
            epoch_inference_sec = float(eval_profile.get("inference_sec", epoch_eval_sec))
            epoch_testing_sec = float(eval_profile.get("testing_sec", 0.0))
            run_eval_peak_cpu = max(run_eval_peak_cpu, eval_info["cpu_peak_rss_mb"])
            run_eval_peak_cuda = max(run_eval_peak_cuda, eval_info["cuda_peak_allocated_mb"])
            run_eval_peak_cuda_reserved = max(run_eval_peak_cuda_reserved, eval_info["cuda_peak_reserved_mb"])
            eval_peak_cpu = max(eval_peak_cpu, eval_info["cpu_peak_rss_mb"])
            eval_peak_cuda = max(eval_peak_cuda, eval_info["cuda_peak_allocated_mb"])
            eval_peak_cuda_reserved = max(eval_peak_cuda_reserved, eval_info["cuda_peak_reserved_mb"])
            run_eval_sec += epoch_eval_sec
            run_inference_sec += epoch_inference_sec
            run_testing_sec += epoch_testing_sec
            total_eval_sec += epoch_eval_sec
            total_inference_sec += epoch_inference_sec
            total_testing_sec += epoch_testing_sec
            selected_key = _find_result_key(results_rank, metric_key)
            if selected_key is None:
                raise KeyError(f"Selection metric '{metric_key}' not found in results. Available: {list(results_rank.keys())}")
            (_, val_selected, test_selected) = results_rank[selected_key]
            metric_label = selected_key
            improved = val_selected is not None and float(val_selected) > best_val_selected
            if improved:
                best_val_selected = float(val_selected)
                best_test_selected_for_run = None if test_selected is None else float(test_selected)
                best_test_other_metrics = dict(results_rank)
                best_state_dict = _snapshot_state_dict_cpu(
                    model.state_dict() if reference_protocol else _node2vec_state_dict(encoder, predictor)
                )
                best_epoch = epoch
                patience_counter = 0
            else:
                patience_counter += 1
            pbar.set_postfix(
                loss=f"{train_loss:.4f}",
                eval=f"{epoch_eval_sec:.2f}s",
                infer=f"{epoch_inference_sec:.2f}s",
                test=f"{epoch_testing_sec:.2f}s",
            )
            if _runtime_exceeded(program_t0):
                timed_out = True
                tqdm.write(f"RUNTIME_LIMIT_EXCEEDED after eval epoch {epoch}: elapsed_sec={time.time() - program_t0:.2f}")
                break
            if patience_counter >= args.patience:
                break
        if best_state_dict is not None and (not timed_out):
            if reference_protocol:
                model.load_state_dict(best_state_dict)
            else:
                _load_node2vec_state_dict(encoder, predictor, best_state_dict)
            best_state_dict = None
            encoder.zero_grad(set_to_none=True)
            predictor.zero_grad(set_to_none=True)
            optimizer = None
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if selection_validation_only and (not selection_requires_auc):
                deferred_profile = {}
                deferred_profiler = StageProfiler(device)
                deferred_profiler.start()
                try:
                    if reference_protocol:
                        final_validation_metrics = _planetoid_reference_validation_only(
                            model,
                            data,
                            device,
                            reference_eval_batch_size,
                            profile=deferred_profile,
                            include_auc=compute_auc,
                        )
                    else:
                        final_validation_metrics = validation_only(
                            model,
                            data,
                            x,
                            args.eval_batch_size,
                            profile=deferred_profile,
                            include_auc=compute_auc,
                        )
                finally:
                    deferred_info = deferred_profiler.stop()
                deferred_selected_key = _find_result_key(final_validation_metrics, metric_key)
                if deferred_selected_key is None:
                    raise RuntimeError(f"Restored best Node2Vec checkpoint did not reproduce the selection metric {metric_key!r}.")
                (_, deferred_val, _) = final_validation_metrics[deferred_selected_key]
                if deferred_val is None or float(deferred_val) != float(best_val_selected):
                    raise RuntimeError(
                        f"Restored best Node2Vec checkpoint changed the selected validation value: {deferred_val!r} != {best_val_selected!r}."
                    )
                best_test_other_metrics = dict(final_validation_metrics)
                deferred_sec = float(deferred_info["sec"])
                deferred_inference_sec = float(deferred_profile.get("inference_sec", 0.0))
                deferred_testing_sec = float(deferred_profile.get("testing_sec", 0.0))
                run_eval_sec += deferred_sec
                total_eval_sec += deferred_sec
                run_inference_sec += deferred_inference_sec
                total_inference_sec += deferred_inference_sec
                run_testing_sec += deferred_testing_sec
                total_testing_sec += deferred_testing_sec
                run_eval_peak_cpu = max(run_eval_peak_cpu, float(deferred_info["cpu_peak_rss_mb"]))
                run_eval_peak_cuda = max(run_eval_peak_cuda, float(deferred_info["cuda_peak_allocated_mb"]))
                run_eval_peak_cuda_reserved = max(run_eval_peak_cuda_reserved, float(deferred_info["cuda_peak_reserved_mb"]))
                eval_peak_cpu = max(eval_peak_cpu, float(deferred_info["cpu_peak_rss_mb"]))
                eval_peak_cuda = max(eval_peak_cuda, float(deferred_info["cuda_peak_allocated_mb"]))
                eval_peak_cuda_reserved = max(eval_peak_cuda_reserved, float(deferred_info["cuda_peak_reserved_mb"]))
            final_profile = {}
            final_profiler = StageProfiler(device)
            final_profiler.start()
            try:
                if reference_protocol:
                    final_test_metrics = _planetoid_reference_test_only(
                        model,
                        data,
                        device,
                        reference_eval_batch_size,
                        profile=final_profile,
                        include_auc=compute_auc,
                    )
                else:
                    final_test_metrics = test_only(
                        model,
                        data,
                        x,
                        args.eval_batch_size,
                        profile=final_profile,
                        include_auc=compute_auc,
                    )
            finally:
                final_info = final_profiler.stop()
            run_test_sec = float(final_info["sec"])
            run_test_peak_cpu = float(final_info["cpu_peak_rss_mb"])
            run_test_peak_cuda = float(final_info["cuda_peak_allocated_mb"])
            run_test_peak_cuda_reserved = float(final_info["cuda_peak_reserved_mb"])
            run_test_completed = True
            total_test_sec += run_test_sec
            test_peak_cpu = max(test_peak_cpu, run_test_peak_cpu)
            test_peak_cuda = max(test_peak_cuda, run_test_peak_cuda)
            test_peak_cuda_reserved = max(test_peak_cuda_reserved, run_test_peak_cuda_reserved)
            best_test_other_metrics = _merge_test_metrics(best_test_other_metrics, final_test_metrics)
            final_selected_key = _find_result_key(final_test_metrics, metric_key)
            if final_selected_key is not None:
                final_selected = final_test_metrics[final_selected_key]
                if final_selected is not None:
                    best_test_selected_for_run = float(final_selected)
                    metric_label = final_selected_key
        run_train_secs.append(run_train_sec)
        run_eval_secs.append(run_eval_sec)
        run_test_secs.append(run_test_sec)
        run_inference_secs.append(run_inference_sec)
        run_testing_secs.append(run_testing_sec)
        run_resource_profiles.append(_run_resource_profile(
            run_idx + 1, seed, last_epoch, timed_out, (run_train_sec, run_test_sec, run_eval_sec),
            ((run_train_peak_cpu, run_train_peak_cuda, run_train_peak_cuda_reserved),
             (run_test_peak_cpu, run_test_peak_cuda, run_test_peak_cuda_reserved),
             (run_eval_peak_cpu, run_eval_peak_cuda, run_eval_peak_cuda_reserved)), run_test_completed))
        recorded = _record_test_metrics(best_test_selected_for_run, best_test_other_metrics, test_selected_metrics,
                                        {"AUC": test_aucs, "AP": test_aps, "MRR": test_mrrs}, test_hits_any)
        if recorded:
            tqdm.write(
                f"[Run {run_idx + 1}] Best Val {metric_label}={best_val_selected} | Test {metric_label}={best_test_selected_for_run}"
            )
        else:
            tqdm.write(f"[Run {run_idx + 1}] No valid selected metric computed; skipping recording for this run.")
        tqdm.write(
            f"[Run {run_idx + 1}] train_time_sec={run_train_sec:.2f} test_time_sec={run_test_sec:.2f}"
            f" eval_model_selection_sec={run_eval_sec:.2f} train_peak_cpu_rss_mb={run_train_peak_cpu:.2f}"
            f" test_peak_cpu_rss_mb={run_test_peak_cpu:.2f} train_peak_cuda_allocated_mb={run_train_peak_cuda:.2f}"
            f" test_peak_cuda_allocated_mb={run_test_peak_cuda:.2f}"
            f" train_peak_cuda_reserved_mb={run_train_peak_cuda_reserved:.2f}"
            f" test_peak_cuda_reserved_mb={run_test_peak_cuda_reserved:.2f} inference_sec={run_inference_sec:.2f}"
            f" metric_computation_sec={run_testing_sec:.2f}"
        )
        if best_epoch is None:
            checkpoint_state_dict = _snapshot_state_dict_cpu(
                model.state_dict() if reference_protocol else _node2vec_state_dict(encoder, predictor)
            )
            checkpoint_epoch = last_epoch
            checkpoint_type = "final_model_state"
        else:
            checkpoint_state_dict = (
                best_state_dict
                if best_state_dict is not None
                else _snapshot_state_dict_cpu(model.state_dict() if reference_protocol else _node2vec_state_dict(encoder, predictor))
            )
            checkpoint_epoch = best_epoch
            checkpoint_type = "best_validation_model_state"
        if reference_protocol:
            reference_recipe = reference_embedding_metadata["recipe"]
            model_config = {
                "protocol": "reference", "model_implementation": "node2vec-heart-reference-two-stage-v2", "num_nodes": int(num_nodes),
                "raw_feature_dim": int(x.size(1)), "base_feature_sha256": reference_base_feature_sha256,
                "n2v_embedding_dim": int(PLANETOID_N2V_PRETRAIN["embedding_dim"]), "input_channels": int(reference_features.size(1)),
                "hidden_channels": int(reference_config["hidden_channels"]), "num_layers": int(reference_config["num_layers"]),
                "predictor_layers": int(reference_config["predictor_layers"]), "dropout": float(reference_config["dropout"]),
                "node_encode_batch_size": 262144, "evaluation_edge_batch_size": int(reference_eval_batch_size),
                "learning_rate": float(reference_config["learning_rate"]), "weight_decay": float(reference_config["weight_decay"]),
                "batch_size": int(reference_config["batch_size"]), "optimizer_implementation": "adam-scalar-foreach-false",
                "matmul_precision": reference_matmul_precision, "epochs": int(args.epochs), "eval_steps": int(args.eval_steps),
                "patience": int(args.patience), "model_selection_evaluation": "validation-only",
                "final_test_evaluation": "once-from-restored-best-state", "n2v_embedding_path": reference_embedding_path,
                "embedding_sha256": reference_embedding_metadata.get("embedding_sha256"),
                "embedding_recipe_digest": _reference_embedding_recipe_digest(reference_recipe),
                "edge_index_sha256": reference_recipe.get("edge_index_sha256"),
                "directed_edge_index_sha256": reference_recipe.get("edge_index_sha256"), "graph_semantics": reference_recipe.get("graph_semantics"),
                "feature_source": data.get("heart_feature_source"), "pretrain_seed": int(PLANETOID_N2V_PRETRAIN["seed"]),
                "pretrain_epochs": int(PLANETOID_N2V_PRETRAIN["epochs"]), "pretrain_walk_length": int(PLANETOID_N2V_PRETRAIN["walk_length"]),
                "pretrain_context_size": int(PLANETOID_N2V_PRETRAIN["context_size"]),
                "pretrain_walks_per_node": int(PLANETOID_N2V_PRETRAIN["walks_per_node"]),
                "pretrain_batch_size": int(PLANETOID_N2V_PRETRAIN["batch_size"]), "pretrain_workers": int(PLANETOID_N2V_PRETRAIN["workers"]),
                "pretrain_learning_rate": float(PLANETOID_N2V_PRETRAIN["learning_rate"]), "torch_version": reference_recipe.get("torch_version"),
                "torch_geometric_version": reference_recipe.get("torch_geometric_version"), "cuda_version": reference_recipe.get("cuda_version"),
            }
        else:
            model_config = {
                "num_nodes": num_nodes, "emb_dim": args.emb_dim, "walk_length": args.walk_length,
                "context_size": args.context_size, "walks_per_node": args.walks_per_node,
                "num_negative_samples": args.num_neg_samples, "p": args.p, "q": args.q, "sparse": True,
            }
        checkpoint_path = _save_model_checkpoint(
            checkpoint_state_dict, framework="pyg", mode=args.mode, dataset=args.dataset, model_name="n2v", run_number=seed + 1,
            seed=seed, epoch=checkpoint_epoch, timed_out=timed_out, metric_name=metric_label, best_val=best_val_selected,
            best_test=best_test_selected_for_run, args=args, model_config=model_config, checkpoint_type=checkpoint_type,
        )
        tqdm.write(f"[Run {run_idx + 1}] Saved checkpoint: {checkpoint_path}")
        if args.save_gnn_feature and reference_protocol and (run_idx == args.num_runs - 1) and (not timed_out):
            tqdm.write(
                f"WARNING: --save-gnn-feature is ignored by the reference protocol; the shared Node2Vec table is already stored separately at {reference_embedding_path}, so the input gnn_feature is preserved."
            )
        elif args.save_gnn_feature and run_idx == args.num_runs - 1 and (not timed_out):
            encoder.eval()
            with torch.no_grad():
                z = encoder().detach().cpu()
            out_dir = os.path.join(args.root, args.dataset)
            os.makedirs(out_dir, exist_ok=True)
            torch.save({"entity_embedding": z}, os.path.join(out_dir, "gnn_feature"))
        del checkpoint_state_dict
        optimizer = None
        del encoder, predictor, model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if timed_out:
            print(f"Stopping remaining runs because runtime exceeded 24 hours. elapsed_sec={time.time() - program_t0:.2f}", flush=True)
            break
    total_wall_sec = time.time() - program_t0
    header_lines = [
        "\n" + "=" * 80, "Timing summary", f"dataset: {args.dataset}", f"mode: {args.mode}", "model: n2v",
        f"n2v_protocol_requested: {args.n2v_protocol}", f"n2v_protocol_effective: {protocol}", f"device: {device}",
        f"compute_auc: {args.compute_auc}",
        f"evaluation_positive_cap: {args.eval_cap}",
        "evaluation_scope: " + ("configured_heart_validation_and_test_rows" if heart_validation_only else "configured_validation_and_test_rows"),
        "model_selection_evaluation: " + ("validation_final_test_only" if selection_validation_only else "validation_and_test"),
        f"reference_embedding_path: {reference_embedding_path or 'not-applicable'}",
        f"reference_embedding_sha256: {reference_embedding_metadata.get('embedding_sha256', 'not-applicable')}",
        f"reference_base_feature_sha256: {reference_base_feature_sha256 or 'not-applicable'}", f"reference_shared_pretrain_sec: {reference_pretrain_sec:.2f}",
    ]
    if reference_protocol:
        for key in ("hidden_channels", "num_layers", "predictor_layers", "dropout", "learning_rate", "weight_decay", "batch_size"):
            header_lines.append(f"reference_downstream_{key}: {reference_config[key]}")
        header_lines += ["reference_downstream_optimizer: adam-scalar-foreach-false",
                         f"reference_downstream_matmul_precision: {reference_matmul_precision}"]
    header_lines += [f"{key}: {(value if value is not None else 'not-applicable')}" for key, value in heart_candidate_metadata.items()]
    header_lines += [f"ranking_device: {device}", f"runtime_limit_exceeded: {timed_out}",
                     "status: " + ("exceeded 24 hour runtime limit" if timed_out else "completed within 24 hour runtime limit")]
    summary_values = {
        "runtime_limit_sec": RUNTIME_LIMIT_SEC, "data_load_sec": data_load_sec, "train_total_sec": total_train_sec,
        "test_total_sec": total_test_sec, "eval_total_sec": total_eval_sec, "inference_total_sec": total_inference_sec,
        "testing_total_sec": total_testing_sec, "total_wall_sec": total_wall_sec, "cpu_rss_mb_current": current_cpu_rss_mb(),
        "cpu_rss_mb_peak_process": peak_cpu_rss_mb(), "train_peak_cpu_rss_mb_max": train_peak_cpu,
        "test_peak_cpu_rss_mb_max": test_peak_cpu, "eval_peak_cpu_rss_mb_max": eval_peak_cpu,
        "train_peak_cuda_allocated_mb_max": train_peak_cuda, "test_peak_cuda_allocated_mb_max": test_peak_cuda,
        "eval_peak_cuda_allocated_mb_max": eval_peak_cuda, "train_peak_cuda_reserved_mb_max": train_peak_cuda_reserved,
        "test_peak_cuda_reserved_mb_max": test_peak_cuda_reserved, "eval_peak_cuda_reserved_mb_max": eval_peak_cuda_reserved,
    }
    per_run_series = {
        "train_sec_per_run": run_train_secs, "test_sec_per_run": run_test_secs, "eval_sec_per_run": run_eval_secs,
        "inference_sec_per_run": run_inference_secs, "testing_sec_per_run": run_testing_secs,
        "train_peak_cpu_rss_mb_per_run": [p["train_peak_cpu_rss_mb"] for p in run_resource_profiles],
        "test_peak_cpu_rss_mb_per_run": [p["test_peak_cpu_rss_mb"] for p in run_resource_profiles],
        "train_peak_cuda_allocated_mb_per_run": [p["train_peak_cuda_allocated_mb"] for p in run_resource_profiles],
        "test_peak_cuda_allocated_mb_per_run": [p["test_peak_cuda_allocated_mb"] for p in run_resource_profiles],
    }
    _write_timing_summary(
        log_path=log_path, header_lines=header_lines, summary_values=summary_values, per_run_series=per_run_series,
        run_profiles=run_resource_profiles, test_selected=test_selected_metrics, test_aucs=test_aucs, test_aps=test_aps,
        test_mrrs=test_mrrs, test_hits=test_hits_any, dataset=args.dataset, base_seed=args.seed, metric=metric_key,
    )


if __name__ == "__main__":
    main()
