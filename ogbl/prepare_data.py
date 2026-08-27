import os
import torch
from torch_sparse import SparseTensor
from .data_core import (
    _load_citation2_base,
    _load_noncitation_base,
    _resolve_pool_request,
    _pp_unique_undirected_uv,
    _sample_train_val,
    parse_pool_argument,
)
from .heart_generation import _load_or_build_generated_heart
from .ranked_candidates import load_or_build_ranked_valid_pool
from .protocol import resolve_ogbl_eval_cap
from utils.heart_protocol import GENERATED_HEART_NEGATIVES_TOTAL, heart_negative_count_metadata

_BUNDLE_KEYS = ("data", "split_edge", "adj", "train_pos", "train_val", "valid_pos", "valid_neg", "test_pos", "test_neg", "x")


def _dataset_local_negative_cache_dir(root, data_name):
    normalized = str(data_name).strip().lower().replace("_", "-")
    dataset_dirname = "ogbl-ppa" if normalized == "ogbl-ppa" else normalized.replace("-", "_")
    return os.path.join(root, dataset_dirname, "heart_cache")


def _canonical_generated_eval_cap(out, data_name, requested_eval_cap):
    cap = int(requested_eval_cap or 0)
    del data_name
    if cap <= 0:
        return cap
    if int(out["all_valid_pos"].size(0)) <= cap and torch.equal(out["valid_pos"], out["all_valid_pos"]):
        out["effective_eval_cap"] = 0
        out["effective_eval_seed"] = 0
        return 0
    return cap


def _restore_complete_test_split(out, data_name):
    del data_name
    out["test_pos"] = out["all_test_pos"].to(torch.long).contiguous()
    out["test_neg"] = None
    return out


def _bundle(out, **updates):
    bundle = {key: out[key] for key in _BUNDLE_KEYS}
    bundle.update(updates)
    return bundle


def read_ogbl_all_test(data_name, seed=0, root="dataset", eval_cap=0, pool=10000, all_negatives=None, ranked_backend="auto", negative_cache_dir=None, cache_negatives=True):
    strict_pool_max = 10000
    if data_name == "ogbl-citation2":
        out = _load_citation2_base(data_name, root, eval_cap, seed)
    else:
        out = _load_noncitation_base(data_name, root, eval_cap, seed)
    if all_negatives is not None:
        requested_total = int(all_negatives)
        if requested_total <= 0 or requested_total % 2 != 0:
            raise ValueError("all_negatives must be a positive even integer; it is split equally across both endpoints.")
        requested_pool = requested_total // 2
        pool_request = f"legacy-total:{requested_total}"
        full_graph_pool = False
        print(f"Using legacy all_negatives={requested_total}; equivalent pool_per_side={requested_pool}.", flush=True)
    else:
        (requested_pool, pool_request, full_graph_pool) = _resolve_pool_request(pool, out["num_nodes"])
        if full_graph_pool:
            print(
                f"pool=all uses the graph's {requested_pool} nodes as the maximum candidate universe per side before legal-candidate filtering.",
                flush=True,
            )
    original_requested_pool = int(requested_pool)
    requested_pool = min(original_requested_pool, strict_pool_max)
    if requested_pool < original_requested_pool:
        print(
            f"all-mode pool request reduced from {original_requested_pool} to the strict maximum of {strict_pool_max} candidates per side.",
            flush=True,
        )
    if negative_cache_dir is None:
        negative_cache_dir = _dataset_local_negative_cache_dir(root, data_name)
    _discard_stock_ogb_negatives(out)
    (valid_pool, test_pool, backend, cache_path) = load_or_build_ranked_valid_pool(
        out, data_name, requested_pool, seed, eval_cap, ranked_backend, negative_cache_dir, cache_negatives
    )
    side_counts = torch.cat([valid_pool.side_counts.reshape(-1), test_pool.side_counts.reshape(-1)])
    row_counts = torch.cat([valid_pool.rowptr[1:] - valid_pool.rowptr[:-1], test_pool.rowptr[1:] - test_pool.rowptr[:-1]])
    pool_per_side = int(side_counts.max().item()) if side_counts.numel() else 0
    pool_per_side_min = int(side_counts.min().item()) if side_counts.numel() else 0
    pool_per_side_mean = float(side_counts.to(torch.float64).mean().item()) if side_counts.numel() else 0.0
    pool_total_max = int(row_counts.max().item()) if row_counts.numel() else 0
    pool_total_min = int(row_counts.min().item()) if row_counts.numel() else 0
    pool_total_mean = float(row_counts.to(torch.float64).mean().item()) if row_counts.numel() else 0.0
    return _bundle(
        out,
        valid_neg=valid_pool,
        test_neg=test_pool,
        mode="all",
        pool_setting=pool_request,
        pool_request=pool_request,
        pool_full_graph=full_graph_pool,
        pool_cap_applied=requested_pool < out["num_nodes"],
        pool_sampling="positive-ra-ppr-ranked-valid-unpadded",
        pool_graph_nodes_per_side=out["num_nodes"],
        pool_per_side=pool_per_side,
        pool_per_side_min=pool_per_side_min,
        pool_per_side_mean=pool_per_side_mean,
        pool_per_side_max=pool_per_side,
        pool_requested_per_side=requested_pool,
        pool_user_requested_per_side=original_requested_pool,
        pool_total=pool_total_max,
        pool_total_min=pool_total_min,
        pool_total_mean=pool_total_mean,
        pool_total_max=pool_total_max,
        pool_requested_total=2 * requested_pool,
        pool_user_requested_total=2 * original_requested_pool,
        all_negatives=pool_total_max,
        all_negatives_requested=2 * requested_pool,
        ranked_backend=backend,
        negative_cache_path=cache_path,
    )


def _discard_stock_ogb_negatives(out):
    split_edge = out.get("split_edge")
    if isinstance(split_edge, dict):
        for split_name in ("valid", "test"):
            split_payload = split_edge.get(split_name)
            if not isinstance(split_payload, dict):
                continue
            split_payload.pop("edge_neg", None)
            split_payload.pop("target_node_neg", None)
    out["valid_neg"] = None
    out["test_neg"] = None


def _apply_generated_ppa_query_panel(out, panel, seed):
    valid_pos = panel["valid_pos"].to(torch.long).contiguous()
    test_pos = panel["test_pos"].to(torch.long).contiguous()
    num_nodes = int(out["num_nodes"])
    out["valid_pos"] = valid_pos
    out["test_pos"] = test_pos
    out["valid_input_pos"] = valid_pos
    out["valid_input_weight"] = None
    out["train_val"] = _sample_train_val(out["train_pos"], valid_pos.size(0), int(seed) + 3)
    data = out["data"]
    data.csr_tv_rowptr = None
    data.csr_tv_col = None
    out["tv_uv"] = None
    out["csr_tv_rowptr"] = None
    out["csr_tv_col"] = None
    valid_uv = _pp_unique_undirected_uv(valid_pos, num_nodes)
    tv_uv = torch.cat([out["train_uv"], valid_uv], dim=1).contiguous()
    valid_edge_index = torch.cat([valid_uv, valid_uv.flip(0)], dim=1)
    valid_adj = SparseTensor.from_edge_index(
        valid_edge_index, torch.ones(valid_edge_index.size(1), dtype=torch.float32), (num_nodes, num_nodes)
    )
    tv_adj = out["adj"] + valid_adj
    (tv_rowptr, tv_col, _) = tv_adj.csr()
    data.csr_tv_rowptr = tv_rowptr
    data.csr_tv_col = tv_col
    out["tv_uv"] = tv_uv
    out["csr_tv_rowptr"] = tv_rowptr
    out["csr_tv_col"] = tv_col
    out["effective_eval_cap"] = int(valid_pos.size(0))
    out["effective_eval_seed"] = int(seed) if panel["mode"] == "local-seeded" else 0
    for output_key, panel_key in (("heart_ppa_query_panel_mode", "mode"), ("heart_ppa_query_panel_recipe", "recipe"), ("heart_ppa_query_panel_identity_sha256", "identity_sha256"), ("heart_ppa_query_scope", "query_scope")):
        out[output_key] = panel[panel_key]
    out["heart_ppa_reference_positive_query_scope"] = bool(panel["reference_positive_query_scope"])
    for key in (
        "valid_index_sha256",
        "test_index_sha256",
        "valid_file_sha256",
        "test_file_sha256",
        "valid_query_sha256",
        "test_query_sha256",
        "valid_index_path",
        "test_index_path",
    ):
        out[f"heart_ppa_query_panel_{key}"] = panel.get(key)
    return out


def _load_generated_ppa_base(root, eval_cap, seed):
    from .ppa_query_panel import load_ppa_query_panel

    cap = int(eval_cap or 0)
    provisional_cap = cap or 100000
    out = _load_noncitation_base("ogbl-ppa", root, provisional_cap, seed)
    panel = load_ppa_query_panel(
        valid_split=out["all_valid_pos"], test_split=out["all_test_pos"], root=root, seed=seed, eval_cap=cap
    )
    return _apply_generated_ppa_query_panel(out, panel, seed)


def read_ogbl_generated_positive_scope(data_name, seed=0, root="dataset", eval_cap=0):
    """Load the exact positive-query scope used by generated HeaRT evaluation.

    This loader never builds, loads, or restores negative candidates.  It is
    intended for paired exhaustive evaluation that must share the generated
    protocol's positive rows and known-positive filters.
    """
    eval_cap = resolve_ogbl_eval_cap(eval_cap, "heart", data_name)
    generated_eval_cap = int(eval_cap or 0)
    if data_name == "ogbl-citation2":
        out = _load_citation2_base(data_name, root, eval_cap, seed)
        _restore_complete_test_split(out, data_name)
    elif data_name == "ogbl-ppa":
        out = _load_generated_ppa_base(root, eval_cap, seed)
    else:
        out = _load_noncitation_base(data_name, root, eval_cap, seed)
        _restore_complete_test_split(out, data_name)
    generated_eval_cap = _canonical_generated_eval_cap(out, data_name, generated_eval_cap)
    _discard_stock_ogb_negatives(out)
    out.update(
        {
            "mode": "heart",
            "paired_heart_positive_scope": True,
            "heart_source": "generated-online",
            "heart_source_policy": "generated-only",
            "heart_positive_scope_protocol": "generated-heart-positive-scope-v1",
            "heart_negative_candidates_loaded": False,
            "generated_eval_cap": generated_eval_cap,
        }
    )
    return out


def read_ogbl_heart(data_name, seed=0, root="dataset", eval_cap=0, heart_negatives=500, pool=10000, ranked_backend="auto", negative_cache_dir=None, cache_negatives=True):
    requested_total = int(heart_negatives)
    if requested_total != int(GENERATED_HEART_NEGATIVES_TOTAL):
        raise ValueError(
            "Generated OGB HeaRT requires --heart-negatives 500 (exactly 250 generated candidate slots per corruption side)."
        )
    requested_per_side = int(GENERATED_HEART_NEGATIVES_TOTAL) // 2
    out = read_ogbl_generated_positive_scope(data_name, seed=seed, root=root, eval_cap=eval_cap)
    generated_eval_cap = int(out["generated_eval_cap"])
    print("HeaRT candidates are generated locally from the full legal graph.", flush=True)
    del pool
    if negative_cache_dir is None:
        negative_cache_dir = _dataset_local_negative_cache_dir(root, data_name)
    (valid_neg, test_neg, draw_per_side, backend, cache_path, generated_heart_metadata) = _load_or_build_generated_heart(
        out, data_name, requested_per_side, seed, generated_eval_cap, ranked_backend, negative_cache_dir, cache_negatives
    )
    print(f"HeaRT selection=full-graph-candidate-ranking selected_total={2 * draw_per_side} selected_per_side={draw_per_side}", flush=True)
    return _bundle(
        out,
        valid_neg=valid_neg,
        test_neg=test_neg,
        mode="heart",
        heart_source="generated-online",
        heart_source_policy="generated-only",
        heart_negatives_per_side=draw_per_side,
        heart_negatives_requested_per_side=requested_per_side,
        heart_negatives_total=2 * draw_per_side,
        heart_negatives_requested_total=requested_total,
        **heart_negative_count_metadata(requested_total, effective_total=2 * draw_per_side),
        heart_selection="full-graph-candidate-ranking",
        heart_candidate_universe="full-legal-graph",
        heart_candidate_graph_nodes=out["num_nodes"],
        ranked_backend=backend,
        **generated_heart_metadata,
    )


def read_data(data, mode, eval_cap=500, seed=0, root="dataset", all_negatives=None, ranked_backend="auto", negative_cache_dir=None, cache_negatives=True, heart_negatives=500, pool=10000):
    mode = str(mode).strip().lower()
    if mode == "all":
        out = read_ogbl_all_test(
            data,
            seed=seed,
            root=root,
            eval_cap=eval_cap,
            pool=pool,
            all_negatives=all_negatives,
            ranked_backend=ranked_backend,
            negative_cache_dir=negative_cache_dir,
            cache_negatives=cache_negatives,
        )
    else:
        out = read_ogbl_heart(
            data,
            seed=seed,
            root=root,
            eval_cap=eval_cap,
            heart_negatives=heart_negatives,
            pool=pool,
            ranked_backend=ranked_backend,
            negative_cache_dir=negative_cache_dir,
            cache_negatives=cache_negatives,
        )
    valid_input_pos = getattr(out.get("data"), "valid_input_pos", None)
    if torch.is_tensor(valid_input_pos):
        out["valid_input_pos"] = valid_input_pos
    valid_input_weight = getattr(out.get("data"), "valid_input_weight", None)
    if torch.is_tensor(valid_input_weight):
        out["valid_input_weight"] = valid_input_weight
    out["mode"] = mode
    return out


def load_ogbl_splits(name, root="dataset"):
    if name == "ogbl-citation2":
        out = _load_citation2_base(name, root, 0, 0)
    else:
        out = _load_noncitation_base(name, root, 0, 0)
    eval_edges = {
        "pos_train_edge": out["train_pos"],
        "train_val_edge": out["train_val"],
        "pos_valid_edge": out["valid_pos"],
        "neg_valid_edge": out["valid_neg"],
        "pos_test_edge": out["test_pos"],
        "neg_test_edge": out["test_neg"],
    }
    return (out["data"], out["split_edge"], eval_edges)
