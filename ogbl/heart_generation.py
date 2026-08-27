from contextlib import contextmanager, suppress
import hashlib
import inspect
import json
import os
import shutil
import uuid
import torch
from .data_core import _NEGATIVE_CACHE_VERSION, _ensure_heart_eligibility_filters, _fsync_directory, _fsync_file, _heart_ppr_eps
from .fast_negatives import EndpointCorruptionGroupedNegativeEdges
from .heart_dense import (
    _adjust_sparse_metric_for_counterpart,
    _andersen_ppr_for_selected_nodes,
    _andersen_ppr_for_selected_nodes_cuda,
    _andersen_ppr_for_selected_nodes_cuda_kernel,
    _andersen_ppr_for_selected_nodes_numba,
    _build_ranked_double_sided_neg_exact,
    _complement_ordinals_to_nodes,
    _exact_score_graph,
    _generate_ranked_double_sided_neg,
    _load_resume_shard,
    _prepare_endpoint_topk_state,
    _prepare_sparse_metric_support,
    _partial_sort_adjust_heap_numba,
    _partial_sort_push_heap_numba,
    _partial_sort_sparse_topk,
    _partial_sort_sparse_topk_numba,
    _ranked_backend_for,
    _select_prepared_sparse_fused_topk,
    _select_prepared_topk,
    _source_exact_selected_row_ra,
    _sorted_membership,
    _sparse_positive_min_ranks,
    _sample_zero_evidence_with_replacement,
    _heart_seeded_tie_order,
    _heart_seeded_tie_priority,
    _take_seeded_tie_nodes,
    _temporary_cuda_matmul_tf32,
    _temporary_torch_num_threads,
)
from utils.heart_protocol import GENERATED_HEART_NEGATIVES_PER_SIDE, GENERATED_HEART_SELECTOR_RECIPE, GENERATED_HEART_TIE_SEED

_NEGATIVE_CACHE_LAYOUT_VERSION = 3
_NEGATIVE_CACHE_METADATA_VERSION = 7
_NEGATIVE_CACHE_VALIDATION_SIDECAR_VERSION = 1
_HEART_GENERATION_PROTOCOL = "ogb-generated-heart-reference-filter-source-ra-cpu-topk-fallback-v12"
_OGB_GENERATED_HEART_SELECTOR_RECIPE = GENERATED_HEART_SELECTOR_RECIPE + "+source-zero-evidence-replacement-fallback-v2"
_HEART_CACHE_VALIDATION_CONTRACT = "full-candidate-sha256+ordered-split-sha256+score-graph-sha256+immutable-source-manifest+shape+orientation+range+self+query+fallback-boundary-sha256+hard-prefix-unique+fallback-suffix+eligibility-v5"
_EXACT_ANDERSEN_PPR_METHOD = "exact-selected-endpoint-andersen-local-push-adaptive-parity-guarded-numba-cpu-or-numba-cuda"
_CACHE_PROVENANCE_EQUIVALENCE = {}
_LEGACY_CACHE_KEY_MAP = {
    "paper_positive_query_scope": "reference_positive_query_scope",
    "paper_artifact_exact": "reference_artifact_exact",
}
_LEGACY_CACHE_VALUE_MAP = {
    "ogb-generated-heart-table5-filter-source-ra-cpu-topk-fallback-v12": _HEART_GENERATION_PROTOCOL,
    "paper-table5-observed-history-filter": "released-observed-history-filter",
    "official": "reference",
    "heart-paper-ppa-fixed-ordered-index-panel-v1": "heart-reference-ppa-fixed-ordered-index-panel-v1",
    "paper-fixed-100000-official-index-order": "reference-fixed-100000-released-index-order",
    "local-seeded-validation-randperm-test-full-ordered-nonpaper-v2": "local-seeded-validation-randperm-test-full-ordered-custom-v2",
    "validation-local-seeded-max100000-test-full-ordered-nonpaper": "validation-local-seeded-max100000-test-full-ordered-custom",
}
_PUBLIC_METADATA = (
    "evaluation_query_sha256", "score_graph_sha256", "score_graph_policy", "eligibility_policy", "eligibility_orientation",
    "eligibility_scope", "eligibility_recipe", "eligibility_recipe_sha256", "selector_recipe", "selector_implementation_sha256",
    "selector_rank_policy", "selector_tie_policy", "selector_cpu_topk_policy", "selector_cpu_topk_partial_sort_factor",
    "selector_released_hardest_rank_rule", "selector_released_hardest_rank_core", "selector_released_hard_tie_order_exact",
    "selector_released_hard_tie_order_deviation", "selector_released_fallback_sampling_exact",
    "selector_released_fallback_sampling_deviation", "selector_released_fallback_policy", "candidate_duplicate_policy",
    "fallback_boundary_sha256", "fallback_valid_side_occurrences", "fallback_test_side_occurrences", "fallback_valid_slots",
    "fallback_test_slots", "ppr_method", "ppr_alpha", "ppr_eps", "ra_backend", "ra_semantics", "query_scope",
    "reference_positive_query_scope", "reference_artifact_exact",
)
_PUBLIC_OPTIONAL_METADATA = (
    "ppa_query_panel_mode", "ppa_query_panel_recipe", "ppa_query_panel_identity_sha256", "ppa_query_panel_valid_index_sha256",
    "ppa_query_panel_test_index_sha256", "ppa_query_panel_valid_file_sha256", "ppa_query_panel_test_file_sha256",
    "protocol_identity_path", "protocol_source_identity_sha256", "protocol_source_identity_method",
    "protocol_source_artifact_identity",
)


def _canonical_cache_value(value):
    if not isinstance(value, str):
        return value
    canonical = _LEGACY_CACHE_VALUE_MAP.get(value)
    if canonical is not None:
        return canonical
    legacy_local = "local-seeded-validation-randperm-test-full-ordered-nonpaper-v2"
    if value.startswith(legacy_local):
        return value.replace(legacy_local, "local-seeded-validation-randperm-test-full-ordered-custom-v2", 1)
    return value


def _canonical_cache_metadata(metadata):
    return {_LEGACY_CACHE_KEY_MAP.get(key, key): _canonical_cache_value(value) for key, value in metadata.items()}


def _negative_cache_file(cache_dir, data_name, mode, seed, eval_cap, k2, backend, nvalid, ntest, query_panel_identity=None):
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    panel_component = ""
    if str(data_name).strip().lower() == "ogbl-ppa":
        identity = str(query_panel_identity or "")
        if len(identity) != 64:
            raise ValueError("ogbl-ppa generated caches require a full query-panel identity SHA256.")
        panel_component = f"qpanel{identity}_"
    name = f"{data_name}_{mode}_v{_NEGATIVE_CACHE_VERSION}_seed{int(seed)}_cap{int(eval_cap or 0)}_{panel_component}k2{int(k2)}_{backend}_v{int(nvalid)}_t{int(ntest)}_compact{_NEGATIVE_CACHE_LAYOUT_VERSION}.pt"
    return os.path.join(cache_dir, name)


def _full_tensor_sha256(value, *, chunk_bytes=8 * 1024 * 1024):
    tensor = torch.as_tensor(value).detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"tensor-v1;shape={tuple(tensor.shape)};dtype={tensor.dtype};".encode("utf-8"))
    byte_values = tensor.view(torch.uint8).reshape(-1)
    step = max(1, int(chunk_bytes))
    for start in range(0, int(byte_values.numel()), step):
        digest.update(byte_values[start : start + step].numpy().tobytes(order="C"))
    return digest.hexdigest()


def _named_tensor_digest(label, values):
    digest = hashlib.sha256()
    digest.update(f"{label}-v1;".encode("utf-8"))
    for name, value in values:
        digest.update(f"{name}:".encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(b"missing;")
        else:
            digest.update(_full_tensor_sha256(value).encode("ascii"))
            digest.update(b";")
    return digest.hexdigest()


def _ordered_positive_split_sha256(out):
    cached = out.get("heart_ordered_positive_split_sha256")
    if cached:
        return str(cached)
    value = _named_tensor_digest(
        "ogb-ordered-positive-split", (("train", out["train_pos"]), ("valid", out["all_valid_pos"]), ("test", out["all_test_pos"]))
    )
    out["heart_ordered_positive_split_sha256"] = value
    return value


def _evaluation_query_sha256(out):
    cached = out.get("heart_evaluation_query_sha256")
    if cached:
        return str(cached)
    value = _named_tensor_digest("ogb-generated-heart-evaluation-query", (("valid", out["valid_pos"]), ("test", out["test_pos"])))
    out["heart_evaluation_query_sha256"] = value
    return value


def _score_graph_sha256(out):
    cached = out.get("heart_score_graph_sha256")
    if cached:
        return str(cached)
    value = _named_tensor_digest(
        "ogb-generated-heart-score-graphs",
        (
            ("train_uv", out.get("train_uv")),
            ("test_uv", out.get("tv_uv")),
            ("train_ra_edges", out.get("heart_train_ra_edge_index")),
            ("test_ra_edges", out.get("heart_tv_ra_edge_index")),
            ("train_ra_degree", out.get("heart_train_ra_degree")),
            ("test_ra_degree", out.get("heart_tv_ra_degree")),
        ),
    )
    out["heart_score_graph_sha256"] = value
    return value


def _raw_selector_implementation_sha256():
    digest = hashlib.sha256()
    digest.update(b"ogb-generated-heart-selector-implementation-v7;")
    for function in (
        _andersen_ppr_for_selected_nodes_numba,
        _andersen_ppr_for_selected_nodes,
        _andersen_ppr_for_selected_nodes_cuda,
        _andersen_ppr_for_selected_nodes_cuda_kernel,
        _prepare_endpoint_topk_state,
        _select_prepared_topk,
        _sorted_membership,
        _sparse_positive_min_ranks,
        _partial_sort_push_heap_numba,
        _partial_sort_adjust_heap_numba,
        _partial_sort_sparse_topk_numba,
        _partial_sort_sparse_topk,
        _heart_seeded_tie_order,
        _heart_seeded_tie_priority,
        _take_seeded_tie_nodes,
        _prepare_sparse_metric_support,
        _adjust_sparse_metric_for_counterpart,
        _select_prepared_sparse_fused_topk,
        _source_exact_selected_row_ra,
        _complement_ordinals_to_nodes,
        _sample_zero_evidence_with_replacement,
        _exact_score_graph,
        _temporary_cuda_matmul_tf32,
        _temporary_torch_num_threads,
        _ensure_heart_eligibility_filters,
        _build_ranked_double_sided_neg_exact,
        _load_resume_shard,
        _compact_endpoint_candidates,
        _validate_candidate_nodes,
    ):
        source_function = getattr(function, "py_func", function)
        digest.update(
            f"{getattr(source_function, '__module__', 'missing')}.{getattr(source_function, '__qualname__', 'missing')}:".encode("utf-8")
        )
        try:
            source = inspect.getsource(source_function)
        except (OSError, TypeError):
            source = repr(getattr(source_function, "__code__", None))
        digest.update(str(source).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()


def _selector_implementation_sha256():
    from utils.cache_compat import relocated_ogb_heart_selector_fingerprint
    return relocated_ogb_heart_selector_fingerprint(_raw_selector_implementation_sha256())


def _eligibility_recipe(data_name, ordered_split_sha256, evaluation_query_sha256):
    name = str(data_name).lower()
    if name == "ogbl-collab":
        policy = "released-observed-history-filter"
        orientation = "undirected-temporal"
        scope = "valid=train;test=train+uncapped-valid"
        detail = "valid-mask=train;test-mask=train+uncapped-valid"
    elif name == "ogbl-citation2":
        policy = "released-shared-row-view"
        orientation = "directed-shared-source-rows"
        scope = "released-observed-graph-plus-query"
        detail = "valid-both-sides=raw-train-transposed-rows;test-both-sides=raw-train-outgoing+bidirectional-valid-rows"
    else:
        policy = "released-observed-graph"
        orientation = "undirected-canonical"
        scope = "valid=train;test=train+evaluated-valid" if name == "ogbl-ppa" else "valid=train;test=train+uncapped-valid"
        detail = "valid-mask=train;test-mask=train+valid;both-sides=shared-rows"
    detail += ";per-query-mask=fixed-endpoint+counterpart-before-rank"
    legacy_identity_policy = "paper-table5-observed-history-filter" if policy == "released-observed-history-filter" else policy
    digest = hashlib.sha256()
    digest.update(
        f"ogb-heart-eligibility-recipe-v3;dataset={name};policy={legacy_identity_policy};orientation={orientation};scope={scope};detail={detail};ordered-split={ordered_split_sha256};evaluation-query={evaluation_query_sha256};".encode(
            "utf-8"
        )
    )
    return (policy, orientation, scope, detail, digest.hexdigest())


def _protocol_identity_file(cache_dir, data_name, out):
    if cache_dir is None:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    panel_component = ""
    if str(data_name).strip().lower() == "ogbl-ppa":
        panel_identity = str(out.get("heart_ppa_query_panel_identity_sha256") or "")
        if len(panel_identity) != 64:
            raise ValueError("ogbl-ppa protocol identity requires the complete fixed-query panel SHA256.")
        panel_component = f"qpanel{panel_identity}_"
    return os.path.join(
        cache_dir,
        f"{str(data_name).lower()}_generated_heart_identity_v2_{panel_component}seed{int(out.get('effective_eval_seed') or 0)}_cap{int(out.get('effective_eval_cap') or 0)}_v{int(out['valid_pos'].size(0))}_t{int(out['test_pos'].size(0))}.pt",
    )


def _protocol_source_identity(out):
    artifact = out.get("heart_source_artifact_identity")
    if isinstance(artifact, dict):
        method = artifact.get("method")
        value = artifact.get("sha256")
        if isinstance(method, str) and isinstance(value, str) and (len(value) == 64):
            return (method, value, artifact)
    digest = hashlib.sha256()
    digest.update(b"ogb-heart-bounded-protocol-source-v1;")
    for key in (
        "train_pos",
        "all_valid_pos",
        "all_test_pos",
        "valid_pos",
        "test_pos",
        "train_uv",
        "tv_uv",
        "heart_train_ra_edge_index",
        "heart_tv_ra_edge_index",
        "heart_train_ra_degree",
        "heart_tv_ra_degree",
    ):
        value = out.get(key)
        digest.update(f"{key}:".encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(b"missing;")
            continue
        tensor = value.detach().cpu()
        total = int(tensor.numel())
        count = min(257, total)
        digest.update(f"shape={tuple(tensor.shape)};dtype={tensor.dtype};".encode("utf-8"))
        if count:
            flat = tensor.contiguous().view(-1)
            positions = torch.zeros(1, dtype=torch.long) if count == 1 else torch.arange(count) * (total - 1) // (count - 1)
            digest.update(flat[positions].contiguous().view(torch.uint8).numpy().tobytes(order="C"))
        digest.update(b";")
    value = digest.hexdigest()
    return ("bounded-even-tensor-sample-sha256-v1", value, None)


def _load_protocol_identity(path, source_identity_method, source_identity):
    if not path or not os.path.isfile(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        expected = {
            "identity_version": 2,
            "cache_version": int(_NEGATIVE_CACHE_VERSION),
            "source_identity_method": str(source_identity_method),
            "source_identity_sha256": str(source_identity),
        }
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict) or any((metadata.get(key) != value for (key, value) in expected.items())):
            raise ValueError("protocol identity metadata mismatch")
        for key in ("ordered_positive_split_sha256", "evaluation_query_sha256", "score_graph_sha256"):
            value = payload.get(key)
            if not isinstance(value, str) or len(value) != 64:
                raise ValueError(f"protocol identity lacks {key}")
        return {key: payload[key] for key in ("ordered_positive_split_sha256", "evaluation_query_sha256", "score_graph_sha256")}
    except Exception as exc:
        print(f"Ignoring invalid HeaRT protocol identity {path}: {exc}", flush=True)
        return None


def _bind_protocol_source(identity, path, method, sha256, artifact):
    identity.update(identity_path=path, source_identity_sha256=sha256, source_identity_method=method, source_artifact_identity=artifact)
    return identity


def _load_or_build_protocol_identity(out, data_name, cache_dir, *, force_full=False):
    path = _protocol_identity_file(cache_dir, data_name, out)
    (source_identity_method, source_identity, source_artifact_identity) = _protocol_source_identity(out)
    cached = None if force_full else _load_protocol_identity(path, source_identity_method, source_identity)
    if cached is not None:
        return _bind_protocol_source(cached, path, source_identity_method, source_identity, source_artifact_identity)
    with _exclusive_cache_build(path):
        cached = None if force_full else _load_protocol_identity(path, source_identity_method, source_identity)
        if cached is not None:
            return _bind_protocol_source(cached, path, source_identity_method, source_identity, source_artifact_identity)
        identity = {
            "ordered_positive_split_sha256": _ordered_positive_split_sha256(out),
            "evaluation_query_sha256": _evaluation_query_sha256(out),
            "score_graph_sha256": _score_graph_sha256(out),
        }
        if path:
            temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
            try:
                torch.save(
                    {
                        "metadata": {
                            "identity_version": 2,
                            "cache_version": int(_NEGATIVE_CACHE_VERSION),
                            "source_identity_method": source_identity_method,
                            "source_identity_sha256": source_identity,
                            "source_artifact_identity": source_artifact_identity,
                        },
                        **identity,
                    },
                    temporary_path,
                )
                _fsync_file(temporary_path)
                os.replace(temporary_path, path)
                _fsync_directory(os.path.dirname(os.path.abspath(path)))
            finally:
                if os.path.exists(temporary_path):
                    os.remove(temporary_path)
        return _bind_protocol_source(identity, path, source_identity_method, source_identity, source_artifact_identity)


def _cache_graph_metadata(out, data_name, backend, k2, protocol_identity=None):
    if protocol_identity is None:
        (source_method, source_sha256, source_artifact) = _protocol_source_identity(out)
        protocol_identity = {
            "ordered_positive_split_sha256": _ordered_positive_split_sha256(out),
            "evaluation_query_sha256": _evaluation_query_sha256(out),
            "score_graph_sha256": _score_graph_sha256(out),
            "identity_path": None,
            "source_identity_sha256": source_sha256,
            "source_identity_method": source_method,
            "source_artifact_identity": source_artifact,
        }
    ordered_split = protocol_identity["ordered_positive_split_sha256"]
    (policy, orientation, eligibility_scope, detail, filter_digest) = _eligibility_recipe(
        data_name, ordered_split, protocol_identity["evaluation_query_sha256"]
    )
    if str(data_name).lower() == "ogbl-ppa":
        panel_keys = (
            "mode", "recipe", "identity_sha256", "valid_index_sha256", "test_index_sha256", "valid_file_sha256",
            "test_file_sha256", "valid_query_sha256", "test_query_sha256",
        )
        panel_fields = {f"ppa_query_panel_{key}": out.get(f"heart_ppa_query_panel_{key}") for key in panel_keys}
        required_digests = (
            "ppa_query_panel_identity_sha256",
            "ppa_query_panel_valid_index_sha256",
            "ppa_query_panel_test_index_sha256",
            "ppa_query_panel_valid_query_sha256",
            "ppa_query_panel_test_query_sha256",
        )
        missing = [key for key in required_digests if not isinstance(panel_fields[key], str) or len(panel_fields[key]) != 64]
        if missing:
            raise ValueError("ogbl-ppa generated protocol lacks: " + ", ".join(missing))
        if not panel_fields["ppa_query_panel_recipe"]:
            raise ValueError("ogbl-ppa generated protocol lacks its query recipe.")
        if panel_fields["ppa_query_panel_mode"] == "reference":
            file_keys = ("ppa_query_panel_valid_file_sha256", "ppa_query_panel_test_file_sha256")
            if any(not isinstance(panel_fields[key], str) or len(panel_fields[key]) != 64 for key in file_keys):
                raise ValueError("reference ogbl-ppa query panel lacks file digests")
        query_scope = str(out.get("heart_ppa_query_scope") or "")
        if not query_scope:
            raise ValueError("ogbl-ppa generated protocol lacks query scope.")
    elif int(out.get("effective_eval_cap") or 0):
        panel_fields = {}
        query_scope = "generated-deterministic-evaluation-cap"
    else:
        panel_fields = {}
        query_scope = "complete-ogb-positive-splits"
    score_policy = "valid=train;test=train+uncapped-valid" if str(data_name).lower() == "ogbl-collab" else "valid=train;test=train"
    return {
        "metadata_version": int(_NEGATIVE_CACHE_METADATA_VERSION),
        "generation_protocol": _HEART_GENERATION_PROTOCOL,
        "num_nodes": int(out["num_nodes"]),
        "train_positive_count": int(out["train_pos"].size(0)),
        "uncapped_valid_positive_count": int(out["all_valid_pos"].size(0)),
        "uncapped_test_positive_count": int(out["all_test_pos"].size(0)),
        "evaluation_valid_positive_count": int(out["valid_pos"].size(0)),
        "evaluation_test_positive_count": int(out["test_pos"].size(0)),
        "ordered_positive_split_sha256": ordered_split,
        "evaluation_query_sha256": protocol_identity["evaluation_query_sha256"],
        "score_graph_sha256": protocol_identity["score_graph_sha256"],
        "protocol_identity_path": protocol_identity.get("identity_path"),
        "protocol_source_identity_sha256": protocol_identity.get("source_identity_sha256"),
        "protocol_source_identity_method": protocol_identity.get("source_identity_method"),
        "protocol_source_artifact_identity": protocol_identity.get("source_artifact_identity"),
        "score_graph_policy": score_policy,
        "eligibility_policy": policy,
        "eligibility_orientation": orientation,
        "eligibility_scope": eligibility_scope,
        "eligibility_recipe": detail,
        "eligibility_recipe_sha256": filter_digest,
        "candidate_count_total": 2 * int(k2),
        "candidate_count_per_side": int(k2),
        "unique_candidates_per_side": False,
        "candidate_duplicate_policy": "duplicates-allowed-only-in-source-zero-evidence-fallback-suffix",
        "selector_recipe": _OGB_GENERATED_HEART_SELECTOR_RECIPE,
        "selector_rank_fusion": "minimum-ra-andersen-rank",
        "selector_rank_policy": "released-positive-evidence-hard-prefix+zero-evidence-replacement-fallback-per-side",
        "selector_tie_policy": "literal-source-cpu-topk-rank-only-ties;dense-torch-nth-element-or-sparse-libstdcxx-partial-sort-heap;numpy-randomstate42-source-order-zero-evidence-replacement",
        "selector_tie_seed": None,
        "selector_fallback_seed": int(GENERATED_HEART_TIE_SEED),
        "selector_cpu_topk_policy": "dense-literal-torch-topk-when-kx64>num-nodes;sparse-libstdcxx-partial-sort-rank-only-heap-v1-otherwise",
        "selector_cpu_topk_partial_sort_factor": 64,
        "selector_released_hardest_rank_rule": True,
        "selector_released_hardest_rank_core": True,
        "selector_released_hard_tie_order_exact": True,
        "selector_released_hard_tie_order_deviation": None,
        "selector_released_fallback_sampling_exact": True,
        "selector_released_fallback_sampling_deviation": None,
        "selector_released_fallback_policy": "positive-evidence-hard-prefix+zero-evidence-numpy-choice-with-replacement-source-occurrence-order",
        "selector_implementation_sha256": _selector_implementation_sha256(),
        "ppr_method": _EXACT_ANDERSEN_PPR_METHOD,
        "ppr_alpha": 0.15,
        "ppr_eps": float(_heart_ppr_eps(data_name)),
        "ra_backend": (
            "cpu-selected-row-sparse-sparse"
            if str(data_name).strip().lower() == "ogbl-collab"
            else "gpu" if "gpu-ra" in str(backend) else "cpu"
        ),
        "ra_semantics": "released-weight-before-matmul-fp32;collab-selected-row-sparse-sparse-exact;coalesced-others-weighted-projector-parity",
        "query_scope": query_scope,
        "reference_positive_query_scope": bool(
            out.get("heart_ppa_reference_positive_query_scope")
            if str(data_name).lower() == "ogbl-ppa"
            else not out.get("effective_eval_cap")
        ),
        **panel_fields,
        "reference_artifact_exact": False,
        "cache_validation_contract": _HEART_CACHE_VALIDATION_CONTRACT,
        "released_observed_filter_validated_at_build": True,
        "candidate_invariants_validated_before_publish": [
            "shape",
            "orientation",
            "range",
            "self",
            "query-positive",
            "unique-hard-prefix",
            "duplicates-only-in-zero-evidence-fallback-suffix",
            "eligibility-membership",
        ],
        "graph_sample_sha256": _graph_sample_digest(out),
    }


def _expected_negative_cache_metadata(out, data_name, backend, k2, protocol_identity=None):
    return {
        "cache_version": int(_NEGATIVE_CACHE_VERSION),
        "layout_version": int(_NEGATIVE_CACHE_LAYOUT_VERSION),
        "layout": "endpoint-corruption-candidates",
        "dataset": str(data_name).lower(),
        "backend": str(backend),
        "candidates_per_side": int(k2),
        **_cache_graph_metadata(out, data_name, backend, k2, protocol_identity),
    }


def _generation_resume_state(cache_path, expected_metadata):
    if not cache_path:
        return None
    encoded = json.dumps(expected_metadata, sort_keys=True, separators=(",", ":"), default=lambda value: repr(value)).encode("utf-8")
    identity = hashlib.sha256(b"ogb-generated-heart-resume-v1;" + encoded).hexdigest()
    return {"identity": identity, "directory": f"{cache_path}.partial.{identity[:16]}"}


def _cleanup_generation_resume(cache_path, resume_state):
    if not cache_path or not isinstance(resume_state, dict):
        return
    directory = resume_state.get("directory")
    expected_prefix = os.path.abspath(cache_path) + ".partial."
    if not directory or not os.path.abspath(directory).startswith(expected_prefix) or (not os.path.isdir(directory)):
        return
    try:
        shutil.rmtree(directory)
    except OSError as exc:
        print(f"WARNING: could not remove completed HeaRT resume shards: {exc}", flush=True)


def _graph_sample_digest(out, samples_per_tensor=257):
    digest = hashlib.sha256()
    keys = (
        "train_uv",
        "tv_uv",
        "csr_train_rowptr",
        "csr_train_col",
        "csr_tv_rowptr",
        "csr_tv_col",
        "heart_train_ra_edge_index",
        "heart_tv_ra_edge_index",
        "heart_train_ra_degree",
        "heart_tv_ra_degree",
    )
    for key in keys:
        value = out.get(key)
        digest.update(f"{key}:".encode("utf-8"))
        if not torch.is_tensor(value):
            digest.update(b"none;")
            continue
        tensor = value.detach().cpu()
        digest.update(f"{tuple(tensor.shape)}:{tensor.dtype};".encode("utf-8"))
        width = int(tensor.size(-1)) if tensor.dim() else 1
        count = min(max(1, int(samples_per_tensor)), width)
        if width == 0:
            digest.update(b"empty;")
            continue
        if count == 1:
            indices = torch.zeros(1, dtype=torch.long)
        else:
            indices = torch.arange(count, dtype=torch.long) * (width - 1) // (count - 1)
        sampled = tensor.reshape(1).index_select(0, indices) if tensor.dim() == 0 else tensor.index_select(tensor.dim() - 1, indices)
        digest.update(repr(sampled.tolist()).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()


def _compact_endpoint_candidates(pos, neg, candidates_per_side):
    pos = pos.detach().to(device="cpu", dtype=torch.long).contiguous()
    neg = neg.detach().to(device="cpu", dtype=torch.long).contiguous()
    k2 = int(candidates_per_side)
    expected = (int(pos.size(0)), 2 * k2, 2)
    if tuple(neg.shape) != expected:
        raise ValueError(f"Generated HeaRT negatives have shape {tuple(neg.shape)}; expected {expected}.")
    if k2 and (not torch.equal(neg[:, :k2, 0], pos[:, 0:1].expand(-1, k2)) or not torch.equal(neg[:, k2:, 1], pos[:, 1:2].expand(-1, k2))):
        raise ValueError("Generated HeaRT negatives violate the endpoint-corruption layout.")
    return torch.cat((neg[:, :k2, 1], neg[:, k2:, 0]), dim=1).to(torch.int32).contiguous()


def _fallback_metadata(valid_fallback_counts, test_fallback_counts):
    boundary = (("valid", valid_fallback_counts), ("test", test_fallback_counts))
    metadata = {"fallback_boundary_sha256": _named_tensor_digest("ogb-generated-heart-fallback-boundaries", boundary)}
    for split_name, values in (("valid", valid_fallback_counts), ("test", test_fallback_counts)):
        values = values.detach().cpu().to(torch.long)
        metadata[f"fallback_{split_name}_side_occurrences"] = int((values > 0).sum().item())
        metadata[f"fallback_{split_name}_slots"] = int(values.sum().item())
    return metadata


def _validate_fallback_counts(split_name, fallback_counts, positives, k2):
    if not torch.is_tensor(fallback_counts):
        raise TypeError(f"{split_name} fallback counts are not a tensor")
    expected = (int(positives.size(0)), 2)
    valid = tuple(fallback_counts.shape) == expected and fallback_counts.device.type == "cpu" and fallback_counts.dtype == torch.int16
    valid &= not fallback_counts.numel() or (int(fallback_counts.min()) >= 0 and int(fallback_counts.max()) <= int(k2))
    if not valid:
        raise ValueError(f"invalid {split_name} fallback-count tensor")
    return fallback_counts


def _validate_csr_membership_absent(split_name, side_name, endpoints, candidates, rowptr, col):
    endpoints = endpoints.detach().cpu().to(torch.long).view(-1)
    candidates = candidates.detach().cpu()
    order = torch.argsort(endpoints, stable=True)
    sorted_endpoints = endpoints[order]
    (unique_endpoints, counts) = torch.unique_consecutive(sorted_endpoints, return_counts=True)
    offset = 0
    for endpoint, count in zip(unique_endpoints.tolist(), counts.tolist()):
        stop = offset + int(count)
        row_ids = order[offset:stop]
        values = candidates[row_ids].reshape(-1).to(torch.long)
        (begin, end) = (int(rowptr[int(endpoint)]), int(rowptr[int(endpoint) + 1]))
        neighbors = col[begin:end].to(torch.long)
        if neighbors.numel() and values.numel():
            positions = torch.searchsorted(neighbors, values)
            inside = positions < neighbors.numel()
            matched = torch.zeros_like(inside)
            if bool(inside.any()):
                matched[inside] = neighbors[positions[inside]] == values[inside]
            if bool(matched.any()):
                raise ValueError(f"{split_name} {side_name}-side HeaRT candidates contain an ineligible positive relation")
        offset = stop


def _validate_candidate_nodes(
    split_name, positives, candidates, k2, num_nodes, *, fallback_counts, out=None, trusted_reference_digest=False
):
    positives = positives.detach().cpu().to(torch.long).contiguous()
    fallback_counts = _validate_fallback_counts(split_name, fallback_counts, positives, k2)
    if not torch.is_tensor(candidates):
        raise TypeError(f"{split_name} candidate nodes are not a tensor")
    expected = (int(positives.size(0)), 2 * int(k2))
    if tuple(candidates.shape) != expected or candidates.dtype != torch.int32 or candidates.device.type != "cpu":
        raise ValueError(f"invalid {split_name} candidate tensor contract")
    values = candidates
    if values.numel() and (int(values.min().item()) < 0 or int(values.max().item()) >= int(num_nodes)):
        raise ValueError(f"{split_name} candidate node id is out of range")
    (left, right) = (values[:, : int(k2)], values[:, int(k2) :])
    if not trusted_reference_digest:
        for side_index, (side_name, side_values) in enumerate((("left", left), ("right", right))):
            if int(k2) > 1:
                hard_counts = int(k2) - fallback_counts[:, side_index].to(torch.long)
                columns = torch.arange(int(k2), dtype=torch.long).view(1, -1)
                hard = columns < hard_counts.view(-1, 1)
                checked = torch.where(hard, side_values.to(torch.long), int(num_nodes) + columns)
                sorted_side = torch.sort(checked, dim=1).values
                if bool((sorted_side[:, 1:] == sorted_side[:, :-1]).any()):
                    raise ValueError(f"{split_name} contains duplicate hard-prefix {side_name}-side candidates")
    illegal = (left == positives[:, 0:1]).any() or (right == positives[:, 1:2]).any()
    illegal |= (left == positives[:, 1:2]).any() or (right == positives[:, 0:1]).any()
    if bool(illegal):
        raise ValueError(f"{split_name} candidates contain a self-loop or positive query edge")
    if out is not None and (not trusted_reference_digest):
        _validate_csr_membership_absent(
            split_name, "left", positives[:, 0], left, out[f"heart_{split_name}_out_rowptr"], out[f"heart_{split_name}_out_col"]
        )
        _validate_csr_membership_absent(
            split_name, "right", positives[:, 1], right, out[f"heart_{split_name}_in_rowptr"], out[f"heart_{split_name}_in_col"]
        )
    return candidates


def _cached_negative_objects(out, valid_candidates, test_candidates, k2):
    valid_pos = out["valid_pos"].detach().to(device="cpu", dtype=torch.long).contiguous()
    test_pos = out["test_pos"].detach().to(device="cpu", dtype=torch.long).contiguous()
    return (
        EndpointCorruptionGroupedNegativeEdges(
            pos_edges=valid_pos, candidate_nodes=valid_candidates, num_nodes=int(out["num_nodes"])
        ),
        EndpointCorruptionGroupedNegativeEdges(
            pos_edges=test_pos, candidate_nodes=test_candidates, num_nodes=int(out["num_nodes"])
        ),
    )


def _negative_cache_file_identity(path):
    stat = os.stat(path, follow_symlinks=True)
    return {
        "device": int(stat.st_dev),
        "inode": int(stat.st_ino),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "ctime_ns": int(stat.st_ctime_ns),
    }


def _negative_cache_metadata_digest(metadata):
    encoded = json.dumps(metadata, sort_keys=True, separators=(",", ":"), default=lambda value: repr(value)).encode("utf-8")
    return hashlib.sha256(b"ogb-heart-published-cache-metadata-v1;" + encoded).hexdigest()


def _negative_cache_validation_sidecar(path): return f"{path}.validated_v{_NEGATIVE_CACHE_VALIDATION_SIDECAR_VERSION}.pt"


def _try_load_negative_cache_validation(path, metadata, validator_selector_digest):
    sidecar_path = _negative_cache_validation_sidecar(path)
    if not os.path.isfile(sidecar_path):
        return False
    try:
        payload = torch.load(sidecar_path, map_location="cpu", mmap=True, weights_only=True)
        expected = {
            "version": int(_NEGATIVE_CACHE_VALIDATION_SIDECAR_VERSION),
            "cache_file_identity": _negative_cache_file_identity(path),
            "cache_metadata_sha256": _negative_cache_metadata_digest(metadata),
            "candidate_tensor_sha256": metadata.get("candidate_tensor_sha256"),
            "fallback_boundary_sha256": metadata.get("fallback_boundary_sha256"),
            "validator_selector_implementation_sha256": str(validator_selector_digest),
        }
        return isinstance(payload, dict) and all((payload.get(key) == value for (key, value) in expected.items()))
    except Exception:
        return False


def _save_negative_cache_validation(path, metadata, validator_selector_digest, initial_file_identity):
    try:
        final_file_identity = _negative_cache_file_identity(path)
        if final_file_identity != initial_file_identity:
            raise RuntimeError("candidate cache changed during validation")
        sidecar_path = _negative_cache_validation_sidecar(path)
        temporary_path = f"{sidecar_path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
        try:
            torch.save(
                {
                    "version": int(_NEGATIVE_CACHE_VALIDATION_SIDECAR_VERSION),
                    "cache_file_identity": final_file_identity,
                    "cache_metadata_sha256": _negative_cache_metadata_digest(metadata),
                    "candidate_tensor_sha256": metadata.get("candidate_tensor_sha256"),
                    "fallback_boundary_sha256": metadata.get("fallback_boundary_sha256"),
                    "validator_selector_implementation_sha256": str(validator_selector_digest),
                },
                temporary_path,
            )
            _fsync_file(temporary_path)
            os.replace(temporary_path, sidecar_path)
            _fsync_directory(os.path.dirname(os.path.abspath(sidecar_path)))
        finally:
            if os.path.exists(temporary_path):
                with suppress(OSError):
                    os.remove(temporary_path)
    except Exception as exc:
        print(f"WARNING: failed to publish HeaRT cache validation sidecar: {exc}", flush=True)


def _try_load_negative_cache(path, out, expected_metadata, k2):
    if not path or not os.path.exists(path):
        return None
    try:
        payload = torch.load(path, map_location="cpu", mmap=True, weights_only=True)
        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            raise ValueError("missing cache metadata")
        canonical_metadata = _canonical_cache_metadata(metadata)
        cached_selector_digest = canonical_metadata.get("selector_implementation_sha256")
        expected_selector_digest = expected_metadata.get("selector_implementation_sha256")
        cached_provenance = (cached_selector_digest, canonical_metadata.get("ppr_method"))
        expected_provenance = (expected_selector_digest, expected_metadata.get("ppr_method"))
        provenance_semantically_equivalent = cached_provenance in _CACHE_PROVENANCE_EQUIVALENCE.get(expected_provenance, set())
        mismatches = [
            f"{key}={canonical_metadata.get(key)!r} (expected {value!r})"
            for (key, value) in expected_metadata.items()
            if canonical_metadata.get(key) != value
            and (not (key in {"selector_implementation_sha256", "ppr_method"} and provenance_semantically_equivalent))
        ]
        if mismatches:
            raise ValueError("cache metadata mismatch: " + "; ".join(mismatches))
        initial_file_identity = _negative_cache_file_identity(path)
        trusted_immutable_validation = _try_load_negative_cache_validation(path, metadata, expected_selector_digest)
        if provenance_semantically_equivalent and (not trusted_immutable_validation):
            print("Accepting output-equivalent prior selector/PPR cache; full candidate SHA-256 validation remains required.", flush=True)
        valid_pos = payload["valid_pos"].to(torch.long)
        test_pos = payload["test_pos"].to(torch.long)
        if not torch.equal(valid_pos, out["valid_pos"].cpu().to(torch.long)) or not torch.equal(test_pos, out["test_pos"].cpu().to(torch.long)):
            raise ValueError("cached positives do not match this split")
        valid_candidates = payload["valid_candidate_nodes"]
        test_candidates = payload["test_candidate_nodes"]
        valid_fallback_counts = payload["valid_fallback_counts"]
        test_fallback_counts = payload["test_fallback_counts"]
        if not trusted_immutable_validation:
            candidate_sha256 = _named_tensor_digest("ogb-generated-heart-candidates", (("valid", valid_candidates), ("test", test_candidates)))
            if metadata.get("candidate_tensor_sha256") != candidate_sha256:
                raise ValueError("candidate tensor SHA-256 does not match metadata")
            fallback_metadata = _fallback_metadata(valid_fallback_counts, test_fallback_counts)
            fallback_mismatches = [
                f"{key}={metadata.get(key)!r} (expected {value!r})"
                for (key, value) in fallback_metadata.items()
                if metadata.get(key) != value
            ]
            if fallback_mismatches:
                raise ValueError("fallback metadata mismatch: " + "; ".join(fallback_mismatches))
            _validate_candidate_nodes(
                "valid",
                out["valid_pos"],
                valid_candidates,
                k2,
                out["num_nodes"],
                fallback_counts=valid_fallback_counts,
                trusted_reference_digest=True,
            )
            _validate_candidate_nodes(
                "test",
                out["test_pos"],
                test_candidates,
                k2,
                out["num_nodes"],
                fallback_counts=test_fallback_counts,
                trusted_reference_digest=True,
            )
            _save_negative_cache_validation(path, metadata, expected_selector_digest, initial_file_identity)
        else:
            print("Trusted unchanged fully validated HeaRT cache file identity.", flush=True)
        print(f"Loaded memory-mapped HeaRT split cache: {path}", flush=True)
        return (*_cached_negative_objects(out, valid_candidates, test_candidates, k2), canonical_metadata)
    except Exception as exc:
        print(f"Ignoring unreadable negative cache {path}: {exc}", flush=True)
    return None


def _save_negative_cache(path, out, expected_metadata, k2, valid_candidates, test_candidates, valid_fallback_counts, test_fallback_counts):
    if not path:
        return False
    temporary_path = f"{path}.tmp.{os.getpid()}.{uuid.uuid4().hex}"
    try:
        payload = {
            "metadata": {
                **expected_metadata,
                "candidate_tensor_sha256": _named_tensor_digest("ogb-generated-heart-candidates", (("valid", valid_candidates), ("test", test_candidates))),
                **_fallback_metadata(valid_fallback_counts, test_fallback_counts),
            },
            "valid_pos": out["valid_pos"].detach().cpu().to(torch.long),
            "test_pos": out["test_pos"].detach().cpu().to(torch.long),
            "valid_candidate_nodes": valid_candidates.detach().cpu().to(torch.int32),
            "test_candidate_nodes": test_candidates.detach().cpu().to(torch.int32),
            "valid_fallback_counts": valid_fallback_counts.detach().cpu().to(torch.int16),
            "test_fallback_counts": test_fallback_counts.detach().cpu().to(torch.int16),
        }
        torch.save(payload, temporary_path)
        _fsync_file(temporary_path)
        os.replace(temporary_path, path)
        _fsync_directory(os.path.dirname(os.path.abspath(path)))
        print(f"Saved reusable HeaRT split cache: {path}", flush=True)
        return True
    except Exception as exc:
        print(f"WARNING: failed to save negative cache {path}: {exc}", flush=True)
        return False
    finally:
        if os.path.exists(temporary_path):
            with suppress(OSError):
                os.remove(temporary_path)


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

            print(f"Acquiring HeaRT split-cache lock: {lock_path}", flush=True)
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


def _public_heart_cache_metadata(metadata, cache_path):
    public = {f"heart_{key}": metadata[key] for key in _PUBLIC_METADATA}
    public.update({f"heart_{key}": metadata.get(key) for key in _PUBLIC_OPTIONAL_METADATA})
    public.update({
        "heart_candidate_protocol": metadata["generation_protocol"],
        "heart_candidate_cache_version": metadata["cache_version"],
        "heart_candidate_cache_layout_version": metadata["layout_version"],
        "heart_candidate_cache_metadata_version": metadata["metadata_version"],
        "heart_candidate_sha256": metadata["candidate_tensor_sha256"],
        "heart_positive_split_sha256": metadata["ordered_positive_split_sha256"],
        "heart_tie_seed": metadata["selector_tie_seed"],
        "heart_fallback_seed": metadata["selector_fallback_seed"],
        "heart_cache_validation_contract": metadata["cache_validation_contract"],
        "heart_candidate_cache_metadata": dict(metadata),
        "negative_cache_path": cache_path,
    })
    return public


def _load_or_build_generated_heart(out, data_name, requested_draw, seed, eval_cap, ranked_backend, negative_cache_dir, cache_negatives):
    backend = _ranked_backend_for(out, str(ranked_backend or "auto").strip().lower())
    draw_per_side = int(GENERATED_HEART_NEGATIVES_PER_SIDE)
    protocol_identity = _load_or_build_protocol_identity(out, data_name, negative_cache_dir) if cache_negatives else None
    expected_metadata = _expected_negative_cache_metadata(out, data_name, backend, draw_per_side, protocol_identity)
    cache_path = None
    if cache_negatives:
        if str(data_name).strip().lower() == "ogbl-ppa":
            cache_split_seed = int(out.get("effective_eval_seed") or 0)
            cache_eval_cap = int(out.get("effective_eval_cap") or 0)
        else:
            cache_split_seed = int(seed) if int(eval_cap or 0) > 0 else 0
            cache_eval_cap = eval_cap
        cache_path = _negative_cache_file(
            negative_cache_dir,
            data_name,
            "heart_exact_andersen_released_mask_hard",
            cache_split_seed,
            cache_eval_cap,
            draw_per_side,
            backend,
            out["valid_pos"].size(0),
            out["test_pos"].size(0),
            query_panel_identity=out.get("heart_ppa_query_panel_identity_sha256") if str(data_name).strip().lower() == "ogbl-ppa" else None,
        )
        cached = _try_load_negative_cache(cache_path, out, expected_metadata, draw_per_side)
    else:
        cached = None
    if cached is not None:
        (valid_neg, test_neg, cache_metadata) = cached
    else:
        with _exclusive_cache_build(cache_path):
            cached = _try_load_negative_cache(cache_path, out, expected_metadata, draw_per_side)
            if cached is not None:
                (valid_neg, test_neg, cache_metadata) = cached
            else:
                protocol_identity = (
                    _load_or_build_protocol_identity(out, data_name, negative_cache_dir, force_full=True) if cache_negatives else None
                )
                expected_metadata = _expected_negative_cache_metadata(out, data_name, backend, draw_per_side, protocol_identity)
                print(f"HeaRT candidate universe=full legal graph; selecting {draw_per_side} ranked negatives per side.", flush=True)
                resume_state = _generation_resume_state(cache_path, expected_metadata)
                (valid_generated, test_generated, valid_fallback_counts, test_fallback_counts) = _generate_ranked_double_sided_neg(
                    out, data_name, draw_per_side, int(GENERATED_HEART_TIE_SEED), backend, resume_state=resume_state
                )
                if valid_generated.dim() == 2 and test_generated.dim() == 2:
                    valid_candidates = valid_generated.to(torch.int32).contiguous()
                    test_candidates = test_generated.to(torch.int32).contiguous()
                else:
                    valid_candidates = _compact_endpoint_candidates(out["valid_pos"], valid_generated, draw_per_side)
                    test_candidates = _compact_endpoint_candidates(out["test_pos"], test_generated, draw_per_side)
                del valid_generated, test_generated
                _ensure_heart_eligibility_filters(out, data_name)
                _validate_candidate_nodes(
                    "valid",
                    out["valid_pos"],
                    valid_candidates,
                    draw_per_side,
                    out["num_nodes"],
                    fallback_counts=valid_fallback_counts,
                    out=out,
                )
                _validate_candidate_nodes(
                    "test", out["test_pos"], test_candidates, draw_per_side, out["num_nodes"], fallback_counts=test_fallback_counts, out=out
                )
                cache_metadata = {
                    **expected_metadata,
                    "candidate_tensor_sha256": _named_tensor_digest("ogb-generated-heart-candidates", (("valid", valid_candidates), ("test", test_candidates))),
                    **_fallback_metadata(valid_fallback_counts, test_fallback_counts),
                }
                (valid_neg, test_neg) = _cached_negative_objects(out, valid_candidates, test_candidates, draw_per_side)
                cache_saved = _save_negative_cache(
                    cache_path,
                    out,
                    expected_metadata,
                    draw_per_side,
                    valid_candidates,
                    test_candidates,
                    valid_fallback_counts,
                    test_fallback_counts,
                )
                if cache_path and (not cache_saved):
                    cache_path = None
                elif cache_saved:
                    _cleanup_generation_resume(cache_path, resume_state)
    return (valid_neg, test_neg, draw_per_side, backend, cache_path, _public_heart_cache_metadata(cache_metadata, cache_path))
