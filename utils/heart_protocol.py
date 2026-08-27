import os
import warnings

GENERATED_HEART_NEGATIVES_TOTAL = 500
GENERATED_HEART_NEGATIVES_PER_SIDE = GENERATED_HEART_NEGATIVES_TOTAL // 2
GENERATED_HEART_TIE_SEED = 42
GENERATED_HEART_SELECTOR_RECIPE = (
    "min-ra-andersen-rank+literal-cpu-topk-nth-element-or-sparse-libstdcxx-partial-sort-heap+numpy42-fallback-v4"
)


def _serializable_heart_metadata_value(value):
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, dict):
        return {str(key): _serializable_heart_metadata_value(item) for (key, item) in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serializable_heart_metadata_value(item) for item in value]
    item = getattr(value, "item", None)
    numel = getattr(value, "numel", None)
    try:
        if callable(item) and (not callable(numel) or int(numel()) == 1):
            return _serializable_heart_metadata_value(item())
    except (TypeError, ValueError, RuntimeError):
        pass
    return str(value)


def heart_candidate_metadata(bundle):
    metadata = {}
    for key, value in sorted(dict(bundle or {}).items()):
        if not (str(key).startswith("heart_") or key in {"ranked_backend", "negative_cache_path"}):
            continue
        metadata[str(key)] = _serializable_heart_metadata_value(value)
    source = metadata.get("heart_source")
    if source is not None:
        metadata["heart_source_resolved"] = source
    backend = metadata.get("ranked_backend")
    if backend is not None:
        metadata["heart_backend_resolved"] = backend
    return metadata


def persist_heart_candidate_metadata(args, bundle):
    metadata = heart_candidate_metadata(bundle)
    for key, value in metadata.items():
        if key in {"heart_source", "heart_backend"}:
            continue
        setattr(args, key, value)
    args.heart_source_resolved = metadata.get("heart_source_resolved")
    args.heart_backend_resolved = metadata.get("heart_backend_resolved")
    args.heart_candidate_metadata = dict(metadata)
    return metadata


def heart_negative_count_metadata(requested_total, *, effective_total=None):
    requested = int(requested_total)
    effective = requested if effective_total is None else int(effective_total)
    compatible = effective == GENERATED_HEART_NEGATIVES_TOTAL
    return {
        "heart_negatives_generated_contract_total": GENERATED_HEART_NEGATIVES_TOTAL,
        "heart_negatives_generated_contract_per_side": GENERATED_HEART_NEGATIVES_PER_SIDE,
        "heart_negative_count_protocol": "generated-fixed-500-total" if compatible else f"generated-custom-{effective}-total",
        "heart_negative_count_requested_contract_compatible": requested == GENERATED_HEART_NEGATIVES_TOTAL,
        "heart_negative_count_contract_compatible": compatible,
    }


def warn_if_custom_heart_negative_count(requested_total, *, effective_total=None, source=None):
    requested = int(requested_total)
    if requested == GENERATED_HEART_NEGATIVES_TOTAL:
        return
    effective = requested if effective_total is None else int(effective_total)
    source_text = str(source or "unknown")
    if effective == requested:
        effect = f"The requested {requested} total remains in effect, so this run does not match the standard generated-candidate count."
    else:
        effect = f"The {source_text} generator has a fixed effective count of {effective}; the custom request is not applied."
    warnings.warn(
        f"Nonstandard generated HeaRT candidate count: {requested} total negatives per positive. The generated-candidate contract uses exactly 500 total (250 per corruption side) for every dataset, including ogbl-citation2. {effect}",
        UserWarning,
        stacklevel=2,
    )
