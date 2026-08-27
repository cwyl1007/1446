from __future__ import annotations
import argparse
import hashlib
import json
import os
from typing import Iterable, Optional
import torch

PPA_QUERY_PANEL_ENV = "OGBL_PPA_QUERY_PANEL"
PPA_QUERY_INDEX_ROOT_ENV = "OGBL_PPA_QUERY_INDEX_ROOT"
PPA_REFERENCE_QUERY_COUNT = 100000
PPA_VALID_INDEX_NAME = "valid_samples_index.pt"
PPA_TEST_INDEX_NAME = "test_samples_index.pt"
PPA_REFERENCE_RECIPE = "heart-reference-ppa-fixed-ordered-index-panel-v1"
PPA_LOCAL_RECIPE = "local-seeded-validation-randperm-test-full-ordered-custom-v2"
PPA_LOCAL_SCOPE = "validation-local-seeded-max100000-test-full-ordered-custom"


def resolve_ppa_query_panel_mode(value: Optional[str] = None) -> str:
    raw = os.environ.get(PPA_QUERY_PANEL_ENV, "local-seeded") if value is None else value
    mode = str(raw).strip().lower().replace("_", "-")
    aliases = {
        "official": "reference",
        "fixed": "reference",
        "official-fixed": "reference",
        "released": "reference",
        "local": "local-seeded",
        "seeded": "local-seeded",
        "random": "local-seeded",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"reference", "local-seeded"}:
        raise ValueError(
            f"{PPA_QUERY_PANEL_ENV} must be 'reference' or 'local-seeded'; got {raw!r}. The local-seeded mode is a custom panel."
        )
    return mode


def _unique_directories(paths: Iterable[Optional[str]]) -> list[str]:
    seen = set()
    result = []
    for value in paths:
        if not value:
            continue
        path = os.path.abspath(os.path.expanduser(str(value)))
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def ppa_query_index_directories(*, root: str = "dataset", query_index_root: Optional[str] = None) -> list[str]:
    cwd = os.getcwd()
    explicit = os.environ.get(PPA_QUERY_INDEX_ROOT_ENV)
    bases = _unique_directories((query_index_root, explicit, root, cwd))
    candidates = []
    for base in bases:
        candidates.extend(
            (
                base,
                os.path.join(base, "ogbl-ppa"),
                os.path.join(base, "dataset", "ogbl-ppa"),
                os.path.join(base, "data", "ppa_subset"),
                os.path.join(base, "HeaRT", "data", "ppa_subset"),
            )
        )
    candidates.extend(
        (
            os.path.join(cwd, "dataset", "ogbl-ppa"),
            os.path.join(cwd, "HeaRT", "data", "ppa_subset"),
            os.path.join(os.path.dirname(cwd), "HeaRT", "data", "ppa_subset"),
        )
    )
    return _unique_directories(candidates)


def find_reference_ppa_query_indices(*, root: str = "dataset", query_index_root: Optional[str] = None) -> tuple[str, str]:
    checked = []
    partial = []
    for directory in ppa_query_index_directories(root=root, query_index_root=query_index_root):
        valid_path = os.path.join(directory, PPA_VALID_INDEX_NAME)
        test_path = os.path.join(directory, PPA_TEST_INDEX_NAME)
        checked.extend((valid_path, test_path))
        valid_exists = os.path.isfile(valid_path)
        test_exists = os.path.isfile(test_path)
        if valid_exists and test_exists:
            return (valid_path, test_path)
        if valid_exists or test_exists:
            partial.append(directory)
    detail = "\n  ".join(checked) if checked else "<no candidate paths>"
    partial_detail = "\nFound an incomplete pair in: " + ", ".join(partial) if partial else ""
    raise FileNotFoundError(
        f"Generated ogbl-ppa reference mode requires the released ordered query indices {PPA_VALID_INDEX_NAME} and"
        f" {PPA_TEST_INDEX_NAME}. Neither a seeded replacement nor an old generated cache is accepted. Place both files in"
        " dataset/ogbl-ppa, set OGBL_PPA_QUERY_INDEX_ROOT, or set OGBL_PPA_QUERY_PANEL=local-seeded for a custom"
        f" run.{partial_detail}\nChecked:\n  {detail}"
    )


def _torch_load_raw(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(f"tensor-v1;shape={tuple(tensor.shape)};dtype={tensor.dtype};".encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _file_sha256(path: str, chunk_bytes: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(int(chunk_bytes))
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def validate_reference_ppa_index(raw, *, split_name: str, split_rows: Optional[int] = None) -> torch.Tensor:
    if not torch.is_tensor(raw):
        raise TypeError(f"ogbl-ppa {split_name} reference query index must be a raw torch.Tensor.")
    if raw.dtype != torch.int64:
        raise TypeError(f"ogbl-ppa {split_name} reference query index must have dtype torch.int64, got {raw.dtype}.")
    if raw.dim() != 1:
        raise ValueError(
            f"ogbl-ppa {split_name} reference query index must be rank 1, got shape {tuple(raw.shape)}. Refusing to flatten because row order is protocol data."
        )
    if raw.numel() != PPA_REFERENCE_QUERY_COUNT:
        raise ValueError(
            f"ogbl-ppa {split_name} reference query index must contain exactly {PPA_REFERENCE_QUERY_COUNT} ordered rows, got {raw.numel()}."
        )
    index = raw.detach().cpu().contiguous()
    if index.numel() and int(index.min()) < 0:
        raise ValueError(f"ogbl-ppa {split_name} reference query index contains a negative row.")
    if split_rows is not None and index.numel() and (int(index.max()) >= int(split_rows)):
        raise ValueError(
            f"ogbl-ppa {split_name} reference query index must be within [0,{int(split_rows)}), got maximum {int(index.max())}."
        )
    if torch.unique(index).numel() != index.numel():
        raise ValueError(f"ogbl-ppa {split_name} reference query index contains duplicate rows.")
    return index


def _ordered_query_sha256(full_split: torch.Tensor, index: torch.Tensor) -> tuple[torch.Tensor, str]:
    split = torch.as_tensor(full_split)
    if split.dim() != 2 or split.size(1) != 2:
        raise ValueError(f"OGB positive split must have shape [N,2], got {tuple(split.shape)}.")
    selected = split.detach().cpu().index_select(0, index).to(torch.int64).contiguous()
    return (selected, _tensor_sha256(selected))


def _identity(metadata: dict) -> str:
    identity_metadata = dict(metadata)
    if identity_metadata.get("mode") == "reference":
        identity_metadata["mode"] = "official"
    if identity_metadata.get("scope") == "reference-fixed-100000-released-index-order":
        identity_metadata["scope"] = "paper-fixed-100000-official-index-order"
    if identity_metadata.get("recipe") == PPA_REFERENCE_RECIPE:
        identity_metadata["recipe"] = "heart-paper-ppa-fixed-ordered-index-panel-v1"
    if identity_metadata.get("scope") == PPA_LOCAL_SCOPE:
        identity_metadata["scope"] = "validation-local-seeded-max100000-test-full-ordered-nonpaper"
    recipe = identity_metadata.get("recipe")
    if isinstance(recipe, str) and recipe.startswith(PPA_LOCAL_RECIPE):
        identity_metadata["recipe"] = recipe.replace(PPA_LOCAL_RECIPE, "local-seeded-validation-randperm-test-full-ordered-nonpaper-v2", 1)
    encoded = json.dumps(identity_metadata, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(b"ogbl-ppa-query-panel-identity-v1;" + encoded).hexdigest()


def _local_validation_cap(eval_cap: int) -> int:
    cap = int(eval_cap or 0)
    if cap < 0:
        raise ValueError("ogbl-ppa local query eval_cap must be non-negative.")
    if cap == 0:
        return PPA_REFERENCE_QUERY_COUNT
    return min(cap, PPA_REFERENCE_QUERY_COUNT)


def load_ppa_query_panel(
    *,
    valid_split: torch.Tensor,
    test_split: torch.Tensor,
    root: str = "dataset",
    query_index_root: Optional[str] = None,
    mode: Optional[str] = None,
    seed: int = 0,
    eval_cap: int = PPA_REFERENCE_QUERY_COUNT,
) -> dict:
    mode = resolve_ppa_query_panel_mode(mode)
    valid_rows = int(valid_split.size(0))
    test_rows = int(test_split.size(0))
    cap = int(eval_cap or 0)
    if mode == "reference":
        if cap not in {0, PPA_REFERENCE_QUERY_COUNT}:
            raise ValueError(
                f"Reference ogbl-ppa query mode permits only eval_cap=0 or {PPA_REFERENCE_QUERY_COUNT}; use local-seeded for a custom cap."
            )
        (valid_path, test_path) = find_reference_ppa_query_indices(root=root, query_index_root=query_index_root)
        valid_index = validate_reference_ppa_index(_torch_load_raw(valid_path), split_name="valid", split_rows=valid_rows)
        test_index = validate_reference_ppa_index(_torch_load_raw(test_path), split_name="test", split_rows=test_rows)
        recipe = PPA_REFERENCE_RECIPE
        reference_scope = True
        valid_file_sha256 = _file_sha256(valid_path)
        test_file_sha256 = _file_sha256(test_path)
    else:
        validation_cap = _local_validation_cap(cap)
        valid_count = min(validation_cap, valid_rows)
        valid_generator = torch.Generator().manual_seed(int(seed) + 100)
        valid_index = torch.randperm(valid_rows, generator=valid_generator)[:valid_count]
        test_index = torch.arange(test_rows, dtype=torch.long)
        recipe = f"{PPA_LOCAL_RECIPE};seed={int(seed)};valid_seed={int(seed) + 100};valid_cap={validation_cap};test=full-original-order"
        reference_scope = False
        valid_path = None
        test_path = None
        valid_file_sha256 = None
        test_file_sha256 = None
    (valid_pos, valid_query_sha256) = _ordered_query_sha256(valid_split, valid_index)
    (test_pos, test_query_sha256) = _ordered_query_sha256(test_split, test_index)
    scope = "reference-fixed-100000-released-index-order" if reference_scope else PPA_LOCAL_SCOPE
    identity_fields = {"mode": mode, "scope": scope, "recipe": recipe}
    if mode == "reference":
        identity_fields.update(
            {
                "valid_index_sha256": _tensor_sha256(valid_index),
                "test_index_sha256": _tensor_sha256(test_index),
                "valid_file_sha256": valid_file_sha256,
                "test_file_sha256": test_file_sha256,
            }
        )
    panel_identity = _identity(identity_fields)
    return {
        "valid_pos": valid_pos,
        "test_pos": test_pos,
        "valid_index": valid_index,
        "test_index": test_index,
        "mode": mode,
        "recipe": recipe,
        "reference_positive_query_scope": reference_scope,
        "query_scope": scope,
        "identity_sha256": panel_identity,
        "valid_index_sha256": _tensor_sha256(valid_index),
        "test_index_sha256": _tensor_sha256(test_index),
        "valid_file_sha256": valid_file_sha256,
        "test_file_sha256": test_file_sha256,
        "valid_query_sha256": valid_query_sha256,
        "test_query_sha256": test_query_sha256,
        "valid_index_path": valid_path,
        "test_index_path": test_path,
    }


def preflight_ppa_query_panel(*, root: str = "dataset", query_index_root: Optional[str] = None, mode: Optional[str] = None, seed: int = 0, eval_cap: int = PPA_REFERENCE_QUERY_COUNT) -> dict:
    mode = resolve_ppa_query_panel_mode(mode)
    cap = int(eval_cap or 0)
    if mode == "local-seeded":
        validation_cap = _local_validation_cap(cap)
        fields = {
            "mode": mode,
            "scope": PPA_LOCAL_SCOPE,
            "recipe": f"{PPA_LOCAL_RECIPE};seed={int(seed)};valid_seed={int(seed) + 100};valid_cap={validation_cap};test=full-original-order",
        }
        return {**fields, "identity_sha256": _identity(fields)}
    if cap not in {0, PPA_REFERENCE_QUERY_COUNT}:
        raise ValueError("Reference ogbl-ppa query mode requires eval_cap=0 or 100000.")
    valid_path, test_path = find_reference_ppa_query_indices(root=root, query_index_root=query_index_root)
    valid_index = validate_reference_ppa_index(_torch_load_raw(valid_path), split_name="valid")
    test_index = validate_reference_ppa_index(_torch_load_raw(test_path), split_name="test")
    fields = {
        "mode": mode, "scope": "reference-fixed-100000-released-index-order", "recipe": PPA_REFERENCE_RECIPE,
        "valid_index_sha256": _tensor_sha256(valid_index), "test_index_sha256": _tensor_sha256(test_index),
        "valid_file_sha256": _file_sha256(valid_path), "test_file_sha256": _file_sha256(test_path),
    }
    return {**fields, "identity_sha256": _identity(fields)}


def _main() -> int:
    parser = argparse.ArgumentParser(description="Validate ogbl-ppa query panel inputs.")
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--root", default="dataset")
    parser.add_argument("--query-index-root", default=None)
    parser.add_argument("--mode", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--eval-cap", type=int, default=PPA_REFERENCE_QUERY_COUNT)
    parser.add_argument("--field", default=None)
    args = parser.parse_args()
    payload = preflight_ppa_query_panel(
        root=args.root,
        query_index_root=args.query_index_root,
        mode=args.mode,
        seed=args.seed,
        eval_cap=args.eval_cap,
    )
    if args.field not in payload and args.field is not None:
        raise KeyError(f"Unknown preflight field: {args.field}")
    print(payload[args.field] if args.field else json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
