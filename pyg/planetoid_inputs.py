"""Fixed positive-split and feature inputs for Planetoid evaluation."""

from __future__ import annotations

import os
from typing import Iterable, Optional

import torch


FIXED_PLANETOID_POSITIVE_SPLIT_DATASETS = frozenset(
    {"cora", "citeseer", "pubmed"}
)
SPLIT_FILENAMES = {
    "train": "train_pos.txt",
    "valid": "valid_pos.txt",
    "test": "test_pos.txt",
}
FEATURE_FILENAME = "gnn_feature"


def _unique_existing_directories(paths: Iterable[str]) -> list[str]:
    seen = set()
    directories = []
    for path in filter(None, paths):
        resolved = os.path.abspath(os.path.expanduser(str(path)))
        if resolved not in seen and os.path.isdir(resolved):
            seen.add(resolved)
            directories.append(resolved)
    return directories


def candidate_planetoid_input_dirs(
    data_name: str,
    root: Optional[str] = None,
    input_root: Optional[str] = None,
) -> list[str]:
    """Return existing locations that may contain fixed Planetoid inputs."""

    name = str(data_name).lower()
    cwd = os.getcwd()
    roots = []
    for base in (
        input_root,
        root,
        os.environ.get("PLANETOID_INPUT_ROOT"),
        cwd,
    ):
        if not base:
            continue
        base = os.path.abspath(os.path.expanduser(str(base)))
        roots.extend(
            [
                base,
                os.path.join(base, name),
                os.path.join(base, "dataset", name),
                os.path.join(base, "HeaRT", "dataset", name),
            ]
        )
    roots.extend(
        [
            os.path.join(cwd, "dataset", name),
            os.path.join(cwd, "HeaRT", "dataset", name),
            os.path.join(os.path.dirname(cwd), "dataset", name),
            os.path.join(os.path.dirname(cwd), "HeaRT", "dataset", name),
        ]
    )
    return _unique_existing_directories(roots)


def _find_positive_split_dir(
    data_name: str,
    root: Optional[str],
    input_root: Optional[str],
) -> Optional[str]:
    required = tuple(SPLIT_FILENAMES.values())
    for directory in candidate_planetoid_input_dirs(
        data_name, root=root, input_root=input_root
    ):
        if all(os.path.isfile(os.path.join(directory, name)) for name in required):
            return directory
    return None


def _read_positive_edges(path: str) -> torch.Tensor:
    edges = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            fields = line.strip().replace(",", " ").split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"Expected two node ids at {path}:{line_number}, got {line!r}."
                )
            source, target = (int(value) for value in fields)
            if source != target:
                edges.append((source, target))
    if not edges:
        return torch.empty((0, 2), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long)


def load_fixed_planetoid_positive_split(
    data_name: str,
    *,
    num_nodes: Optional[int] = None,
    root: Optional[str] = None,
    input_root: Optional[str] = None,
) -> Optional[dict]:
    """Load the fixed positive-edge partition without loading negatives."""

    name = str(data_name).lower()
    if name not in FIXED_PLANETOID_POSITIVE_SPLIT_DATASETS:
        return None
    directory = _find_positive_split_dir(name, root, input_root)
    if directory is None:
        return None
    paths = {
        split_name: os.path.join(directory, filename)
        for split_name, filename in SPLIT_FILENAMES.items()
    }
    split_edges = {
        split_name: _read_positive_edges(path).contiguous()
        for split_name, path in paths.items()
    }
    all_edges = torch.cat(list(split_edges.values()), dim=0)
    if all_edges.numel():
        minimum, maximum = int(all_edges.min()), int(all_edges.max())
        if minimum < 0:
            raise ValueError(
                f"Fixed split node ids must be non-negative, found {minimum} "
                f"under {directory}."
            )
        if num_nodes is not None and maximum >= int(num_nodes):
            raise ValueError(
                f"Fixed split node ids [{minimum}, {maximum}] are outside "
                f"the local graph with {int(num_nodes)} nodes."
            )
    return {
        "train_pos": split_edges["train"],
        "all_valid_pos": split_edges["valid"],
        "valid_pos": split_edges["valid"],
        "test_pos": split_edges["test"],
        "artifact_dir": directory,
        "split_paths": paths,
        "positive_split_source": "fixed-planetoid-positive-txt",
    }


def missing_fixed_planetoid_positive_split_message(
    data_name: str,
    root: Optional[str] = None,
    input_root: Optional[str] = None,
) -> str:
    searched = candidate_planetoid_input_dirs(
        str(data_name).lower(), root=root, input_root=input_root
    )
    locations = "\n".join(f"  - {path}" for path in searched)
    if not locations:
        locations = "  - no existing candidate dataset directories found"
    required = ", ".join(SPLIT_FILENAMES.values())
    return (
        f"Generated HeaRT evaluation for {data_name} requires the fixed "
        f"positive split files: {required}.\n"
        f"Searched:\n{locations}\nThe matching gnn_feature artifact must "
        "accompany the split files; candidate negatives are generated locally. "
        "Refusing to silently substitute a different seeded "
        "train/validation/test split."
    )


def _torch_load_cpu(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def load_fixed_planetoid_features(directory: str, num_nodes: int):
    path = os.path.join(directory, FEATURE_FILENAME)
    if not os.path.isfile(path):
        return (None, None)
    payload = _torch_load_cpu(path)
    feature = payload.get("entity_embedding") if isinstance(payload, dict) else payload
    if not torch.is_tensor(feature) or feature.dim() != 2:
        raise ValueError(f"Expected {path} to contain a 2-D entity_embedding tensor.")
    if int(feature.size(0)) != int(num_nodes):
        raise ValueError(
            f"Feature rows ({feature.size(0)}) do not match num_nodes "
            f"({num_nodes}) in {path}."
        )
    return (feature.to(torch.float32).contiguous(), path)
