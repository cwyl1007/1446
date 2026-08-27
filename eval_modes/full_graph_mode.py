"""Independent evaluator dispatch for exhaustive full-graph candidates."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping

from eval_modes import evaluator_helpers as _core


POLICY = "exhaustive"
GROUPED = False
TEST_ONLY = True
ORIENT_QUERY_ENDPOINTS = False
USES_FIXED_PLANETOID_INPUTS = True
TARGET_CHECKPOINT_MODE = None
CANDIDATE_LABEL = "full-graph"
EVALUATION_PROTOCOL = "exact_streaming_full_graph_two_sided"
RESULT_FILENAME = "full_graph_evaluation.json"
FORMAT_VERSION = 6
OUTPUT_INCLUDES_CHECKPOINT_MODE = True


def add_evaluator_arguments(parser):
    return parser


def validate_cli(args, dataset):
    return 0


def evaluator_context(args, dataset, framework, device):
    validate_cli(args, dataset)
    return {
        "policy": POLICY,
        "dataset": str(dataset),
        "framework": str(framework),
        "device": device,
        "load_seconds": 0.0,
        "heuristic": bool(getattr(args, "heuristic", None)),
    }


def _requested_mode(args, checkpoint):
    requested = getattr(args, "mode", None)
    if requested:
        return str(requested).strip().lower()
    if isinstance(checkpoint, Mapping):
        arguments = checkpoint.get("arguments") or {}
        requested = arguments.get("mode") or checkpoint.get("mode")
        if requested:
            return str(requested).strip().lower()
    return "all"


def resolve_cap(args, checkpoint, framework, dataset):
    value = str(getattr(args, "eval_cap", "checkpoint")).strip().lower()
    if value != "checkpoint" or checkpoint is not None:
        return _core.resolve_cap(value, checkpoint, framework, dataset)
    mode = _requested_mode(args, checkpoint)
    if framework == "pyg":
        from pyg.prepare_data import resolve_pyg_eval_cap

        return resolve_pyg_eval_cap(None, mode, dataset)
    from ogbl.protocol import resolve_ogbl_eval_cap

    return resolve_ogbl_eval_cap(None, mode, dataset)


def data_seed(context, target_seed):
    return int(target_seed)


def _load_plan(args, context, checkpoint, cap):
    framework = context["framework"]
    dataset = context["dataset"]
    mode = _requested_mode(args, checkpoint)
    generated_pyg = _core.use_generated_pyg_heart_protocol(framework, mode, dataset)
    generated_ogb = (
        framework == "ogb"
        and mode == "heart"
        and dataset.strip().lower() in _core.HEART_BENCHMARK_OGB_DATASETS
    )
    return {
        "mode": mode,
        "generated_pyg": generated_pyg,
        "generated_ogb": generated_ogb,
    }


def cache_key(args, context, data_seed, cap, checkpoint=None):
    plan = _load_plan(args, context, checkpoint, cap)
    cache_seed = int(data_seed)
    if (plan["generated_pyg"] or plan["generated_ogb"]) and int(cap) == 0:
        cache_seed = 0
    if plan["generated_pyg"]:
        return ("fixed-planetoid-positive-exhaustive", cache_seed, int(cap))
    if plan["generated_ogb"]:
        return ("generated-ogb-positive-scope-exhaustive", cache_seed, int(cap))
    return (POLICY, cache_seed, int(cap))


def load_bundle(framework, dataset, root, seed, cap, device=None, checkpoint=None, args=None, context=None):
    device = device or "cpu"
    args = args or SimpleNamespace(mode="all", planetoid_input_root=None)
    context = context or evaluator_context(args, dataset, framework, device)
    plan = _load_plan(args, context, checkpoint, cap)
    if plan["generated_pyg"]:
        bundle = _core.load_fixed_planetoid_positive_base(
            dataset,
            root,
            seed,
            cap,
            planetoid_input_root=getattr(args, "planetoid_input_root", None),
        )
    elif plan["generated_ogb"]:
        from ogbl.prepare_data import read_ogbl_generated_positive_scope

        bundle = read_ogbl_generated_positive_scope(
            dataset,
            seed=int(seed),
            root=str(root),
            eval_cap=int(cap),
        )
    else:
        bundle = _core.load_bundle(framework, dataset, root, seed, cap)
    return bundle


def prepare_bundle(bundle):
    return bundle


def install_evaluator_candidates(bundle, framework, dataset, device, args, context, data_seed, root):
    del bundle, framework, dataset, device, args, context, data_seed, root
    return {}


def selector_metadata(args, context, data_seed):
    return {}


def candidate_pool(bundle, context, both_sides):
    del context, both_sides
    scope = bundle.get("test_positive_scope") or {}
    rows = scope.get("test_positive_rows")
    if rows is None and hasattr(bundle.get("test_pos"), "size"):
        rows = int(bundle["test_pos"].size(0))
    output = {
        "setting": "all",
        "per_side_before_legality_filter": "num_nodes",
        "both_sides_before_legality_filter": "2 * num_nodes",
        "positive_query_scope": scope.get("policy", "unspecified"),
        "positive_query_rows": rows,
        "sampling": None,
        "maximum": None,
        "paired_heart_positive_scope": bool(bundle.get("paired_heart_positive_scope", False)),
    }
    if "full_graph_known_positive_filter" in bundle:
        output["known_positive_filter"] = dict(bundle["full_graph_known_positive_filter"])
    if output["paired_heart_positive_scope"]:
        output["loaded_metadata"] = _core.heart_bundle_metadata(bundle)
    return output


def result_fields(bundle):
    output = {}
    if "full_graph_known_positive_filter" in bundle:
        output["known_positive_filter"] = dict(bundle["full_graph_known_positive_filter"])
    if bundle.get("paired_heart_positive_scope"):
        output.update(heart_candidates=_core.heart_bundle_metadata(bundle), paired_heart_positive_scope=True)
    return output


def output_root(args, project_root):
    del args
    return Path(project_root) / "results" / "full_graph"


def format_version(framework, dataset, *, heuristic):
    if heuristic and str(framework).lower() == "ogb" and str(dataset).lower() == "ogbl-collab":
        return 8
    return FORMAT_VERSION


def main():
    return _core.main_for_mode(sys.modules[__name__])


if __name__ == "__main__":
    raise SystemExit(main())
