import math
import torch

_REFERENCE_MF_SOURCE_SCHEDULES = {
    "ogbl-collab": (9999, 1, 100),
    "ogbl-ddi": (9999, 1, 100),
    "ogbl-ppa": (9999, 1, 100),
    "ogbl-citation2": (300, 1, 20),
}
_REFERENCE_MF_SOURCE_BATCH_SIZE = 65536


def reference_mf_batch_count(num_examples, batch_size, max_batches=None):
    count = int(math.ceil(int(num_examples) / max(1, int(batch_size))))
    if max_batches is not None and int(max_batches) > 0:
        count = min(count, int(max_batches))
    return count


def reference_mf_cpu_shuffle_batches(num_examples, batch_size, max_batches=None):
    num_examples = int(num_examples)
    if num_examples <= 0:
        return
    torch.empty((), dtype=torch.int64).random_()
    sampler_seed = int(torch.empty((), dtype=torch.int64).random_().item())
    sampler_generator = torch.Generator()
    sampler_generator.manual_seed(sampler_seed)
    order = torch.randperm(num_examples, generator=sampler_generator)
    batch_size = max(1, int(batch_size))
    limit = None if max_batches is None or int(max_batches) <= 0 else int(max_batches)
    for batch_index, start in enumerate(range(0, num_examples, batch_size)):
        if limit is not None and batch_index >= limit:
            break
        yield order[start : min(start + batch_size, num_examples)]


def reference_mf_random_negative_edge(dataset_name, pos_edge, num_nodes, device):
    if str(dataset_name).strip().lower() == "ogbl-citation2":
        neg_target = torch.randint(0, int(num_nodes), pos_edge[0].shape, dtype=torch.long, device=device)
        return torch.stack([pos_edge[0], neg_target], dim=0)
    return torch.randint(0, int(num_nodes), pos_edge.size(), dtype=torch.long, device=device)


def reference_mf_decode_batch(model, node_features, pos_edge, num_nodes, device):
    pos_logits = model.decode(node_features, pos_edge).view(-1)
    neg_edge = reference_mf_random_negative_edge(str(getattr(model, "dataset_name", "")), pos_edge, num_nodes, device)
    neg_logits = model.decode(node_features, neg_edge).view(-1)
    return (pos_logits, neg_logits)


def reference_mf_runtime_metadata(dataset_name, *, epochs, eval_steps, patience, optimizer_batch_size):
    dataset_key = str(dataset_name).strip().lower()
    (source_epochs, source_eval_steps, source_patience) = _REFERENCE_MF_SOURCE_SCHEDULES[dataset_key]
    effective_schedule = (int(epochs), int(eval_steps), int(patience))
    source_schedule = (source_epochs, source_eval_steps, source_patience)
    source_batch_size = int(_REFERENCE_MF_SOURCE_BATCH_SIZE)
    effective_batch_size = int(optimizer_batch_size)
    schedule_faithful = effective_schedule == source_schedule
    batch_faithful = effective_batch_size == source_batch_size
    deviations = []
    if not schedule_faithful:
        deviations.append("runtime-budgeted-training-schedule")
    if not batch_faithful:
        deviations.append("throughput-adapted-optimizer-batch")
    return {
        "reference_mf_source_epochs": int(source_epochs),
        "reference_mf_source_eval_steps": int(source_eval_steps),
        "reference_mf_source_patience": int(source_patience),
        "reference_mf_effective_epochs": int(epochs),
        "reference_mf_effective_eval_steps": int(eval_steps),
        "reference_mf_effective_patience": int(patience),
        "reference_mf_training_schedule_faithful": bool(schedule_faithful),
        "reference_mf_source_optimizer_batch_size": source_batch_size,
        "reference_mf_effective_optimizer_batch_size": effective_batch_size,
        "reference_mf_optimizer_batch_faithful": bool(batch_faithful),
        "reference_mf_optimizer_schedule_fidelity": "released" if not deviations else "+".join(deviations),
    }
