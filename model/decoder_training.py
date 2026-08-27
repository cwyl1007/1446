from __future__ import annotations
from typing import Callable, Optional
import torch
import torch.nn.functional as F


def _binary_prediction_loss(model, logits, positive, reduction="mean"):
    if bool(getattr(model, "reference_probability_loss", False)):
        probability = torch.sigmoid(logits.float())
        values = -torch.log(probability + 1e-15) if positive else -torch.log(1.0 - probability + 1e-15)
        if reduction == "sum":
            return values.sum()
        if reduction == "none":
            return values
        return values.mean()
    labels = torch.ones_like(logits) if positive else torch.zeros_like(logits)
    return F.binary_cross_entropy_with_logits(logits, labels, reduction=reduction)


def _clear_decode_cache(model, callback: Optional[Callable] = None) -> None:
    if callback is not None:
        callback(model)
        return
    clear = getattr(model, "clear_decode_cache", None)
    if callable(clear):
        clear()


def _backward_with_weighted_embedding_grad(loss: torch.Tensor, z: torch.Tensor, weight: float, *, retain_graph: bool, additional_leaves=()) -> None:
    handles = []
    scale = float(weight)
    leaves = (z, *tuple(additional_leaves))
    seen = set()
    for leaf in leaves:
        if not isinstance(leaf, torch.Tensor) or not leaf.requires_grad:
            continue
        identity = id(leaf)
        if identity in seen:
            continue
        seen.add(identity)
        handles.append(leaf.register_hook(lambda grad, scale=scale: grad * scale))
    try:
        loss.backward(retain_graph=retain_graph)
    finally:
        for handle in handles:
            handle.remove()


def train_cached_decoder_minibatches(model, optimizer, z_graph: torch.Tensor, pos_edge: torch.Tensor, neg_edge: torch.Tensor, *, auxiliary_loss: Optional[torch.Tensor] = None, update_batch_size: int = 1024, decode_micro_batch_size: Optional[int] = None, positive_weight: float = 0.5, negative_weight: float = 0.5, clear_decode_cache: Optional[Callable] = None, seed: Optional[int] = None, show_progress: bool = False, progress_desc: str = "cached decoder batches") -> float:
    if pos_edge.dim() != 2 or pos_edge.size(0) != 2:
        raise ValueError(f"Expected positive edges [2,E], got {tuple(pos_edge.shape)}")
    if neg_edge.dim() != 2 or neg_edge.size(0) != 2:
        raise ValueError(f"Expected negative edges [2,E], got {tuple(neg_edge.shape)}")
    if pos_edge.size(1) != neg_edge.size(1):
        raise ValueError(
            f"Cached decoder minibatch training expects equal positive and negative counts: {pos_edge.size(1)} != {neg_edge.size(1)}"
        )
    num_edges = int(pos_edge.size(1))
    if num_edges == 0:
        return 0.0
    update_batch_size = max(1, min(int(update_batch_size), num_edges))
    if decode_micro_batch_size is not None:
        decode_micro_batch_size = max(1, min(int(decode_micro_batch_size), update_batch_size))
    positive_weight = float(positive_weight)
    negative_weight = float(negative_weight)
    if seed is None:
        order = torch.randperm(num_edges, device=pos_edge.device)
    else:
        generator = torch.Generator(device=pos_edge.device)
        generator.manual_seed(int(seed))
        order = torch.randperm(num_edges, device=pos_edge.device, generator=generator)
    pos_edge = pos_edge[:, order].contiguous()
    neg_edge = neg_edge[:, order].contiguous()
    optimizer.zero_grad(set_to_none=True)
    _clear_decode_cache(model, clear_decode_cache)
    z = z_graph.detach().requires_grad_(True) if z_graph.requires_grad else z_graph
    total_loss = 0.0
    streamed_loss_terms = []
    num_minibatches = (num_edges + update_batch_size - 1) // update_batch_size
    if decode_micro_batch_size is not None:
        begin_deferred_neighbor_sum = getattr(model, "begin_deferred_neighbor_sum_backward", None)
        if callable(begin_deferred_neighbor_sum):
            begin_deferred_neighbor_sum()
        starts = range(0, num_edges, update_batch_size)
        if show_progress:
            from tqdm import tqdm

            starts = tqdm(starts, total=num_minibatches, desc=progress_desc, leave=False)
        for start in starts:
            end = min(start + update_batch_size, num_edges)
            count = end - start
            optimizer.zero_grad(set_to_none=True)
            phases = ((pos_edge, positive_weight, False), (neg_edge, negative_weight, True))
            for phase_edges, class_weight, is_negative in phases:
                for micro_start in range(start, end, decode_micro_batch_size):
                    micro_end = min(micro_start + decode_micro_batch_size, end)
                    logits = model.decode(z, phase_edges[:, micro_start:micro_end]).view(-1)
                    weighted_loss_sum = class_weight * _binary_prediction_loss(model, logits, positive=not is_negative, reduction="sum")
                    decoder_loss = weighted_loss_sum / float(count)
                    cached_neighbor_sum = getattr(model, "_neighbor_sum", None)
                    deferred_leaves = ()
                    get_deferred_leaves = getattr(model, "deferred_decode_gradient_leaves", None)
                    if callable(get_deferred_leaves):
                        deferred_leaves = get_deferred_leaves()
                    final_term = is_negative and micro_end == num_edges
                    retain_shared_graph = not final_term and cached_neighbor_sum is not None and (cached_neighbor_sum.grad_fn is not None)
                    _backward_with_weighted_embedding_grad(
                        decoder_loss,
                        z,
                        float(count) / float(num_edges),
                        retain_graph=retain_shared_graph,
                        additional_leaves=deferred_leaves,
                    )
                    streamed_loss_terms.append(weighted_loss_sum.detach())
                    del logits, weighted_loss_sum, decoder_loss
            trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
        if streamed_loss_terms:
            total_loss = float(torch.stack(streamed_loss_terms).sum().item())
        del streamed_loss_terms
    else:
        starts = range(0, num_edges, update_batch_size)
        if show_progress:
            from tqdm import tqdm

            starts = tqdm(starts, total=num_minibatches, desc=progress_desc, leave=False)
        for minibatch_id, start in enumerate(starts, 1):
            end = min(start + update_batch_size, num_edges)
            count = end - start
            optimizer.zero_grad(set_to_none=True)
            pos_logits = model.decode(z, pos_edge[:, start:end]).view(-1)
            neg_logits = model.decode(z, neg_edge[:, start:end]).view(-1)
            pos_loss = _binary_prediction_loss(model, pos_logits, positive=True)
            neg_loss = _binary_prediction_loss(model, neg_logits, positive=False)
            loss = positive_weight * pos_loss + negative_weight * neg_loss
            cached_neighbor_sum = getattr(model, "_neighbor_sum", None)
            retain_shared_graph = minibatch_id < num_minibatches and cached_neighbor_sum is not None and cached_neighbor_sum.grad_fn is not None
            _backward_with_weighted_embedding_grad(loss, z, float(count) / float(num_edges), retain_graph=retain_shared_graph)
            trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
            if trainable:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            optimizer.step()
            total_loss += float(loss.detach().item()) * count
    flush_deferred_neighbor_sum = getattr(model, "flush_deferred_neighbor_sum_backward", None)
    if callable(flush_deferred_neighbor_sum):
        flush_deferred_neighbor_sum()
    auxiliary_value = 0.0
    needs_graph_step = z_graph.requires_grad and z.grad is not None
    needs_auxiliary_step = auxiliary_loss is not None and auxiliary_loss.requires_grad
    if needs_graph_step or needs_auxiliary_step:
        optimizer.zero_grad(set_to_none=True)
        if needs_auxiliary_step:
            auxiliary_value = float(auxiliary_loss.detach().item())
            auxiliary_loss.backward(retain_graph=needs_graph_step)
        if needs_graph_step:
            z_graph.backward(z.grad)
        trainable = [parameter for parameter in model.parameters() if parameter.grad is not None]
        if trainable:
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
    _clear_decode_cache(model, clear_decode_cache)
    return total_loss / float(num_edges) + auxiliary_value


__all__ = ["train_cached_decoder_minibatches"]
