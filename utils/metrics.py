import torch

HITS_K = (1, 3, 5, 10, 20, 50, 100)


def _score_dtype(value):
    return torch.float64 if value.dtype == torch.float64 else torch.float32


def _eval_mrr(y_pred_pos, y_pred_neg):
    y_pred_pos = y_pred_pos.view(-1, 1)
    optimistic_rank = (y_pred_neg >= y_pred_pos).sum(dim=1)
    pessimistic_rank = (y_pred_neg > y_pred_pos).sum(dim=1)
    rank = 0.5 * (optimistic_rank + pessimistic_rank) + 1
    return {**{f"hits@{k}_list": (rank <= k).to(torch.float32) for k in HITS_K}, "mrr_list": 1.0 / rank.to(torch.float32)}


def _binary_clf_curve(y_true, y_score):
    y_true = y_true.to(torch.float32).view(-1)
    y_score = y_score.to(_score_dtype(y_score)).view(-1)
    if y_true.numel() == 0:
        empty = torch.empty(0, dtype=torch.float32, device=y_score.device)
        return (empty, empty)
    order = torch.argsort(y_score, descending=True)
    y_true = y_true[order]
    y_score = y_score[order]
    if y_score.numel() == 1:
        threshold_idxs = torch.zeros(1, dtype=torch.long, device=y_score.device)
    else:
        distinct = torch.nonzero(y_score[1:] != y_score[:-1], as_tuple=False).view(-1)
        threshold_idxs = torch.cat([distinct, torch.tensor([y_true.numel() - 1], dtype=torch.long, device=y_score.device)])
    tps = torch.cumsum(y_true, dim=0)[threshold_idxs]
    fps = (threshold_idxs + 1).to(torch.float32) - tps
    return (tps, fps)


def evaluate_mrr(evaluator, pos_val_pred, neg_val_pred):
    pos = pos_val_pred.view(-1)
    comparison_dtype = torch.float64 if pos.dtype == torch.float64 or neg_val_pred.dtype == torch.float64 else torch.float32
    if neg_val_pred.dim() == 2:
        if neg_val_pred.size(0) != pos.numel():
            raise ValueError(f"Expected grouped negatives [Npos,K], got {tuple(neg_val_pred.shape)} for Npos={pos.numel()}.")
        mrr_output = _eval_mrr(pos.to(comparison_dtype), neg_val_pred.to(device=pos.device, dtype=comparison_dtype))
    else:
        pos_dev = pos.detach().to(comparison_dtype)
        neg_dev = neg_val_pred.detach().view(-1).to(device=pos_dev.device, dtype=comparison_dtype)
        neg_sorted = torch.sort(neg_dev).values
        N = neg_sorted.numel()
        lower = torch.searchsorted(neg_sorted, pos_dev, right=False)
        upper = torch.searchsorted(neg_sorted, pos_dev, right=True)
        optimistic = (N - lower).to(torch.float32)
        pessimistic = (N - upper).to(torch.float32)
        rank = 0.5 * (optimistic + pessimistic) + 1.0
        mrr_output = {"mrr_list": 1.0 / rank, **{f"hits@{k}_list": (rank <= k).to(torch.float32) for k in HITS_K}}
    result = {"MRR": round(mrr_output["mrr_list"].mean().item(), 4)}
    result.update({f"mrr_hit{k}": round(mrr_output[f"hits@{k}_list"].mean().item(), 4) for k in HITS_K})
    return result


def evaluate_auc(val_pred, val_true):
    source = val_pred.detach() if torch.is_tensor(val_pred) else torch.as_tensor(val_pred)
    y_score = source.to(dtype=_score_dtype(source)).view(-1)
    device = y_score.device
    truth = val_true.detach() if torch.is_tensor(val_true) else torch.as_tensor(val_true)
    y_true = truth.to(device=device, dtype=torch.float32).view(-1)
    pos_total = y_true.sum()
    neg_total = float(y_true.numel()) - pos_total
    if float(pos_total.item()) == 0.0 or float(neg_total.item()) == 0.0:
        return {"AUC": 0.0, "AP": 0.0}
    (tps, fps) = _binary_clf_curve(y_true, y_score)
    tpr = tps / pos_total
    fpr = fps / neg_total
    tpr = torch.cat([torch.zeros(1, device=device), tpr])
    fpr = torch.cat([torch.zeros(1, device=device), fpr])
    auc = torch.trapz(tpr, fpr)
    precision = tps / (tps + fps)
    recall = tps / pos_total
    recall_prev = torch.cat([torch.zeros(1, device=device), recall[:-1]])
    ap = torch.sum((recall - recall_prev) * precision)
    return {"AUC": round(float(auc.item()), 4), "AP": round(float(ap.item()), 4)}


class StreamingAUCAP:

    def __init__(self, positive_scores, buffer_size=1048576):
        positive_source = torch.as_tensor(positive_scores).detach()
        self.score_dtype = _score_dtype(positive_source)
        positive = positive_source.to(dtype=self.score_dtype).reshape(-1)
        self.device = positive.device
        self.positive_count = int(positive.numel())
        self.negative_count = 0
        self.buffer_size = max(1, int(buffer_size))
        self._pending_negative = []
        self._pending_weight = []
        self._pending_count = 0
        self._all_finite = torch.isfinite(positive).all()
        self.positive_sorted = torch.sort(positive).values.contiguous()
        if self.positive_count:
            (self.positive_thresholds, self.positive_counts) = torch.unique_consecutive(self.positive_sorted, return_counts=True)
        else:
            self.positive_thresholds = self.positive_sorted
            self.positive_counts = torch.empty(0, dtype=torch.long, device=self.device)
        self.negative_bins = torch.zeros(self.positive_thresholds.numel() + 1, dtype=torch.long, device=self.device)
        self.auc_numerator = torch.zeros((), dtype=torch.float64, device=self.device)

    def _process_negative(self, negative, weight=None):
        lower = torch.searchsorted(self.positive_sorted, negative, right=False)
        upper = torch.searchsorted(self.positive_sorted, negative, right=True)
        wins = (self.positive_count - upper).to(torch.float64) + 0.5 * (upper - lower).to(torch.float64)
        if weight is None:
            self.auc_numerator.add_(wins.sum())
        else:
            self.auc_numerator.add_((wins * weight.to(torch.float64)).sum())
        bins = torch.searchsorted(self.positive_thresholds, negative, right=True)
        if weight is None:
            self.negative_bins.add_(torch.bincount(bins, minlength=self.positive_thresholds.numel() + 1))
        else:
            self.negative_bins.index_add_(0, bins, weight)

    def _flush(self):
        if not self._pending_negative:
            return
        if len(self._pending_negative) == 1:
            negative = self._pending_negative[0]
            weight = self._pending_weight[0]
        else:
            negative = torch.cat(self._pending_negative)
            weights = self._pending_weight
            weight = (
                None
                if all((value is None for value in weights))
                else torch.cat(
                    [
                        torch.ones(tensor.numel(), dtype=torch.long, device=self.device) if value is None else value
                        for (tensor, value) in zip(self._pending_negative, weights)
                    ]
                )
            )
        self._pending_negative = []
        self._pending_weight = []
        self._pending_count = 0
        self._process_negative(negative, weight)

    def update(self, negative_scores):
        negative = torch.as_tensor(negative_scores, device=self.device).detach().to(dtype=self.score_dtype).reshape(-1).contiguous()
        count = int(negative.numel())
        if count == 0:
            return
        self.negative_count += count
        self._all_finite.logical_and_(torch.isfinite(negative).all())
        if self.positive_count == 0:
            return
        if not self._pending_negative and count >= self.buffer_size:
            self._process_negative(negative)
            return
        self._pending_negative.append(negative)
        self._pending_weight.append(None)
        self._pending_count += count
        if self._pending_count >= self.buffer_size:
            self._flush()

    def update_weighted(self, negative_scores, counts):
        negative = torch.as_tensor(negative_scores, device=self.device).detach().to(dtype=self.score_dtype).reshape(-1).contiguous()
        weight = torch.as_tensor(counts, device=self.device, dtype=torch.long).detach().reshape(-1).contiguous()
        if negative.numel() != weight.numel():
            raise ValueError("Weighted AUC/AP scores and counts must align.")
        if bool((weight < 0).any()):
            raise ValueError("Weighted AUC/AP counts must be non-negative.")
        keep = weight > 0
        if not bool(keep.any()):
            return
        negative = negative[keep]
        weight = weight[keep]
        occurrence_count = int(weight.sum().item())
        self.negative_count += occurrence_count
        self._all_finite.logical_and_(torch.isfinite(negative).all())
        if self.positive_count == 0:
            return
        if not self._pending_negative and int(negative.numel()) >= self.buffer_size:
            self._process_negative(negative, weight)
            return
        self._pending_negative.append(negative)
        self._pending_weight.append(weight)
        self._pending_count += int(negative.numel())
        if self._pending_count >= self.buffer_size:
            self._flush()

    def compute(self):
        self._flush()
        if not bool(self._all_finite.item()):
            raise ValueError("AUC/AP predictions must contain only finite scores.")
        if self.positive_count == 0 or self.negative_count == 0:
            return {"AUC": 0.0, "AP": 0.0}
        auc = self.auc_numerator / float(self.positive_count * self.negative_count)
        negative_ge_ascending = torch.flip(torch.cumsum(torch.flip(self.negative_bins[1:], dims=[0]), dim=0), dims=[0])
        positive_counts_desc = torch.flip(self.positive_counts, dims=[0]).to(torch.float64)
        negative_ge_desc = torch.flip(negative_ge_ascending, dims=[0]).to(torch.float64)
        cumulative_true_positives = torch.cumsum(positive_counts_desc, dim=0)
        precision = cumulative_true_positives / (cumulative_true_positives + negative_ge_desc)
        ap = torch.sum(positive_counts_desc / float(self.positive_count) * precision)
        return {"AUC": round(float(auc.item()), 4), "AP": round(float(ap.item()), 4)}
