import torch
import torch.nn as nn
import torch.nn.functional as F
from .pairwise_models import FastAdvancedPredictor, MLP


class LPFormerPredictor(FastAdvancedPredictor):
    def __init__(self, encoder, hidden_dim, pred_layers=3, dropout=0.0, decode_batch_size=8192, heads=4, max_attend=64, max_neighbors=64, max_twohop_neighbors=2, thresh_cn=0.0, thresh_1hop=0.0, thresh_far=0.0, att_drop=0.0, use_ncnc_aux=True):
        nn.Module.__init__(self)
        self.encoder = encoder
        self.mode = "lpformer"
        self.implementation_name = "lpformer-optimized-adaptation"
        self.decode_is_dedup_safe = True
        self.out_channels = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.decode_batch_size = int(decode_batch_size)
        self.evaluation_decode_batch_size = max(int(decode_batch_size), 65536)
        self.evaluation_decode_policy = "strict-no-fallback"
        self.pe_dim = 0
        self.node_emb = None
        FastAdvancedPredictor.clear_graph_cache(self)
        self._limited_neighbor_cache = {}
        self.max_attend = max(0, int(max_attend))
        self.max_neighbors = max(0, int(max_neighbors))
        self.max_twohop_neighbors = max(0, int(max_twohop_neighbors))
        self.thresh_cn = float(thresh_cn)
        self.thresh_1hop = float(thresh_1hop)
        self.thresh_far = float(thresh_far)
        self.att_drop = float(att_drop)
        self.use_ncnc_aux = bool(use_ncnc_aux)
        h = int(hidden_dim)
        heads = max(1, int(heads))
        heads = min(heads, h)
        while h % heads != 0 and heads > 1:
            heads -= 1
        self.heads = heads
        self.head_dim = h // heads
        self.att_dim = self.heads * self.head_dim
        self.endpoint_lin = nn.Linear(h, self.att_dim)
        self.node_lin = nn.Linear(2 * h, self.att_dim)
        self.att = nn.Parameter(torch.empty(self.heads, self.head_dim))
        nn.init.xavier_uniform_(self.att)
        self.rpe_cn = MLP(2, h, h, 2, dropout)
        self.rpe_onehop = MLP(2, h, h, 2, dropout)
        self.rpe_far = MLP(2, h, h, 2, dropout)
        self.att_norm = nn.LayerNorm(self.att_dim)
        self.pairwise_lin = MLP(self.att_dim, h, h, 2, dropout)
        final_in = 5 * h + 13 if self.use_ncnc_aux else 3 * h + 13
        self.pred = MLP(final_in, h, 1, pred_layers, dropout)

    def _limited_neighbor_csr(self, limit):
        limit = int(limit)
        key = (self._cache_key, limit)
        cached = self._limited_neighbor_cache.get(key)
        if cached is not None:
            return cached
        counts = self._rowptr[1:] - self._rowptr[:-1]
        take = torch.clamp(counts, max=limit)
        rows = torch.repeat_interleave(torch.arange(int(self._n), device=self._col.device), take)
        total = rows.numel()
        limited_rowptr = torch.empty_like(self._rowptr)
        limited_rowptr[0] = 0
        limited_rowptr[1:] = torch.cumsum(take, dim=0)
        if total == 0:
            limited_col = torch.empty(0, dtype=torch.long, device=self._col.device)
        else:
            take_rep = take[rows].clamp_min(1)
            count_rep = counts[rows].clamp_min(1)
            local = torch.arange(total, device=self._col.device) - limited_rowptr[rows]
            offsets = torch.div(local * count_rep, take_rep, rounding_mode="floor")
            starts = self._rowptr[:-1]
            limited_col = self._col[starts[rows] + offsets]
        cached = (limited_rowptr, limited_col)
        self._limited_neighbor_cache[key] = cached
        while len(self._limited_neighbor_cache) > 4:
            oldest_key = next(iter(self._limited_neighbor_cache))
            del self._limited_neighbor_cache[oldest_key]
        return cached

    def _row_keys_limited(self, nodes, limit):
        limit = int(limit)
        if limit <= 0 or nodes.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=nodes.device)
            return (empty, empty, empty)
        (rowptr, col) = self._limited_neighbor_csr(limit)
        starts = rowptr[nodes]
        counts = rowptr[nodes + 1] - rowptr[nodes]
        rows = torch.repeat_interleave(torch.arange(nodes.numel(), device=nodes.device), counts)
        total = rows.numel()
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=nodes.device)
            return (empty, empty, empty)
        start_rep = starts[rows]
        row_offsets = torch.cumsum(counts, dim=0) - counts
        local = torch.arange(total, device=nodes.device) - row_offsets[rows]
        cols = col[start_rep + local]
        keys = rows * int(self._n) + cols
        return (keys, rows, cols)

    def _cap_keys_per_edge(self, keys, limit, return_indices=False, return_counts=False):
        def pack(capped, selected, counts):
            result = [capped]
            if return_indices:
                result.append(selected)
            if return_counts:
                result.append(counts)
            return tuple(result) if len(result) > 1 else capped

        limit = int(limit)
        if limit <= 0 or keys.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=keys.device)
            counts = torch.zeros(int(self._cur_b), dtype=torch.long, device=keys.device)
            return pack(empty, empty, counts)
        batch = keys // int(self._n)
        b = int(self._cur_b)
        counts = torch.bincount(batch, minlength=b)
        take = torch.clamp(counts, max=limit)
        rows = torch.repeat_interleave(torch.arange(b, device=keys.device), take)
        total = rows.numel()
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=keys.device)
            return pack(empty, empty, take)
        starts = torch.cumsum(counts, dim=0) - counts
        take_rep = torch.repeat_interleave(take, take, output_size=total).clamp_min(1)
        count_rep = torch.repeat_interleave(counts, take, output_size=total).clamp_min(1)
        row_offsets = torch.cumsum(take, dim=0) - take
        local = torch.arange(total, device=keys.device) - torch.repeat_interleave(row_offsets, take, output_size=total)
        offsets = torch.div(local * count_rep, take_rep, rounding_mode="floor")
        selected = starts[rows] + offsets
        capped = keys[selected]
        return pack(capped, selected, take)

    def _expand_keys_one_step(self, keys, limit, unique=True):
        limit = int(limit)
        if limit <= 0 or keys.numel() == 0:
            return torch.empty(0, dtype=torch.long, device=keys.device)
        n = int(self._n)
        batch = keys // n
        nodes = keys % n
        (rowptr, col) = self._limited_neighbor_csr(limit)
        starts = rowptr[nodes]
        counts = rowptr[nodes + 1] - rowptr[nodes]
        parent = torch.repeat_interleave(torch.arange(keys.numel(), device=keys.device), counts)
        total = parent.numel()
        if total == 0:
            return torch.empty(0, dtype=torch.long, device=keys.device)
        start_rep = starts[parent]
        row_offsets = torch.cumsum(counts, dim=0) - counts
        local = torch.arange(total, device=keys.device) - row_offsets[parent]
        cols = col[start_rep + local]
        expanded = batch[parent] * n + cols
        return torch.unique(expanded, sorted=True) if unique else expanded

    def _sorted_union_with_origin(self, left, right):
        combined = torch.cat([left, right], dim=0)
        (union, inverse) = torch.unique(combined, sorted=True, return_inverse=True)
        in_left = torch.zeros(union.numel(), dtype=torch.bool, device=combined.device)
        in_right = torch.zeros_like(in_left)
        left_count = left.numel()
        if left_count:
            in_left.scatter_(0, inverse[:left_count], True)
        if right.numel():
            in_right.scatter_(0, inverse[left_count:], True)
        return (union, in_left, in_right)

    def _count_keys(self, keys, b, dtype):
        out = torch.zeros(b, dtype=dtype, device=keys.device if keys.numel() else self._deg.device)
        if keys.numel() == 0:
            return out
        batch = keys // int(self._n)
        out.index_add_(0, batch, torch.ones_like(batch, dtype=dtype))
        return out

    def _segment_softmax(self, score, index, num_segments):
        if score.numel() == 0:
            return score
        h = score.size(1)
        expanded_index = index.view(-1, 1).expand(-1, h)
        max_per = torch.full((num_segments, h), -torch.inf, dtype=score.dtype, device=score.device)
        if hasattr(max_per, "scatter_reduce_"):
            max_per.scatter_reduce_(0, expanded_index, score, reduce="amax", include_self=True)
        else:
            for i in torch.unique(index).tolist():
                mask = index == int(i)
                max_per[int(i)] = score[mask].max(dim=0).values
        score_exp = torch.exp(score - max_per[index].clamp_min(torch.finfo(score.dtype).min / 2))
        denom = torch.zeros((num_segments, h), dtype=score.dtype, device=score.device)
        denom.index_add_(0, index, score_exp)
        return score_exp / denom[index].clamp_min(1e-12)

    def _make_rpe(self, ppr_src, ppr_dst, node_type):
        if ppr_src.numel() == 0:
            return torch.empty((0, self.hidden_dim), dtype=ppr_src.dtype, device=ppr_src.device)
        inp = torch.stack([ppr_src, ppr_dst], dim=-1)
        inp_flip = torch.stack([ppr_dst, ppr_src], dim=-1)
        out = torch.empty((ppr_src.numel(), self.hidden_dim), dtype=ppr_src.dtype, device=ppr_src.device)
        for value, module in enumerate((self.rpe_cn, self.rpe_onehop, self.rpe_far)):
            mask = node_type == value
            if bool(mask.any()):
                out[mask] = self._symmetric_rpe(module, inp[mask], inp_flip[mask])
        return out

    def _symmetric_rpe(self, module, inp, inp_flip):
        if len(module.lins) != 2:
            return module(inp) + module(inp_flip)
        hidden = module.lins[0](inp)
        hidden = F.dropout(F.relu(hidden), p=module.dropout, training=module.training)
        hidden_flip = module.lins[0](inp_flip)
        hidden_flip = F.dropout(F.relu(hidden_flip), p=module.dropout, training=module.training)
        out = module.lins[1](hidden + hidden_flip)
        if module.lins[1].bias is not None:
            out = out + module.lins[1].bias
        return out

    def _lpformer_candidates(self, z, src, dst, endpoint_degrees=None):
        b = src.numel()
        self._cur_b = b
        if endpoint_degrees is None:
            deg_u = self._deg[src].to(z.dtype)
            deg_v = self._deg[dst].to(z.dtype)
        else:
            (deg_u, deg_v) = endpoint_degrees
        (common_keys, common_batch, common_col) = self._common_compact(src, dst)
        common_info = (common_keys, common_batch, common_col)
        cn_budget = max(1, self.max_attend // 2) if self.max_attend > 0 else 0
        remaining = max(0, self.max_attend - cn_budget)
        onehop_budget = 2 * remaining // 3
        far_budget = max(0, remaining - onehop_budget)
        (cn_att, cn_take) = self._cap_keys_per_edge(common_keys, cn_budget, return_counts=True)
        (src_keys_l, _, _) = self._row_keys_limited(src, self.max_neighbors)
        (dst_keys_l, _, _) = self._row_keys_limited(dst, self.max_neighbors)
        if src_keys_l.numel() or dst_keys_l.numel():
            (union_keys, union_in_src, union_in_dst) = self._sorted_union_with_origin(src_keys_l, dst_keys_l)
            union_batch = union_keys // int(self._n)
            union_nodes = union_keys % int(self._n)
            remove = (union_nodes == src[union_batch]) | (union_nodes == dst[union_batch])
            if common_keys.numel() > 0:
                remove = remove | self._isin_sorted(union_keys, common_keys)
            onehop = union_keys[~remove]
            onehop_in_src = union_in_src[~remove]
            onehop_in_dst = union_in_dst[~remove]
            (onehop_att, selected, onehop_take) = self._cap_keys_per_edge(onehop, onehop_budget, return_indices=True, return_counts=True)
            onehop_in_src = onehop_in_src[selected]
            onehop_in_dst = onehop_in_dst[selected]
        else:
            onehop_att = torch.empty(0, dtype=torch.long, device=z.device)
            src_keys_l = dst_keys_l = onehop_att
            onehop_in_src = onehop_in_dst = torch.empty(0, dtype=torch.bool, device=z.device)
            onehop_take = torch.zeros(b, dtype=torch.long, device=z.device)
        src_twohop = self._expand_keys_one_step(src_keys_l, self.max_twohop_neighbors, unique=False)
        dst_twohop = self._expand_keys_one_step(dst_keys_l, self.max_twohop_neighbors, unique=False)
        if far_budget > 0 and (src_twohop.numel() or dst_twohop.numel()):
            (twohop_union, twohop_in_src, twohop_in_dst) = self._sorted_union_with_origin(src_twohop, dst_twohop)
            twohop_batch = twohop_union // int(self._n)
            twohop_nodes = twohop_union % int(self._n)
            remove = (twohop_nodes == src[twohop_batch]) | (twohop_nodes == dst[twohop_batch])
            remove = remove | self._keys_in_endpoint_rows(twohop_union, src)
            remove = remove | self._keys_in_endpoint_rows(twohop_union, dst)
            far = twohop_union[~remove]
            far_in_src = twohop_in_src[~remove]
            far_in_dst = twohop_in_dst[~remove]
            (far_att, selected, far_take) = self._cap_keys_per_edge(far, far_budget, return_indices=True, return_counts=True)
            far_in_src = far_in_src[selected]
            far_in_dst = far_in_dst[selected]
        else:
            far_att = torch.empty(0, dtype=torch.long, device=z.device)
            far_in_src = far_in_dst = torch.empty(0, dtype=torch.bool, device=z.device)
            far_take = torch.zeros(b, dtype=torch.long, device=z.device)
        keys_parts = []
        type_parts = []
        psrc_parts = []
        pdst_parts = []
        if cn_att.numel() > 0:
            batch = cn_att // int(self._n)
            deg_src = deg_u[batch].clamp_min(1.0)
            deg_dst = deg_v[batch].clamp_min(1.0)
            p_src = 1.0 / deg_src
            p_dst = 1.0 / deg_dst
            if self.thresh_cn > 0.0:
                keep = (p_src >= self.thresh_cn) & (p_dst >= self.thresh_cn)
                cn_att = cn_att[keep]
                p_src = p_src[keep]
                p_dst = p_dst[keep]
            if cn_att.numel() > 0:
                keys_parts.append(cn_att)
                type_parts.append(torch.zeros(cn_att.numel(), dtype=torch.long, device=z.device))
                psrc_parts.append(p_src)
                pdst_parts.append(p_dst)
        if onehop_att.numel() > 0:
            batch = onehop_att // int(self._n)
            deg_src = deg_u[batch].clamp_min(1.0)
            deg_dst = deg_v[batch].clamp_min(1.0)
            p_src = onehop_in_src.to(z.dtype) / deg_src
            p_dst = onehop_in_dst.to(z.dtype) / deg_dst
            if self.thresh_1hop > 0.0:
                keep = p_src + p_dst >= self.thresh_1hop
                onehop_att = onehop_att[keep]
                p_src = p_src[keep]
                p_dst = p_dst[keep]
            if onehop_att.numel() > 0:
                keys_parts.append(onehop_att)
                type_parts.append(torch.ones(onehop_att.numel(), dtype=torch.long, device=z.device))
                psrc_parts.append(p_src)
                pdst_parts.append(p_dst)
        if far_att.numel() > 0:
            batch = far_att // int(self._n)
            deg_src = deg_u[batch].clamp_min(1.0)
            deg_dst = deg_v[batch].clamp_min(1.0)
            p_src = 0.5 * far_in_src.to(z.dtype) / deg_src
            p_dst = 0.5 * far_in_dst.to(z.dtype) / deg_dst
            if self.thresh_far > 0.0:
                keep = p_src + p_dst >= self.thresh_far
                far_att = far_att[keep]
                p_src = p_src[keep]
                p_dst = p_dst[keep]
            if far_att.numel() > 0:
                keys_parts.append(far_att)
                type_parts.append(torch.full((far_att.numel(),), 2, dtype=torch.long, device=z.device))
                psrc_parts.append(p_src)
                pdst_parts.append(p_dst)
        cn_count = self._count_keys(cn_att, b, z.dtype) if self.thresh_cn > 0.0 else cn_take.to(z.dtype)
        onehop_count = self._count_keys(onehop_att, b, z.dtype) if self.thresh_1hop > 0.0 else onehop_take.to(z.dtype)
        far_count = self._count_keys(far_att, b, z.dtype) if self.thresh_far > 0.0 else far_take.to(z.dtype)
        if not keys_parts:
            empty = torch.empty(0, dtype=torch.long, device=z.device)
            counts = torch.log1p(torch.stack([cn_count, onehop_count, far_count], dim=-1))
            return (empty, empty, empty, empty, empty, counts, common_info)
        keys = torch.cat(keys_parts, dim=0)
        node_type = torch.cat(type_parts, dim=0)
        p_src = torch.cat(psrc_parts, dim=0)
        p_dst = torch.cat(pdst_parts, dim=0)
        if self.training and self.att_drop > 0 and (keys.numel() > 0):
            random_values = torch.rand(keys.numel(), device=keys.device)
            keep = random_values >= self.att_drop
            keep[random_values.argmax()] = True
            (keys, node_type, p_src, p_dst) = (keys[keep], node_type[keep], p_src[keep], p_dst[keep])
        counts = torch.log1p(torch.stack([cn_count, onehop_count, far_count], dim=-1))
        return (keys, keys // int(self._n), keys % int(self._n), node_type, (p_src, p_dst), counts, common_info)

    def _lpformer_pairwise(self, z, src, dst, return_common=False, endpoint_z=None, endpoint_degrees=None):
        b = src.numel()
        (keys, batch, nodes, node_type, pprs, counts, common_info) = self._lpformer_candidates(
            z, src, dst, endpoint_degrees=endpoint_degrees
        )
        if keys.numel() == 0:
            pairwise = z.new_zeros((b, self.hidden_dim))
            return (pairwise, counts, common_info) if return_common else (pairwise, counts)
        if endpoint_z is None:
            (zu, zv) = (z[src], z[dst])
        else:
            (zu, zv) = endpoint_z
        (p_src, p_dst) = pprs
        rpe = self._make_rpe(p_src, p_dst, node_type)
        node_input = torch.cat([z[nodes], rpe], dim=-1)
        node_state = self.node_lin(node_input).view(-1, self.heads, self.head_dim)
        query = self.endpoint_lin(zu + zv)
        if self.endpoint_lin.bias is not None:
            query = query + self.endpoint_lin.bias
        query = query.view(b, self.heads, self.head_dim)
        att_state = F.leaky_relu(node_state * query[batch], negative_slope=0.2)
        score = (att_state * self.att.view(1, self.heads, self.head_dim)).sum(dim=-1)
        alpha = self._segment_softmax(score, batch, b)
        msg = node_state * alpha.unsqueeze(-1)
        out = z.new_zeros((b, self.heads, self.head_dim))
        out.index_add_(0, batch, msg)
        out = out.reshape(b, self.att_dim)
        out = self.att_norm(out)
        out = F.dropout(out, p=self.att_drop, training=self.training)
        pairwise = self.pairwise_lin(out)
        return (pairwise, counts, common_info) if return_common else (pairwise, counts)

    def _decode_block(self, z, edge_index):
        (src, dst) = edge_index
        (zu, zv) = (z[src], z[dst])
        deg_u = self._deg[src].to(z.dtype)
        deg_v = self._deg[dst].to(z.dtype)
        (pairwise, counts, common_info) = self._lpformer_pairwise(
            z, src, dst, return_common=True, endpoint_z=(zu, zv), endpoint_degrees=(deg_u, deg_v)
        )
        (stats, zc, zcomp, comp_stats) = self._stats(
            z,
            src,
            dst,
            common_emb=True,
            completion=True,
            precomputed_common=common_info,
            endpoint_z=(zu, zv),
            endpoint_degrees=(deg_u, deg_v),
        )
        scalar = torch.cat([stats, comp_stats, counts], dim=-1)
        if self.use_ncnc_aux:
            feat = torch.cat([zu * zv, torch.abs(zu - zv), pairwise, zc, zcomp, scalar], dim=-1)
        else:
            feat = torch.cat([zu * zv, torch.abs(zu - zv), pairwise, scalar], dim=-1)
        return self.pred(feat).view(-1)

    def _evaluation_block_size(self, z, num_edges):
        del z
        target = int(self.evaluation_decode_batch_size)
        if target < 65536:
            raise RuntimeError(f"LPFormer evaluation decode contract is invalid: configured={target}; minimum=65536.")
        return max(1, min(int(num_edges), target))

    def _decode_evaluation_blocks(self, z, edge_label_index):
        block_size = self._evaluation_block_size(z, int(edge_label_index.size(1)))
        outputs = []
        for start in range(0, edge_label_index.size(1), block_size):
            outputs.append(self._decode_block(z, edge_label_index[:, start : start + block_size]))
        return torch.cat(outputs, dim=0)

    def _uses_wide_evaluation_blocks(self):
        dataset = str(getattr(self, "dataset_name", "")).strip().lower()
        return dataset in {
            "cora",
            "citeseer",
            "pubmed",
            "amazon-c",
            "amazon-p",
            "wiki-chameleon",
            "wiki-squirrel",
            "reddit",
            "github",
            "facebook",
            "ogbl-collab",
            "ogbl-ddi",
            "ogbl-ppa",
            "ogbl-citation2",
        }

    def decode(self, z, edge_label_index):
        if edge_label_index.size(0) != 2:
            edge_label_index = edge_label_index.t().contiguous()
        edge_label_index = edge_label_index.to(device=z.device, dtype=torch.long, non_blocking=True)
        if edge_label_index.size(1) == 0:
            return z.new_empty((0,))
        if self.training:
            return super().decode(z, edge_label_index)
        if torch.is_grad_enabled():
            raise RuntimeError(
                "LPFormer evaluation requires torch.no_grad() or torch.inference_mode(); narrow grad-enabled fallback is disabled."
            )
        if not self._uses_wide_evaluation_blocks():
            raise RuntimeError(
                f"LPFormer strict evaluation batching is not validated for dataset={getattr(self, 'dataset_name', None)!r}; narrow fallback is disabled."
            )
        return self._decode_evaluation_blocks(z, edge_label_index)
