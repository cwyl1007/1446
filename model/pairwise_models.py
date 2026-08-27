import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import coalesce, to_undirected
from . import models as base_models

_ADVANCED_NAMES = {
    "seal",
    "buddy",
    "neognn",
    "ncn",
    "ncnc",
    "nbfnet",
    "peg",
    "lpformer",
    "lpf",
}
_PLANETOID_NAMES = {"cora", "citeseer", "pubmed"}
_OGBL_NAMES = {"ogbl-collab", "ogbl-ddi", "ogbl-ppa", "ogbl-citation2"}
_OGBL_REFERENCE_BASE_CELLS = {
    "ogbl-collab": {"mlp", "gcn", "gat", "sage"},
    "ogbl-ddi": {"gcn", "gat", "sage"},
    "ogbl-ppa": {"mlp", "gcn", "sage"},
    "ogbl-citation2": {"mlp", "gcn", "sage"},
}
_REFERENCE_PLANETOID_RESET_NAMES = {"mf", "mlp", "mlpip", "gcn", "gat", "sage", "gae"}
_INNER_PRODUCT_MLP_FAMILIES = {"mlpip": "mlp", "concatip": "concat"}


def _is_planetoid(params):
    return str(params.get("dataset_name", "")).strip().lower() in _PLANETOID_NAMES


def _is_ogbl(params):
    return str(params.get("dataset_name", "")).strip().lower() in _OGBL_NAMES


def _normalize_name(model_name):
    return str(model_name).strip().lower().replace("-", "").replace("_", "")


def _public_advanced_name(model_name):
    normalized = _normalize_name(model_name)
    return "neo-gnn" if normalized == "neognn" else normalized


def _gat_heads(params):
    default = 1 if _is_planetoid(params) or _is_ogbl(params) else 4
    return max(1, int(params.get("gat_heads", params.get("gat_head", default))))


def _gat_fused_sparse_propagation(params):
    value = params.get("gat_fused_sparse_propagation", False)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def reset_reference_planetoid_model(model, *, model_name, dataset_name, seed, seed_fn, device, emb_size, pred_layers, dropout):
    normalized = str(model_name).lower().replace("-", "").replace("_", "")
    dataset = str(dataset_name).strip().lower()
    if dataset not in _PLANETOID_NAMES or normalized not in _REFERENCE_PLANETOID_RESET_NAMES:
        return False
    unused_gae_scorer = None
    if normalized == "gae":
        unused_gae_scorer = base_models.MLPDecoder(int(emb_size), int(emb_size), 1, int(pred_layers), float(dropout)).to(device)
    seed_fn(int(seed))
    reset = getattr(model, "reset_parameters", None)
    if not callable(reset):
        raise RuntimeError(f"Reference Planetoid model {model_name!r} has no reset_parameters")
    reset()
    if unused_gae_scorer is not None:
        unused_gae_scorer.reset_parameters()
        del unused_gae_scorer
    return True


class MLP(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, layers, dropout):
        super().__init__()
        layers = max(1, int(layers))
        dims = [in_dim] + [hidden_dim] * (layers - 1) + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x):
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i != len(self.lins) - 1:
                x = F.dropout(F.relu(x), p=self.dropout, training=self.training)
        return x


class PEGEncoder(nn.Module):
    def __init__(self, in_channels, pe_dim, hidden_dim, layers, dropout):
        super().__init__()
        self.enc = base_models.GCNEncoder(in_channels + pe_dim, hidden_dim, hidden_dim, layers, dropout)

    def forward(self, x, adj, pe):
        return self.enc(torch.cat([x, pe], dim=-1), adj)


class FastAdvancedPredictor(nn.Module):
    def __init__(self, encoder, hidden_dim, mode, pred_layers=3, dropout=0.0, decode_batch_size=65536, pe_dim=16):
        super().__init__()
        self.encoder = encoder
        self.mode = _normalize_name(mode)
        self.implementation_name = _public_advanced_name(self.mode)
        self.execution_path = "scalable-pairwise"
        self.decode_is_dedup_safe = True
        self.symmetrize_decode = False
        self.out_channels = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.decode_batch_size = int(decode_batch_size)
        self.pe_dim = int(pe_dim)
        self.node_emb = None
        self.clear_graph_cache()
        h = int(hidden_dim)
        p = int(pe_dim)
        if self.mode == "seal":
            in_dim = 4 * h + 8
        elif self.mode in {"buddy", "ncn", "nbfnet"}:
            in_dim = 5 * h + 7
        elif self.mode == "ncnc":
            in_dim = 6 * h + 10
        elif self.mode == "neognn":
            in_dim = 4 * h + 7
        elif self.mode == "peg":
            in_dim = 4 * h + 3 * p + 1
        else:
            in_dim = 4 * h
        self.pred = MLP(in_dim, h, 1, pred_layers, dropout)

    def _graph_key(self, data):
        rowptr = getattr(data, "csr_rowptr", None)
        col = getattr(data, "csr_col", None)
        if rowptr is not None and col is not None:
            return (id(rowptr), id(col))
        adj_t = getattr(data, "adj_t", None)
        if adj_t is not None:
            return id(adj_t)
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            return id(edge_index)
        raise ValueError("data.edge_index, data.adj_t, or CSR adjacency is required")

    def _edge_index(self, data):
        edge_index = getattr(data, "edge_index", None)
        if edge_index is not None:
            return edge_index
        adj_t = getattr(data, "adj_t", None)
        if adj_t is None:
            raise ValueError("data.edge_index or data.adj_t is required")
        (row, col, _) = adj_t.coo()
        return torch.stack([row, col], dim=0)

    def _adj(self, data):
        return data.adj_t if hasattr(data, "adj_t") and data.adj_t is not None else data.edge_index

    def _num_nodes(self, data):
        if hasattr(data, "num_nodes") and data.num_nodes is not None:
            return int(data.num_nodes)
        return int(data.x.size(0))

    def _ensure_cache(self, data, device):
        n = self._num_nodes(data)
        key = (self._graph_key(data), n, device.type, device.index)
        if self._cache_key == key:
            return
        rowptr = getattr(data, "csr_rowptr", None)
        col = getattr(data, "csr_col", None)
        if rowptr is not None and col is not None:
            rowptr = rowptr.to(device=device, dtype=torch.long, non_blocking=True)
            col = col.to(device=device, dtype=torch.long, non_blocking=True)
        else:
            adj_t = getattr(data, "adj_t", None)
            if adj_t is not None and hasattr(adj_t, "csr"):
                (rowptr, col, _) = adj_t.csr()
                rowptr = rowptr.to(device=device, dtype=torch.long, non_blocking=True)
                col = col.to(device=device, dtype=torch.long, non_blocking=True)
            else:
                edge_index = self._edge_index(data).to(device=device, dtype=torch.long)
                if edge_index.size(0) != 2:
                    edge_index = edge_index.t().contiguous()
                edge_index = to_undirected(edge_index, num_nodes=n)
                (edge_index, _) = coalesce(edge_index, None, n, n)
                (row, col) = edge_index
                counts = torch.bincount(row, minlength=n).long()
                rowptr = torch.empty(n + 1, dtype=torch.long, device=device)
                rowptr[0] = 0
                rowptr[1:] = torch.cumsum(counts, dim=0)
        if rowptr.numel() != n + 1:
            raise RuntimeError(f"CSR rowptr has length {rowptr.numel()}, expected {n + 1}.")
        deg = (rowptr[1:] - rowptr[:-1]).to(torch.float32)
        self._n = n
        self._rowptr = rowptr
        self._col = col
        self._deg = deg
        self._ra = torch.where(deg > 0, 1.0 / deg.clamp_min(1.0), torch.zeros_like(deg))
        aa_denom = torch.log(deg.clamp_min(2.0))
        self._aa = torch.where(aa_denom > 0, 1.0 / aa_denom, torch.zeros_like(deg))
        adj_t = getattr(data, "adj_t", None)
        self._adj_t = adj_t if adj_t is not None and hasattr(adj_t, "matmul") else None
        self._sparse_csr = None
        self.clear_decode_cache()
        self._edge_ids_key = None
        self._edge_ids = None
        self._cache_key = key

    def clear_decode_cache(self):
        self._neighbor_sum_key = None
        self._neighbor_sum = None
        self._neighbor_sum_graph = None
        self._defer_neighbor_sum_backward = False

    def clear_graph_cache(self):
        self.clear_decode_cache()
        self._cache_key = None
        self._n = None
        self._rowptr = None
        self._col = None
        self._deg = None
        self._ra = None
        self._aa = None
        self._adj_t = None
        self._sparse_csr = None
        self._edge_ids_key = None
        self._edge_ids = None
        self._pe_key = None
        self._pe = None

    def begin_deferred_neighbor_sum_backward(self):
        self._defer_neighbor_sum_backward = True

    def deferred_decode_gradient_leaves(self):
        neighbor_sum = self._neighbor_sum
        if self._defer_neighbor_sum_backward and neighbor_sum is not None and neighbor_sum.is_leaf and neighbor_sum.requires_grad:
            return (neighbor_sum,)
        return ()

    def flush_deferred_neighbor_sum_backward(self):
        try:
            if self._neighbor_sum_graph is not None and self._neighbor_sum is not None and (self._neighbor_sum.grad is not None):
                self._neighbor_sum_graph.backward(self._neighbor_sum.grad)
        finally:
            self._defer_neighbor_sum_backward = False
            self._neighbor_sum_graph = None
            self._neighbor_sum_key = None
            self._neighbor_sum = None

    def _neighbor_sum_for(self, z):
        key = (id(z), int(z.data_ptr()), tuple(z.shape), z.dtype, z.device.type, z.device.index)
        if self._neighbor_sum_key == key and self._neighbor_sum is not None:
            return self._neighbor_sum
        if self._adj_t is not None:
            out = self._adj_t.matmul(z)
        else:
            rebuild = self._sparse_csr is None
            if not rebuild:
                rebuild = self._sparse_csr.device != z.device or self._sparse_csr.dtype != z.dtype
            if rebuild:
                values = torch.ones(self._col.numel(), dtype=z.dtype, device=z.device)
                self._sparse_csr = torch.sparse_csr_tensor(
                    self._rowptr, self._col, values, size=(int(self._n), int(self._n)), device=z.device
                )
            out = torch.sparse.mm(self._sparse_csr, z)
        self._neighbor_sum_key = key
        if self._defer_neighbor_sum_backward and out.requires_grad:
            self._neighbor_sum_graph = out
            self._neighbor_sum = out.detach().requires_grad_(True)
        else:
            self._neighbor_sum_graph = None
            self._neighbor_sum = out
        return self._neighbor_sum

    def _edge_ids_for_graph(self):
        key = (self._cache_key, self._col.device.type, self._col.device.index)
        if self._edge_ids_key == key and self._edge_ids is not None:
            return self._edge_ids
        counts = self._rowptr[1:] - self._rowptr[:-1]
        rows = torch.repeat_interleave(
            torch.arange(int(self._n), device=self._col.device, dtype=torch.long), counts, output_size=int(self._col.numel())
        )
        self._edge_ids = rows * int(self._n) + self._col
        self._edge_ids_key = key
        return self._edge_ids

    def _edge_exists(self, src, dst):
        edge_ids = self._edge_ids_for_graph()
        query = src.to(torch.long) * int(self._n) + dst.to(torch.long)
        if edge_ids.numel() == 0:
            return torch.zeros(query.numel(), dtype=torch.bool, device=query.device)
        pos = torch.searchsorted(edge_ids, query)
        inside = pos < edge_ids.numel()
        safe_pos = pos.clamp_max(edge_ids.numel() - 1)
        return inside & (edge_ids[safe_pos] == query)

    def _row_keys(self, nodes):
        b = nodes.numel()
        device = nodes.device
        starts = self._rowptr[nodes]
        counts = self._rowptr[nodes + 1] - self._rowptr[nodes]
        rows = torch.repeat_interleave(torch.arange(b, device=device), counts)
        total = rows.numel()
        if total == 0:
            empty = torch.empty(0, dtype=torch.long, device=device)
            return (empty, empty, empty)
        starts_rep = torch.repeat_interleave(starts, counts, output_size=total)
        row_offsets = torch.cumsum(counts, dim=0) - counts
        offsets = torch.arange(total, device=device) - torch.repeat_interleave(row_offsets, counts, output_size=total)
        cols = self._col[starts_rep + offsets]
        keys = rows * int(self._n) + cols
        return (keys, rows, cols)

    def _isin_sorted(self, vals, ref):
        out = torch.zeros(vals.numel(), dtype=torch.bool, device=vals.device)
        if vals.numel() == 0 or ref.numel() == 0:
            return out
        idx = torch.searchsorted(ref, vals)
        valid = idx < ref.numel()
        out[valid] = ref[idx[valid]] == vals[valid]
        return out

    def _common_against_repeated_reference(self, reference, other, output_rows):
        if reference.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=reference.device)
            return (empty, empty)
        (_, other_rows, other_cols) = self._row_keys(other)
        if other_cols.numel() == 0:
            empty = torch.empty(0, dtype=torch.long, device=reference.device)
            return (empty, empty)
        matched = self._edge_exists(reference[other_rows], other_cols)
        return (output_rows[other_rows[matched]], other_cols[matched])

    def _keys_in_endpoint_rows(self, keys, endpoints):
        if keys.numel() == 0:
            return torch.empty(0, dtype=torch.bool, device=keys.device)
        batch = keys // int(self._n)
        nodes = keys % int(self._n)
        return self._edge_exists(endpoints[batch], nodes)

    def _common_compact(self, src, dst):
        b = int(src.numel())
        if b == 0:
            empty = torch.empty(0, dtype=torch.long, device=src.device)
            return (empty, empty, empty)
        src_degree = self._rowptr[src + 1] - self._rowptr[src]
        dst_degree = self._rowptr[dst + 1] - self._rowptr[dst]
        use_src = src_degree >= dst_degree
        rows_all = torch.arange(b, device=src.device, dtype=torch.long)
        reference = torch.where(use_src, src, dst)
        other = torch.where(use_src, dst, src)
        (common_rows, common_cols) = self._common_against_repeated_reference(reference, other, rows_all)
        common_batch = common_rows
        common_col = common_cols
        common_keys = common_batch * int(self._n) + common_col
        return (common_keys, common_batch, common_col)

    def _pair_structures(self, z, src, dst, common_emb=False, completion=False, union_pool=False, precomputed_common=None, endpoint_z=None, endpoint_degrees=None):
        b = src.numel()
        self._cur_b = b
        if endpoint_degrees is None:
            deg_u = self._deg[src].to(z.dtype)
            deg_v = self._deg[dst].to(z.dtype)
        else:
            (deg_u, deg_v) = endpoint_degrees
        if endpoint_z is None:
            (zu, zv) = (z[src], z[dst])
        else:
            (zu, zv) = endpoint_z
        if precomputed_common is None:
            (common_keys, common_batch, common_col) = self._common_compact(src, dst)
        else:
            (common_keys, common_batch, common_col) = precomputed_common
        cn = torch.zeros(b, dtype=z.dtype, device=z.device)
        ra = torch.zeros_like(cn)
        aa = torch.zeros_like(cn)
        if common_keys.numel() > 0:
            one = torch.ones_like(common_batch, dtype=z.dtype)
            cn.index_add_(0, common_batch, one)
            ra.index_add_(0, common_batch, self._ra[common_col].to(z.dtype))
            aa.index_add_(0, common_batch, self._aa[common_col].to(z.dtype))
        stats = torch.stack(
            [
                torch.log1p(cn),
                ra,
                aa,
                cn / (deg_u + deg_v - cn).clamp_min(1.0),
                torch.log1p(deg_u * deg_v),
                torch.log1p(deg_u),
                torch.log1p(deg_v),
            ],
            dim=-1,
        )
        zc = torch.zeros((b, z.size(1)), dtype=z.dtype, device=z.device)
        zcomp = torch.zeros_like(zc)
        comp_stats = torch.zeros((b, 3), dtype=z.dtype, device=z.device)
        union_mean = None
        union_count = None
        need_common_sum = common_emb or completion or union_pool
        common_sum = torch.zeros_like(zc)
        if need_common_sum and common_keys.numel() > 0:
            common_sum.index_add_(0, common_batch, z[common_col])
        if common_emb:
            zc = common_sum / cn.clamp_min(1.0).unsqueeze(-1)
            zc = zc * (cn > 0).to(z.dtype).unsqueeze(-1)
        if completion or union_pool:
            neighbor_sum = self._neighbor_sum_for(z)
            connected = self._edge_exists(src, dst)
            connected_f = connected.to(z.dtype)
        if completion:
            comp_sum = neighbor_sum[src] + neighbor_sum[dst] - 2.0 * common_sum
            comp_sum = comp_sum - connected_f.unsqueeze(-1) * (zu + zv)
            count = (deg_u + deg_v - 2.0 * cn - 2.0 * connected_f).clamp_min(0.0)
            zcomp = comp_sum / count.clamp_min(1.0).unsqueeze(-1)
            zcomp = zcomp * (count > 0).to(z.dtype).unsqueeze(-1)
            comp_stats = torch.stack([torch.log1p(count), count / (deg_u + deg_v).clamp_min(1.0), torch.log1p(deg_u + deg_v - cn)], dim=-1)
        if union_pool:
            union_mean = torch.zeros_like(zc)
            union_count = torch.zeros(b, dtype=z.dtype, device=z.device)
            union_sum = neighbor_sum[src] + neighbor_sum[dst] - common_sum
            missing_endpoints = (~connected).to(z.dtype)
            union_sum = union_sum + missing_endpoints.unsqueeze(-1) * (zu + zv)
            union_count = deg_u + deg_v - cn + 2.0 * missing_endpoints
            union_mean = union_sum / union_count.clamp_min(1.0).unsqueeze(-1)
            union_mean = union_mean * (union_count > 0).to(z.dtype).unsqueeze(-1)
        return (stats, zc, zcomp, comp_stats, union_mean, union_count)

    def _stats(self, z, src, dst, common_emb=False, completion=False, precomputed_common=None, endpoint_z=None, endpoint_degrees=None):
        (stats, zc, zcomp, comp_stats, _, _) = self._pair_structures(
            z,
            src,
            dst,
            common_emb=common_emb,
            completion=completion,
            precomputed_common=precomputed_common,
            endpoint_z=endpoint_z,
            endpoint_degrees=endpoint_degrees,
        )
        return (stats, zc, zcomp, comp_stats)

    def _lap_pe(self, data, device):
        n = self._num_nodes(data)
        key = (self._graph_key(data), n, self.pe_dim, device.type, device.index)
        if self._pe_key == key and self._pe is not None:
            return self._pe
        self._ensure_cache(data, device)
        inv_sqrt = torch.where(self._deg > 0, self._deg.rsqrt(), torch.zeros_like(self._deg))
        adj_t = getattr(data, "adj_t", None)
        sparse_adj = None
        if adj_t is None or not hasattr(adj_t, "matmul"):
            values = torch.ones(self._col.numel(), dtype=torch.float32, device=device)
            sparse_adj = torch.sparse_csr_tensor(self._rowptr, self._col, values, size=(n, n), device=device)
        gen = torch.Generator(device=device)
        gen.manual_seed(0)
        pe = torch.randn((n, self.pe_dim), device=device, generator=gen)
        for _ in range(4):
            scaled = pe * inv_sqrt.unsqueeze(-1)
            if sparse_adj is None:
                pe = adj_t.matmul(scaled)
            else:
                pe = torch.sparse.mm(sparse_adj, scaled)
            pe = pe * inv_sqrt.unsqueeze(-1)
        pe = pe - pe.mean(dim=0, keepdim=True)
        pe = pe / (pe.std(dim=0, keepdim=True) + 1e-06)
        self._pe = pe
        self._pe_key = key
        return self._pe

    def embed(self, data):
        self.clear_decode_cache()
        self._ensure_cache(data, data.x.device)
        if self.mode == "peg":
            x = self.node_emb.weight if self.node_emb is not None else data.x
            pe = self._lap_pe(data, x.device)
            return self.encoder(x, self._adj(data), pe)
        return self.encoder(data.x, self._adj(data))

    def _base(self, z, src, dst):
        zu = z[src]
        zv = z[dst]
        return torch.cat([zu, zv, zu * zv, torch.abs(zu - zv)], dim=-1)

    @staticmethod
    def _reverse_stats(stats):
        return torch.cat([stats[:, :5], stats[:, 6:7], stats[:, 5:6]], dim=-1)

    def _decode_block(self, z, edge_index):
        (src, dst) = edge_index
        symmetric = bool(getattr(self, "symmetrize_decode", False))
        if symmetric:
            canonical_src = torch.minimum(src, dst)
            canonical_dst = torch.maximum(src, dst)
            (src, dst) = (canonical_src, canonical_dst)
        base = self._base(z, src, dst)
        reverse_base = self._base(z, dst, src) if symmetric else None
        if self.mode == "seal":
            (stats, _, _, _, pooled, size) = self._pair_structures(z, src, dst, union_pool=True)
            feat = torch.cat([z[src], z[dst], z[src] * z[dst], pooled, stats, torch.log1p(size).unsqueeze(-1)], dim=-1)
            if symmetric:
                reverse_feat = torch.cat(
                    [z[dst], z[src], z[src] * z[dst], pooled, self._reverse_stats(stats), torch.log1p(size).unsqueeze(-1)], dim=-1
                )
        elif self.mode in {"buddy", "ncn", "nbfnet"}:
            (stats, zc, _, _) = self._stats(z, src, dst, common_emb=True)
            feat = torch.cat([base, zc, stats], dim=-1)
            if symmetric:
                reverse_feat = torch.cat([reverse_base, zc, self._reverse_stats(stats)], dim=-1)
        elif self.mode == "ncnc":
            (stats, zc, zcomp, comp_stats) = self._stats(z, src, dst, common_emb=True, completion=True)
            feat = torch.cat([base, zc, zcomp, stats, comp_stats], dim=-1)
            if symmetric:
                reverse_feat = torch.cat([reverse_base, zc, zcomp, self._reverse_stats(stats), comp_stats], dim=-1)
        elif self.mode == "neognn":
            (stats, _, _, _) = self._stats(z, src, dst)
            feat = torch.cat([base, stats], dim=-1)
            if symmetric:
                reverse_feat = torch.cat([reverse_base, self._reverse_stats(stats)], dim=-1)
        elif self.mode == "peg":
            pe = self._pe.to(z.device)
            pu = pe[src]
            pv = pe[dst]
            d = pu - pv
            feat = torch.cat([base, pu * pv, torch.abs(d), d * d, (d * d).sum(dim=-1, keepdim=True)], dim=-1)
            if symmetric:
                reverse_feat = torch.cat([reverse_base, pu * pv, torch.abs(d), d * d, (d * d).sum(dim=-1, keepdim=True)], dim=-1)
        else:
            feat = base
        if not symmetric:
            return self.pred(feat).view(-1)
        count = int(feat.size(0))
        oriented = self.pred(torch.cat([feat, reverse_feat], dim=0)).view(-1)
        return 0.5 * (oriented[:count] + oriented[count:])

    def decode(self, z, edge_label_index):
        if edge_label_index.size(0) != 2:
            edge_label_index = edge_label_index.t().contiguous()
        edge_label_index = edge_label_index.to(device=z.device, dtype=torch.long, non_blocking=True)
        if edge_label_index.size(1) == 0:
            return z.new_empty((0,))
        block_size = max(1, int(self.decode_batch_size))
        while True:
            outs = []
            try:
                for start in range(0, edge_label_index.size(1), block_size):
                    outs.append(self._decode_block(z, edge_label_index[:, start : start + block_size]))
                return torch.cat(outs, dim=0)
            except RuntimeError as exc:
                is_oom = "out of memory" in str(exc).lower()
                if z.device.type != "cuda" or not is_oom or block_size <= 256:
                    raise
                del outs
                torch.cuda.empty_cache()
                block_size = max(256, block_size // 2)
                self.decode_batch_size = block_size

    def forward(self, data):
        z = self.embed(data)
        return self.decode(z, data.edge_label_index)


def _make_encoder(params, default_backbone="gcn"):
    backbone = str(params.get("backbone", default_backbone)).lower()
    emb_size = int(params["emb_size"])
    use_node_emb = bool(params.get("use_node_emb", False))
    in_channels = emb_size if use_node_emb else int(params["in_channels"])
    layers = int(params.get("layers", 2))
    dropout = float(params.get("dropout", 0.0))
    if backbone == "gcn":
        enc = base_models.GCNEncoder(in_channels, emb_size, emb_size, layers, dropout)
    elif backbone == "gat":
        enc = base_models.GATEncoder(
            in_channels,
            emb_size,
            emb_size,
            layers,
            dropout,
            head=_gat_heads(params),
            fused_sparse_propagation=_gat_fused_sparse_propagation(params),
        )
    elif backbone == "sage":
        enc = base_models.SAGEEncoder(in_channels, emb_size, emb_size, layers, dropout)
    elif backbone == "mlp":
        enc = base_models.MLPEncoder(in_channels, emb_size, emb_size, layers, dropout)
    else:
        raise ValueError(f"Unsupported backbone: {backbone}")
    if use_node_emb:
        enc = base_models.NodeEmbeddingEncoder(int(params["num_nodes"]), emb_size, enc)
    return enc


def is_advanced_model(model_name):
    return _normalize_name(model_name) in _ADVANCED_NAMES


def _configure_training(model, params):
    model.train_samples_per_epoch = max(0, int(params.get("train_samples_per_epoch", 0)))
    model.stage1_train_samples_per_epoch = max(0, int(params.get("stage1_train_samples_per_epoch", model.train_samples_per_epoch)))
    model.strict_train_negatives = bool(params.get("strict_train_negatives", getattr(model, "strict_train_negatives", True)))
    model.dataset_name = str(params.get("dataset_name", ""))
    return model


def _configure_scalable_advanced_protocol(model, params, mode):
    if _is_ogbl(params):
        return model
    normalized_mode = _normalize_name(mode)
    model.implementation_name = _public_advanced_name(normalized_mode)
    model.execution_path = "scalable-pairwise"
    model.symmetrize_decode = True
    model.decode_is_symmetric = True
    if not _is_planetoid(params):
        model.training_protocol = "pyg-extension-full-graph-scalable-v1"
    else:
        loose_modes = {"neognn", "nbfnet", "peg"}
        masked_modes = {"ncn", "ncnc"}
        strict_negatives = bool(params.get("strict_train_negatives", normalized_mode not in loose_modes))
        model.planetoid_target_edge_masked = normalized_mode in masked_modes
        model.training_protocol = (
            "planetoid-target-edge-masked-scalable-v1" if model.planetoid_target_edge_masked else "planetoid-full-graph-scalable-v1"
        )
        model.strict_train_negatives = strict_negatives
        model.train_negative_protocol = "strict-unobserved-nonself-edge" if strict_negatives else "uniform-random-endpoints"
    return model


def get_advanced_model(model_name, params):
    name = _normalize_name(model_name)
    hidden_dim = int(params["emb_size"])
    dropout = float(params.get("dropout", 0.0))
    pred_layers = int(params.get("pred_layers", 3))
    default_bs = {"ncnc": 32768, "lpformer": 8192, "lpf": 8192}.get(name, 65536)
    decode_batch_size = int(params.get("decode_batch_size", default_bs))
    if name == "peg":
        pe_dim = int(params.get("pe_dim", min(16, hidden_dim)))
        use_node_emb = bool(params.get("use_node_emb", False))
        in_channels = hidden_dim if use_node_emb else int(params["in_channels"])
        enc = PEGEncoder(in_channels, pe_dim, hidden_dim, int(params.get("layers", 2)), dropout)
        model = FastAdvancedPredictor(enc, hidden_dim, name, pred_layers, dropout, decode_batch_size, pe_dim)
        if use_node_emb:
            model.node_emb = nn.Embedding(int(params["num_nodes"]), in_channels)
            nn.init.xavier_uniform_(model.node_emb.weight)
        model = _configure_scalable_advanced_protocol(model, params, name)
        return _configure_training(model, params)
    if name in {"lpformer", "lpf"}:
        from .lpformer import LPFormerPredictor

        enc = _make_encoder(params, str(params.get("backbone", "gcn")))
        model = LPFormerPredictor(
            enc,
            hidden_dim,
            pred_layers=pred_layers,
            dropout=dropout,
            decode_batch_size=decode_batch_size,
            heads=int(params.get("lpformer_heads", params.get("num_heads", 4))),
            max_attend=int(params.get("lpformer_max_attend", 64)),
            max_neighbors=int(params.get("lpformer_max_neighbors", 64)),
            max_twohop_neighbors=int(params.get("lpformer_max_twohop_neighbors", 2)),
            thresh_cn=float(params.get("lpformer_thresh_cn", 0.0)),
            thresh_1hop=float(params.get("lpformer_thresh_1hop", 0.0)),
            thresh_far=float(params.get("lpformer_thresh_far", 0.0)),
            att_drop=float(params.get("lpformer_att_drop", min(dropout, 0.2))),
            use_ncnc_aux=bool(params.get("lpformer_use_ncnc_aux", True)),
        )
        return _configure_training(model, params)
    backbone = "mlp" if name == "buddy" else "gcn"
    model = FastAdvancedPredictor(_make_encoder(params, backbone), hidden_dim, name, pred_layers, dropout, decode_batch_size)
    model = _configure_scalable_advanced_protocol(model, params, name)
    return _configure_training(model, params)


def get_model(model_name, params):
    name = str(model_name).strip().lower()
    compact_name = _normalize_name(name)
    model_family = _INNER_PRODUCT_MLP_FAMILIES.get(compact_name, compact_name)
    inner_product_mlp = compact_name in _INNER_PRODUCT_MLP_FAMILIES
    emb_size = int(params["emb_size"])
    dropout = float(params["dropout"])
    pred_layers = 0 if inner_product_mlp else int(params["pred_layers"])
    layers = int(params["layers"])
    use_node_emb = bool(params.get("use_node_emb", False))
    in_channels = emb_size if use_node_emb else int(params["in_channels"])
    if is_advanced_model(name):
        return get_advanced_model(name, params)
    if name == "mf":
        if _is_planetoid(params):
            enc = base_models.MFEncoder(
                num_nodes=int(params["num_nodes"]),
                emb_dim=int(params["in_channels"]),
                hidden_channels=emb_size,
                out_channels=emb_size,
                num_layers=layers,
                dropout=dropout,
                reference_style=True,
            )
            model = base_models.LinkPredictor(enc, pred_layers, dropout, dot=False)
            model.reference_planetoid_mf = True
            model.reference_optimizer = True
            model.optimizer_protocol = "adam-scalar-foreach-false"
            model.reference_evaluation_transform = "sigmoid"
            model.reference_evaluation_row_batch_size = 1024
            model.reference_evaluation_negative_layout = "grouped"
            return _configure_training(model, params)
        if _is_ogbl(params):
            dataset_name = str(params["dataset_name"]).strip().lower()
            enc = base_models.MFEncoder(
                num_nodes=int(params["num_nodes"]),
                emb_dim=emb_size,
                num_layers=0,
                reference_style=True,
                embedding_init_std=0.2 if dataset_name == "ogbl-citation2" else None,
            )
            model = base_models.LinkPredictor(enc, pred_layers, dropout, dot=False)
            model.implementation_name = "heart-ogbl-reference-mf"
            model.training_protocol = "heart-ogbl-mf-random-endpoint-minibatch-v2"
            model.train_negative_protocol = (
                "fixed-positive-source+uniform-random-target-with-replacement"
                if dataset_name == "ogbl-citation2"
                else "uniform-random-endpoints-with-replacement"
            )
            model.protocol_fidelity = "released-reference-model"
            model.reference_ogbl_mf = True
            model.reference_optimizer = True
            model.optimizer_protocol = "adam-scalar-foreach-false"
            model.reference_mf_embedding_initialization = "normal-std-0.2" if dataset_name == "ogbl-citation2" else "torch-embedding-reset"
            model.reference_mf_gradient_clipping = False
            model.reference_probability_loss = True
            model.reference_evaluation_transform = "sigmoid"
            model.reference_random_endpoint_negatives = True
            model.strict_train_negatives = False
            return _configure_training(model, params)
        enc = base_models.MFEncoder(num_nodes=int(params["num_nodes"]), emb_dim=emb_size)
        return _configure_training(base_models.LinkPredictor(enc, pred_layers, dropout, dot=True), params)
    if name in {"gcn", "gat", "sage"} or model_family in {"mlp", "ppr", "concat"}:
        enc = _make_encoder(params, "mlp" if model_family in {"mlp", "ppr", "concat"} else name)
    elif name == "gae":
        enc = base_models.GCNEncoder(in_channels, emb_size, emb_size, layers, dropout)
        if use_node_emb:
            enc = base_models.NodeEmbeddingEncoder(int(params["num_nodes"]), emb_size, enc)
        model = base_models.LinkPredictor(enc, pred_layers, dropout, dot=True)
        if _is_planetoid(params):
            model.reference_dense_gae = True
            model.reference_optimizer = True
            model.reference_evaluation_transform = "sigmoid"
            model.reference_evaluation_row_batch_size = 1024
            model.reference_evaluation_negative_layout = "grouped"
        elif str(params.get("dataset_name", "")).strip().lower() == "ogbl-ddi":
            model.implementation_name = "canonical-dense-ddi-gae"
            model.training_protocol = "ddi-full-dense-adjacency-reconstruction-v1"
            model.train_negative_protocol = "all-zero-adjacency-cells"
            model.protocol_fidelity = "canonical-evaluator-repair-of-released-untrained-scorer"
            model.ogbl_dense_gae = True
            model.reference_optimizer = True
            model.optimizer_protocol = "adam-scalar-foreach-false"
            model.reference_evaluation_transform = "sigmoid"
            model.dense_gae_loss = "unweighted-full-adjacency-bce"
            model.dense_gae_evaluation = "symmetric-dot-product"
            model.gradient_clipping_protocol = "global-norm-1.0"
        return _configure_training(model, params)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    model = base_models.LinkPredictor(enc, pred_layers, dropout, dot=inner_product_mlp)
    if name == "gat" and _gat_fused_sparse_propagation(params):
        model.implementation_name = base_models.FUSED_GAT_IMPLEMENTATION
        model.execution_path = base_models.FUSED_GAT_EXECUTION_PATH
    if _is_ogbl(params) and model_family in {"mlp", "gcn", "gat", "sage"}:
        dataset_name = str(params["dataset_name"]).strip().lower()
        reference_cell = model_family in _OGBL_REFERENCE_BASE_CELLS[dataset_name]
        model.implementation_name = (
            f"heart-ogbl-reference-{model_family}"
            if reference_cell
            else f"optimized-ogbl-{model_family}-unreported-reference-cell"
        )
        model.training_protocol = "heart-ogbl-base-link-predictor-v1"
        model.train_negative_protocol = "uniform-random-endpoints-with-replacement"
        model.protocol_fidelity = (
            "released-reference-model-fast-large-graph-adapter" if reference_cell else "framework-extension-no-comparable-reference-cell"
        )
        model.reference_ogbl_base = reference_cell
        model.reference_probability_loss = True
        model.reference_evaluation_transform = "sigmoid"
        model.reference_random_endpoint_negatives = True
        model.strict_train_negatives = False
        if model_family == "gat":
            model.gat_heads = _gat_heads(params)
    if _is_planetoid(params) and model_family in {"mlp", "gcn", "gat", "sage"}:
        model.reference_optimizer = True
        model.reference_evaluation_transform = "sigmoid"
        model.reference_evaluation_row_batch_size = 1024
        model.reference_evaluation_negative_layout = "grouped"
    if inner_product_mlp:
        model.implementation_name = f"{compact_name}-mlp-encoder-inner-product-decoder"
        model.model_family = model_family
        model.decoder_type = "inner-product"
        model.predictor_depth = 0
        model.decoder_output = "raw-inner-product-logit"
        model.ranking_score = "raw-inner-product-logit"
        model.probability_transform = "sigmoid"
        model.reference_evaluation_transform = "sigmoid"
    return _configure_training(model, params)
