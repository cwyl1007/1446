from collections import defaultdict
import time
import torch
from torch_geometric.utils import to_undirected, coalesce
from torch_sparse import SparseTensor


def _as_2xE(edge):
    if edge.dim() != 2:
        raise ValueError(f"edge must be 2D, got {edge.shape}")
    if edge.size(0) == 2:
        return edge
    if edge.size(1) == 2:
        return edge.t().contiguous()
    raise ValueError(f"edge must be [2,E] or [E,2], got {edge.shape}")


def build_graph_structures(train_edge_index, num_nodes, make_undirected=True, edge_weight=None):
    ei = _as_2xE(train_edge_index)
    values = None
    if edge_weight is not None:
        values = edge_weight.detach().to(torch.float32).reshape(-1)
        if int(values.numel()) != int(ei.size(1)):
            raise ValueError("edge_weight must contain one value per input edge")
    if make_undirected:
        if values is None:
            ei = to_undirected(ei, num_nodes=num_nodes)
        else:
            ei = torch.cat([ei, ei.flip(0)], dim=1)
            values = torch.cat([values, values], dim=0)
    (ei, values) = coalesce(ei, values, num_nodes=num_nodes, reduce="sum")
    adj = SparseTensor(row=ei[0], col=ei[1], value=values, sparse_sizes=(num_nodes, num_nodes)).coalesce()
    (rowptr, col, csr_value) = adj.csr()
    rowptr = rowptr.cpu()
    col = col.cpu()
    if csr_value is None:
        deg = (rowptr[1:] - rowptr[:-1]).to(torch.long)
    else:
        deg = adj.sum(dim=0).to_dense().to(torch.float32).cpu()
    return (rowptr, col, deg, adj)


def _gpu_enabled(device):
    return device is not None and str(device).startswith("cuda") and torch.cuda.is_available()


def _effective_source_batch_size(num_nodes, source_batch_size=None, max_dense_elems=4000000):
    if source_batch_size is None:
        source_batch_size = 256
    return max(1, min(int(source_batch_size), max(1, max_dense_elems // max(1, int(num_nodes)))))


def _group_edges_by_source(source_nodes):
    (unique_source, inverse) = torch.unique(source_nodes, sorted=True, return_inverse=True)
    counts = torch.bincount(inverse, minlength=int(unique_source.numel()))
    offsets = torch.zeros(int(unique_source.numel()) + 1, dtype=torch.long)
    offsets[1:] = torch.cumsum(counts, dim=0)
    already_grouped = inverse.numel() < 2 or bool((inverse[1:] >= inverse[:-1]).all())
    order = None if already_grouped else torch.argsort(inverse)
    return (unique_source, inverse, order, offsets)


def _source_edge_block(inverse, order, offsets, start, end):
    lower = int(offsets[start])
    upper = int(offsets[end])
    if order is None:
        edge_index = torch.arange(lower, upper, dtype=torch.long)
    else:
        edge_index = order[lower:upper]
    return (edge_index, inverse[edge_index] - int(start))


def _check_deadline(deadline, stage):
    if deadline is not None and time.time() >= float(deadline):
        raise TimeoutError(f"Heuristic runtime deadline exceeded during {stage}.")


def _notify_progress(progress_callback, completed, total):
    if progress_callback is not None:
        progress_callback(int(completed), int(total))


def _make_weighted_adj(adj, weights):
    if weights is None:
        return adj
    (_, col, value) = adj.coo()
    if value is None:
        value = torch.ones(col.numel(), dtype=torch.float32, device=col.device)
    return adj.set_value(value.to(torch.float32) * weights[col], layout="coo")


def _csr_edge_values(adj):
    if adj is None or not adj.has_value():
        return None
    (_, _, value) = adj.csr()
    if value is None:
        return None
    return value.detach().to(device="cpu", dtype=torch.float32)


def _common_neighbor_sum_cpu(rowptr, col, weights, edges_2xE, edge_values=None, deadline=None, progress_callback=None, progress_interval=4096):
    edges = _as_2xE(edges_2xE).cpu()
    (u_all, v_all) = (edges[0], edges[1])
    out = torch.zeros(u_all.numel(), dtype=torch.float32)
    for i in range(u_all.numel()):
        if i % max(1, int(progress_interval)) == 0:
            _check_deadline(deadline, "CPU neighbor-overlap scoring")
            _notify_progress(progress_callback, i, u_all.numel())
        u = int(u_all[i])
        v = int(v_all[i])
        a0 = int(rowptr[u])
        a1 = int(rowptr[u + 1])
        b0 = int(rowptr[v])
        b1 = int(rowptr[v + 1])
        (ia, ib) = (a0, b0)
        s = 0.0
        while ia < a1 and ib < b1:
            ca = int(col[ia])
            cb = int(col[ib])
            if ca == cb:
                contribution = 1.0
                if edge_values is not None:
                    contribution *= float(edge_values[ia])
                    contribution *= float(edge_values[ib])
                if weights is not None:
                    contribution *= float(weights[ca])
                s += contribution
                ia += 1
                ib += 1
            elif ca < cb:
                ia += 1
            else:
                ib += 1
        out[i] = s
    _notify_progress(progress_callback, u_all.numel(), u_all.numel())
    return out


def _neighbor_overlap_gpu(adj, edges, device, weights=None, edge_batch_size=65536, deadline=None, progress_callback=None):
    edges = _as_2xE(edges)
    u_all = edges[0].to(torch.long).cpu()
    v_all = edges[1].to(torch.long).cpu()
    num_edges = int(u_all.numel())
    if num_edges == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    adj_dev = adj.to(device)
    valued_adj = adj_dev if adj_dev.has_value() else adj_dev.fill_value(1.0)
    score_adj = _make_weighted_adj(valued_adj, None if weights is None else weights.to(device=device, dtype=torch.float32))
    batch_size = max(1, int(edge_batch_size))
    parts = []
    for start in range(0, num_edges, batch_size):
        _check_deadline(deadline, "CUDA neighbor-overlap scoring")
        end = min(start + batch_size, num_edges)
        src = u_all[start:end].to(device)
        dst = v_all[start:end].to(device)
        left_rows = valued_adj[src]
        right_rows = score_adj[dst]
        parts.append(left_rows.mul(right_rows).sum(dim=1).to(torch.float32))
        _notify_progress(progress_callback, end, num_edges)
    return torch.cat(parts, dim=0)


def _score_neighbor_overlap(rowptr, col, weights, edges, adj, device, edge_batch_size, deadline, progress_callback):
    if _gpu_enabled(device) and adj is not None:
        return _neighbor_overlap_gpu(
            adj, edges, device, weights=weights, edge_batch_size=edge_batch_size, deadline=deadline, progress_callback=progress_callback
        )
    return _common_neighbor_sum_cpu(
        rowptr, col, weights, edges, edge_values=_csr_edge_values(adj), deadline=deadline, progress_callback=progress_callback
    )


def score_cn(rowptr, col, edges, adj=None, device=None, source_batch_size=None, edge_batch_size=65536, deadline=None, progress_callback=None):
    return _score_neighbor_overlap(rowptr, col, None, edges, adj, device, edge_batch_size, deadline, progress_callback)


def score_aa(rowptr, col, deg, edges, adj=None, device=None, source_batch_size=None, edge_batch_size=65536, deadline=None, progress_callback=None):
    deg_f = deg.to(torch.float32)
    w = torch.zeros_like(deg_f)
    mask = deg_f > 1
    w[mask] = 1.0 / torch.log(deg_f[mask])
    return _score_neighbor_overlap(rowptr, col, w, edges, adj, device, edge_batch_size, deadline, progress_callback)


def score_ra(rowptr, col, deg, edges, adj=None, device=None, source_batch_size=None, edge_batch_size=65536, deadline=None, progress_callback=None):
    deg_f = deg.to(torch.float32)
    w = torch.zeros_like(deg_f)
    mask = deg_f > 0
    w[mask] = 1.0 / deg_f[mask]
    return _score_neighbor_overlap(rowptr, col, w, edges, adj, device, edge_batch_size, deadline, progress_callback)


def _shortest_path_cpu(rowptr, col, edges, cutoff=None, transform="inv", unreachable_distance=None, self_score=None, deadline=None, progress_callback=None):
    edges = _as_2xE(edges).cpu()
    (u_all, v_all) = (edges[0], edges[1])
    n = int(rowptr.numel() - 1)
    by_src = defaultdict(list)
    for i in range(u_all.numel()):
        by_src[int(u_all[i])].append((i, int(v_all[i])))
    out = torch.zeros(u_all.numel(), dtype=torch.float32)
    if transform == "inv" and unreachable_distance is not None:
        out.fill_(1.0 / float(unreachable_distance))
    elif transform != "inv":
        out.fill_(-1000000000.0)
    num_sources = len(by_src)
    for source_index, (src, items) in enumerate(by_src.items()):
        _check_deadline(deadline, "CPU shortest-path scoring")
        dist = [-1] * n
        frontier = [src]
        dist[src] = 0
        head = 0
        while head < len(frontier):
            x = frontier[head]
            head += 1
            dx = dist[x]
            if cutoff is not None and dx >= cutoff:
                continue
            start = int(rowptr[x])
            end = int(rowptr[x + 1])
            for y in col[start:end].tolist():
                if dist[y] == -1:
                    dist[y] = dx + 1
                    frontier.append(y)
        for idx, v in items:
            d = dist[v]
            if d > 0:
                out[idx] = 1.0 / float(d) if transform == "inv" else -float(d)
        _notify_progress(progress_callback, source_index + 1, num_sources)
    if self_score is not None:
        out[u_all == v_all] = float(self_score)
    return out


def _shortest_path_gpu(adj, edges, cutoff=None, transform="inv", unreachable_distance=None, self_score=None, device=None, source_batch_size=None, max_dense_elems=4000000, deadline=None, progress_callback=None):
    edges = _as_2xE(edges)
    u_all = edges[0].to(torch.long).cpu()
    v_all = edges[1].to(torch.long)
    num_edges = int(u_all.numel())
    if num_edges == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    adj_dev = adj.to(device).fill_value(1.0)
    n = int(adj_dev.sparse_sizes()[0])
    cutoff = n - 1 if cutoff is None else int(cutoff)
    batch_size = _effective_source_batch_size(n, source_batch_size, max_dense_elems=max_dense_elems)
    (unique_src, edge_to_src, source_order, source_offsets) = _group_edges_by_source(u_all)
    if transform == "inv":
        fill = 0.0 if unreachable_distance is None else 1.0 / float(unreachable_distance)
        out = torch.full((num_edges,), fill, dtype=torch.float32, device=device)
    else:
        out = torch.full((num_edges,), -1000000000.0, dtype=torch.float32, device=device)
    for start in range(0, unique_src.numel(), batch_size):
        _check_deadline(deadline, "CUDA shortest-path scoring")
        end = min(start + batch_size, unique_src.numel())
        src_batch = unique_src[start:end].to(device)
        bsz = int(src_batch.numel())
        (edge_idx, local_source) = _source_edge_block(edge_to_src, source_order, source_offsets, start, end)
        if edge_idx.numel() == 0:
            continue
        pending_edge_idx = edge_idx.to(device)
        pending_dst = v_all[edge_idx].to(device)
        pending_cols = local_source.to(device)
        nonself = pending_dst.ne(src_batch[pending_cols])
        pending_edge_idx = pending_edge_idx[nonself]
        pending_dst = pending_dst[nonself]
        pending_cols = pending_cols[nonself]
        frontier = torch.zeros((n, bsz), dtype=torch.float32, device=device)
        frontier[src_batch, torch.arange(bsz, device=device)] = 1.0
        visited = frontier > 0
        for dist in range(1, cutoff + 1):
            if pending_edge_idx.numel() == 0:
                break
            _check_deadline(deadline, "CUDA shortest-path propagation")
            frontier = adj_dev.matmul(frontier)
            frontier = frontier > 0
            frontier = frontier & ~visited
            if not frontier.any():
                break
            hits = frontier[pending_dst, pending_cols]
            if hits.any():
                if transform == "inv":
                    out[pending_edge_idx[hits]] = 1.0 / float(dist)
                else:
                    out[pending_edge_idx[hits]] = -float(dist)
            visited |= frontier
            unresolved = ~hits
            pending_edge_idx = pending_edge_idx[unresolved]
            pending_dst = pending_dst[unresolved]
            pending_cols = pending_cols[unresolved]
            if pending_edge_idx.numel() == 0:
                break
            live_cols = frontier.any(dim=0)
            target_cols = torch.bincount(pending_cols, minlength=bsz).gt(0)
            keep_cols = live_cols & target_cols
            if not keep_cols.any():
                break
            if not bool(keep_cols.all()):
                old_to_new = torch.full((bsz,), -1, dtype=torch.long, device=device)
                new_bsz = int(keep_cols.sum().item())
                old_to_new[keep_cols] = torch.arange(new_bsz, dtype=torch.long, device=device)
                keep_edges = keep_cols[pending_cols]
                pending_edge_idx = pending_edge_idx[keep_edges]
                pending_dst = pending_dst[keep_edges]
                pending_cols = old_to_new[pending_cols[keep_edges]]
                frontier = frontier[:, keep_cols]
                visited = visited[:, keep_cols]
                bsz = new_bsz
            frontier = frontier.to(torch.float32)
        _notify_progress(progress_callback, end, unique_src.numel())
    if self_score is not None:
        self_mask = u_all.eq(v_all)
        if bool(self_mask.any()):
            out[self_mask.to(device)] = float(self_score)
    return out


def score_shortest_path(rowptr, col, edges, cutoff=None, transform="inv", unreachable_distance=None, self_score=None, adj=None, device=None, source_batch_size=None, max_dense_elems=4000000, deadline=None, progress_callback=None):
    if _gpu_enabled(device) and adj is not None:
        return _shortest_path_gpu(
            adj,
            edges,
            cutoff=cutoff,
            transform=transform,
            unreachable_distance=unreachable_distance,
            self_score=self_score,
            device=device,
            source_batch_size=source_batch_size,
            max_dense_elems=max_dense_elems,
            deadline=deadline,
            progress_callback=progress_callback,
        )
    return _shortest_path_cpu(
        rowptr,
        col,
        edges,
        cutoff=cutoff,
        transform=transform,
        unreachable_distance=unreachable_distance,
        self_score=self_score,
        deadline=deadline,
        progress_callback=progress_callback,
    )


def _katz_length_two_sparse(adj, edges, beta, device, *, edge_batch_size=65536, is_symmetric=False, deadline=None, progress_callback=None):
    edges = _as_2xE(edges)
    u_all = edges[0].to(torch.long).cpu()
    v_all = edges[1].to(torch.long).cpu()
    num_edges = int(u_all.numel())
    if num_edges == 0:
        return torch.empty(0, dtype=torch.float32, device=device)
    binary_adj = adj.to(device).fill_value(1.0)
    incoming_adj = binary_adj if is_symmetric else binary_adj.t()
    num_nodes = int(binary_adj.sparse_sizes()[1])
    batch_size = max(1, int(edge_batch_size))
    parts = []
    for start in range(0, num_edges, batch_size):
        _check_deadline(deadline, "length-two Katz scoring")
        end = min(start + batch_size, num_edges)
        src = u_all[start:end].to(device)
        dst = v_all[start:end].to(device)
        size = end - start
        outgoing_rows = binary_adj[src]
        incoming_rows = incoming_adj[dst]
        length_two = outgoing_rows.mul(incoming_rows).sum(dim=1)
        destination_rows = SparseTensor(
            row=torch.arange(size, dtype=torch.long, device=device),
            col=dst,
            value=torch.ones(size, dtype=torch.float32, device=device),
            sparse_sizes=(size, num_nodes),
        )
        length_one = outgoing_rows.mul(destination_rows).sum(dim=1)
        parts.append(float(beta) * length_one.to(torch.float32) + float(beta) ** 2 * length_two.to(torch.float32))
        _notify_progress(progress_callback, end, num_edges)
    return torch.cat(parts, dim=0)


def score_katz(adj, edges, beta=0.01, max_length=5, device=None, source_batch_size=None, edge_batch_size=65536, max_dense_elems=4000000, is_symmetric=False, self_score=None, deadline=None, progress_callback=None):
    edges = _as_2xE(edges)
    u_all = edges[0].to(torch.long).cpu()
    v_all = edges[1].to(torch.long)
    num_edges = int(u_all.numel())
    if num_edges == 0:
        return torch.empty(0, dtype=torch.float32, device=device if _gpu_enabled(device) else "cpu")
    if device is None:
        device = torch.device("cpu")
    if int(max_length) == 2:
        result = _katz_length_two_sparse(
            adj,
            edges,
            beta,
            device,
            edge_batch_size=edge_batch_size,
            is_symmetric=is_symmetric,
            deadline=deadline,
            progress_callback=progress_callback,
        )
        if self_score is not None:
            self_mask = u_all.eq(v_all).to(result.device)
            if bool(self_mask.any()):
                result[self_mask] = float(self_score)
        return result
    adj_dev = adj.to(device).fill_value(1.0)
    n = int(adj_dev.sparse_sizes()[0])
    batch_size = _effective_source_batch_size(n, source_batch_size, max_dense_elems=max_dense_elems)
    (unique_src, edge_to_src, source_order, source_offsets) = _group_edges_by_source(u_all)
    out = torch.zeros(num_edges, dtype=torch.float32, device=device)
    for start in range(0, unique_src.numel(), batch_size):
        _check_deadline(deadline, "Katz scoring")
        end = min(start + batch_size, unique_src.numel())
        src_batch = unique_src[start:end].to(device)
        bsz = int(src_batch.numel())
        (edge_idx, local_source) = _source_edge_block(edge_to_src, source_order, source_offsets, start, end)
        if edge_idx.numel() == 0:
            continue
        edge_idx_dev = edge_idx.to(device)
        dst = v_all[edge_idx].to(device)
        local_cols = local_source.to(device)
        x = torch.zeros((n, bsz), dtype=torch.float32, device=device)
        x[src_batch, torch.arange(bsz, device=device)] = 1.0
        for l in range(1, max_length + 1):
            _check_deadline(deadline, "Katz propagation")
            x = adj_dev.matmul(x)
            out[edge_idx_dev] += beta**l * x[dst, local_cols]
        _notify_progress(progress_callback, end, unique_src.numel())
    if self_score is not None:
        self_mask = u_all.eq(v_all).to(out.device)
        if bool(self_mask.any()):
            out[self_mask] = float(self_score)
    return out


def score_edges(method, rowptr, col, deg, adj, edges, **kwargs):
    method = method.lower()
    common = {
        "device": kwargs.get("device"),
        "source_batch_size": kwargs.get("source_batch_size"),
        "deadline": kwargs.get("deadline"),
        "progress_callback": kwargs.get("progress_callback"),
    }
    edge_batch_size = kwargs.get("edge_batch_size", 65536)
    if method in ("cn", "common_neighbors"):
        return score_cn(rowptr, col, edges, adj=adj, edge_batch_size=edge_batch_size, **common)
    if method in ("aa", "adamic_adar", "adamicadar"):
        return score_aa(rowptr, col, deg, edges, adj=adj, edge_batch_size=edge_batch_size, **common)
    if method in ("ra", "resource_allocation", "resourceallocation"):
        return score_ra(rowptr, col, deg, edges, adj=adj, edge_batch_size=edge_batch_size, **common)
    if method in ("shortest_path", "sp"):
        return score_shortest_path(
            rowptr,
            col,
            edges,
            cutoff=kwargs.get("cutoff", None),
            transform=kwargs.get("transform", "inv"),
            unreachable_distance=kwargs.get("unreachable_distance", None),
            self_score=kwargs.get("self_score", None),
            adj=adj,
            max_dense_elems=kwargs.get("max_dense_elems", 4000000),
            **common,
        )
    if method == "katz":
        return score_katz(
            adj,
            edges,
            beta=float(kwargs.get("beta", 0.01)),
            max_length=int(kwargs.get("max_length", 5)),
            edge_batch_size=edge_batch_size,
            max_dense_elems=kwargs.get("max_dense_elems", 4000000),
            is_symmetric=bool(kwargs.get("is_symmetric", False)),
            self_score=kwargs.get("self_score", None),
            **common,
        )
    raise ValueError(f"Unknown heuristic method: {method}")
