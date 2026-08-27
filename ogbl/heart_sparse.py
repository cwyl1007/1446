import torch


def _rank_positive_scores(indices, values):
    if indices.numel() == 0:
        return (torch.empty((0,), dtype=torch.long), torch.empty((0,), dtype=torch.long))
    order = torch.argsort(values, descending=False, stable=True)
    sorted_vals = values[order]
    ascending_ranks = torch.empty(sorted_vals.size(0), dtype=torch.long)
    start = 0
    while start < sorted_vals.size(0):
        end = start + 1
        v = sorted_vals[start]
        while end < sorted_vals.size(0) and bool(sorted_vals[end] == v):
            end += 1
        ascending_ranks[start:end] = start + 1
        start = end
    ranks_sorted = ascending_ranks.max() - ascending_ranks + 1
    inv = torch.empty_like(order)
    inv[order] = torch.arange(order.numel(), dtype=torch.long)
    return (indices, ranks_sorted[inv])


def _csr_neighbors(rowptr, col, node):
    (s, e) = (int(rowptr[node]), int(rowptr[node + 1]))
    return col[s:e]


def _ra_sparse_scores_for_node(u, rowptr, col, inv_deg):
    neigh = _csr_neighbors(rowptr, col, u)
    if neigh.numel() == 0:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float))
    parts = []
    weights = []
    for intermediate in neigh.tolist():
        nbrs = _csr_neighbors(rowptr, col, int(intermediate))
        if nbrs.numel() == 0:
            continue
        parts.append(nbrs)
        weights.append(torch.full((nbrs.numel(),), float(inv_deg[int(intermediate)]), dtype=torch.float))
    if not parts:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float))
    idx = torch.cat(parts).to(torch.long)
    scores = torch.zeros(int(rowptr.numel() - 1), dtype=torch.float)
    scores.scatter_add_(0, idx, torch.cat(weights))
    nz = torch.nonzero(scores > 0, as_tuple=False).view(-1)
    return (nz, scores[nz])


def _ppr_push_scores_for_node(u, rowptr, col, deg, alpha=0.15, eps=5e-05, max_pushes=None):
    u = int(u)
    alpha_eps = float(alpha) * float(eps)
    p = {u: 0.0}
    r = {u: float(alpha)}
    q = [u]
    in_q = {u}
    pushes = 0
    while q and (max_pushes is None or pushes < int(max_pushes)):
        x = q.pop()
        in_q.discard(x)
        res = float(r.get(x, 0.0))
        dx = int(deg[x]) if deg.numel() else 0
        if dx <= 0:
            p[x] = p.get(x, 0.0) + res
            r[x] = 0.0
            pushes += 1
            continue
        if res < alpha_eps * dx:
            continue
        p[x] = p.get(x, 0.0) + res
        r[x] = 0.0
        inc = (1.0 - float(alpha)) * res / dx
        (s, e) = (int(rowptr[x]), int(rowptr[x + 1]))
        for v in col[s:e].tolist():
            v = int(v)
            nr = float(r.get(v, 0.0)) + inc
            r[v] = nr
            dv = int(deg[v]) if deg.numel() else 0
            if dv > 0 and nr >= alpha_eps * dv and (v not in in_q):
                q.append(v)
                in_q.add(v)
        pushes += 1
    if not p:
        return (torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.float))
    idx = torch.tensor(list(p.keys()), dtype=torch.long)
    vals = torch.tensor([p[int(i)] for i in idx.tolist()], dtype=torch.float)
    mask = vals > 0
    return (idx[mask], vals[mask])
