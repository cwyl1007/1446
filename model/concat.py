import torch
import torch.nn.functional as F

CONCAT_PREPROCESSING_RECIPE = "l2-x-dmhalf-a-dphalf-x-no-self-v1"


def _literal_normalized_training_adjacency(training_graph, num_nodes, *, dtype, device):
    num_nodes = int(num_nodes)
    if hasattr(training_graph, "coo") and hasattr(training_graph, "remove_diag"):
        if tuple(training_graph.sparse_sizes()) != (num_nodes, num_nodes):
            raise ValueError("Training adjacency shape does not match node features.")
        adjacency = training_graph.to(device)
        if not adjacency.is_coalesced():
            adjacency = adjacency.coalesce()
        adjacency = adjacency.remove_diag()
        adjacency = adjacency.set_value(torch.ones(adjacency.nnz(), dtype=dtype, device=device), layout="coo")
        degree = adjacency.sum(dim=1).to(dtype=dtype)
        inv_sqrt_degree = torch.zeros_like(degree)
        nonisolated = degree > 0
        inv_sqrt_degree[nonisolated] = degree[nonisolated].rsqrt()
        sqrt_degree = degree.sqrt()
        (row, col, _) = adjacency.coo()
        return adjacency.set_value(inv_sqrt_degree[row] * sqrt_degree[col], layout="coo")
    if not torch.is_tensor(training_graph):
        raise ValueError("concat requires a training edge index or sparse adjacency.")
    edges = training_graph.to(device=device, dtype=torch.long, non_blocking=True)
    if edges.ndim != 2:
        raise ValueError("train_edge_index must be a rank-2 tensor.")
    if edges.size(0) != 2 and edges.size(1) == 2:
        edges = edges.t()
    if edges.size(0) != 2:
        raise ValueError("train_edge_index must have shape [2, E] or [E, 2].")
    if edges.numel():
        if int(edges.min()) < 0 or int(edges.max()) >= num_nodes:
            raise ValueError("Training edges contain an out-of-range node index.")
        edges = edges[:, edges[0] != edges[1]]
        edges = torch.cat([edges, edges.flip(0)], dim=1)
    adjacency = torch.sparse_coo_tensor(
        edges, torch.ones(edges.size(1), dtype=dtype, device=device), (num_nodes, num_nodes), device=device
    ).coalesce()
    binary_edges = adjacency.indices()
    adjacency = torch.sparse_coo_tensor(
        binary_edges, torch.ones(binary_edges.size(1), dtype=dtype, device=device), (num_nodes, num_nodes), device=device
    ).coalesce()
    (row, col) = adjacency.indices()
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, row, adjacency.values())
    inv_sqrt_degree = torch.zeros_like(degree)
    nonisolated = degree > 0
    inv_sqrt_degree[nonisolated] = degree[nonisolated].rsqrt()
    normalized_values = inv_sqrt_degree[row] * degree[col].sqrt()
    return torch.sparse_coo_tensor(adjacency.indices(), normalized_values, adjacency.size(), device=device).coalesce()


def concatenate_node_features(x, training_graph):
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError("concat requires a rank-2 node-feature tensor.")
    if not x.is_floating_point():
        raise ValueError("concat requires floating-point node features.")
    normalized_adjacency = _literal_normalized_training_adjacency(training_graph, int(x.size(0)), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        normalized_x = F.normalize(x, p=2, dim=1, out=x)
        if torch.is_tensor(normalized_adjacency):
            propagated_x = torch.sparse.mm(normalized_adjacency, normalized_x)
        else:
            propagated_x = normalized_adjacency.matmul(normalized_x)
        return torch.cat((normalized_x, propagated_x), dim=1).detach()
