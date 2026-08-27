import torch
import torch.nn.functional as F
from .concat import CONCAT_PREPROCESSING_RECIPE, concatenate_node_features

_AGGREGATED_MLP_METHODS = {
    "ppr": "ppr",
    "concat": "concat",
    "concatip": "concat",
}
PPR_PREPROCESSING_RECIPE = "concat-normalized-x-anorm-x-v2"


def normalized_model_name(model_name):
    return str(model_name).strip().lower().replace("-", "").replace("_", "")


def is_aggregated_mlp(model_name):
    return normalized_model_name(model_name) in _AGGREGATED_MLP_METHODS


def aggregated_mlp_method(model_name):
    normalized = normalized_model_name(model_name)
    return _AGGREGATED_MLP_METHODS.get(normalized)


def aggregated_mlp_recipe(model_name):
    method = aggregated_mlp_method(model_name)
    if method == "ppr":
        return PPR_PREPROCESSING_RECIPE
    if method == "concat":
        return CONCAT_PREPROCESSING_RECIPE
    raise ValueError(f"Unknown aggregated MLP method: {model_name}")


def _normalized_training_adjacency(training_graph, num_nodes, *, dtype, device):
    num_nodes = int(num_nodes)
    if hasattr(training_graph, "coo") and hasattr(training_graph, "set_diag"):
        if tuple(training_graph.sparse_sizes()) != (num_nodes, num_nodes):
            raise ValueError("Training adjacency shape does not match node features.")
        adjacency = training_graph.to(device)
        if not adjacency.is_coalesced():
            adjacency = adjacency.coalesce()
        adjacency = adjacency.set_value(torch.ones(adjacency.nnz(), dtype=dtype, device=device), layout="coo").set_diag(
            torch.ones(num_nodes, dtype=dtype, device=device)
        )
        degree = adjacency.sum(dim=1).to(dtype=dtype)
        inv_sqrt_degree = degree.rsqrt()
        (row, col, _) = adjacency.coo()
        return adjacency.set_value(inv_sqrt_degree[row] * inv_sqrt_degree[col], layout="coo")
    if not torch.is_tensor(training_graph):
        raise ValueError("ppr requires a training edge index or sparse adjacency.")
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
    values = torch.ones(edges.size(1), dtype=dtype, device=device)
    adjacency = torch.sparse_coo_tensor(edges, values, (num_nodes, num_nodes), device=device).coalesce()
    binary_edges = adjacency.indices()
    loops = torch.arange(num_nodes, dtype=torch.long, device=device)
    indices = torch.cat([binary_edges, torch.stack([loops, loops], dim=0)], dim=1)
    values = torch.ones(indices.size(1), dtype=dtype, device=device)
    adjacency = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes), device=device).coalesce()
    (row, col) = adjacency.indices()
    degree = torch.zeros(num_nodes, dtype=dtype, device=device)
    degree.scatter_add_(0, row, adjacency.values())
    inv_sqrt_degree = degree.rsqrt()
    normalized_values = adjacency.values() * inv_sqrt_degree[row] * inv_sqrt_degree[col]
    return torch.sparse_coo_tensor(adjacency.indices(), normalized_values, adjacency.size(), device=device).coalesce()


def aggregate_node_features(x, training_graph):
    if not torch.is_tensor(x) or x.ndim != 2:
        raise ValueError("ppr requires a rank-2 node-feature tensor.")
    if not x.is_floating_point():
        raise ValueError("ppr requires floating-point node features.")
    normalized_adjacency = _normalized_training_adjacency(training_graph, int(x.size(0)), dtype=x.dtype, device=x.device)
    with torch.no_grad():
        normalized_x = F.normalize(x, p=2, dim=1, out=x)
        if torch.is_tensor(normalized_adjacency):
            propagated_x = torch.sparse.mm(normalized_adjacency, normalized_x)
        else:
            propagated_x = normalized_adjacency.matmul(normalized_x)
        return torch.cat((normalized_x, propagated_x), dim=1).detach()


def fixed_identity_sketch(num_nodes, num_features, *, dtype, device, seed=0):
    num_nodes = int(num_nodes)
    num_features = int(num_features)
    if num_features <= 0:
        raise ValueError("Featureless input width must be positive.")
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    features = torch.randint(0, 2, (num_nodes, num_features), generator=generator, dtype=torch.int8, device="cpu")
    features = features.to(dtype=dtype).mul_(2).sub_(1)
    return features.to(device=device, non_blocking=True)


def preprocess_aggregated_mlp(model_name, dataset_name, x, training_graph, featureless_dim=None, featureless_seed=0):
    method = aggregated_mlp_method(model_name)
    if method is None:
        return x
    if training_graph is None:
        raise ValueError(f"{method} requires the training graph.")
    transform = concatenate_node_features if method == "concat" else aggregate_node_features
    if str(dataset_name).strip().lower() == "ogbl-ddi":
        if x is None:
            raise ValueError("ogbl-ddi preprocessing requires its node count.")
        if featureless_dim is None:
            raise ValueError(f"ogbl-ddi {method} preprocessing requires featureless_dim.")
        fixed_x = fixed_identity_sketch(
            int(x.size(0)),
            int(featureless_dim),
            dtype=x.dtype if x.is_floating_point() else torch.float32,
            device=x.device,
            seed=featureless_seed,
        )
        return transform(fixed_x, training_graph)
    if x is None:
        raise ValueError(
            f"{method} requires meaningful fixed node features; this dataset does not provide them and needs separate handling."
        )
    return transform(x, training_graph)
