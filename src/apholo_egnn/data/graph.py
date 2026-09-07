"""Build a heterogeneous protein graph for inference."""
import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.data import HeteroData
from torch_geometric.nn import radius_graph

from apholo_egnn.utils.constants import RES_IDS
from apholo_egnn.utils.grid import sample_fibonacci_grid


def _res_to_one_hot(res_names: np.ndarray) -> Tensor:
    ids = np.array([RES_IDS.get(r, RES_IDS["X"]) for r in res_names.tolist()])
    return F.one_hot(torch.from_numpy(ids), num_classes=len(RES_IDS)).float()


def build_graph(
    res_coords: np.ndarray,
    esm_embeddings: np.ndarray,
    res_names: np.ndarray,
    num_virtual_nodes: int = 8,
    max_neighbours: int = 10,
    dist_cutoff: float = 10.0,
) -> HeteroData:
    """
    Construct the heterogeneous protein graph consumed by VNEGNN.

    Args:
        res_coords:        [N, 3] alpha-carbon coordinates in Angstroms.
        esm_embeddings:    [N, D] per-residue ESM embeddings.
        res_names:         [N]    three-letter residue codes (unknown → "X").
        num_virtual_nodes: K virtual nodes placed on a Fibonacci sphere.
        max_neighbours:    maximum atom neighbours in the radius graph.
        dist_cutoff:       radius (Å) for atom-atom edges.

    Returns:
        HeteroData with node types "atom" and "global_node" and the three
        edge types the VNEGNN model expects.
    """
    N = len(res_coords)
    K = num_virtual_nodes

    coords_t = torch.from_numpy(np.ascontiguousarray(res_coords, dtype=np.float32))
    esm_t    = torch.from_numpy(np.ascontiguousarray(esm_embeddings, dtype=np.float32))

    # Atom features: [one-hot residue | ESM embedding]
    one_hot = _res_to_one_hot(np.asarray(res_names))  # [N, 21]
    x_atom  = torch.cat([one_hot, esm_t], dim=-1)     # [N, 21+D]

    # Global-node features: mean ESM, zero-padded to match atom dim
    esm_mean = esm_embeddings.astype(np.float32).mean(axis=0)
    x_gn = torch.from_numpy(np.stack([esm_mean] * K))  # [K, D]
    pad  = x_atom.shape[-1] - x_gn.shape[-1]
    if pad > 0:
        x_gn = F.pad(x_gn, (0, pad))  # [K, 21+D]

    # Atom-atom edges (radius graph)
    # radius_graph CUDA kernel requires a contiguous tensor
    edge_aa = radius_graph(coords_t, r=dist_cutoff, max_num_neighbors=max_neighbours)

    # Atom ↔ global-node edges (fully connected bipartite)
    src_a = torch.arange(N).repeat(K)
    dst_v = torch.arange(K).repeat_interleave(N)
    edge_ag = torch.stack([src_a, dst_v])
    edge_ga = torch.stack([dst_v, src_a])

    # Initial virtual-node positions (Fibonacci sphere)
    centroid = coords_t.mean(dim=0)
    radius   = (coords_t - centroid).norm(dim=1).max().unsqueeze(0)
    pos_gn   = sample_fibonacci_grid(centroid, radius, K, random_rotations=False)

    data = HeteroData()
    data["atom"].x                               = x_atom
    data["atom"].pos                             = coords_t
    data["global_node"].x                        = x_gn
    data["global_node"].pos                      = pos_gn
    data["atom",         "to", "atom"       ].edge_index = edge_aa
    data["atom",         "to", "global_node"].edge_index = edge_ag
    data["global_node",  "to", "atom"       ].edge_index = edge_ga
    data.centroid = centroid
    data.radius   = radius

    return data
