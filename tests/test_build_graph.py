import numpy as np
import pytest

from apholo8_egnn.data.graph import build_graph

N, D, K = 10, 4, 3


@pytest.fixture
def inputs():
    coords = np.random.randn(N, 3).astype(np.float32)
    esm = np.random.randn(N, D).astype(np.float32)
    names = np.array(["ALA"] * N)
    return coords, esm, names


def test_output_shapes(inputs):
    coords, esm, names = inputs
    data = build_graph(coords, esm, names, num_virtual_nodes=K)

    assert data["atom"].x.shape == (N, 21 + D)
    assert data["atom"].pos.shape == (N, 3)
    assert data["global_node"].x.shape == (K, 21 + D)
    assert data["global_node"].pos.shape == (K, 3)


def test_edge_types_present(inputs):
    coords, esm, names = inputs
    data = build_graph(coords, esm, names, num_virtual_nodes=K)

    assert ("atom", "to", "atom") in data.edge_index_dict
    assert ("atom", "to", "global_node") in data.edge_index_dict
    assert ("global_node", "to", "atom") in data.edge_index_dict


def test_empty_coords_raises():
    with pytest.raises(ValueError, match="empty"):
        build_graph(np.empty((0, 3)), np.empty((0, D)), np.array([]))


def test_esm_length_mismatch_raises(inputs):
    coords, _, names = inputs
    bad_esm = np.random.randn(N + 1, D).astype(np.float32)
    with pytest.raises(ValueError, match="does not match"):
        build_graph(coords, bad_esm, names)


def test_invalid_virtual_nodes_raises(inputs):
    coords, esm, names = inputs
    with pytest.raises(ValueError, match="num_virtual_nodes"):
        build_graph(coords, esm, names, num_virtual_nodes=0)


def test_invalid_dist_cutoff_raises(inputs):
    coords, esm, names = inputs
    with pytest.raises(ValueError, match="dist_cutoff"):
        build_graph(coords, esm, names, dist_cutoff=0.0)
