import numpy as np

from apholo8_egnn.utils.nms import nms_order


def test_returns_full_permutation():
    coords = np.array([[0, 0, 0], [10, 0, 0], [20, 0, 0]], dtype=float)
    conf = np.array([0.9, 0.8, 0.7])
    result = nms_order(coords, conf, radius=2.0)
    assert sorted(result) == [0, 1, 2]


def test_nearby_duplicate_pushed_to_end():
    # Site 0 and 1 are 1 Å apart (below radius); site 2 is far.
    # After picking 0, site 2 satisfies the distance constraint and is picked
    # second; site 1 falls back to the end.
    coords = np.array([[0, 0, 0], [1, 0, 0], [50, 0, 0]], dtype=float)
    conf = np.array([0.9, 0.8, 0.7])
    result = nms_order(coords, conf, radius=5.0)
    assert result == [0, 2, 1]


def test_nan_confidence_sorts_last():
    coords = np.array([[0, 0, 0], [100, 0, 0]], dtype=float)
    conf = np.array([float("nan"), 0.9])
    result = nms_order(coords, conf, radius=2.0)
    assert result[0] == 1
    assert result[-1] == 0
