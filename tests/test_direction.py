from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from direction_protocol import combine, temporal_pair


@pytest.mark.parametrize("swap", [False, True])
def test_temporal_swap_keeps_the_same_binary_target(swap):
    a, b = np.zeros((3, 5)), np.ones((3, 5))
    truth = np.eye(3, 5, dtype=bool)
    x, y, g = temporal_pair(a, b, truth, swap)
    assert x is (b if swap else a) and y is (a if swap else b)
    assert g is truth


@pytest.mark.parametrize("mode", ["mean", "max"])
def test_symmetric_probability_and_masks_are_order_invariant(mode):
    a = np.array([[0.12, 0.65, 0.8]], dtype=np.float32)
    b = np.array([[0.52, 0.15, 0.9]], dtype=np.float32)
    forward, reverse = combine(a, b, mode), combine(b, a, mode)
    np.testing.assert_array_equal(forward, reverse)
    expected = [[0.32, 0.40, 0.85]] if mode == "mean" else [[0.52, 0.65, 0.9]]
    np.testing.assert_allclose(forward, expected)
    for cutoff in (.3, .4, .5):
        np.testing.assert_array_equal(forward >= cutoff, reverse >= cutoff)
