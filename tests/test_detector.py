import numpy as np
from detector import detect_change, score


def test_identical_images_have_empty_mask():
    a = np.zeros((32, 32, 3), dtype=np.uint8)
    mask, _ = detect_change(a, a)
    assert not mask.any()


def test_detects_synthetic_square_and_scores():
    a = np.zeros((64, 64, 3), dtype=np.uint8)
    b = a.copy(); b[20:40, 20:40] = 255
    truth = np.zeros((64, 64), dtype=np.uint8); truth[20:40, 20:40] = 255
    mask, overlay = detect_change(a, b, threshold=20, blur=1)
    metrics = score(mask, truth)
    assert metrics["f1"] == 1.0
    assert overlay.shape == b.shape
