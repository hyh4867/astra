"""Lightweight image differencing utilities for ASTRA."""
import cv2
import numpy as np


def detect_change(before: np.ndarray, after: np.ndarray, threshold: int = 30, blur: int = 5):
    """Return binary mask and red overlay. Inputs are RGB uint8 arrays."""
    if before.shape != after.shape:
        raise ValueError("Images must have identical dimensions")
    if before.ndim != 3 or before.shape[2] != 3:
        raise ValueError("Images must be RGB")
    a, b = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    delta = cv2.absdiff(a, b)
    if blur > 1:
        blur = blur if blur % 2 else blur + 1
        delta = cv2.GaussianBlur(delta, (blur, blur), 0)
    mask = (delta >= threshold).astype(np.uint8) * 255
    overlay = after.copy()
    red = np.zeros_like(overlay); red[..., 0] = 255
    overlay[mask > 0] = (0.55 * overlay[mask > 0] + 0.45 * red[mask > 0]).astype(np.uint8)
    return mask, overlay


def score(mask: np.ndarray, truth: np.ndarray):
    pred, gt = mask > 0, truth > 0
    tp = int(np.count_nonzero(pred & gt)); fp = int(np.count_nonzero(pred & ~gt)); fn = int(np.count_nonzero(~pred & gt))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "tp": tp, "fp": fp, "fn": fn}
