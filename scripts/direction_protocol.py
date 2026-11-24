"""Shared definitions for the preregistered temporal direction study."""
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs/direction"
CACHE = ROOT / "data/raw/direction"
SEEDS = (42, 1042, 2042)
FAMILIES = ("A", "A+T", "S", "S+T")
CUTOFFS = (0.5, 0.4, 0.3)


def temporal_pair(before, after, truth, swap):
    return (after, before, truth) if swap else (before, after, truth)


def combine(forward, reverse, mode):
    if mode == "mean":
        return (forward + reverse) * 0.5
    if mode == "max":
        return np.maximum(forward, reverse)
    raise ValueError(f"Unknown temporal combination {mode!r}")


def weights(family, seed):
    if family == "A" and seed == 42:
        return ROOT / "data/raw/generalization/work/adapted.pth"
    return CACHE / f"{family}-{seed}" / "best.pth"
