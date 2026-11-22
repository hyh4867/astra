"""Descriptive paired patch-bootstrap intervals from the published test counts."""
import csv
import json
from pathlib import Path

import numpy as np

RESULTS = Path(__file__).resolve().parents[1] / "outputs/generalization"
METHODS = ("original", "calibrated", "adapted")


def summarize(seed=42, draws=10000):
    by_method = []
    for method in METHODS:
        with (RESULTS / "final" / method / "counts.csv").open() as stream:
            rows = list(csv.DictReader(stream))
        counts = {r["name"]: [int(r[k]) for k in ("tp", "fp", "fn")] for r in rows}
        if len(counts) != len(rows) or len(counts) != 979:
            raise ValueError(f"Unexpected or duplicate patch IDs for {method}")
        by_method.append(counts)
    names = sorted(by_method[0])
    if any(counts.keys() != by_method[0].keys() for counts in by_method):
        raise ValueError("The three methods must score the same patches")
    counts = np.array([[method[name] for method in by_method] for name in names])

    def micro_f1(sums):
        tp, fp, fn = np.moveaxis(sums, -1, 0)
        return 2 * tp / (2 * tp + fp + fn)

    rng = np.random.default_rng(seed)
    estimates = np.empty((draws, len(METHODS)))
    for start in range(0, draws, 100):
        chunk = min(100, draws - start)
        indices = rng.integers(len(names), size=(chunk, len(names)))
        estimates[start:start + chunk] = micro_f1(counts[indices].sum(axis=1))
    observed = micro_f1(counts.sum(axis=0))

    def interval(point, bootstrap):
        lo, hi = np.quantile(bootstrap, [.025, .975])
        return dict(estimate=point, lower=lo, upper=hi)

    return dict(seed=seed, draws=draws, pairs=len(names), unit="image pair",
                interval="95% percentile, paired patch resampling",
                f1={m: interval(observed[i], estimates[:, i]) for i, m in enumerate(METHODS)},
                changes={f"{a}_to_{b}": interval(observed[i+1] - observed[i],
                                                 estimates[:, i+1] - estimates[:, i])
                         for i, (a, b) in enumerate(zip(METHODS, METHODS[1:]))})


if __name__ == "__main__":
    print(json.dumps(summarize(), indent=2, allow_nan=False))
