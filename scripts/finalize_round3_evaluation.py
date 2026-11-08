"""Finalize Round 3 comparison CSV and chart from completed evaluation tables."""
from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/diagnosis_round"
MODULE_PATH = ROOT / "scripts/evaluate_round3_models.py"
spec = importlib.util.spec_from_file_location("round3_eval", MODULE_PATH)
assert spec and spec.loader
evaluator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluator)


def read(name: str) -> list[dict]:
    with (OUT / name).open(newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    full = read("round3_test_metrics.csv")
    objects = read("round3_object_completeness.csv")
    robust = read("round3_robustness.csv")
    swaps = read("round3_swap.csv")
    bootstrap = read("round3_bootstrap_delta_f1.csv")
    if len(full) != 9:
        raise RuntimeError(f"Expected 9 clean-test model rows, found {len(full)}")
    for row in full:
        if int(row["pixels"]) != 128 * 1024 * 1024:
            raise RuntimeError(f"Confusion-matrix pixel total failed: {row['method']}")
    for row in robust:
        row["strength"] = float(row["strength"])
        for key in ("f1", "precision", "recall", "fp"):
            row[key] = float(row[key])
    evaluator.save_csv(OUT / "round3_v2_comparison.csv",
                       evaluator.v2_comparison(full, objects, robust, swaps, bootstrap))
    evaluator.build_robustness_png(robust)
    print(f"Validated {len(full)} full-test rows and {len(robust)} robustness rows.")
    print("Wrote outputs/diagnosis_round/round3_v2_comparison.csv")
    print("Wrote outputs/diagnosis_round/robustness_f1_round3.png")


if __name__ == "__main__":
    main()
