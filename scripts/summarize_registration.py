"""Aggregate outputs of evaluate_registration.py into P/R/F1 tables.

Counts are summed over pairs first (same micro aggregation as
scripts/scientific_evaluation.py). Held pairs stay in every total: in the
"pipeline" condition they are scored with B left untouched.

    python scripts/summarize_registration.py outputs/registration/test_per_pair.csv
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "registration"
ROBUST_LIST = ROOT / "data" / "manifests" / "robustness_test_seed42.txt"
EXISTING = ROOT / "outputs" / "robustness_test_seed42.csv"


def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def summarize(rows, subset_name, names):
    rows = [r for r in rows if r["name"] in names]
    clean_est = {r["name"]: (float(r["est_dx"]), float(r["est_dy"])) for r in rows
                 if r["condition"] == "uncorrected" and (float(r["inj_dx"]), float(r["inj_dy"])) == (0.0, 0.0)}
    keys = sorted({(float(r["inj_dx"]), float(r["inj_dy"]), r["condition"]) for r in rows},
                  key=lambda k: (k[0] != k[1], abs(k[0]) + abs(k[1]), k[0], k[2]))
    out = []
    for sx, sy, condition in keys:
        group = [r for r in rows if (float(r["inj_dx"]), float(r["inj_dy"]), r["condition"]) == (sx, sy, condition)]
        tp = sum(int(r["tp"]) for r in group); fp = sum(int(r["fp"]) for r in group); fn = sum(int(r["fn"]) for r in group)
        p, r_, f = prf(tp, fp, fn)
        err = np.array([np.hypot(float(r["est_dx"]) - sx, float(r["est_dy"]) - sy) for r in group])
        rel = np.array([np.hypot(float(r["est_dx"]) - sx - clean_est[r["name"]][0], float(r["est_dy"]) - sy - clean_est[r["name"]][1])
                        for r in group if r["name"] in clean_est])
        held = sum(r["status"] == "review" for r in group)
        moved = sum(float(r["applied_dx"]) != 0 or float(r["applied_dy"]) != 0 for r in group)
        accepted_err = np.array([e for e, r in zip(err, group) if r["status"] != "review"])
        out.append({"subset": subset_name, "pairs": len(group), "inj_dx": sx, "inj_dy": sy, "condition": condition,
                    "precision": p, "recall": r_, "f1": f, "tp": tp, "fp": fp, "fn": fn,
                    "held_pairs": held, "held_fraction": held / len(group), "moved_pairs": moved,
                    "est_error_vs_injected_median_px": float(np.median(err)), "est_error_vs_injected_p90_px": float(np.percentile(err, 90)),
                    "est_error_vs_injected_max_px": float(err.max()),
                    "est_error_accepted_median_px": float(np.median(accepted_err)) if len(accepted_err) else "",
                    "est_error_vs_clean_estimate_median_px": float(np.median(rel)) if len(rel) else "",
                    "est_error_vs_clean_estimate_max_px": float(rel.max()) if len(rel) else "",
                    "scored_pixels": sum(int(r["scored_pixels"]) for r in group)})
    return out


def check_against_existing(summary):
    """Compare the uncorrected rows with the published robustness CSV."""
    if not EXISTING.is_file():
        return
    with EXISTING.open(newline="") as f:
        old = {(r["factor"], float(r["strength"])): r for r in csv.DictReader(f)
               if r["method"] == "TinyCD" and r["factor"] in ("control", "shift")}
    print("\nReproduction check vs outputs/robustness_test_seed42.csv (TinyCD, 32 pairs, 8 px crop):")
    for row in summary:
        if row["subset"] != "robust32" or row["condition"] != "uncorrected" or row["inj_dx"] != row["inj_dy"] or row["inj_dx"] < 0:
            continue
        key = ("control", 0.0) if row["inj_dx"] == 0 else ("shift", row["inj_dx"])
        if key not in old:
            continue
        o = old[key]
        print(f"  shift {row['inj_dx']:g}: F1 new {row['f1']:.6f} / published {float(o['f1']):.6f} · "
              f"FP new {row['fp']} / published {o['fp_pixels']} · max |ΔP,ΔR,ΔF1| = "
              f"{max(abs(row['precision'] - float(o['precision'])), abs(row['recall'] - float(o['recall'])), abs(row['f1'] - float(o['f1']))):.2e}")


def main():
    source = Path(sys.argv[1])
    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    robust = set(ROBUST_LIST.read_text().split())
    all_names = {r["name"] for r in rows}
    summary = []
    if rows[0]["split"] == "test":
        if robust <= all_names:
            summary += summarize(rows, "robust32", robust)
        test_names = set((ROOT / "data/manifests/test_filenames.txt").read_text().split())
        if all_names == test_names:
            summary += [row for row in summarize(rows, f"test{len(test_names)}", test_names)
                        if row["pairs"] == len(test_names)]
    else:
        summary += summarize(rows, f"{rows[0]['split']}{len(all_names)}", all_names)
    target = source.with_name(source.name.replace("_per_pair", "_summary"))
    with target.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(summary)
    print("saved", target)
    for row in summary:
        print(f"{row['subset']:9s} n={row['pairs']:3d} shift=({row['inj_dx']:+g},{row['inj_dy']:+g}) {row['condition']:17s} "
              f"P {row['precision']:.4f} R {row['recall']:.4f} F1 {row['f1']:.4f} held {row['held_pairs']:3d} moved {row['moved_pairs']:3d} "
              f"err med {row['est_error_vs_injected_median_px']:.2f} rel med {row['est_error_vs_clean_estimate_median_px'] if row['est_error_vs_clean_estimate_median_px']=='' else round(row['est_error_vs_clean_estimate_median_px'],3)}")
    if source.name == "test_per_pair.csv":
        check_against_existing(summary)


if __name__ == "__main__":
    main()
