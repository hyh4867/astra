"""Choose the shift-estimator settings on the LEVIR-CD validation split.

No labels and no detector are used here. Each validation pair gets known
shifts injected into B; an estimate is "consistent" when
    estimate(shifted) - estimate(unshifted) - injected
is within 1 px. The grid search then looks for the acceptance thresholds that
never accept an inconsistent estimate while holding back as few pairs as
possible.

Stage 1 (measure): python scripts/select_registration_config.py measure
Stage 2 (grid):    python scripts/select_registration_config.py grid
Stage 3 (dead band, needs outputs of evaluate_registration.py on val):
                   python scripts/select_registration_config.py deadband
"""
from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from registration import DEFAULT_CONFIG, judge_measurement, measure_shift, shift_image, with_overrides  # noqa: E402

OUT = ROOT / "outputs" / "registration"
VAL = ROOT / "data" / "benchmark" / "val"
MEASUREMENTS = OUT / "val_shift_measurements.csv"
GRID = OUT / "val_estimator_grid.csv"
SHIFTS = [(0, 0), (1, 1), (2, 2), (3, 3), (5, 5), (-5, -5), (3, -2), (-2, 4), (-4, 0), (0, 5), (2.5, -1.5), (8, -6)]
CONSISTENT_PX = 1.0
MAX_WRONG_ACCEPT = 0          # selection rule: no accepted estimate may be inconsistent on val
DEADBAND_GRID = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 8.0)
DEADBAND_MAX_CLEAN_F1_LOSS = 0.002


def measure():
    OUT.mkdir(parents=True, exist_ok=True)
    names = sorted(p.name for p in (VAL / "A").glob("*.png"))
    with MEASUREMENTS.open("w", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["name", "inj_dx", "inj_dy", "preprocess", "global_dx", "global_dy", "global_peak_ratio", "tile_size", "tiles_json"])
        for index, name in enumerate(names, 1):
            a = np.asarray(Image.open(VAL / "A" / name).convert("RGB"))
            b = np.asarray(Image.open(VAL / "B" / name).convert("RGB"))
            for sx, sy in SHIFTS:
                shifted = b if (sx, sy) == (0, 0) else shift_image(b, sx, sy)
                for preprocess in ("gray", "gradient"):
                    m = measure_shift(a, shifted, with_overrides(preprocess=preprocess))
                    gdx, gdy, gratio = m["global"]
                    tiles = [[t[0], t[1], round(t[2], 4), round(t[3], 4), round(t[4], 4), round(t[5], 2)] for t in m["tiles"]]
                    writer.writerow([name, sx, sy, preprocess, f"{gdx:.4f}", f"{gdy:.4f}", f"{gratio:.4f}", m["tile_size"], json.dumps(tiles)])
            if index % 8 == 0 or index == len(names):
                print(f"measured {index}/{len(names)}", flush=True)
    print("saved", MEASUREMENTS.relative_to(ROOT))


def load_measurements():
    data = {}
    with MEASUREMENTS.open(newline="") as f:
        for row in csv.DictReader(f):
            key = (row["preprocess"], row["name"], float(row["inj_dx"]), float(row["inj_dy"]))
            data[key] = {"too_small": False, "tile_size": int(row["tile_size"]),
                         "global": (float(row["global_dx"]), float(row["global_dy"]), float(row["global_peak_ratio"])),
                         "tiles": [tuple(t) for t in json.loads(row["tiles_json"])]}
    return data


def grid():
    data = load_measurements()
    names = sorted({k[1] for k in data})
    shifts = sorted({(k[2], k[3]) for k in data})
    rows = []
    options = itertools.product(("gray", "gradient"), ("global", "tile_median"), (1.2, 1.5, 2.0, 3.0),
                                (0.125, 0.25, 0.375, 0.5), (0.5, 0.6, 0.75, 0.9), (1.0, 1.2, 1.5, 2.0))
    for preprocess, estimator, tile_ratio, informative, agreement, global_ratio in options:
        config = with_overrides(preprocess=preprocess, estimator=estimator, min_tile_peak_ratio=tile_ratio,
                                min_informative_fraction=informative, min_tile_agreement=agreement,
                                min_global_peak_ratio=global_ratio, max_auto_px=1e9)
        judged = {(n, s): judge_measurement(data[(preprocess, n, s[0], s[1])], config) for n in names for s in shifts}
        accepted = wrong = shifted_total = 0
        errors = []
        for n in names:
            clean = judged[(n, (0.0, 0.0))]
            for s in shifts:
                if s == (0.0, 0.0):
                    continue
                shifted_total += 1
                j = judged[(n, s)]
                if j["reasons"]:
                    continue
                accepted += 1
                error = float(np.hypot(j["dx"] - clean["dx"] - s[0], j["dy"] - clean["dy"] - s[1]))
                errors.append(error)
                wrong += error > CONSISTENT_PX
        clean_accept = sum(not judged[(n, (0.0, 0.0))]["reasons"] for n in names)
        # A pair is "stable" when it gets the same accept/hold decision for every shift.
        stable = sum(len({bool(judged[(n, s)]["reasons"]) for s in shifts}) == 1 for n in names)
        rows.append({"preprocess": preprocess, "estimator": estimator, "min_tile_peak_ratio": tile_ratio,
                     "min_informative_fraction": informative, "min_tile_agreement": agreement,
                     "min_global_peak_ratio": global_ratio, "pairs": len(names), "shifted_cases": shifted_total,
                     "accepted_shifted": accepted, "accept_rate_shifted": accepted / shifted_total,
                     "accepted_clean_pairs": clean_accept, "wrong_accepted": wrong,
                     "median_consistency_error_px": float(np.median(errors)) if errors else "",
                     "max_consistency_error_px": float(np.max(errors)) if errors else "",
                     "decision_stable_pairs": stable})
    rows.sort(key=lambda r: (r["wrong_accepted"] > MAX_WRONG_ACCEPT, -r["accepted_shifted"], -r["decision_stable_pairs"],
                             r["min_tile_peak_ratio"], r["min_global_peak_ratio"]))
    with GRID.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    print("saved", GRID.relative_to(ROOT), len(rows), "settings")
    print("rule: wrong_accepted <=", MAX_WRONG_ACCEPT, "then most accepted shifted cases, then most stable pairs")
    for r in rows[:8]:
        print(r)
    current = next((r for r in rows if all(r[k] == getattr(DEFAULT_CONFIG, k) for k in (
        "preprocess", "estimator", "min_tile_peak_ratio", "min_informative_fraction", "min_tile_agreement", "min_global_peak_ratio"))), None)
    print("DEFAULT_CONFIG row:", current)


def f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return p, r, (2 * p * r / (p + r) if p + r else 0.0)


def deadband():
    """Pick the dead band from validation detections: the smallest threshold
    whose automatic correction costs at most DEADBAND_MAX_CLEAN_F1_LOSS of F1
    on unshifted validation pairs. Reads outputs/registration/val_per_pair.csv."""
    source = OUT / "val_per_pair.csv"
    table = {}
    with source.open(newline="") as f:
        for row in csv.DictReader(f):
            table[(row["name"], float(row["inj_dx"]), float(row["inj_dy"]), row["condition"])] = row
    names = sorted({k[0] for k in table})
    shifts = sorted({(k[1], k[2]) for k in table})
    out = []
    for threshold in DEADBAND_GRID:
        for s in shifts:
            sums = {"uncorrected": np.zeros(3, int), "policy": np.zeros(3, int)}
            applied = held = complete = 0
            for n in names:
                base = table.get((n, s[0], s[1], "uncorrected")); always = table.get((n, s[0], s[1], "estimated_always"))
                if base is None or always is None:
                    continue
                complete += 1
                confident = base["gate_reasons"] == ""
                magnitude = float(np.hypot(float(base["est_dx"]), float(base["est_dy"])))
                use = always if (confident and magnitude >= threshold) else base
                applied += use is always and (float(always["applied_dx"]) != 0 or float(always["applied_dy"]) != 0)
                held += not confident
                sums["uncorrected"] += [int(base[k]) for k in ("tp", "fp", "fn")]
                sums["policy"] += [int(use[k]) for k in ("tp", "fp", "fn")]
            if not complete:
                continue
            pu, ru, fu = f1(*sums["uncorrected"]); pp, rp, fpol = f1(*sums["policy"])
            out.append({"deadband_px": threshold, "inj_dx": s[0], "inj_dy": s[1], "pairs": complete,
                        "held_pairs": held, "corrected_pairs": applied,
                        "f1_uncorrected": fu, "f1_policy": fpol, "f1_change": fpol - fu,
                        "precision_policy": pp, "recall_policy": rp})
    path = OUT / "val_deadband_selection.csv"
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out[0].keys(), lineterminator="\n")
        writer.writeheader(); writer.writerows(out)
    print("saved", path.relative_to(ROOT))
    for r in out:
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
    clean = [r for r in out if (r["inj_dx"], r["inj_dy"]) == (0.0, 0.0)]
    ok = [r["deadband_px"] for r in clean if r["f1_change"] >= -DEADBAND_MAX_CLEAN_F1_LOSS]
    print(f"rule: smallest dead band with clean-val F1 loss <= {DEADBAND_MAX_CLEAN_F1_LOSS}:", min(ok) if ok else "none in grid")


if __name__ == "__main__":
    {"measure": measure, "grid": grid, "deadband": deadband}[sys.argv[1] if len(sys.argv) > 1 else "grid"]()
