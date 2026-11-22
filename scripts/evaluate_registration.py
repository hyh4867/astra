"""Does estimating and undoing an x/y shift restore TinyCD on LEVIR-CD?

For every pair and injected shift this writes one row per condition:
    uncorrected       detection on the pair as given (shift 0,0 = clean baseline)
    pipeline          registration.register_pair with the frozen DEFAULT_CONFIG,
                      then detection; held or dead-band pairs keep B untouched
    estimated_always  the estimate is applied even when the pipeline would not
                      (ablation; only for the shifts listed in --always)
    oracle            B moved back by the injected shift (control)

Protocol kept from scripts/scientific_evaluation.py: B is shifted with the
same warp, the label mask stays on A's grid, an 8 px border is cropped in
every condition, and TP/FP/FN are summed over pairs before P/R/F1.
Pixels inside the scoring region where a corrected B has no real data count
as "no candidate". The estimator sees only the two images.

Resumable: rows are appended per pair and finished (pair, shift) keys are
skipped on restart.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from registration import DEFAULT_CONFIG, register_pair, shift_image, shift_valid_mask  # noqa: E402

CROP = 8  # same border crop as scientific_evaluation.CROP
OUT = ROOT / "outputs" / "registration"
FIELDS = ["split", "name", "inj_dx", "inj_dy", "condition", "tp", "fp", "fn", "scored_pixels",
          "est_dx", "est_dy", "status", "gate_reasons", "applied_dx", "applied_dy",
          "global_dx", "global_dy", "global_peak_ratio", "informative_fraction", "tile_agreement",
          "detector_runs", "seconds", "device"]


def parse_shifts(text: str):
    return [tuple(float(v) for v in item.split(",")) for item in text.split(";") if item.strip()]


def counts(pred: np.ndarray, truth: np.ndarray, valid: np.ndarray | None):
    if valid is not None:
        pred = pred & valid  # no real B data there: nothing is shown as a candidate
    p = pred[CROP:-CROP, CROP:-CROP]; t = truth[CROP:-CROP, CROP:-CROP]
    return int(np.count_nonzero(p & t)), int(np.count_nonzero(p & ~t)), int(np.count_nonzero(~p & t)), int(p.size)


def main():
    import torch
    from evaluate_benchmark import DATA, ChangeClassifier, Sample, load_samples, tinycd_predict
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=["val", "test"])
    parser.add_argument("--shifts", required=True, help='e.g. "0,0;1,1;5,5;3,-2"')
    parser.add_argument("--always", default="", help="shifts that also get the estimated_always ablation")
    parser.add_argument("--names", default="", help="optional text file with one pair name per line")
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-oracle", action="store_true")
    args = parser.parse_args()

    shifts = parse_shifts(args.shifts); always = set(parse_shifts(args.always))
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    existing = {}
    if out.is_file():
        with out.open(newline="") as f:
            rows = list(csv.DictReader(f))
        for row in rows:
            existing.setdefault((row["name"], float(row["inj_dx"]), float(row["inj_dy"])), set()).add(row["condition"])
        for key, conditions in existing.items():
            need = {"uncorrected", "pipeline"} | ({"estimated_always"} if (key[1], key[2]) in always else set())
            if (key[1], key[2]) != (0.0, 0.0) and not args.no_oracle:
                need.add("oracle")
            if need <= conditions:
                done.add(key)
    else:
        with out.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n").writeheader()

    wanted = [line.strip() for line in Path(args.names).read_text().splitlines() if line.strip()] if args.names else None
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT / "models" / "tinycd_levir_best.pth", map_location="cpu", weights_only=True), strict=True)
    print(f"device={device} split={args.split} shifts={shifts} config={DEFAULT_CONFIG}", flush=True)

    names = sorted(path.name for path in (DATA / args.split / "A").glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No images for {args.split}")
    if wanted is not None:
        if set(wanted) - set(names) or len(wanted) != len(set(wanted)):
            raise ValueError("Requested names are missing or duplicated")
        names = sorted(wanted)
    pending = [name for name in names if any((name, dx, dy) not in done for dx, dy in shifts)]
    for index, sample in enumerate(load_samples(args.split, pending), 1):
        started = time.perf_counter()
        for sx, sy in shifts:
            if (sample.name, sx, sy) in done:
                continue
            t0 = time.perf_counter(); runs = 0
            clean = (sx, sy) == (0.0, 0.0)
            b_in = sample.after if clean else shift_image(sample.after, sx, sy)

            def detect(after):
                nonlocal runs
                runs += 1
                return tinycd_predict(model, Sample(sample.name, sample.before, after, sample.truth, None), device, batch_size=4)

            # The estimator receives the two images and nothing else.
            result = register_pair(sample.before, b_in, DEFAULT_CONFIG)
            base_pred = detect(b_in)
            cache = {(0.0, 0.0): (base_pred, None)}

            def corrected(move_x, move_y):
                key = (float(move_x), float(move_y))
                if key not in cache:
                    cache[key] = (detect(shift_image(b_in, *key)), shift_valid_mask(sample.truth.shape, *key))
                return cache[key]

            info = {"split": args.split, "name": sample.name, "inj_dx": sx, "inj_dy": sy,
                    "est_dx": f"{result.dx:.4f}", "est_dy": f"{result.dy:.4f}", "status": result.status,
                    "gate_reasons": "|".join(result.reasons), "global_dx": f"{result.global_dx:.4f}",
                    "global_dy": f"{result.global_dy:.4f}", "global_peak_ratio": f"{result.global_peak_ratio:.4f}",
                    "informative_fraction": f"{result.informative_fraction:.4f}",
                    "tile_agreement": f"{result.tile_agreement:.4f}", "device": str(device)}
            rows = []

            def add(condition, pred, valid, move):
                tp, fp, fn, n = counts(pred, sample.truth, valid)
                rows.append({**info, "condition": condition, "tp": tp, "fp": fp, "fn": fn, "scored_pixels": n,
                             "applied_dx": move[0], "applied_dy": move[1]})

            add("uncorrected", base_pred, None, (0.0, 0.0))
            pipe_move = (result.applied_dx, result.applied_dy) if result.applied else (0.0, 0.0)
            add("pipeline", *corrected(*pipe_move), pipe_move)
            if (sx, sy) in always:
                always_move = (float(round(-result.dx)), float(round(-result.dy)))
                add("estimated_always", *corrected(*always_move), always_move)
            if not clean and not args.no_oracle:
                # Control only: the injected shift is used here and nowhere above.
                oracle_pred = detect(shift_image(b_in, -sx, -sy))
                add("oracle", oracle_pred, shift_valid_mask(sample.truth.shape, -sx, -sy), (-sx, -sy))
            seconds = round(time.perf_counter() - t0, 2)
            with out.open("a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS, lineterminator="\n")
                for row in rows:
                    if row["condition"] in existing.get((sample.name, sx, sy), set()):
                        continue
                    writer.writerow({**row, "detector_runs": runs, "seconds": seconds})
        print(f"{args.split} {index}/{len(pending)} {sample.name} {time.perf_counter() - started:.1f}s", flush=True)
    print("done", out, flush=True)


if __name__ == "__main__":
    main()
