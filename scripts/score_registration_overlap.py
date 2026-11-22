"""Score registration conditions on their common real-image rectangle.

Reuse fixed-crop confusion counts wherever that crop already excludes padding.
For integer translations, output coordinate x accesses x - move in the
injected image and x - move - injection in the original B (likewise for y).
"""
import csv
from collections import defaultdict
from itertools import groupby

import numpy as np

from evaluate_registration import CROP, OUT, ROOT, shift_image


def axis_bounds(length, injection, moves):
    offsets = [offset for move in moves for offset in (move, move + injection)]
    return max(CROP, max(offsets)), length + min(-CROP, min(offsets))


def main():
    import torch
    from evaluate_benchmark import ChangeClassifier, Sample, load_samples, tinycd_predict
    source = OUT / "test_per_pair.csv"
    target = OUT / "test_overlap_per_pair.csv"
    with source.open(newline="") as f:
        rows = list(csv.DictReader(f))
    names = set((ROOT / "data/manifests/test_filenames.txt").read_text().split())
    if {row["name"] for row in rows} != names:
        raise ValueError("Finish the full test checkpoint before scoring")
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT / "models/tinycd_levir_best.pth", map_location="cpu", weights_only=True))
    part = target.with_suffix(".csv.tmp")
    adjusted = 0
    with part.open("w", newline="") as f:
        fields = [k for k in rows[0] if k not in ("seconds", "detector_runs", "device")]
        writer = csv.DictWriter(f, fieldnames=fields + ["x0", "x1", "y0", "y1", "score_source"],
                                extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        for name, group in groupby(sorted(rows, key=lambda r: r["name"]), key=lambda r: r["name"]):
            sample = None
            by_shift = defaultdict(list)
            for row in group:
                by_shift[int(float(row["inj_dx"])), int(float(row["inj_dy"]))].append(row)
            for (sx, sy), conditions in by_shift.items():
                moves = [(int(float(r["applied_dx"])), int(float(r["applied_dy"]))) for r in conditions]
                x0, x1 = axis_bounds(1024, sx, [x for x, _ in moves])
                y0, y1 = axis_bounds(1024, sy, [y for _, y in moves])
                source_label = "fixed_crop_checkpoint"
                if (x1 - x0) * (y1 - y0) != (1024 - 2*CROP)**2:
                    adjusted += 1
                    source_label = "common_overlap_rescore"
                    if sample is None:
                        sample = next(load_samples("test", [name]))
                    b = shift_image(sample.after, sx, sy) if (sx, sy) != (0, 0) else sample.after
                    truth = sample.truth[y0:y1, x0:x1]
                    scores = {}
                    for row, move in zip(conditions, moves):
                        if move not in scores:
                            after = shift_image(b, *move) if move != (0, 0) else b
                            pred = tinycd_predict(model, Sample(name, sample.before, after, sample.truth, None),
                                                  device, batch_size=4)[y0:y1, x0:x1]
                            scores[move] = dict(tp=np.count_nonzero(pred & truth), fp=np.count_nonzero(pred & ~truth),
                                                fn=np.count_nonzero(~pred & truth), scored_pixels=truth.size)
                        row.update(scores[move])
                for row in conditions:
                    writer.writerow(dict(row, x0=x0, x1=x1, y0=y0, y1=y1, score_source=source_label))
    part.replace(target)
    print(f"Scored {len(rows)} condition rows. Narrowed the common rectangle in {adjusted} pair-shifts.")


if __name__ == "__main__":
    main()
