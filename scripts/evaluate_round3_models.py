"""Unified CPU evaluation for Original, v1, and available v2 TinyCD checkpoints."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "vendor" / "tinycd"))
from evaluate_benchmark import Sample, load_samples, tinycd_predict, ChangeClassifier
from followup_scientific_checks import gt_objects
from scientific_evaluation import perturb

OUT = ROOT / "outputs/diagnosis_round"
CROP = 8
SEED = 42
BOOTSTRAPS = 10_000


def counts(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    p, t = pred.astype(bool), truth.astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    assert tp + fp + fn + tn == truth.size
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def summarize(c: dict[str, int]) -> dict[str, float | int]:
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    iou = c["tp"] / (c["tp"] + c["fp"] + c["fn"]) if c["tp"] + c["fp"] + c["fn"] else 0.0
    return {**c, "precision": p, "recall": r, "f1": f1, "iou": iou}


def save_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def method_group(name: str) -> str:
    if name.startswith("CONTROL-v2"): return "CONTROL-v2"
    if name.startswith("ROBUST-v2"): return "ROBUST-v2"
    return name


def aggregate_rows(rows: list[dict], keys: tuple[str, ...], metrics: tuple[str, ...]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for row in rows:
        key = tuple(method_group(row[k]) if k == "method" else row[k] for k in keys)
        buckets.setdefault(key, []).append(row)
    out = []
    for key, group in buckets.items():
        result = {k: v for k, v in zip(keys, key)}
        result["n_seeds"] = len(group)
        for metric in metrics:
            values = np.asarray([float(r[metric]) for r in group if r.get(metric) not in (None, "")], dtype=float)
            result[f"{metric}_mean"] = float(values.mean()) if len(values) else ""
            result[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0 if len(values) == 1 else ""
        out.append(result)
    return out


def mean_std(rows: list[dict], key: str) -> tuple[float | str, float | str]:
    vals = np.asarray([float(r[key]) for r in rows if r.get(key) not in (None, "")], dtype=float)
    if not len(vals): return "", ""
    return float(vals.mean()), float(vals.std(ddof=1)) if len(vals) > 1 else 0.0


def v2_comparison(full_rows, object_rows, robust_rows, swap_rows, bootstrap_rows) -> list[dict]:
    result = []
    for name in ("Original", "CONTROL-v2", "ROBUST-v2"):
        full = [r for r in full_rows if method_group(r["method"]) == name]
        if not full: continue
        row = {"method": name, "n_seeds": len(full)}
        for key in ("precision", "recall", "f1", "iou"):
            row[f"clean_{key}_mean"], row[f"clean_{key}_std"] = mean_std(full, key)
        for scope, prefix in (("all", "all_objects"), ("isolated", "isolated_objects")):
            group = [r for r in object_rows if method_group(r["method"]) == name and r["scope"] == scope]
            for key in ("completeness", "chance_completeness", "object_skill"):
                row[f"{prefix}_{key}_mean"], row[f"{prefix}_{key}_std"] = mean_std(group, key)
        for condition, strength, label in (("blur_B", 2, "B_blur_sigma2"), ("blur_A+B", 2, "AB_blur_sigma2"), ("blur_A", 2, "A_blur_sigma2")):
            group = [r for r in robust_rows if method_group(r["method"]) == name and r["condition"] == condition and r["strength"] == strength and r["frame"] == "A-frame"]
            for key in ("precision", "recall", "f1", "fp"):
                row[f"{label}_{key}_mean"], row[f"{label}_{key}_std"] = mean_std(group, key)
        swapped = [r for r in swap_rows if method_group(r["method"]) == name]
        row["swap_recall_mean"], row["swap_recall_std"] = mean_std(swapped, "recall")
        if name != "Original":
            individual = [r for r in bootstrap_rows if method_group(r["method"]) == name]
            for comparison in ("clean_test", "B_blur_sigma2"):
                matches = [r for r in individual if r["comparison"] == comparison]
                row[f"{comparison}_bootstrap_CI_by_seed"] = "; ".join(
                    f"{r['method'].rsplit('-',1)[-1]}: [{float(r['ci95_low']):.5f}, {float(r['ci95_high']):.5f}]"
                    for r in matches
                )
        result.append(row)
    return result


def load_model(path: Path) -> torch.nn.Module:
    model = ChangeClassifier(pretrained=False)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def predict(model: torch.nn.Module, sample: Sample, before=None, after=None) -> np.ndarray:
    current = Sample(sample.name, sample.before if before is None else before,
                     sample.after if after is None else after, sample.truth, sample.gray_diff)
    return tinycd_predict(model, current, torch.device("cpu"))


def f1_counts(c: np.ndarray) -> np.ndarray:
    p = c[:, 0] / np.maximum(c[:, 0] + c[:, 1], 1)
    r = c[:, 0] / np.maximum(c[:, 0] + c[:, 2], 1)
    return np.divide(2 * p * r, p + r, out=np.zeros_like(p), where=(p + r) > 0)


def paired_bootstrap(original: np.ndarray, candidate: np.ndarray, label: str, rng: np.random.Generator) -> dict:
    n = len(original)
    index = rng.integers(0, n, size=(BOOTSTRAPS, n))
    base_f1 = f1_counts(original[index].sum(axis=1))
    new_f1 = f1_counts(candidate[index].sum(axis=1))
    delta = new_f1 - base_f1
    return {"comparison": label, "n_images": n, "resamples": BOOTSTRAPS,
            "seed": SEED, "mean_delta_f1": float(delta.mean()),
            "ci95_low": float(np.quantile(delta, 0.025)),
            "ci95_high": float(np.quantile(delta, 0.975))}


def object_scores(samples: list[Sample], masks: dict[str, np.ndarray]) -> list[dict]:
    null_file = ROOT / "data/manifests/object_null_derangement_seed42.csv"
    with null_file.open(newline="") as f:
        null_map = {r["gt_filename"]: r["prediction_filename"] for r in csv.DictReader(f)}
    if set(null_map) != set(masks):
        raise ValueError("The fixed deranged-null map must cover all 128 test images.")
    ag = {scope: {key: 0 for key in ("n", "real", "chance")} for scope in ("all", "isolated")}
    for sample in samples:
        labels, areas, isolated = gt_objects(sample.truth)
        actual = np.bincount(labels[masks[sample.name]], minlength=int(labels.max()) + 1)
        chance = np.bincount(labels[masks[null_map[sample.name]]], minlength=int(labels.max()) + 1)
        for i, area in enumerate(areas, 1):
            detected = actual[i] >= 0.30 * area
            chance_detected = chance[i] >= 0.30 * area
            for scope in (("all", "isolated") if isolated[i - 1] else ("all",)):
                ag[scope]["n"] += 1
                ag[scope]["real"] += int(detected)
                ag[scope]["chance"] += int(chance_detected)
    rows = []
    for scope, c in ag.items():
        n = c["n"]
        real = c["real"] / n if n else 0.0
        chance = c["chance"] / n if n else 0.0
        skill = (real - chance) / (1 - chance) if chance < 1 else ""
        rows.append({"scope": scope, "n_objects": n, "detected": c["real"],
                     "chance_detected": c["chance"], "completeness": real,
                     "chance_completeness": chance, "object_skill": skill,
                     "coverage_rule": 0.30, "null_seed": SEED})
    return rows


def shifted_truth(truth: np.ndarray, offset: int) -> np.ndarray:
    matrix = np.float32([[1, 0, offset], [0, 1, offset]])
    return cv2.warpAffine(truth.astype(np.uint8), matrix, (truth.shape[1], truth.shape[0]),
                          flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REFLECT_101).astype(bool)


def object_robust_stats(pred: np.ndarray, truth: np.ndarray) -> dict[str, int]:
    n_gt, labels, gt_stats, _ = cv2.connectedComponentsWithStats(
        truth.astype(np.uint8), connectivity=8
    )
    areas = gt_stats[1:, cv2.CC_STAT_AREA]
    hits = np.bincount(labels[pred], minlength=int(labels.max()) + 1)
    detected = int(sum(hits[i + 1] >= 0.30 * area for i, area in enumerate(areas)))
    n_pred, pred_labels, pred_stats, _ = cv2.connectedComponentsWithStats(
        pred.astype(np.uint8), connectivity=8
    )
    pred_areas = pred_stats[1:, cv2.CC_STAT_AREA]
    overlaps = np.bincount(pred_labels[truth], minlength=n_pred)[1:]
    false_alarms = int(np.count_nonzero((pred_areas >= 20) & (overlaps / np.maximum(pred_areas, 1) < 0.10)))
    return {"n_gt_objects": int(len(areas)), "detected_gt_objects": detected,
            "false_alarm_objects_ge20": false_alarms}


def condition_inputs(sample: Sample, condition: str, strength: float, rng: np.random.Generator):
    a, b = sample.before, sample.after
    if condition in ("gain", "gamma", "noise"):
        b = perturb(b, condition, strength, rng)
    elif condition == "shift":
        b = perturb(b, "shift", strength, rng)
    elif condition == "blur_B":
        b = perturb(b, "blur", strength, rng)
    elif condition == "blur_A":
        a = perturb(a, "blur", strength, rng)
    elif condition == "blur_A+B":
        a = perturb(a, "blur", strength, rng); b = perturb(b, "blur", strength, rng)
    return a, b


def eval_robustness(model, method: str, byname: dict[str, Sample], names: list[str]):
    conditions = [("control", 0), *(('shift', x) for x in (1, 2, 3, 5)),
                  *(('gain', x) for x in (0.8, 1.2)), *(('gamma', x) for x in (0.7, 1.4)),
                  *(('blur_B', x) for x in (1, 1.5, 1.75, 2)),
                  ("blur_A", 2), ("blur_A+B", 2), *(('noise', x) for x in (5, 10))]
    rows, object_rows, blur2_counts, clean_counts = [], [], {}, {}
    for ci, (condition, strength) in enumerate(conditions):
        sums = {"A-frame": np.zeros(4, dtype=np.int64)}
        if condition == "shift": sums["B-frame"] = np.zeros(4, dtype=np.int64)
        elapsed = {frame: 0.0 for frame in sums}
        object_totals = {frame: {"n_gt_objects": 0, "detected_gt_objects": 0,
                                 "false_alarm_objects_ge20": 0} for frame in sums}
        per_image = {frame: {} for frame in sums}
        for ii, name in enumerate(names):
            sample = byname[name]
            rng = np.random.default_rng(SEED + ci * 1000 + ii)
            a, b = condition_inputs(sample, condition, strength, rng)
            start = time.perf_counter()
            pred = predict(model, sample, a, b)[CROP:-CROP, CROP:-CROP]
            duration = time.perf_counter() - start
            gt_a = sample.truth[CROP:-CROP, CROP:-CROP]
            ca = counts(pred, gt_a)
            oa = object_robust_stats(pred, gt_a)
            for key, value in oa.items(): object_totals["A-frame"][key] += value
            vector = np.array([ca[k] for k in ("tp", "fp", "fn", "tn")], dtype=np.int64)
            sums["A-frame"] += vector; elapsed["A-frame"] += duration
            per_image["A-frame"][name] = vector
            if condition == "shift":
                gt_b = shifted_truth(sample.truth, int(strength))[CROP:-CROP, CROP:-CROP]
                cb = counts(pred, gt_b)
                ob = object_robust_stats(pred, gt_b)
                for key, value in ob.items(): object_totals["B-frame"][key] += value
                vb = np.array([cb[k] for k in ("tp", "fp", "fn", "tn")], dtype=np.int64)
                sums["B-frame"] += vb; elapsed["B-frame"] += duration
                per_image["B-frame"][name] = vb
            if condition == "control": clean_counts[name] = vector
            if condition == "blur_B" and strength == 2: blur2_counts[name] = vector
        for frame, vector in sums.items():
            score = summarize({k: int(v) for k, v in zip(("tp", "fp", "fn", "tn"), vector)})
            rows.append({"method": method, "condition": condition, "strength": strength,
                         "frame": frame, **score, "runtime_seconds": elapsed[frame],
                         "delta_fp_vs_control": "", "delta_fn_vs_control": "",
                         "delta_fp_delta_fn_ratio": "",
                         "images": len(names), "crop_border_px": CROP, "seed": SEED, "device": "cpu"})
            obj = object_totals[frame]
            km2 = len(names) * (1024 - 2 * CROP) ** 2 * 0.25 / 1_000_000
            object_rows.append({"method": method, "condition": condition, "strength": strength,
                                "frame": frame, **obj,
                                "completeness": obj["detected_gt_objects"] / obj["n_gt_objects"] if obj["n_gt_objects"] else "",
                                "evaluated_area_km2": km2,
                                "false_alarms_per_km2": obj["false_alarm_objects_ge20"] / km2,
                                "images": len(names), "crop_border_px": CROP})
    for frame in ("A-frame", "B-frame"):
        base = next((r for r in rows if r["condition"] == "control" and r["frame"] == frame), None)
        if base is None: continue
        for row in rows:
            if row["condition"] != "shift" or row["frame"] != frame: continue
            row["delta_fp_vs_control"] = row["fp"] - base["fp"]
            row["delta_fn_vs_control"] = row["fn"] - base["fn"]
            row["delta_fp_delta_fn_ratio"] = (row["delta_fp_vs_control"] / row["delta_fn_vs_control"]
                                               if row["delta_fn_vs_control"] else "")
    return rows, object_rows, blur2_counts, clean_counts


def build_robustness_png(rows: list[dict]) -> None:
    from PIL import Image, ImageDraw, ImageFont
    methods = [m for m in ("Original", "CONTROL-v1", "ROBUST-v1", "CONTROL-v2", "ROBUST-v2")
               if any(method_group(r["method"]) == m for r in rows)]
    colors = ["#2364aa", "#e07a26", "#31935b", "#8c58a8", "#d64f65"]
    panels = [("B shift · A-frame", "shift", [1, 2, 3, 5], "A-frame"),
              ("B shift · B-frame", "shift", [1, 2, 3, 5], "B-frame"),
              ("B blur", "blur_B", [1, 1.5, 1.75, 2], "A-frame"),
              ("Blur placement σ=2", "placement", [0, 1, 2], "A-frame"),
              ("Gain", "gain", [0.8, 1.2], "A-frame"),
              ("Gamma", "gamma", [0.7, 1.4], "A-frame"),
              ("Noise σ", "noise", [5, 10], "A-frame")]
    w, h = 1900, 1420
    image = Image.new("RGB", (w, h), "white"); draw = ImageDraw.Draw(image)
    regular = "/System/Library/Fonts/Supplemental/Arial.ttf"
    bold = "/System/Library/Fonts/Supplemental/Arial Bold.ttf"
    def font(size, is_bold=False):
        path = bold if is_bold else regular
        return ImageFont.truetype(path, size) if Path(path).exists() else ImageFont.load_default()
    draw.text((30, 15), "Round 3 unified pipeline · fixed 32-image subset · post-inference 8 px crop", fill="#111", font=font(30, True))
    pw, ph, gx, gy = 590, 360, 25, 28
    for pi, (title, condition, strengths, frame) in enumerate(panels):
        col, row = pi % 3, pi // 3
        x0, y0 = 30 + col * (pw + gx), 70 + row * (ph + gy)
        left, right, top, bottom = x0 + 60, x0 + pw - 18, y0 + 45, y0 + ph - 62
        draw.text((x0 + 4, y0 + 5), title, fill="#111", font=font(22, True))
        for y in (0, .25, .5, .75, 1):
            yy = int(bottom - y * (bottom - top)); draw.line((left, yy, right, yy), fill="#ddd")
            draw.text((x0 + 4, yy - 9), f"{y:.2g}", fill="#444", font=font(14))
        if condition == "shift": labels = [str(x) for x in strengths]
        elif condition == "placement": labels = ["B σ2", "A σ2", "A+B σ2"]
        else: labels = [str(x) for x in strengths]
        xs = [int(left + i * (right-left) / max(1, len(strengths)-1)) for i in range(len(strengths))]
        for x, label in zip(xs, labels): draw.text((x-14, bottom+9), label, fill="#444", font=font(14))
        for mi, method in enumerate(methods):
            vals, stds = [], []
            for strength in strengths:
                if condition == "placement":
                    key = [("blur_B", "B"), ("blur_A", "A"), ("blur_A+B", "A+B")][strength]
                    found = [r for r in rows if method_group(r["method"]) == method and r["condition"] == key[0] and r["strength"] == 2 and r["frame"] == frame]
                else:
                    found = [r for r in rows if method_group(r["method"]) == method and r["condition"] == condition and r["strength"] == strength and r["frame"] == frame]
                values = [float(r["f1"]) for r in found]
                vals.append(float(np.mean(values)) if values else 0.0)
                stds.append(float(np.std(values, ddof=1)) if len(values) > 1 else 0.0)
            points = [(x, int(bottom-v*(bottom-top))) for x,v in zip(xs, vals)]
            if len(points) > 1: draw.line(points, fill=colors[mi % len(colors)], width=4)
            for x,y,std in zip(xs, points, stds):
                yy=y[1]; err=int(std*(bottom-top)); draw.line((x, yy-err, x, yy+err),fill=colors[mi%len(colors)],width=2)
                draw.ellipse((x-5,yy-5,x+5,yy+5), fill=colors[mi % len(colors)])
    ly = 1380
    for i, method in enumerate(methods):
        x = 80 + i * 350; draw.line((x,ly+12,x+42,ly+12),fill=colors[i%len(colors)],width=5)
        draw.text((x+52,ly),method,fill="#222",font=font(19,True))
    image.save(OUT / "robustness_f1_round3.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    samples = list(load_samples("test"))
    if len(samples) != 128: raise RuntimeError(f"Expected 128 test pairs, got {len(samples)}")
    names = sorted((ROOT / "data/manifests/robustness_test_seed42.txt").read_text().splitlines())
    if len(names) != 32: raise RuntimeError("The fixed seed-42 subset must contain 32 names")
    byname = {s.name: s for s in samples}
    if not set(names) <= set(byname): raise RuntimeError("Subset file includes names outside test split")
    checkpoints = [("Original", ROOT / "models/tinycd_levir_best.pth"),
                   ("CONTROL-v1", ROOT / "models/finetuned/tinycd_control_best.pth"),
                   ("ROBUST-v1", ROOT / "models/finetuned/tinycd_robust_best.pth")]
    for mode in ("control", "robust"):
        for path in sorted((ROOT / "models/finetuned").glob(f"tinycd_{mode}_v2_seed*_best.pth")):
            checkpoints.append((f"{mode.upper()}-v2-{path.stem.split('seed')[-1]}", path))
    if not any("v2" in name for name, _ in checkpoints):
        print("No v2 checkpoints found; evaluating Original and existing v1 checkpoints only.", flush=True)
    full_rows, object_rows, robust_rows, robust_object_rows, swap_rows = [], [], [], [], []
    clean_counts, blur_counts = {}, {}
    rng_boot = np.random.default_rng(SEED)
    for method, path in checkpoints:
        if not path.is_file():
            print(f"skip missing checkpoint {method}: {path}", flush=True); continue
        print(f"begin {method} ({path.name})", flush=True)
        model = load_model(path); masks = {}; per_image = []
        total = np.zeros(4, dtype=np.int64); started = time.perf_counter()
        for i, sample in enumerate(samples, 1):
            pred = predict(model, sample)
            c = counts(pred, sample.truth)
            vector = np.array([c[k] for k in ("tp", "fp", "fn", "tn")], dtype=np.int64)
            total += vector; per_image.append(vector); masks[sample.name] = pred
            if i % 16 == 0: print(f"{method} clean test {i}/128", flush=True)
        score = summarize({k: int(v) for k, v in zip(("tp", "fp", "fn", "tn"), total)})
        score.update({"method": method, "images": 128, "pixels": int(total.sum()),
                      "runtime_seconds": time.perf_counter()-started, "device": "cpu", "threshold": .5})
        assert score["pixels"] == 128 * 1024 * 1024
        full_rows.append(score); clean_counts[method] = np.stack(per_image)
        object_rows.extend({"method": method, **r} for r in object_scores(samples, masks))
        rrows, orows, b2, _ = eval_robustness(model, method, byname, names)
        robust_rows.extend(rrows); robust_object_rows.extend(orows)
        blur_counts[method] = np.stack([b2[n] for n in names])
        # Unmodified B,A swap; score in the original label frame, with the common 8px crop.
        total_swap = np.zeros(4, dtype=np.int64); swap_runtime = 0.0
        for name in names:
            s = byname[name]; start = time.perf_counter()
            pred = predict(model, s, s.after, s.before)[CROP:-CROP, CROP:-CROP]
            swap_runtime += time.perf_counter()-start
            total_swap += np.array([counts(pred, s.truth[CROP:-CROP, CROP:-CROP])[k] for k in ("tp","fp","fn","tn")])
        swap_rows.append({"method": method, "input_order": "B,A", "images": len(names),
                          **summarize({k: int(v) for k, v in zip(("tp","fp","fn","tn"), total_swap)}),
                          "runtime_seconds": swap_runtime, "device": "cpu", "crop_border_px": CROP})
        print(f"end {method} clean F1={score['f1']:.5f}; total seconds={score['runtime_seconds']:.1f}", flush=True)
        del model
    save_csv(OUT / "round3_test_metrics.csv", full_rows)
    save_csv(OUT / "round3_object_completeness.csv", object_rows)
    save_csv(OUT / "round3_robustness.csv", robust_rows)
    save_csv(OUT / "round3_robustness_objects.csv", robust_object_rows)
    save_csv(OUT / "round3_swap.csv", swap_rows)
    save_csv(OUT / "round3_method_summary.csv", aggregate_rows(full_rows, ("method",),
             ("precision", "recall", "f1", "iou", "runtime_seconds")))
    save_csv(OUT / "round3_object_summary.csv", aggregate_rows(object_rows, ("method", "scope"),
             ("completeness", "chance_completeness", "object_skill")))
    save_csv(OUT / "round3_robustness_seed_summary.csv", aggregate_rows(robust_rows,
             ("method", "condition", "strength", "frame"),
             ("precision", "recall", "f1", "fp", "runtime_seconds")))
    save_csv(OUT / "round3_robustness_object_summary.csv", aggregate_rows(robust_object_rows,
             ("method", "condition", "strength", "frame"),
             ("n_gt_objects", "detected_gt_objects", "completeness", "false_alarm_objects_ge20", "false_alarms_per_km2")))
    save_csv(OUT / "round3_swap_seed_summary.csv", aggregate_rows(swap_rows, ("method",),
             ("precision", "recall", "f1", "iou", "runtime_seconds")))
    original = "Original"
    bootstrap_rows = []
    if original in clean_counts and original in blur_counts:
        for method in clean_counts:
            if method == original: continue
            bootstrap_rows.append({"method": method, **paired_bootstrap(clean_counts[original], clean_counts[method], "clean_test", rng_boot)})
            bootstrap_rows.append({"method": method, **paired_bootstrap(blur_counts[original], blur_counts[method], "B_blur_sigma2", rng_boot)})
    save_csv(OUT / "round3_bootstrap_delta_f1.csv", bootstrap_rows)
    save_csv(OUT / "round3_v2_comparison.csv", v2_comparison(full_rows, object_rows, robust_rows, swap_rows, bootstrap_rows))
    build_robustness_png(robust_rows)
    print("Round 3 unified evaluation complete; all metrics and figures remain local.", flush=True)


if __name__ == "__main__":
    main()
