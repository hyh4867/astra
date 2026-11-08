"""Re-evaluate the local NAIP window with VAL-selected checkpoints only.

All outputs are qualitative: the NAIP pair has no ground-truth mask.
"""
from __future__ import annotations

import csv
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from rasterio import Affine

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["ASTRA_FORCE_CPU"] = "1"

from geospatial import align_pair, object_features, read_image_bytes
from inference import load_tinycd_model, make_overlay, probability_heatmap, tinycd_probability

PAIR = ROOT / "data/demo_naip/celina_town_2020_2022_overlap"
OUT = ROOT / "outputs/naip_demo"
CHECKPOINTS = [
    ("Original", ROOT / "models/tinycd_levir_best.pth", None),
    ("CONTROL-v2 seed 42", ROOT / "models/finetuned/tinycd_control_v2_seed42_best.pth", 0.909601),
    ("ROBUST-v2 seed 42", ROOT / "models/finetuned/tinycd_robust_v2_seed42_best.pth", 0.905744),
]


def save_figure(a: np.ndarray, b: np.ndarray, probability: np.ndarray, title: str, destination: Path) -> None:
    panels = [("A · before · 2020", a), ("B · after · 2022", b),
              ("Probability · 0–1", probability_heatmap(probability)),
              ("Overlay · cutoff 0.5", make_overlay(b, probability >= 0.5))]
    panel_w, panel_h, label_h = 480, 600, 48
    canvas = Image.new("RGB", (4 * panel_w, panel_h + label_h + 42), "white")
    draw = ImageDraw.Draw(canvas)
    bold_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    font = ImageFont.truetype(str(bold_path), 18) if bold_path.exists() else ImageFont.load_default()
    draw.text((12, 7), f"QUALITATIVE ONLY · NAIP · NO GROUND TRUTH · {title}", fill="#9b1c1c", font=font)
    for i, (label, array) in enumerate(panels):
        x = i * panel_w
        draw.text((x + 10, label_h), label, fill="#10243a", font=font)
        image = Image.fromarray(array)
        image.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (panel_w - image.width) // 2,
                             label_h + 26 + (panel_h - image.height) // 2))
    destination.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(destination)


def save_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run_scale(a: np.ndarray, b: np.ndarray, geo: dict, gsd: float, scale_name: str) -> dict:
    scale = gsd / 0.5 if scale_name == "0p5m" else 1.0
    if scale_name == "0p5m":
        a = cv2.resize(a, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        b = cv2.resize(b, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        geo = dict(geo)
        geo["transform"] = geo["transform"] * Affine.scale(1.0 / scale)
        geo["gsd_m"] = 0.5
        geo["pixel_area_m2"] = 0.25
        target_gsd = 0.5
    else:
        target_gsd = float(gsd)

    results = []
    for label, checkpoint, val_f1 in CHECKPOINTS:
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, device = load_tinycd_model(checkpoint)
        if device.type != "cpu":
            raise RuntimeError(f"ASTRA_FORCE_CPU was requested but {label} loaded on {device}")
        started = time.perf_counter()
        probability = tinycd_probability(model, a, b, device)
        runtime = time.perf_counter() - started
        mask = probability >= 0.5
        rows, collection = object_features(mask, probability, geo, target_gsd)
        for feature in collection["features"]:
            feature["properties"]["evaluation_status"] = "QUALITATIVE_ONLY_NO_GROUND_TRUTH"
        slug = label.lower().replace(" ", "_").replace("-", "_")
        stem = f"round4_{slug}_{scale_name}"
        csv_path = OUT / f"{stem}_objects.csv"
        geojson_path = OUT / f"{stem}_objects.geojson"
        figure_path = OUT / f"figure_{stem}.png"
        save_rows(csv_path, rows)
        geojson_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n")
        save_figure(a, b, probability, f"{label} · {target_gsd:.1f} m/px", figure_path)
        result = {
            "model": label,
            "checkpoint": str(checkpoint.relative_to(ROOT)),
            "clean_val_f1_for_selection": val_f1,
            "selection_rule": "highest clean VAL F1 across seeds; no TEST-based selection",
            "scale": scale_name,
            "gsd_m_per_px": target_gsd,
            "device": str(device),
            "runtime_seconds": runtime,
            "change_objects": len(rows),
            "changed_pixels": int(mask.sum()),
            "total_area_m2": float(sum(float(row["area_m2"]) for row in rows)),
            "figure": figure_path.name,
            "objects_csv": csv_path.name,
            "objects_geojson": geojson_path.name,
            "qualitative_only": True,
            "ground_truth": False,
        }
        results.append(result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
        del model
    return {"scale": scale_name, "results": results}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a0, geo_a = read_image_bytes((PAIR / "A_before_2020.tif").read_bytes(), "A_before_2020.tif")
    b0, geo_b = read_image_bytes((PAIR / "B_after_2022.tif").read_bytes(), "B_after_2022.tif")
    a, b, geo, gsd = align_pair(a0, geo_a, b0, geo_b)
    clean_val = {name: value for name, _, value in CHECKPOINTS}
    result = {
        "pair": "Celina, Texas; 2020-11-02 to 2022-07-04; existing window",
        "checkpoint_selection": "CONTROL-v2 seed 42 (VAL F1 0.909601) and ROBUST-v2 seed 42 (VAL F1 0.905744) were selected by clean VAL F1 across seeds; Original is the pretrained reference.",
        "clean_val_f1": clean_val,
        "gsd_native_m_per_px": float(gsd),
        "device": "cpu; host MPS matmul failed during environment check",
        "qualitative_only": True,
        "ground_truth": False,
        "models": [],
    }
    result["models"].extend(run_scale(a, b, geo, float(gsd), "0p6m")["results"])
    result["models"].extend(run_scale(a, b, geo, float(gsd), "0p5m")["results"])
    summary_path = OUT / "round4_naip_checkpoint_selection.json"
    summary_path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(f"summary={summary_path.relative_to(ROOT)}", flush=True)


if __name__ == "__main__":
    main()
