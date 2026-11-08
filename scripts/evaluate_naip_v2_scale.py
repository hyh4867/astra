"""Qualitative-only v2 scale check on the existing Celina NAIP demo pair.

This is the already-downloaded weak demo window, not a candidate from the pending
contact sheet. It has no ground truth and is not used for model selection.
"""
from __future__ import annotations

import csv
import json
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
sys.path.insert(0, str(ROOT / "vendor/tinycd"))

from geospatial import align_pair, object_features, read_image_bytes
from inference import make_overlay, probability_heatmap, tinycd_probability
from models.change_classifier import ChangeClassifier

PAIR = ROOT / "data/demo_naip/celina_town_2020_2022_overlap"
OUT = ROOT / "outputs/naip_demo"
CHECKPOINT = ROOT / "models/finetuned/tinycd_control_v2_seed44_best.pth"


def save_figure(a: np.ndarray, b: np.ndarray, probability: np.ndarray, name: str) -> None:
    panels = [("A · before", a), ("B · after", b),
              ("Probability · 0–1", probability_heatmap(probability)),
              ("Overlay", make_overlay(b, probability >= .5))]
    panel_w, panel_h, label_h = 480, 600, 46
    canvas = Image.new("RGB", (panel_w * 4, panel_h + label_h + 34), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 6), "QUALITATIVE ONLY · NAIP · NO GROUND TRUTH · CONTROL-v2 seed 44",
              fill="#9b1c1c")
    for index, (label, array) in enumerate(panels):
        x = index * panel_w
        draw.text((x + 10, label_h), label, fill="#10243a")
        image = Image.fromarray(array)
        image.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (panel_w-image.width)//2,
                             label_h + 26 + (panel_h-image.height)//2))
    canvas.save(OUT / name)


def save_inset(a: np.ndarray, b: np.ndarray, mask: np.ndarray, name: str) -> None:
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if count > 1:
        component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        cx, cy = map(int, centroids[component])
    else:
        cx, cy = a.shape[1] // 2, a.shape[0] // 2
    x0 = max(0, min(a.shape[1]-512, cx-256))
    y0 = max(0, min(a.shape[0]-512, cy-256))
    overlay = make_overlay(b, mask)
    images = [("A inset", a), ("B inset", b), ("Overlay inset", overlay)]
    canvas = Image.new("RGB", (1536, 570), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 5), f"QUALITATIVE ONLY · 512×512 inset · x={x0}, y={y0}", fill="#9b1c1c")
    for i, (label, arr) in enumerate(images):
        draw.text((i*512+8, 28), label, fill="#10243a")
        canvas.paste(Image.fromarray(arr[y0:y0+512, x0:x0+512]).resize(
            (512, 512), Image.Resampling.LANCZOS), (i*512, 52))
    canvas.save(OUT / name)


def write_rows(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def infer(model, a, b, geo, gsd: float, label: str) -> dict:
    start = time.perf_counter()
    probability = tinycd_probability(model, a, b, torch.device("cpu"))
    runtime = time.perf_counter() - start
    mask = probability >= .5
    objects, collection = object_features(mask, probability, geo, gsd)
    for feature in collection["features"]:
        feature["properties"]["evaluation_status"] = "QUALITATIVE_ONLY_NO_GROUND_TRUTH"
    tag = f"naip_control_v2_seed44_{label}"
    write_rows(OUT / f"{tag}_objects.csv", objects)
    (OUT / f"{tag}_objects.geojson").write_text(json.dumps(collection, indent=2) + "\n")
    save_figure(a, b, probability, f"figure_control_v2_seed44_{label}.png")
    save_inset(a, b, mask, f"figure_control_v2_seed44_{label}_inset.png")
    return {"objects": len(objects), "area_m2": sum(x["area_m2"] for x in objects),
            "runtime_s": runtime, "changed_pixels": int(mask.sum()),
            "mean_probability": float(probability.mean()),
            "figure": f"figure_control_v2_seed44_{label}.png"}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(CHECKPOINT)
    a, geo_a = read_image_bytes((PAIR / "A_before_2020.tif").read_bytes(), "A_before_2020.tif")
    b, geo_b = read_image_bytes((PAIR / "B_after_2022.tif").read_bytes(), "B_after_2022.tif")
    a, b, geo, gsd = align_pair(a, geo_a, b, geo_b)
    model = ChangeClassifier(pretrained=False)
    model.load_state_dict(torch.load(CHECKPOINT, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    result = {
        "qualitative_only": True,
        "ground_truth": False,
        "pair": "Celina, Texas; 2020-11-02 to 2022-07-04; existing weak demo window",
        "checkpoint": str(CHECKPOINT.relative_to(ROOT)),
        "selection": "best clean VAL F1 among v2 checkpoints (CONTROL seed 44: 0.908415)",
        "device": "cpu",
        "gsd_original_m": float(gsd),
        "gsd_resampled_m": 0.5,
        "models": {},
    }
    result["models"]["Original_existing_reference"] = json.loads(
        (OUT / "naip_diagnosis_summary.json").read_text()
    )["models"]["Original"]
    result["models"]["CONTROL-v2 seed44"] = {
        "0.6m": infer(model, a, b, geo, float(gsd), "0p6m"),
    }
    a05 = cv2.resize(a, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LANCZOS4)
    b05 = cv2.resize(b, None, fx=1.2, fy=1.2, interpolation=cv2.INTER_LANCZOS4)
    geo05 = dict(geo)
    geo05["transform"] = geo["transform"] * Affine.scale(.5 / .6)
    geo05["gsd_m"] = .5
    geo05["pixel_area_m2"] = .25
    result["models"]["CONTROL-v2 seed44"]["0.5m"] = infer(model, a05, b05, geo05, .5, "0p5m")
    (OUT / "naip_control_v2_seed44_scale_summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
