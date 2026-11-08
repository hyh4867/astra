"""Run qualitative-only TinyCD inference on the local public NAIP pair."""
from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from inference import load_tinycd_model, make_overlay, probability_heatmap, tinycd_probability
from geospatial import align_pair, object_features, read_image_bytes

PAIR = ROOT / "data/demo_naip/celina_town_2020_2022_overlap"
OUT = ROOT / "outputs/naip_demo"
CHECKPOINTS = {
    "original": ROOT / "models/tinycd_levir_best.pth",
    "robust": ROOT / "models/finetuned/tinycd_robust_best.pth",
}
ROBUST_READY = ROOT / "models/finetuned/tinycd_robust_ready.json"


def make_sheet(images: list[tuple[str, np.ndarray]], destination: Path) -> None:
    panel_w, panel_h, title_h, header_h = 512, 640, 54, 42
    canvas = Image.new("RGB", (panel_w * len(images), panel_h + title_h + header_h), "white")
    draw = ImageDraw.Draw(canvas)
    regular = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    title_font = ImageFont.truetype(str(bold), 19) if bold.exists() else ImageFont.load_default()
    banner_font = ImageFont.truetype(str(bold), 22) if bold.exists() else ImageFont.load_default()
    draw.text((14, 7), "QUALITATIVE ONLY · NAIP · NO GROUND TRUTH", fill="#9b1c1c", font=banner_font)
    for index, (title, array) in enumerate(images):
        x = index * panel_w
        draw.text((x + 12, header_h + 8), title, fill="#10243a", font=title_font)
        image = Image.fromarray(array)
        image.thumbnail((panel_w, panel_h), Image.Resampling.LANCZOS)
        canvas.paste(image, (x + (panel_w - image.width) // 2, header_h + title_h + (panel_h - image.height) // 2))
    canvas.save(destination)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    a, geo_a = read_image_bytes((PAIR / "A_before_2020.tif").read_bytes(), "A_before_2020.tif")
    b, geo_b = read_image_bytes((PAIR / "B_after_2022.tif").read_bytes(), "B_after_2022.tif")
    a, b, geo, gsd = align_pair(a, geo_a, b, geo_b)
    metadata = json.loads((PAIR / "metadata.json").read_text())
    summary = {"qualitative_only": True, "ground_truth": False, "pair_metadata": metadata, "models": {}}

    for variant, checkpoint in CHECKPOINTS.items():
        if variant == "robust" and not ROBUST_READY.is_file():
            print("ROBUST run is not complete; skipping its provisional checkpoint.", flush=True)
            continue
        if not checkpoint.is_file():
            if variant == "robust":
                print("ROBUST checkpoint not ready; skipping it.", flush=True)
                continue
            raise FileNotFoundError(checkpoint)
        model, device = load_tinycd_model(checkpoint)
        started = time.perf_counter()
        probability = tinycd_probability(model, a, b, device)
        runtime = time.perf_counter() - started
        mask = probability >= 0.5
        rows, collection = object_features(mask, probability, geo, float(gsd))
        for feature in collection["features"]:
            feature["properties"]["evaluation_status"] = "QUALITATIVE_ONLY_NO_GROUND_TRUTH"
        tag = f"naip_{variant}_2020_2022"
        geojson_path = OUT / f"{tag}_objects.geojson"
        geojson_path.write_text(json.dumps(collection, ensure_ascii=False, indent=2) + "\n")
        csv_path = OUT / f"{tag}_objects.csv"
        fields = ["object_id", "area_m2", "centroid_lon", "centroid_lat", "mean_probability", "pixel_count", "area_crs", "evaluation_status", "ground_truth"]
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows([{**row,"evaluation_status":"QUALITATIVE_ONLY_NO_GROUND_TRUTH","ground_truth":False} for row in rows])
        sheet_path = OUT / f"{tag}_A_B_probability_overlay.png"
        make_sheet([
            ("A · before · 2020", a),
            ("B · after · 2022", b),
            ("TinyCD probability · 0–1", probability_heatmap(probability)),
            ("Change overlay · cutoff 0.5", make_overlay(b, mask)),
        ], sheet_path)
        summary["models"][variant] = {
            "checkpoint": checkpoint.name, "device": str(device), "runtime_seconds": runtime,
            "change_objects": len(rows), "changed_pixels": int(mask.sum()),
            "geojson": geojson_path.name, "csv": csv_path.name, "figure": sheet_path.name,
        }
        print(f"{variant}: device={device} runtime={runtime:.2f}s objects={len(rows)}", flush=True)
    (OUT / "naip_qualitative_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
