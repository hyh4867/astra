"""Summarize Round 4 NAIP objects at the 5 and 7.2 m² component floors."""
from __future__ import annotations

import csv
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    ("Original", 0.6): ROOT / "outputs/naip_demo/round4_original_0p6m_objects.csv",
    ("CONTROL-v2 s42", 0.6): ROOT / "outputs/naip_demo/round4_control_v2_seed_42_0p6m_objects.csv",
    ("ROBUST-v2 s42", 0.6): ROOT / "outputs/naip_demo/round4_robust_v2_seed_42_0p6m_objects.csv",
    ("Original", 0.5): ROOT / "outputs/naip_demo/round4_original_0p5m_objects.csv",
    ("CONTROL-v2 s42", 0.5): ROOT / "outputs/naip_demo/round4_control_v2_seed_42_0p5m_objects.csv",
    ("ROBUST-v2 s42", 0.5): ROOT / "outputs/naip_demo/round4_robust_v2_seed_42_0p5m_objects.csv",
}
FLOORS_M2 = (5.0, 7.2)
FIELDS = [
    "method", "gsd_m", "minimum_area_m2", "minimum_pixels",
    "realized_pixel_area_m2", "raw_objects", "raw_area_m2", "kept_objects",
    "kept_area_m2", "sub_floor_components", "raw_pixels", "kept_pixels",
]


def main() -> None:
    output = ROOT / "outputs/naip_demo/round4_naip_filtered_summary.csv"
    rows = []
    for (method, gsd), path in INPUTS.items():
        with path.open(newline="") as stream:
            objects = list(csv.DictReader(stream))
        raw_area = sum(float(obj["area_m2"]) for obj in objects)
        raw_pixels = sum(int(obj["pixel_count"]) for obj in objects)
        for floor_m2 in FLOORS_M2:
            threshold_px = math.ceil(floor_m2 / (gsd * gsd) - 1e-12)
            kept = [obj for obj in objects if int(obj["pixel_count"]) >= threshold_px]
            rows.append({
                "method": method,
                "gsd_m": f"{gsd:.1f}",
                "minimum_area_m2": f"{floor_m2:.1f}",
                "minimum_pixels": threshold_px,
                "realized_pixel_area_m2": f"{threshold_px * gsd * gsd:.2f}",
                "raw_objects": len(objects),
                "raw_area_m2": f"{raw_area:.4f}",
                "kept_objects": len(kept),
                "kept_area_m2": f"{sum(float(obj['area_m2']) for obj in kept):.4f}",
                "sub_floor_components": len(objects) - len(kept),
                "raw_pixels": raw_pixels,
                "kept_pixels": sum(int(obj["pixel_count"]) for obj in kept),
            })
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {output.relative_to(ROOT)}")
    for row in rows:
        print(
            f"{row['method']} {row['gsd_m']}m floor {row['minimum_area_m2']}m² "
            f"(≥{row['minimum_pixels']} px): raw {row['raw_objects']}/"
            f"{float(row['raw_area_m2']):.1f}m² → kept {row['kept_objects']}/"
            f"{float(row['kept_area_m2']):.1f}m²; "
            f"sub-floor {row['sub_floor_components']}"
        )


if __name__ == "__main__":
    main()
