"""Compare filtered Round 4 NAIP masks and component locations in common UTM."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from rasterio.features import bounds, rasterize
from rasterio.transform import from_origin
from rasterio.warp import transform, transform_geom

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/naip_demo"
MODELS = {
    "Original": "original",
    "CONTROL-v2 s42": "control_v2_seed_42",
    "ROBUST-v2 s42": "robust_v2_seed_42",
}
GSD_BY_SCALE = {"0p6m": 0.6, "0p5m": 0.5}
AREA_FLOOR_M2 = 5.0
RASTER_RESOLUTION_M = 0.05
CRS = "EPSG:32614"


def components(model: str, scale: str) -> list[dict]:
    path = OUT / f"round4_{MODELS[model]}_{scale}_objects.geojson"
    data = json.loads(path.read_text())
    gsd = GSD_BY_SCALE[scale]
    minimum_pixels = math.ceil(AREA_FLOOR_M2 / (gsd * gsd) - 1e-12)
    result = []
    for feature in data["features"]:
        props = feature["properties"]
        if int(props["pixel_count"]) < minimum_pixels:
            continue
        geom = transform_geom("EPSG:4326", CRS, feature["geometry"], precision=-1)
        x, y = transform("EPSG:4326", CRS,
                         [float(props["centroid_lon"])],
                         [float(props["centroid_lat"])])
        result.append({
            "model": model,
            "scale": scale,
            "object_id": int(props["object_id"]),
            "pixel_count": int(props["pixel_count"]),
            "area_m2": float(props["area_m2"]),
            "centroid_x": float(x[0]),
            "centroid_y": float(y[0]),
            "geometry": geom,
            "bounds": bounds(geom),
        })
    return result


def intersection_area(a: dict, b: dict) -> float:
    res = RASTER_RESOLUTION_M
    left = max(a["bounds"][0], b["bounds"][0])
    bottom = max(a["bounds"][1], b["bounds"][1])
    right = min(a["bounds"][2], b["bounds"][2])
    top = min(a["bounds"][3], b["bounds"][3])
    if right <= left or top <= bottom:
        return 0.0
    # Snap every pair to the same UTM-origin-aligned 5 cm lattice.
    left = math.floor(left / res) * res
    top = math.ceil(top / res) * res
    width = max(1, math.ceil((right - left) / res))
    height = max(1, math.ceil((top - bottom) / res))
    affine = from_origin(left, top, res, res)
    shape = (height, width)
    mask_a = rasterize([(a["geometry"], 1)], out_shape=shape,
                       transform=affine, fill=0, dtype="uint8")
    mask_b = rasterize([(b["geometry"], 1)], out_shape=shape,
                       transform=affine, fill=0, dtype="uint8")
    return float(np.count_nonzero((mask_a > 0) & (mask_b > 0)) * res * res)


def pair_metrics(a: dict, b: dict) -> tuple[float, float]:
    inter = intersection_area(a, b)
    union = a["area_m2"] + b["area_m2"] - inter
    iou = inter / union if union > 0 else 1.0
    distance = math.hypot(a["centroid_x"] - b["centroid_x"],
                          a["centroid_y"] - b["centroid_y"])
    return iou, distance


def write_pairwise_component_comparison() -> list[dict]:
    original = components("Original", "0p6m")
    robust = components("ROBUST-v2 s42", "0p6m")
    rows = []
    for a in original:
        candidates = []
        for b in robust:
            iou, distance = pair_metrics(a, b)
            candidates.append((b, iou, distance))
        best = max(candidates, key=lambda row: row[1]) if candidates else None
        for b, iou, distance in candidates:
            rows.append({
                "original_object_id": a["object_id"],
                "original_pixels": a["pixel_count"],
                "robust_object_id": b["object_id"],
                "robust_pixels": b["pixel_count"],
                "intersection_over_union": f"{iou:.4f}",
                "centroid_distance_m": f"{distance:.3f}",
                "best_overlap_for_original": b is best[0],
            })
    path = OUT / "round4_original_vs_robust_0p6_component_pairs.csv"
    fields = list(rows[0]) if rows else []
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_mask_comparison() -> list[dict]:
    rows = []
    model_names = list(MODELS)
    for scale, gsd in GSD_BY_SCALE.items():
        for i, first in enumerate(model_names):
            for second in model_names[i + 1:]:
                a_components = components(first, scale)
                b_components = components(second, scale)
                intersection = sum(intersection_area(a, b)
                                   for a in a_components for b in b_components)
                area_a = sum(item["area_m2"] for item in a_components)
                area_b = sum(item["area_m2"] for item in b_components)
                union = area_a + area_b - intersection
                iou = intersection / union if union else 1.0
                dice = 2 * intersection / (area_a + area_b) if area_a + area_b else 1.0
                rows.append({
                    "scale": scale,
                    "gsd_m": gsd,
                    "area_floor_m2": AREA_FLOOR_M2,
                    "model_a": first,
                    "model_b": second,
                    "objects_a": len(a_components),
                    "objects_b": len(b_components),
                    "area_a_m2": f"{area_a:.3f}",
                    "area_b_m2": f"{area_b:.3f}",
                    "intersection_m2_approx": f"{intersection:.3f}",
                    "mask_iou_approx": f"{iou:.4f}",
                    "mask_dice_approx": f"{dice:.4f}",
                    "overlap_grid_resolution_m": RASTER_RESOLUTION_M,
                })
    path = OUT / "round4_naip_mask_overlap.csv"
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def write_cross_scale_comparison() -> tuple[list[dict], list[dict]]:
    scale_rows = []
    for model in MODELS:
        native = components(model, "0p6m")
        resampled = components(model, "0p5m")
        intersection = sum(intersection_area(a, b) for a in native for b in resampled)
        area_native = sum(item["area_m2"] for item in native)
        area_resampled = sum(item["area_m2"] for item in resampled)
        union = area_native + area_resampled - intersection
        iou = intersection / union if union else 1.0
        dice = 2 * intersection / (area_native + area_resampled) if area_native + area_resampled else 1.0
        scale_rows.append({
            "model": model,
            "native_gsd_m": 0.6,
            "resampled_gsd_m": 0.5,
            "area_floor_m2": AREA_FLOOR_M2,
            "objects_0p6m": len(native),
            "objects_0p5m": len(resampled),
            "area_0p6m_m2": f"{area_native:.3f}",
            "area_0p5m_m2": f"{area_resampled:.3f}",
            "intersection_m2_approx": f"{intersection:.3f}",
            "cross_scale_iou_approx": f"{iou:.4f}",
            "cross_scale_dice_approx": f"{dice:.4f}",
            "overlap_grid_resolution_m": RASTER_RESOLUTION_M,
        })

    original_main = max(components("Original", "0p6m"), key=lambda item: item["pixel_count"])
    target_rows = []
    for model in MODELS:
        native = components(model, "0p6m")
        target = max(native, key=lambda item: intersection_area(original_main, item))
        candidates = [(item, intersection_area(target, item))
                      for item in components(model, "0p5m")]
        covered = sum(area for _, area in candidates)
        best, best_intersection = max(candidates, key=lambda item: item[1]) if candidates else (None, 0.0)
        coverage = covered / target["area_m2"] if target["area_m2"] else 0.0
        target_rows.append({
            "model": model,
            "matched_0p6_object_id": target["object_id"],
            "matched_0p6_pixels": target["pixel_count"],
            "0p6_area_m2": f"{target['area_m2']:.3f}",
            "best_0p5_object_id": best["object_id"] if best else "",
            "best_0p5_pixels": best["pixel_count"] if best else 0,
            "best_component_intersection_m2_approx": f"{best_intersection:.3f}",
            "0p6_target_covered_by_0p5_mask_pct": f"{100*coverage:.1f}",
            "detected_at_30pct_overlap_proxy": coverage >= 0.30,
        })

    out_path = OUT / "round4_naip_cross_scale_overlap.csv"
    with out_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(scale_rows[0]))
        writer.writeheader()
        writer.writerows(scale_rows)
    target_path = OUT / "round4_naip_main_component_resampling.csv"
    with target_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)
    return scale_rows, target_rows


def main() -> None:
    component_rows = write_pairwise_component_comparison()
    mask_rows = write_mask_comparison()
    cross_scale_rows, target_rows = write_cross_scale_comparison()
    print("0.6 m Original vs ROBUST-v2 s42 component IoU / centroid distance (5 m² floor):")
    for row in component_rows:
        if row["best_overlap_for_original"]:
            print(
                f"Original #{row['original_object_id']} ({row['original_pixels']} px) ↔ "
                f"ROBUST #{row['robust_object_id']} ({row['robust_pixels']} px): "
                f"IoU {row['intersection_over_union']}, "
                f"centroid distance {row['centroid_distance_m']} m"
            )
    print("Pairwise mask agreement (components ≥5 m²; vector masks rasterized at 0.05 m):")
    for row in mask_rows:
        print(
            f"{row['scale']} {row['model_a']} vs {row['model_b']}: "
            f"IoU {row['mask_iou_approx']}, Dice {row['mask_dice_approx']}"
        )
    print("Same-model native-to-resampled overlap (5 m² floor):")
    for row in cross_scale_rows:
        print(f"{row['model']}: IoU {row['cross_scale_iou_approx']}, Dice {row['cross_scale_dice_approx']}")
    print("Original 425 px main component, matched to each model's 0.6 m mask then compared with its 0.5 m mask:")
    for row in target_rows:
        print(
            f"{row['model']}: target {row['matched_0p6_pixels']} px, "
            f"0.5 m coverage {row['0p6_target_covered_by_0p5_mask_pct']}%, "
            f"best component {row['best_0p5_pixels']} px, "
            f"≥30% overlap {row['detected_at_30pct_overlap_proxy']}"
        )
    print("Wrote outputs/naip_demo/round4_original_vs_robust_0p6_component_pairs.csv")
    print("Wrote outputs/naip_demo/round4_naip_mask_overlap.csv")
    print("Wrote outputs/naip_demo/round4_naip_cross_scale_overlap.csv")
    print("Wrote outputs/naip_demo/round4_naip_main_component_resampling.csv")


if __name__ == "__main__":
    main()
