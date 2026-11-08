"""Inference, visualization and image-quality helpers for the local ASTRA app."""
from __future__ import annotations

import os
import sys
from math import ceil
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent
TILE_SIZE = 256
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)[None, :, None, None]
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)[None, :, None, None]


def pad_to_tile_multiple(image: np.ndarray, tile_size: int = TILE_SIZE):
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Expected an H×W×3 RGB image")
    h, w = image.shape[:2]
    pad_h = (-h) % tile_size
    pad_w = (-w) % tile_size
    mode = cv2.BORDER_REFLECT_101 if h > 1 and w > 1 else cv2.BORDER_REPLICATE
    padded = cv2.copyMakeBorder(image, 0, pad_h, 0, pad_w, mode)
    return padded, (h, w)


def unpad(image: np.ndarray, original_shape: tuple[int, int]):
    h, w = original_shape
    return image[:h, :w, ...]


def iter_tiles(image: np.ndarray, tile_size: int = TILE_SIZE):
    h, w = image.shape[:2]
    if h % tile_size or w % tile_size:
        raise ValueError("Image must be padded to tile multiples first")
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            yield y, x, image[y:y+tile_size, x:x+tile_size]


def prepare_pair(before: np.ndarray, after: np.ndarray, tile_size: int = TILE_SIZE):
    if before.shape != after.shape:
        raise ValueError("Input images must have identical dimensions")
    a, shape = pad_to_tile_multiple(before, tile_size)
    b, _ = pad_to_tile_multiple(after, tile_size)
    return a, b, shape


def tinycd_probability(model, before: np.ndarray, after: np.ndarray, device, batch_size: int = 16):
    """Predict tiled probabilities, padding at bottom/right and cropping back."""
    import torch

    a, b, original_shape = prepare_pair(before, after)
    tile_coords = [(y, x) for y, x, _ in iter_tiles(a)]
    out = np.zeros(a.shape[:2], dtype=np.float32)
    mean = torch.from_numpy(IMAGENET_MEAN).to(device)
    std = torch.from_numpy(IMAGENET_STD).to(device)
    with torch.inference_mode():
        for start in range(0, len(tile_coords), batch_size):
            group = tile_coords[start:start+batch_size]
            ta = np.stack([a[y:y+TILE_SIZE, x:x+TILE_SIZE] for y, x in group])
            tb = np.stack([b[y:y+TILE_SIZE, x:x+TILE_SIZE] for y, x in group])
            ta = torch.from_numpy(ta).to(device).permute(0, 3, 1, 2).float().div_(255)
            tb = torch.from_numpy(tb).to(device).permute(0, 3, 1, 2).float().div_(255)
            ta = (ta - mean) / std
            tb = (tb - mean) / std
            probabilities = model(ta, tb)[:, 0].float().cpu().numpy()
            for (y, x), tile in zip(group, probabilities):
                out[y:y+TILE_SIZE, x:x+TILE_SIZE] = tile
    return unpad(out, original_shape)


def opencv_predictions(before: np.ndarray, after: np.ndarray):
    if before.shape != after.shape:
        raise ValueError("Input images must have identical dimensions")
    gray_a = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY)
    gray_b = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY)
    basic_strength = cv2.absdiff(gray_a, gray_b)
    basic = basic_strength >= 105

    blur_a = cv2.GaussianBlur(gray_a, (3, 3), 0)
    blur_b = cv2.GaussianBlur(gray_b, (3, 3), 0)
    improved_strength = cv2.absdiff(blur_a, blur_b)
    _, improved = cv2.threshold(improved_strength, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    improved = cv2.morphologyEx(improved, cv2.MORPH_OPEN, kernel)
    improved = cv2.morphologyEx(improved, cv2.MORPH_CLOSE, kernel)
    return {
        "OpenCV basic": (basic, basic_strength.astype(np.float32) / 255.0),
        "OpenCV improved": (improved > 0, improved_strength.astype(np.float32) / 255.0),
    }


def quality_checks(before: np.ndarray, after: np.ndarray, gsd_m: float):
    if before.shape != after.shape:
        return {"shift_px": None, "shift_xy_px": None, "sharpness_a": None,
            "sharpness_b": None, "sharpness_ratio": None, "warnings": ["A/B dimensions differ; registration and sharpness checks are unavailable."]}
    ga = cv2.cvtColor(before, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gb = cv2.cvtColor(after, cv2.COLOR_RGB2GRAY).astype(np.float32)
    try:
        (dx, dy), response = cv2.phaseCorrelate(ga, gb)
        shift = float(np.hypot(dx, dy)) if np.isfinite(dx) and np.isfinite(dy) else None
        xy = (float(dx), float(dy)) if shift is not None else None
    except cv2.error:
        shift, xy, response = None, None, 0.0
    sharp_a = float(cv2.Laplacian(ga, cv2.CV_32F).var())
    sharp_b = float(cv2.Laplacian(gb, cv2.CV_32F).var())
    ratio = max(sharp_a, sharp_b) / max(min(sharp_a, sharp_b), 1e-6)
    warnings = []
    if shift is not None and shift > 1.0:
        warnings.append(f"Estimated image registration shift is {shift:.1f}px (warning threshold >1px). Interpret results with caution.")
    if ratio >= 2.0:
        warnings.append(f"A/B sharpness differs substantially (Laplacian variance ratio {ratio:.1f}×; threshold 2×).")
    if abs(gsd_m - 0.5) / 0.5 > 0.20001:
        warnings.append(f"GSD differs from the 0.5 m/px training reference by more than 20% (input {gsd_m:g} m/px).")
    warnings.append(f"Estimated isolated-object detection limit ≈ {209*gsd_m**2:.1f} m² (assumes a pixel-fixed limit; validated only at 0.5 m).")
    return {"shift_px": shift, "shift_xy_px": xy, "phase_response": float(response),
        "sharpness_a": sharp_a, "sharpness_b": sharp_b,
        "sharpness_ratio": float(ratio), "warnings": warnings}


def make_overlay(after: np.ndarray, mask: np.ndarray, color=(255, 0, 0)):
    out = after.copy()
    selected = mask.astype(bool)
    color_array=np.asarray(color,dtype=np.float32).reshape(1,3)
    out[selected] = (0.55*out[selected] + 0.45*color_array).astype(np.uint8)
    return out


def probability_heatmap(probability: np.ndarray):
    values = (np.clip(probability, 0, 1)*255).astype(np.uint8)
    return cv2.cvtColor(cv2.applyColorMap(values, cv2.COLORMAP_INFERNO), cv2.COLOR_BGR2RGB)


def changed_object_summary(mask: np.ndarray, gsd_m: float):
    count, _, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas_px = stats[1:, cv2.CC_STAT_AREA]
    areas_m2 = areas_px.astype(np.float64) * gsd_m**2
    return {"count": int(count-1), "changed_pixels": int(mask.sum()),
        "total_area_m2": float(mask.sum()*gsd_m**2),
        "median_object_area_m2": float(np.median(areas_m2)) if len(areas_m2) else 0.0}


def filter_small_components(mask: np.ndarray, gsd_m: float,
                            min_area_m2: float = 5.0):
    """Remove 8-connected components below a ground-area threshold.

    The area threshold is user-facing in m². The corresponding integer pixel
    cutoff is derived from the input GSD and uses ceil so retained objects meet
    or exceed the requested ground area.
    """
    if not np.isfinite(gsd_m) or gsd_m <= 0:
        raise ValueError("GSD must be a finite positive number")
    if not np.isfinite(min_area_m2) or min_area_m2 < 0:
        raise ValueError("Minimum area must be a finite nonnegative number")

    min_area_m2 = float(min_area_m2)
    min_pixels = (0 if min_area_m2 == 0 else
                  max(1, ceil(min_area_m2 / (gsd_m**2) - 1e-12)))
    binary = np.asarray(mask, dtype=bool)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary.astype(np.uint8), connectivity=8
    )
    areas_px = stats[1:, cv2.CC_STAT_AREA]
    keep_labels = np.flatnonzero(areas_px >= min_pixels) + 1
    filtered = np.isin(labels, keep_labels) if keep_labels.size else np.zeros_like(binary)
    raw_count = int(count - 1)
    kept_count = int(keep_labels.size)
    return filtered, {
        "raw_count": raw_count,
        "kept_count": kept_count,
        "removed_sub_floor_components": raw_count - kept_count,
        "raw_pixels": int(binary.sum()),
        "kept_pixels": int(filtered.sum()),
        "minimum_area_m2": min_area_m2,
        "minimum_pixels_at_gsd": int(min_pixels),
    }


def load_tinycd_model(weights_path: Path | None = None):
    import torch

    weights_path = weights_path or (ROOT/"models"/"tinycd_levir_best.pth")
    if not weights_path.is_file():
        raise FileNotFoundError(f"TinyCD weights not found: {weights_path}")
    sys.path.insert(0, str(ROOT/"vendor"/"tinycd"))
    from models.change_classifier import ChangeClassifier
    force_cpu = os.environ.get("ASTRA_FORCE_CPU") == "1"
    device = torch.device("cpu")
    if not force_cpu and torch.backends.mps.is_available():
        try:
            probe = torch.ones((2, 2), device="mps")
            _ = probe @ probe
            torch.mps.synchronize()
            device = torch.device("mps")
        except Exception as exc:
            print(f"MPS is reported available but the matmul probe failed; using CPU: {exc}", flush=True)
    model = ChangeClassifier(pretrained=False).eval().to(device)
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    return model, device
