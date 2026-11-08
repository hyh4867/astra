"""Crop an all-valid common rectangle from the already-downloaded NAIP pair."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/demo_naip/celina_town_2020_2022"
DEST = ROOT / "data/demo_naip/celina_town_2020_2022_overlap"


def largest_true_rectangle(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return (row0, col0, row1, col1) for the largest all-true rectangle."""
    height, width = mask.shape
    heights = np.zeros(width, dtype=np.int32)
    best_area = 0
    best = (0, 0, height, width)
    for row in range(height):
        heights = np.where(mask[row], heights + 1, 0)
        stack: list[tuple[int, int]] = []
        for col in range(width + 1):
            current = int(heights[col]) if col < width else 0
            start = col
            while stack and stack[-1][1] > current:
                left, bar_height = stack.pop()
                area = bar_height * (col - left)
                if area > best_area:
                    best_area = area
                    best = (row - bar_height + 1, left, row + 1, col)
                start = left
            if current and (not stack or stack[-1][1] < current):
                stack.append((start, current))
    if best_area == 0:
        raise ValueError("The NAIP pair has no common valid pixels.")
    return best


def main() -> None:
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite local output: {DEST}")
    a_path = SOURCE / "A_before_2020.tif"
    b_path = SOURCE / "B_after_2022.tif"
    with rasterio.open(a_path) as src_a, rasterio.open(b_path) as src_b:
        a = src_a.read((1, 2, 3))
        b = src_b.read((1, 2, 3))
        if a.shape != b.shape or src_a.transform != src_b.transform or src_a.crs != src_b.crs:
            raise ValueError("The saved NAIP windows do not share a grid.")
        valid_a = np.any(a != 0, axis=0)
        valid_b = np.any(b != 0, axis=0)
        r0, c0, r1, c1 = largest_true_rectangle(valid_a & valid_b)
        a = a[:, r0:r1, c0:c1]
        b = b[:, r0:r1, c0:c1]
        transform = src_a.window_transform(Window(c0, r0, c1 - c0, r1 - r0))
        profile = src_a.profile.copy()

    rgb_a = a.transpose(1, 2, 0)
    rgb_b = b.transpose(1, 2, 0)
    gray_a = cv2.cvtColor(rgb_a, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(rgb_b, cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b)
    translated = False
    if np.hypot(dx, dy) > 1.0:
        matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        rgb_b = cv2.warpAffine(
            rgb_b, matrix, (rgb_b.shape[1], rgb_b.shape[0]),
            borderMode=cv2.BORDER_REFLECT_101,
        )
        b = rgb_b.transpose(2, 0, 1)
        translated = True

    DEST.mkdir(parents=True)
    profile.update(width=c1 - c0, height=r1 - r0, count=3, transform=transform)
    for name, data in (("A_before_2020.tif", a), ("B_after_2022.tif", b)):
        with rasterio.open(DEST / name, "w", **profile) as dst:
            dst.write(data)
            dst.colorinterp = (
                rasterio.enums.ColorInterp.red,
                rasterio.enums.ColorInterp.green,
                rasterio.enums.ColorInterp.blue,
            )
    metadata = json.loads((SOURCE / "metadata.json").read_text())
    metadata.update(
        source_pair=str(SOURCE.relative_to(ROOT)),
        source_window_px=[2048, 2048],
        window_px=[c1 - c0, r1 - r0],
        common_valid_crop_px=[c0, r0, c1, r1],
        phase_correlate_before_alignment_dx_dy_px=[float(dx), float(dy)],
        phase_response=float(response),
        aligned_by_translation=translated,
        alignment_translation_applied_px=[float(-dx), float(-dy)] if translated else [0.0, 0.0],
        crop_rule="largest rectangle where all RGB channels are not simultaneously zero in both dates",
    )
    (DEST / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"destination": str(DEST), "metadata": metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
