"""Read only a 2048px RGB window from a public USDA NAIP STAC COG pair."""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import planetary_computer as pc
import rasterio
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.warp import reproject, transform as transform_xy
from rasterio.windows import Window

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "data/demo_naip/celina_town_2020_2022"
SIZE = 2048
PAIR = {
    "before": "tx_m_3309642_ne_14_060_20201102",
    "after": "tx_m_3309642_ne_14_060_20220704",
}
CENTER_LONLAT = (-96.784, 33.324)


def read_window(item, lon: float, lat: float):
    with rasterio.open(item.assets["image"].href) as ds:
        x, y = transform_xy("EPSG:4326", ds.crs, [lon], [lat])
        col, row = ds.index(x[0], y[0])
        window = Window(col - SIZE // 2, row - SIZE // 2, SIZE, SIZE)
        rgb = ds.read((1, 2, 3), window=window, boundless=True, fill_value=0)
        return rgb, ds.window_transform(window), ds.crs, ds.res, ds.bounds


def main():
    if DEST.exists():
        raise FileExistsError(f"Refusing to overwrite existing local demo data: {DEST}")
    api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    items = {}
    for time_name, item_id in PAIR.items():
        found = list(api.search(collections=["naip"], ids=[item_id]).items())
        if len(found) != 1:
            raise RuntimeError(f"Expected one STAC item for {item_id}, found {len(found)}")
        items[time_name] = pc.sign(found[0])

    lon, lat = CENTER_LONLAT
    before, out_transform, out_crs, res, bounds = read_window(items["before"], lon, lat)
    after_src, after_transform, after_crs, _, _ = read_window(items["after"], lon, lat)
    after = np.zeros_like(before)
    for band in range(3):
        reproject(
            source=after_src[band], destination=after[band],
            src_transform=after_transform, src_crs=after_crs,
            dst_transform=out_transform, dst_crs=out_crs,
            resampling=Resampling.bilinear, src_nodata=0, dst_nodata=0,
        )

    gray_a = cv2.cvtColor(before.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray_b = cv2.cvtColor(after.transpose(1, 2, 0), cv2.COLOR_RGB2GRAY).astype(np.float32)
    (dx, dy), response = cv2.phaseCorrelate(gray_a, gray_b)
    aligned = False
    if np.hypot(dx, dy) > 1.0:
        matrix = np.float32([[1, 0, -dx], [0, 1, -dy]])
        hwc = after.transpose(1, 2, 0)
        after = cv2.warpAffine(hwc, matrix, (SIZE, SIZE), borderMode=cv2.BORDER_REFLECT_101).transpose(2, 0, 1)
        aligned = True

    DEST.mkdir(parents=True, exist_ok=False)
    profile = {
        "driver": "GTiff", "width": SIZE, "height": SIZE, "count": 3,
        "dtype": "uint8", "crs": out_crs, "transform": out_transform,
        "compress": "deflate", "predictor": 2,
    }
    for label, data in (("A_before_2020.tif", before), ("B_after_2022.tif", after)):
        with rasterio.open(DEST / label, "w", **profile) as dst:
            dst.write(data)
            dst.colorinterp = (rasterio.enums.ColorInterp.red, rasterio.enums.ColorInterp.green, rasterio.enums.ColorInterp.blue)
    metadata = {
        "qualitative_only": True, "ground_truth": False,
        "source": "USDA NAIP via Microsoft Planetary Computer STAC",
        "collection": "naip", "items": {k: {"id": v.id, "datetime": v.datetime.isoformat(), "gsd_m": v.properties.get("gsd"), "asset_url": v.assets["image"].href.split("?")[0]} for k, v in items.items()},
        "center_lonlat": [lon, lat], "window_px": [SIZE, SIZE],
        "crs": str(out_crs), "gsd_m": float(res[0]), "source_tile_bounds": [float(x) for x in bounds],
        "phase_correlate_before_alignment_dx_dy_px": [float(dx), float(dy)],
        "phase_response": float(response), "aligned_by_translation": aligned,
        "alignment_translation_applied_px": [float(-dx), float(-dy)] if aligned else [0.0, 0.0],
        "license_note": "USDA NAIP imagery is public domain; see USDA NAIP metadata and per-item STAC metadata.",
    }
    (DEST / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"destination": str(DEST), "metadata": metadata}, indent=2), flush=True)


if __name__ == "__main__":
    main()
