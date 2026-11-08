"""Rank small NAIP date pairs by a model-independent vegetation/bare-to-built rule."""
from __future__ import annotations

import itertools
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer as pc
import rasterio
from PIL import Image, ImageDraw, ImageFont
from pystac_client import Client
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform as warp_transform

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/naip_demo"
SIZE = 1024
GSD = 0.6
YEARS = ((2020, 2022), (2022, 2024))
TOWNS = {
    "Celina": (33.324, -96.784),
    "Prosper": (33.236, -96.802),
    "Aubrey": (33.304, -96.986),
    "Princeton": (33.181, -96.498),
    "Forney": (32.751, -96.471),
    "Royse City": (32.975, -96.332),
}
OFFSETS = ((0.0, 0.0), (0.017, 0.017), (-0.017, -0.017))


def _ring_covers(ring, lon: float, lat: float) -> bool:
    inside = False
    for p0, p1 in zip(ring, ring[1:]):
        x0, y0 = p0[:2]; x1, y1 = p1[:2]
        cross = (lon-x0)*(y1-y0) - (lat-y0)*(x1-x0)
        if abs(cross) < 1e-10 and min(x0,x1)-1e-10 <= lon <= max(x0,x1)+1e-10 and min(y0,y1)-1e-10 <= lat <= max(y0,y1)+1e-10:
            return True
        if (y0 > lat) != (y1 > lat):
            crossing_lon = x0 + (lat-y0)*(x1-x0)/(y1-y0)
            if lon < crossing_lon:
                inside = not inside
    return inside


def geometry_covers(geometry: dict, lon: float, lat: float) -> bool:
    """Point-in-polygon check for STAC GeoJSON without an extra geometry package."""
    kind, coords = geometry.get("type"), geometry.get("coordinates", [])
    polygons = [coords] if kind == "Polygon" else coords if kind == "MultiPolygon" else []
    for polygon in polygons:
        if polygon and _ring_covers(polygon[0], lon, lat) and not any(_ring_covers(hole, lon, lat) for hole in polygon[1:]):
            return True
    return False


def item_month(item) -> int:
    return item.datetime.month if item.datetime else 7


def gsds(item) -> float:
    try:
        return float(item.properties.get("gsd", 0.6))
    except (TypeError, ValueError):
        return 0.6


def rank_rule(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Fraction of valid pixels changing from green/bare to bright, neutral surfaces."""
    av = np.moveaxis(a, 0, -1).astype(np.float32)
    bv = np.moveaxis(b, 0, -1).astype(np.float32)
    valid = np.all(av > 0, axis=2) & np.all(bv > 0, axis=2)
    if valid.mean() < 0.95:
        return 0.0, float(valid.mean())
    max_a = np.maximum(av.max(axis=2), 1.0)
    max_b = np.maximum(bv.max(axis=2), 1.0)
    sat_a = (av.max(axis=2) - av.min(axis=2)) / max_a
    sat_b = (bv.max(axis=2) - bv.min(axis=2)) / max_b
    lum_a = av.mean(axis=2)
    lum_b = bv.mean(axis=2)
    vegetation_a = (av[:, :, 1] > 1.08 * av[:, :, 0]) & (av[:, :, 1] > 1.03 * av[:, :, 2]) & (lum_a > 45)
    bare_a = (sat_a >= 0.12) & (sat_a <= 0.48) & (lum_a >= 55) & (lum_a <= 205)
    bright_neutral_b = (lum_b >= 145) & (sat_b <= 0.24)
    transition = (vegetation_a | bare_a) & bright_neutral_b & (lum_b - lum_a >= 20) & (sat_a - sat_b >= 0.04)
    return float(transition[valid].mean()), float(valid.mean())


def read_small_window(item, lat: float, lon: float) -> np.ndarray:
    x, y = (coords[0] for coords in warp_transform("EPSG:4326", "EPSG:32614", [lon], [lat]))
    resolution = GSD
    transform = from_origin(x - SIZE * resolution / 2, y + SIZE * resolution / 2, resolution, resolution)
    signed = pc.sign(item)
    with rasterio.open(signed.assets["image"].href) as src:
        with WarpedVRT(src, crs="EPSG:32614", transform=transform, width=SIZE, height=SIZE,
                       resampling=Resampling.bilinear, src_nodata=0, nodata=0) as vrt:
            return vrt.read((1, 2, 3))


def image_thumbnail(chw: np.ndarray, size: int = 224) -> Image.Image:
    hwc = np.moveaxis(chw, 0, -1).astype(np.uint8)
    return Image.fromarray(hwc, "RGB").resize((size, size), Image.Resampling.BILINEAR)


def item_record(item) -> dict:
    dt = item.datetime.isoformat() if item.datetime else "unknown"
    return {"id": item.id, "datetime": dt, "year": item.datetime.year if item.datetime else None,
            "month": item_month(item), "gsd_m": gsds(item)}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    api = Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
    year_items: dict[tuple[str, int], list] = {}
    for town, (lat, lon) in TOWNS.items():
        bbox = [lon - 0.075, lat - 0.060, lon + 0.075, lat + 0.060]
        for year in sorted({y for pair in YEARS for y in pair}):
            search = api.search(collections=["naip"], bbox=bbox,
                                datetime=f"{year}-01-01T00:00:00Z/{year}-12-31T23:59:59Z", limit=100)
            year_items[(town, year)] = list(search.items())
            print(f"STAC {town} {year}: {len(year_items[(town, year)])} item(s)", flush=True)

    candidates = []
    for town, (lat0, lon0) in TOWNS.items():
        for pi, (dy, dx) in enumerate(OFFSETS):
            lat, lon = lat0 + dy, lon0 + dx
            for y0, y1 in YEARS:
                before_items = [x for x in year_items[(town, y0)] if geometry_covers(x.geometry, lon, lat)]
                after_items = [x for x in year_items[(town, y1)] if geometry_covers(x.geometry, lon, lat)]
                if not before_items or not after_items:
                    continue
                combinations = sorted(itertools.product(before_items, after_items),
                    key=lambda pair: (abs(item_month(pair[0])-item_month(pair[1])), gsds(pair[0])+gsds(pair[1])))
                before_item, after_item = combinations[0]
                try:
                    a = read_small_window(before_item, lat, lon)
                    b = read_small_window(after_item, lat, lon)
                    score, valid_fraction = rank_rule(a, b)
                except Exception as exc:
                    print(f"skip {town} point{pi} {y0}-{y1}: {type(exc).__name__}: {exc}", flush=True)
                    continue
                season_match = abs(item_month(before_item)-item_month(after_item)) <= 2
                pair_id = f"{town.lower().replace(' ', '-')}_{y0}-{y1}_p{pi}"
                candidates.append({"window_id": pair_id, "pair_years": [y0, y1], "town": town, "point_index": pi, "center_lat": lat, "center_lon": lon,
                    "score_transition_fraction": score, "valid_fraction": valid_fraction,
                    "season_month_difference": abs(item_month(before_item)-item_month(after_item)),
                    "same_season_pair": season_match, "before": item_record(before_item),
                    "after": item_record(after_item), "a_rgb": a, "b_rgb": b})
                print(f"window {town} {y0}-{y1} score={score:.5f} valid={valid_fraction:.3f} same_season={season_match}", flush=True)

    candidates.sort(key=lambda c: c["score_transition_fraction"], reverse=True)
    top = candidates[:12]
    if not top:
        raise RuntimeError("No candidate windows had overlapping NAIP coverage for the requested years.")
    cols, card_w, card_h = 4, 510, 330
    rows = math.ceil(len(top) / cols)
    sheet = Image.new("RGB", (cols * card_w, rows * card_h), "white")
    draw = ImageDraw.Draw(sheet)
    font_path = Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold_path = Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    font = lambda size, bold=False: ImageFont.truetype(str(bold_path if bold else font_path), size) if (bold_path if bold else font_path).exists() else ImageFont.load_default()
    public_rows = []
    for i, c in enumerate(top, 1):
        col, row = (i-1) % cols, (i-1) // cols
        x, y = col*card_w, row*card_h
        a, b = image_thumbnail(c["a_rgb"]), image_thumbnail(c["b_rgb"])
        draw.text((x+10, y+5), f"#{i} {c['window_id']} · {c['town']} · score {c['score_transition_fraction']:.4f}" + (" · same season" if c["same_season_pair"] else ""), fill="#111", font=font(15, True))
        sheet.paste(a, (x+10, y+34)); sheet.paste(b, (x+244, y+34))
        draw.text((x+10, y+261), f"A {c['before']['datetime'][:10]} | B {c['after']['datetime'][:10]}", fill="#222", font=font(14))
        draw.text((x+10, y+280), f"{c['center_lat']:.5f}, {c['center_lon']:.5f} · GSD {c['before']['gsd_m']:.1f}/{c['after']['gsd_m']:.1f} m", fill="#333", font=font(13))
        draw.text((x+10, y+300), f"{c['before']['id']} → {c['after']['id']}", fill="#555", font=font(10))
        public_rows.append({k: v for k, v in c.items() if k not in ("a_rgb", "b_rgb")})
    out_png = OUT / "candidates_contact_sheet.png"
    sheet.save(out_png)
    meta = {"rule": "score = fraction of valid aligned pixels changing from green vegetation or moderately saturated bare soil in A to bright (mean RGB >=145), low-saturation (<=0.24) surfaces in B, with brightness increase >=20 and saturation decrease >=0.04; RGB only; no TinyCD/model scores used.",
            "window_px": [SIZE, SIZE], "target_gsd_m": GSD, "ranking": "descending score_transition_fraction", "candidate_count": len(candidates), "shown_count": len(top),
            "source": "USDA NAIP via Microsoft Planetary Computer STAC; public domain", "same_season_rule": "acquisition month difference <=2 months",
            "candidates": public_rows, "contact_sheet": str(out_png.relative_to(ROOT))}
    (OUT / "candidates_contact_sheet.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(json.dumps({"candidates": len(candidates), "shown": len(top), "contact_sheet": str(out_png),
                      "same_season_count": sum(c["same_season_pair"] for c in top)}, indent=2), flush=True)


if __name__ == "__main__":
    main()
