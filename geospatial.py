"""GeoTIFF decoding, reprojection, object vectors, and local map helpers."""
from __future__ import annotations

import base64
from io import BytesIO
import json
from math import ceil, hypot

import cv2
import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp, Resampling
from rasterio.features import shapes
from rasterio.io import MemoryFile
from rasterio.warp import reproject, transform, transform_geom


def _to_uint8(bands: np.ndarray, scales, offsets) -> np.ndarray:
    values = bands.astype(np.float32)
    for i in range(values.shape[0]):
        values[i] = values[i] * float(scales[i] or 1.0) + float(offsets[i] or 0.0)
    if bands.dtype == np.uint8 and all(float(s or 1.0) == 1.0 for s in scales[:3]):
        return bands.astype(np.uint8).transpose(1, 2, 0)
    output = []
    for band in values:
        valid = band[np.isfinite(band)]
        if valid.size == 0:
            output.append(np.zeros(band.shape, np.uint8)); continue
        low, high = np.percentile(valid, [2, 98])
        if high <= low:
            high = low + 1
        output.append(np.clip((band - low) * (255.0 / (high - low)), 0, 255).astype(np.uint8))
    return np.stack(output, axis=-1)


def _utm_epsg(lon: float, lat: float) -> int:
    zone = min(60, max(1, int((lon + 180) // 6) + 1))
    return (32600 if lat >= 0 else 32700) + zone


def _pixel_area_and_gsd(transform_affine, crs, width, height):
    col, row = width / 2, height / 2
    p0 = transform_affine @ (col, row)
    p1 = transform_affine @ (col + 1, row)
    p2 = transform_affine @ (col, row + 1)
    lon, lat = transform(crs, "EPSG:4326", [p0[0]], [p0[1]])
    utm = f"EPSG:{_utm_epsg(lon[0], lat[0])}"
    xs, ys = transform(crs, utm, [p0[0], p1[0], p2[0]], [p0[1], p1[1], p2[1]])
    vx=(xs[1]-xs[0], ys[1]-ys[0]); vy=(xs[2]-xs[0], ys[2]-ys[0])
    pixel_area=abs(vx[0]*vy[1]-vx[1]*vy[0])
    return pixel_area, (hypot(*vx) + hypot(*vy)) / 2, utm


def read_image_bytes(data: bytes, filename: str):
    """Return RGB uint8 and optional geospatial metadata from an upload."""
    if filename.lower().endswith((".tif", ".tiff")):
        with MemoryFile(data) as memfile, memfile.open() as src:
            indexes = [1, 2, 3]
            interp = list(src.colorinterp)
            if all(kind in interp for kind in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)):
                indexes = [interp.index(kind) + 1 for kind in (ColorInterp.red, ColorInterp.green, ColorInterp.blue)]
            if src.count < max(indexes):
                indexes = [1, 1, 1]
            rgb = _to_uint8(src.read(indexes), [src.scales[i-1] for i in indexes], [src.offsets[i-1] for i in indexes])
            geo = None
            if src.crs:
                pixel_area, gsd, utm = _pixel_area_and_gsd(src.transform, src.crs, src.width, src.height)
                geo = {"transform": src.transform, "crs": src.crs, "gsd_m": gsd, "pixel_area_m2": pixel_area, "utm_crs": utm}
            return rgb, geo
    image = Image.open(BytesIO(data)).convert("RGB")
    return np.asarray(image), None


def align_pair(before: np.ndarray, geo_a, after: np.ndarray, geo_b):
    """Reproject B to A's grid when both uploads carry spatial metadata."""
    if geo_a is None and geo_b is None:
        if before.shape != after.shape:
            raise ValueError("비지오참조 이미지의 크기가 다릅니다. 같은 크기로 맞춰 업로드하세요.")
        return before, after, None, None
    if geo_a is None or geo_b is None:
        raise ValueError("A와 B 모두 지오참조 GeoTIFF이거나, 둘 다 일반 이미지여야 합니다.")
    if before.shape[:2] != after.shape[:2] or geo_a["crs"] != geo_b["crs"] or geo_a["transform"] != geo_b["transform"]:
        dst = np.zeros((before.shape[0], before.shape[1], 3), dtype=np.uint8)
        for band in range(3):
            reproject(
                source=after[:, :, band], destination=dst[:, :, band],
                src_transform=geo_b["transform"], src_crs=geo_b["crs"],
                dst_transform=geo_a["transform"], dst_crs=geo_a["crs"],
                resampling=Resampling.bilinear,
            )
        after = dst
    if not np.any(after):
        raise ValueError("A/B GeoTIFF의 공간 범위가 겹치지 않습니다.")
    return before, after, geo_a, geo_a["gsd_m"]


def _ring_area(ring):
    return abs(sum(ring[i][0]*ring[(i+1)%len(ring)][1] - ring[(i+1)%len(ring)][0]*ring[i][1] for i in range(len(ring))) / 2)


def _geometry_area(geometry):
    kind=geometry["type"]; coords=geometry["coordinates"]
    if kind == "Polygon":
        return max(0.0, _ring_area(coords[0]) - sum(_ring_area(r) for r in coords[1:]))
    if kind == "MultiPolygon":
        return sum(_geometry_area({"type":"Polygon","coordinates":p}) for p in coords)
    return 0.0


def object_features(mask: np.ndarray, probability: np.ndarray, geo, gsd_m: float):
    """Polygonize 8-connected predicted objects; area uses a local UTM CRS."""
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    rows=[]; features=[]
    for geometry, raw_id in shapes(labels.astype(np.int32), mask=labels > 0,
                                   transform=geo["transform"] if geo else rasterio.Affine.identity(), connectivity=8):
        label_id=int(raw_id)
        if label_id <= 0 or label_id >= count:
            continue
        pixel_count=int(stats[label_id, cv2.CC_STAT_AREA])
        px,py=centroids[label_id]
        mean_prob=float(probability[labels == label_id].mean()) if probability is not None else 1.0
        if geo:
            geom_utm=transform_geom(geo["crs"], geo["utm_crs"], geometry, precision=-1)
            area=_geometry_area(geom_utm)
            geom_wgs84=transform_geom(geo["crs"], "EPSG:4326", geometry, precision=7)
            gx,gy=geo["transform"] @ (float(px)+0.5,float(py)+0.5)
            lon,lat=transform(geo["crs"], "EPSG:4326", [gx],[gy])
            centroid_x,centroid_y=float(lon[0]),float(lat[0])
            feature={"type":"Feature","geometry":geom_wgs84,"properties":{
                "object_id":label_id,"area_m2":area,"area_crs":geo["utm_crs"],
                "centroid_lon":centroid_x,"centroid_lat":centroid_y,
                "mean_probability":mean_prob,"pixel_count":pixel_count}}
        else:
            area=pixel_count*gsd_m**2
            centroid_x,centroid_y=float(px),float(py)
            feature=None
        rows.append({"object_id":label_id,"area_m2":float(area),
            "centroid_lon":centroid_x if geo else None,"centroid_lat":centroid_y if geo else None,
            "centroid_x_px":None if geo else centroid_x,"centroid_y_px":None if geo else centroid_y,
            "mean_probability":mean_prob,"pixel_count":pixel_count,
            "area_crs":geo["utm_crs"] if geo else "pixel_area × GSD²"})
        if feature:
            features.append(feature)
    rows.sort(key=lambda row: row["area_m2"], reverse=True)
    return rows, {"type":"FeatureCollection","features":features} if geo else None


def comparison_slider(before: np.ndarray, after: np.ndarray, height: int = 480):
    """Render a draggable A/B swipe comparison entirely in the local browser."""
    def data_url(image):
        buf=BytesIO(); Image.fromarray(image).save(buf, format="JPEG", quality=82)
        return "data:image/jpeg;base64,"+base64.b64encode(buf.getvalue()).decode("ascii")
    a_url,b_url=data_url(before),data_url(after)
    return f"""<!doctype html><html><head><meta name='viewport' content='width=device-width, initial-scale=1'>
<style>body{{margin:0;font:14px sans-serif;color:#334}}.wrap{{position:relative;width:100%;height:{height}px;overflow:hidden;background:#eee;border-radius:8px}}
.wrap img{{position:absolute;inset:0;width:100%;height:100%;object-fit:contain}}#after{{clip-path:inset(0 50% 0 0)}}
.line{{position:absolute;top:0;bottom:0;left:50%;width:3px;background:white;box-shadow:0 0 4px #333;pointer-events:none}}
input{{width:100%;margin:8px 0}}.labels{{display:flex;justify-content:space-between}}</style></head><body>
<div class='wrap'><img src='{a_url}' alt='A before'><img id='after' src='{b_url}' alt='B after'><div id='line' class='line'></div></div>
<input id='slider' type='range' min='0' max='100' value='50' aria-label='before after comparison slider'>
<div class='labels'><span>A · BEFORE</span><span>B · AFTER</span></div>
<script>const s=document.getElementById('slider'),b=document.getElementById('after'),l=document.getElementById('line');s.oninput=()=>{{b.style.clipPath='inset(0 '+(100-s.value)+'% 0 0)';l.style.left=s.value+'%'}};</script></body></html>"""

