from io import BytesIO
from pathlib import Path
import sys

import numpy as np
import rasterio
from rasterio.enums import ColorInterp
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from geospatial import align_pair, object_features, read_image_bytes
from inference import filter_small_components
from scripts.prepare_naip_overlap import largest_true_rectangle


def make_geotiff():
    profile={"driver":"GTiff","width":8,"height":8,"count":3,"dtype":"uint8",
             "crs":"EPSG:26914","transform":from_origin(500000,3700000,0.6,0.6)}
    mem=MemoryFile()
    with mem.open(**profile) as ds:
        ds.write(np.full((3,8,8),100,dtype=np.uint8))
        ds.colorinterp=(ColorInterp.red,ColorInterp.green,ColorInterp.blue)
    return mem.read()


def test_read_geotiff_returns_rgb_and_gsd():
    rgb,geo=read_image_bytes(make_geotiff(),"pair.tif")
    assert rgb.shape==(8,8,3)
    assert geo["crs"].to_epsg()==26914
    assert abs(geo["gsd_m"]-0.6)<1e-6


def test_align_pair_preserves_matching_geotiff_grid():
    rgb,geo=read_image_bytes(make_geotiff(),"pair.tif")
    a,b,aligned,gsd=align_pair(rgb,geo,rgb.copy(),geo)
    np.testing.assert_array_equal(a,b)
    assert aligned["utm_crs"]=="EPSG:32614"
    assert abs(gsd-.6)<1e-6


def test_object_vectors_have_utm_area_and_wgs84_centroid():
    rgb,geo=read_image_bytes(make_geotiff(),"pair.tif")
    mask=np.zeros((8,8),bool);mask[2:6,2:6]=True;mask[0,0]=True
    mask,filter_info=filter_small_components(mask,.6,5.0)
    rows,collection=object_features(mask,np.full((8,8),.75,np.float32),geo,.6)
    assert filter_info["removed_sub_floor_components"]==1
    assert len(rows)==1 and len(collection["features"])==1
    assert rows[0]["pixel_count"]==16
    assert collection["features"][0]["properties"]["pixel_count"]==16
    assert abs(rows[0]["area_m2"]-16*.36)<1e-5
    assert rows[0]["area_crs"]=="EPSG:32614"
    assert -180<=rows[0]["centroid_lon"]<=180
    assert -90<=rows[0]["centroid_lat"]<=90
    assert rows[0]["mean_probability"]==.75


def test_largest_true_rectangle_excludes_zero_filled_border():
    valid=np.zeros((6,8),dtype=bool)
    valid[1:5,2:7]=True
    assert largest_true_rectangle(valid)==(1,2,5,7)
