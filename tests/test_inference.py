from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

from inference import filter_small_components, iter_tiles, pad_to_tile_multiple, quality_checks, unpad


def test_pad_tile_and_unpad_arbitrary_dimensions():
    image=np.random.default_rng(4).integers(0,256,(513,777,3),dtype=np.uint8)
    padded,shape=pad_to_tile_multiple(image)
    assert padded.shape==(768,1024,3)
    assert padded.shape[0]%256==padded.shape[1]%256==0
    assert len(list(iter_tiles(padded)))==12
    np.testing.assert_array_equal(unpad(padded,shape),image)


def test_minimum_object_area_scales_with_gsd_and_reports_sub_floor_components():
    mask=np.zeros((8,24),bool)
    mask[1,1:21]=True   # 20 px
    mask[5,1:20]=True   # 19 px, separate 8-connected component
    kept,info=filter_small_components(mask,.5,5.0)
    assert info["minimum_area_m2"]==5.0
    assert info["minimum_pixels_at_gsd"]==20
    assert (info["raw_count"],info["kept_count"],info["removed_sub_floor_components"])==(2,1,1)
    assert int(kept.sum())==20

    mask=np.zeros((6,20),bool)
    mask[1,1:14]=True   # 13 px
    mask[4,1:15]=True   # 14 px
    kept,info=filter_small_components(mask,.6,5.0)
    assert info["minimum_pixels_at_gsd"]==14
    assert (info["raw_count"],info["kept_count"],info["removed_sub_floor_components"])==(2,1,1)
    assert int(kept.sum())==14

    kept_72,info_72=filter_small_components(mask,.6,7.2)
    assert info_72["minimum_pixels_at_gsd"]==20
    assert info_72["kept_count"]==0 and info_72["removed_sub_floor_components"]==2


def test_quality_checks_detect_shift_sharpness_and_gsd():
    rng=np.random.default_rng(12)
    before=rng.integers(0,256,(192,192,3),dtype=np.uint8)
    matrix=np.float32([[1,0,3],[0,1,-2]])
    shifted=cv2.warpAffine(before,matrix,(192,192),borderMode=cv2.BORDER_REFLECT_101)
    result=quality_checks(before,shifted,1.6)
    assert result["shift_px"]>1
    assert any("shift" in warning.lower() or "정합" in warning for warning in result["warnings"])
    assert any("GSD" in warning for warning in result["warnings"])

    softened=cv2.GaussianBlur(before,(0,0),sigmaX=3)
    sharp=quality_checks(before,softened,.5)
    assert sharp["sharpness_ratio"]>2
    assert any("sharpness" in warning.lower() or "선명도" in warning for warning in sharp["warnings"])


def tinycd_available():
    """TinyCD needs PyTorch, the upstream code (scripts/install_tinycd.py) and the checkpoint."""
    try:
        import torch  # noqa: F401
    except ImportError:
        return False
    return (ROOT/"vendor"/"tinycd"/"models"/"change_classifier.py").exists() and (ROOT/"models"/"tinycd_levir_best.pth").exists()


@pytest.mark.parametrize("method",["OpenCV basic","OpenCV improved","TinyCD"])
def test_streamlit_app_smoke_on_synthetic_pair(method,monkeypatch):
    if method=="TinyCD" and not tinycd_available():
        pytest.skip("TinyCD needs requirements-ml.txt, scripts/install_tinycd.py and scripts/download_tinycd_weights.py")
    monkeypatch.setenv("ASTRA_TEST_MODE","1")
    monkeypatch.setenv("ASTRA_FORCE_CPU","1")
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=90).run()
    assert not at.exception
    assert at.slider(key="minimum_object_area_m2").value==5.0
    at.selectbox(key="language").set_value("한국어").run()
    at.selectbox(key="image_source").set_value("합성 데모").run()
    at.selectbox(key="method").set_value(method).run()
    assert not at.exception
    assert any("변화 객체" in metric.label for metric in at.metric)
    assert any("기준 미만 component" in caption.value for caption in at.caption)
    assert at.checkbox(key="bidirectional").value is False
    if method=="TinyCD":
        variants=at.selectbox(key="model_variant").options
        assert "Original" in variants
        assert "ROBUST (fine-tuned)" not in variants
        robust_v2 = "ROBUST-v2 (blur/order-robust fine-tune)"
        # The fine-tuned checkpoint is not published; the app offers it only when the file exists.
        if not (ROOT/"models"/"finetuned"/"tinycd_robust_v2_seed42_best.pth").is_file():
            assert variants == ["Original"]
            return
        assert robust_v2 in variants
        at.selectbox(key="model_variant").set_value(robust_v2).run()
        assert not at.exception
        assert at.checkbox(key="bidirectional").disabled
