import inspect
from pathlib import Path
import sys

import cv2
import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"scripts"))

from registration import (
    DEFAULT_CONFIG, STATUS_ALIGNED, STATUS_CORRECTED, STATUS_REVIEW, STATUS_SMALL, estimate_shift, measure_shift,
    register_pair, shift_image, shift_valid_mask, with_overrides,
)

# Tests of the mechanics use a low threshold so that small shifts are applied;
# the shipped threshold (DEFAULT_CONFIG.min_shift_px) has its own tests below.
APPLY_ALL=with_overrides(min_shift_px=0.5)


def textured(shape=(400,520),seed=0):
    rng=np.random.default_rng(seed)
    return cv2.GaussianBlur(rng.integers(0,256,(*shape,3),dtype=np.uint8),(0,0),2)


def stripes(size=384):
    _,xx=np.mgrid[:size,:size]
    return np.repeat((((xx//8)%2)*200+20).astype(np.uint8)[...,None],3,axis=2)


@pytest.mark.parametrize("shift",[(3,-2),(-3,2),(5,0),(0,5),(-4,1),(1,-4),(-5,-5),(5,5)])
def test_estimate_recovers_sign_and_axis(shift):
    """dx is the column (x) displacement, dy the row (y) displacement, both signed."""
    before=textured()
    after=shift_image(before,*shift)
    estimate=estimate_shift(before,after)
    assert estimate["reasons"]==[]
    assert abs(estimate["dx"]-shift[0])<0.2 and abs(estimate["dy"]-shift[1])<0.2


def test_shift_image_moves_content_right_and_down():
    image=np.zeros((40,60,3),np.uint8); image[10,20]=255
    moved=shift_image(image,3,-2,border=cv2.BORDER_CONSTANT)
    assert moved[8,23,0]==255 and moved[10,20,0]==0


def test_shift_image_matches_existing_robustness_perturbation():
    from scientific_evaluation import perturb
    image=textured((128,160),seed=3)
    for strength in (1,2,3,5):
        np.testing.assert_array_equal(shift_image(image,strength,strength),perturb(image,"shift",strength,None))


@pytest.mark.parametrize("shift",[(3,-2),(-4,1),(0,5)])
def test_correction_restores_reference_grid_inside_valid_region(shift):
    before=textured(seed=1)
    result=register_pair(before,shift_image(before,*shift),APPLY_ALL)
    assert result.status==STATUS_CORRECTED and result.applied
    assert (result.applied_dx,result.applied_dy)==(-shift[0],-shift[1])
    valid=result.valid_mask
    np.testing.assert_array_equal(result.after_aligned[valid],before[valid])
    # The padded strip is excluded and it is exactly as wide as the shift.
    assert valid.sum()==(before.shape[0]-abs(shift[1]))*(before.shape[1]-abs(shift[0]))


@pytest.mark.parametrize("move",[(3,-2),(-3,2),(0,4),(-5,0),(2.5,-1.5)])
def test_valid_mask_marks_only_real_pixels(move):
    ones=np.full((50,70),255,np.uint8)
    warped=shift_image(ones,*move,border=cv2.BORDER_CONSTANT)
    mask=shift_valid_mask((50,70),*move)
    assert np.all(warped[mask]==255)
    assert mask.sum()>0
    # Everything outside the mask within one pixel of the border is padded or partly padded.
    assert np.all(warped[~mask]<255)


def test_aligned_pair_is_left_untouched():
    before=textured(seed=2)
    after=before.copy(); after[100:140,200:260]=(230,225,215)
    result=register_pair(before,after)
    assert result.status==STATUS_ALIGNED and not result.applied
    assert result.after_aligned is after and result.valid_mask.all()


def test_flat_scene_is_held_for_review_and_not_corrected():
    rng=np.random.default_rng(5)
    before=np.clip(120+rng.normal(0,2,(300,300,3)),0,255).astype(np.uint8)
    after=np.clip(120+rng.normal(0,2,(300,300,3)),0,255).astype(np.uint8)
    result=register_pair(before,after)
    assert result.status==STATUS_REVIEW and not result.applied
    assert "flat" in result.reasons
    assert result.after_aligned is after and result.valid_mask.all()
    assert result.reason_messages


def test_repeated_pattern_is_held_for_review_and_not_corrected():
    before=stripes(); after=shift_image(before,5,0)
    result=register_pair(before,after)
    assert result.status==STATUS_REVIEW and not result.applied
    assert "repeated" in result.reasons
    assert result.after_aligned is after


def test_large_shift_is_not_applied_automatically():
    before=textured(seed=4)
    result=register_pair(before,shift_image(before,14,0))
    assert result.status==STATUS_REVIEW and "out_of_range" in result.reasons and not result.applied


def test_estimate_below_auto_threshold_is_reported_but_not_applied():
    before=textured(seed=6)
    config=with_overrides(min_shift_px=4.0)
    after=shift_image(before,3,0)
    result=register_pair(before,after,config)
    assert result.status==STATUS_SMALL and not result.applied
    assert result.after_aligned is after and abs(result.dx-3)<0.2
    assert register_pair(before,shift_image(before,-5,0),config).status==STATUS_CORRECTED


def test_default_threshold_applies_shifts_above_it():
    before=textured(seed=7)
    shift=float(np.ceil(DEFAULT_CONFIG.min_shift_px))+1
    result=register_pair(before,shift_image(before,0,-shift))
    assert result.status==STATUS_CORRECTED and (result.applied_dx,result.applied_dy)==(0.0,shift)


def test_estimator_inputs_are_images_only():
    for function in (measure_shift,estimate_shift,register_pair):
        assert list(inspect.signature(function).parameters)==["before","after","config"]


def test_scoring_drops_candidates_outside_valid_region_and_keeps_fixed_crop():
    from evaluate_registration import CROP, counts
    truth=np.zeros((64,64),bool); truth[20:30,20:30]=True
    pred=np.ones((64,64),bool)
    valid=shift_valid_mask((64,64),-12,0)   # the right 12 columns have no real B data
    tp,fp,fn,n=counts(pred,truth,valid)
    assert n==(64-2*CROP)**2                 # the scoring region never changes with the condition
    assert tp==100 and fn==0
    assert fp==(64-2*CROP)*(64-CROP-12)-100  # forced "no candidate" in the invalid strip
    assert counts(pred,truth,None)[1]==(64-2*CROP)**2-100


def run_app(monkeypatch,source,**values):
    monkeypatch.setenv("ASTRA_TEST_MODE","1")
    at=AppTest.from_file(str(ROOT/"app.py"),default_timeout=180).run()
    at.selectbox(key="language").set_value("한국어").run()
    at.selectbox(key="image_source").set_value(source).run()
    at.selectbox(key="method").set_value("OpenCV basic").run()
    for key,value in values.items():
        widget=at.number_input(key=key) if key.startswith("demo_shift") else at.radio(key=key)
        widget.set_value(value)
    at.run()
    assert not at.exception
    return at


def test_app_corrects_injected_shift_and_reports_it(monkeypatch):
    at=run_app(monkeypatch,"합성 데모",demo_shift_dx=5,demo_shift_dy=-2,registration_mode="자동 (믿을 만할 때만 보정)")
    assert any("자동 보정 적용" in s.value and "dx=-5px" in s.value and "dy=+2px" in s.value for s in at.success)
    labels={m.label:m.value for m in at.metric}
    assert labels["보정 적용"]=="예"
    assert abs(float(labels["추정 이동 dx"].split()[0])-5)<0.3 and abs(float(labels["추정 이동 dy"].split()[0])+2)<0.3
    assert not any("정렬 확인 필요" in e.value for e in at.error)


def test_app_leaves_aligned_pair_alone(monkeypatch):
    at=run_app(monkeypatch,"합성 데모")
    assert {m.label:m.value for m in at.metric}["보정 적용"]=="아니오"
    assert any("변화 객체" in m.label for m in at.metric)


def test_app_marks_ambiguous_pair_as_review_only(monkeypatch):
    at=run_app(monkeypatch,"합성 데모 · 반복 패턴")
    assert any("정렬 확인 필요" in e.value for e in at.error)
    assert any("확인이 필요한 이유" in m.value for m in at.markdown)
    labels={m.label:m.value for m in at.metric}
    assert labels["보정 적용"]=="아니오"
    assert any(label.startswith("검토용 후보") for label in labels)
    assert not any(label.startswith("변화 객체") for label in labels)


def test_app_correction_can_be_switched_off(monkeypatch):
    at=run_app(monkeypatch,"합성 데모",demo_shift_dx=5,demo_shift_dy=-2)
    assert at.radio(key="registration_mode").value=="끄기"
    assert {m.label:m.value for m in at.metric}["보정 적용"]=="아니오"
    assert any("보정 끔" in i.value for i in at.info)


def test_common_overlap_bounds_exclude_padding_from_both_warps():
    from score_registration_overlap import axis_bounds
    size=64
    for injection, moves in [(7, [0, -11, -7]), (-6, [0, 10, 6]), (7, [0, 5]), (0, [-15, 9])]:
        # Track original-image validity through both translations with zero padding.
        source=np.ones((size,size,3),np.uint8)
        injected=shift_image(source,injection,0,border=cv2.BORDER_CONSTANT)
        valid=np.logical_and.reduce([shift_image(injected,move,0,border=cv2.BORDER_CONSTANT)[32,:,0]>0 for move in moves])
        valid[:8]=False
        valid[-8:]=False
        lo,hi=axis_bounds(size,injection,moves)
        np.testing.assert_array_equal(np.flatnonzero(valid),np.arange(lo,hi))
