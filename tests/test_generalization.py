from pathlib import Path
import sys

import numpy as np
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'scripts')]
from registration import _phase_peak,shift_image
from study_direction import class_mask
from study_landmarks import assess


@pytest.mark.parametrize('method',['dft','ecc'])
def test_refinement_returns_signed_fractional_displacement(method):
    a=np.random.default_rng(42).normal(size=(128,128)).astype('float32')
    dx,dy,_=_phase_peak(a,shift_image(a,-2.5,1.25),16,method)
    np.testing.assert_allclose([dx,dy],[-2.5,1.25],atol=.23)


def test_direction_masks_preserve_separate_pixel_denominators():
    ann=dict(category_id=3,segmentation=[[0,0,4,0,4,2,0,2]])
    rec=dict(annotations=[ann,dict(ann,category_id=2,segmentation=[[4,0,6,0,6,2,4,2]])])
    a,b=class_mask(rec,3),class_mask(rec,2)
    assert a.sum()==8 and b.sum()==4 and not (a&b).any()


def test_ground_reference_holds_out_disjoint_points():
    pts=[dict(x=x,y=y,u=x+3,v=y-2,quadrant=int(x>=256)+2*int(y>=256))
         for x in (50,100,300,350) for y in (40,110,310,360)]
    result=assess(pts)
    assert result['pass_geometry'] and result['held_out_residual_px']==0
    assert (result['dx'],result['dy'])==(3,-2)


def test_training_tensor_normalizes_hwc_channels_before_transposing(monkeypatch):
    pytest.importorskip("torch", reason="Training tensor check needs the optional ML dependencies")
    import study_train
    rgb=np.full((512,512,3),(255,128,0),np.uint8)
    gt=np.ones((512,512),bool)
    monkeypatch.setattr(study_train,'pair',lambda split,name:(rgb,rgb,gt))
    a,b,y=study_train.train_pair('example',True)
    assert tuple(a.shape)==tuple(b.shape)==(3,256,256)
    assert tuple(y.shape)==(205,205) and y.min()==1
    expected=(np.array([1.,128/255.,0.])-[.485,.456,.406])/[.229,.224,.225]
    np.testing.assert_allclose(a[:,0,0].numpy(),expected,rtol=1e-6)
