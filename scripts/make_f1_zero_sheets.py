"""Write local-only comparison sheets for the five zero-F1 test pairs."""
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"vendor"/"tinycd"))
from models.change_classifier import ChangeClassifier
from evaluate_benchmark import Sample, DATA, OUT, basic_prediction, improved_prediction, tinycd_predict, write_visual

NAMES=["test_59.png","test_61.png","test_63.png","test_88.png","test_98.png"]

def load(name):
    base=DATA/"test"
    a=np.asarray(Image.open(base/"A"/name).convert("RGB"))
    b=np.asarray(Image.open(base/"B"/name).convert("RGB"))
    label=np.asarray(Image.open(base/"label"/name).convert("L"))
    diff=cv2.absdiff(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),cv2.cvtColor(b,cv2.COLOR_RGB2GRAY))
    return Sample(name,a,b,label>127,diff)

if __name__=="__main__":
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT/"models"/"tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True)
    for name in NAMES:
        sample=load(name)
        masks={
            "OpenCV basic":basic_prediction(sample,105),
            "OpenCV improved":improved_prediction(sample,3,5),
            "OpenCV oracle":sample.gray_diff>=98,
            "TinyCD":tinycd_predict(model,sample,device),
        }
        write_visual(sample,masks,"f1zero")
        gt=int(sample.truth.sum()); pred=int(masks["TinyCD"].sum())
        print(f"{name}: GT+={gt}; TinyCD predicted+={pred}; device={device}",flush=True)
