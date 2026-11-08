"""Compare v1 augmented training crops with evaluation-time Gaussian blur."""
import csv,sys
from pathlib import Path
import cv2,numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/"outputs/diagnosis_round"
from finetune_tinycd_robust import TRAIN,make_train_item
from evaluate_benchmark import load_samples
def measure(x):
 g=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32);v=float(g.var());lap=float(cv2.Laplacian(g,cv2.CV_32F).var());return lap/max(v,1e-9)
rows=[];names=sorted(p.name for p in (TRAIN/"A").glob("*.png"))
for i,n in enumerate(names[:8]):
 a,b,y=make_train_item(n,i,3,True)
 for slot,x in (("augmented_A",a),("augmented_B",b)):rows.append({"source":"training_sample","filename":n,"condition":slot,"laplacian_variance_over_image_variance":measure(x)})
subset=(ROOT/"data/manifests/robustness_test_seed42.txt").read_text().splitlines();by={s.name:s for s in load_samples("test")}
for sigma in (0,1,1.5,1.75,2,2.5,3):
 vals=[]
 for n in subset:
  s=by[n];b=s.after if sigma==0 else cv2.GaussianBlur(s.after,(0,0),sigmaX=sigma);vals.append(measure(b))
 rows.append({"source":"evaluation","filename":"subset_mean","condition":f"B_sigma={sigma:g}","laplacian_variance_over_image_variance":float(np.mean(vals))})
with (OUT/"training_vs_evaluation_blur_sharpness.csv").open("w",newline="") as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
print("saved",OUT/"training_vs_evaluation_blur_sharpness.csv")
