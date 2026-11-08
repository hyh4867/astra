"""Quick order-swap check on the fixed 32-image robustness subset."""
import csv
import time

import numpy as np
import torch

from evaluate_benchmark import ChangeClassifier, Sample, ROOT, load_samples, tinycd_predict


def main():
    names=(ROOT/"data/manifests/robustness_test_seed42.txt").read_text().splitlines()
    selected=set(names)
    samples=[s for s in load_samples("test") if s.name in selected]
    if len(samples)!=32: raise RuntimeError(f"Expected 32 subset samples, got {len(samples)}")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT/"models/tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True)
    tp=fp=fn=0; started=time.perf_counter()
    for i,s in enumerate(samples,1):
        reversed_pair=Sample(s.name,s.after,s.before,s.truth,s.gray_diff)
        pred=tinycd_predict(model,reversed_pair,device)[8:-8,8:-8]
        gt=s.truth[8:-8,8:-8]
        tp+=int(np.count_nonzero(pred&gt)); fp+=int(np.count_nonzero(pred&~gt)); fn+=int(np.count_nonzero(~pred&gt))
        if i%16==0: print(f"Swapped order {i}/32",flush=True)
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    row={"method":"TinyCD (B,A)","precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,
        "tp":tp,"fp":fp,"fn":fn,"runtime_seconds":round(time.perf_counter()-started,3),
        "device":str(device),"images":32,"crop_border_px":8,"cutoff":0.5}
    path=ROOT/"outputs/tinycd_swap_check.csv"
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=row.keys(),lineterminator="\n"); w.writeheader(); w.writerow(row)
    print(row,flush=True)


if __name__=="__main__": main()
