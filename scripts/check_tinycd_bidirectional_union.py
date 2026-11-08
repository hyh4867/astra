"""Reference-only forward/reverse TinyCD union score on the fixed subset."""
import csv
import time

import numpy as np
import torch

from evaluate_benchmark import ChangeClassifier, ROOT, Sample, load_samples, tinycd_predict


def counts(pred,truth):
    p=pred[8:-8,8:-8].astype(bool); g=truth[8:-8,8:-8].astype(bool)
    return int(np.count_nonzero(p&g)),int(np.count_nonzero(p&~g)),int(np.count_nonzero(~p&g))


def metrics(tp,fp,fn):
    p=tp/(tp+fp) if tp+fp else 0; r=tp/(tp+fn) if tp+fn else 0
    return p,r,2*p*r/(p+r) if p+r else 0


def main():
    names=(ROOT/"data/manifests/robustness_test_seed42.txt").read_text().splitlines(); selected=set(names)
    samples=[s for s in load_samples("test") if s.name in selected]
    if len(samples)!=32: raise RuntimeError(f"Expected 32 pairs, received {len(samples)}")
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT/"models/tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True)
    totals={key:[0,0,0] for key in ("A,B","B,A","union")}; start=time.perf_counter()
    for i,s in enumerate(samples,1):
        forward=tinycd_predict(model,s,device)
        reverse_sample=Sample(s.name,s.after,s.before,s.truth,s.gray_diff)
        reverse=tinycd_predict(model,reverse_sample,device)
        for key,pred in (("A,B",forward),("B,A",reverse),("union",forward|reverse)):
            c=counts(pred,s.truth)
            totals[key]=[a+b for a,b in zip(totals[key],c)]
        if i%16==0: print(f"Bidirectional union {i}/32",flush=True)
    elapsed=time.perf_counter()-start
    row={"method":"TinyCD (A,B) union (B,A)","precision":metrics(*totals["union"])[0],
        "recall":metrics(*totals["union"])[1],"f1":metrics(*totals["union"])[2],
        "tp":totals["union"][0],"fp":totals["union"][1],"fn":totals["union"][2],
        "forward_tp":totals["A,B"][0],"forward_fp":totals["A,B"][1],"forward_fn":totals["A,B"][2],
        "reverse_tp":totals["B,A"][0],"reverse_fp":totals["B,A"][1],"reverse_fn":totals["B,A"][2],
        "runtime_seconds":round(elapsed,3),"device":str(device),"images":32,"crop_border_px":8,
        "threshold":0.5,"interpretation":"reference only; LEVIR labels do not encode direction"}
    out=ROOT/"outputs/tinycd_bidirectional_union.csv"
    with out.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=row.keys(),lineterminator="\n"); writer.writeheader(); writer.writerow(row)
    print(row,flush=True)


if __name__=="__main__": main()
