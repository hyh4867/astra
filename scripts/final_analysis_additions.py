"""Null completeness and blur breakdown; fixed thresholds and MPS only."""
from __future__ import annotations

import csv
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluate_benchmark import (
    OUT, ROOT, ChangeClassifier, Sample, basic_prediction, improved_prediction,
    load_samples, tinycd_predict,
)
from scientific_evaluation import (
    AREA_EDGES, AREA_NAMES, METHODS, PIXEL_M2, BASIC_THRESHOLD,
    IMPROVED_BLUR, IMPROVED_MORPH, TINYCD_CUTOFF, area_bin, wilson,
)
from followup_scientific_checks import gt_objects

CROP=8
SEED=42


def write_csv(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n")
        w.writeheader(); w.writerows(rows)


def derangement(names):
    rng=np.random.default_rng(SEED); ids=np.arange(len(names))
    while True:
        order=rng.permutation(ids)
        if np.all(order!=ids): return {name:names[int(order[i])] for i,name in enumerate(names)}


def measured_completeness(path,isolated=False):
    source=OUT/("isolated_object_completeness.csv" if isolated else "object_completeness.csv")
    key="coverage_threshold" if isolated else "detection_coverage_threshold"
    rows=list(csv.DictReader(source.open()))
    return {(r["method"],r["gt_area_bin_px"]):float(r["completeness"])
            for r in rows if float(r[key])==.3 and r["completeness"]!=""}


def object_null(samples,preds_by_name):
    names=sorted(s.name for s in samples); mapping=derangement(names)
    with (ROOT/"data/manifests/object_null_derangement_seed42.csv").open("w",newline="") as f:
        w=csv.writer(f,lineterminator="\n"); w.writerow(["gt_filename","prediction_filename"])
        w.writerows(mapping.items())
    all_rows=[]
    real_all=measured_completeness("",False); real_iso=measured_completeness("",True)
    byname={s.name:s for s in samples}
    counts={(typ,m,bi):[0,0] for typ in ("all","isolated") for m in METHODS for bi in range(7)}
    for index,name in enumerate(names,1):
        sample=byname[name]; labels,areas,isolated=gt_objects(sample.truth)
        other=mapping[name]
        for method in METHODS:
            pred=preds_by_name[method][other]
            hits=np.bincount(labels[pred],minlength=int(labels.max())+1)
            for j,(area,iso) in enumerate(zip(areas,isolated),1):
                bi=area_bin(int(area)); hit=int(hits[j])>=.30*int(area)
                c=counts[("all",method,bi)]; c[1]+=1; c[0]+=int(hit)
                if iso:
                    c=counts[("isolated",method,bi)]; c[1]+=1; c[0]+=int(hit)
        if index%32==0: print(f"Null pairing {index}/{len(names)}",flush=True)
    for typ in ("all","isolated"):
        real=real_all if typ=="all" else real_iso
        for method in METHODS:
            for bi,bin_name in enumerate(AREA_NAMES):
                detected,n=counts[(typ,method,bi)]; low,high=wilson(detected,n)
                all_rows.append({"gt_subset":typ,"method":method,"gt_area_bin_px":bin_name,
                    "null_n_objects":n,"null_detected_at_30pct":detected,
                    "null_completeness":detected/n if n else "","null_wilson95_low":low,
                    "null_wilson95_high":high,"real_completeness_30pct":real.get((method,bin_name),""),
                    "chance_minus_real":detected/n-real[(method,bin_name)] if n and (method,bin_name) in real else "",
                    "derangement_seed":SEED,"paired_different_image":True})
    write_csv(OUT/"object_null_completeness_seed42.csv",all_rows)
    return mapping


def score(preds,truth):
    total={k:0 for k in ("tp","fp","fn")}
    for pred,gt in zip(preds,truth):
        p=pred[CROP:-CROP,CROP:-CROP].astype(bool)
        t=gt[CROP:-CROP,CROP:-CROP].astype(bool)
        total["tp"]+=int(np.count_nonzero(p&t)); total["fp"]+=int(np.count_nonzero(p&~t)); total["fn"]+=int(np.count_nonzero(~p&t))
    precision=total["tp"]/(total["tp"]+total["fp"]) if total["tp"]+total["fp"] else 0
    recall=total["tp"]/(total["tp"]+total["fn"]) if total["tp"]+total["fn"] else 0
    return {**total,"precision":precision,"recall":recall,
        "f1":2*precision*recall/(precision+recall) if precision+recall else 0}


def blur_sample(sample,mode,sigma):
    a,b=sample.before,sample.after
    if mode in ("A","A+B"):
        a=cv2.GaussianBlur(a,(0,0),sigmaX=sigma,sigmaY=sigma)
    if mode in ("B","A+B"):
        b=cv2.GaussianBlur(b,(0,0),sigmaX=sigma,sigmaY=sigma)
    return Sample(sample.name,a,b,sample.truth,sample.gray_diff)


def blur_breakdown(samples,model,device):
    names=(ROOT/"data/manifests/robustness_test_seed42.txt").read_text().splitlines()
    byname={s.name:s for s in samples}; subset=[byname[n] for n in names]
    conditions=[("control","control",0.0),("blur","B",1.0),
        ("blur","B",1.25),("blur","B",1.5),("blur","B",1.75),("blur","B",2.0),
        ("blur_location","A",2.0),("blur_location","A+B",2.0)]
    rows=[]
    for ci,(factor,mode,sigma) in enumerate(conditions,1):
        perturbed=[s if mode=="control" else blur_sample(s,mode,sigma) for s in subset]
        method_preds={m:[] for m in METHODS}; runtimes={m:0.0 for m in METHODS}
        for i,sample in enumerate(perturbed,1):
            start=time.perf_counter(); method_preds[METHODS[0]].append(basic_prediction(sample,BASIC_THRESHOLD)); runtimes[METHODS[0]]+=time.perf_counter()-start
            start=time.perf_counter(); method_preds[METHODS[1]].append(improved_prediction(sample,IMPROVED_BLUR,IMPROVED_MORPH)); runtimes[METHODS[1]]+=time.perf_counter()-start
            start=time.perf_counter(); method_preds[METHODS[2]].append(tinycd_predict(model,sample,device)); runtimes[METHODS[2]]+=time.perf_counter()-start
            if i%16==0 or i==len(perturbed): print(f"Blur {mode} sigma={sigma}: {i}/{len(perturbed)}",flush=True)
        for method in METHODS:
            metrics=score(method_preds[method],[s.truth for s in perturbed])
            label="control" if mode=="control" else (f"B sigma={sigma:g}" if mode=="B" else f"{mode} sigma={sigma:g}")
            rows.append({"factor":factor,"perturbation":"none" if mode=="control" else "gaussian_blur",
                "strength":0 if mode=="control" else (sigma if mode=="B" else label),
                "condition":label,"method":method,**{k:metrics[k] for k in ("precision","recall","f1","fp")},
                "fp_pixels":metrics["fp"],"runtime_seconds":round(runtimes[method],3),
                "images":len(subset),"crop_border_px":CROP,"seed":SEED,"device":str(device),
                "blur_mode":mode,"sigma":sigma})
        print(f"Condition {ci}/{len(conditions)} complete: {mode} {sigma}",flush=True)
    write_csv(OUT/"blur_breakdown_mps.csv",rows)
    return rows


def update_robust_csv(newrows):
    path=OUT/"robustness_test_seed42.csv"
    old=list(csv.DictReader(path.open()))
    old=[r for r in old if not (r["factor"]=="blur" or r["factor"]=="control")]
    # Replace earlier B-only blur points and control with the fully timed rerun.
    fields=list(old[0].keys())
    converted=[]
    for r in newrows:
        converted.append({k:r.get(k,"") for k in fields})
    combined=old+converted
    write_csv(path,combined)
    return combined


def chart(rows):
    factors=["control","shift","gain","gamma","blur","blur_location","noise"]
    labels={"control":["0"],"shift":["1","2","3","5"],"gain":["0.8","1.2"],
        "gamma":["0.7","1.4"],"blur":["1","1.25","1.5","1.75","2"],
        "blur_location":["B sigma=2","A sigma=2","A+B sigma=2"],"noise":["5","10"]}
    colors={METHODS[0]:"#2878b5",METHODS[1]:"#ef8354",METHODS[2]:"#3b9b65"}
    titles={"control":"Control","shift":"Shift dx=dy (px)","gain":"Gain","gamma":"Gamma",
        "blur":"Blur B (sigma)","blur_location":"Blur location (sigma=2)","noise":"Noise sigma"}
    width,height=1720,980; im=Image.new("RGB",(width,height),"white"); d=ImageDraw.Draw(im)
    d.text((25,15),"Robustness on fixed 32-image subset; 8 px border cropped",fill="#111")
    pw,ph=320,390; xstart,ystart=28,65; xgap,ygap=16,22
    for pi,factor in enumerate(factors):
        col=pi%5; rr=pi//5; x0=xstart+col*(pw+xgap); y0=ystart+rr*(ph+ygap)
        pl,pr=x0+42,x0+pw-12; pt,pb=y0+45,y0+ph-72
        d.text((x0+8,y0+8),titles[factor],fill="#111")
        for yv in (0,.25,.5,.75,1):
            yy=int(pb-yv*(pb-pt)); d.line((pl,yy,pr,yy),fill="#ddd"); d.text((x0+2,yy-7),f"{yv:.2g}",fill="#333")
        d.line((pl,pt,pl,pb),fill="#333",width=2); d.line((pl,pb,pr,pb),fill="#333",width=2)
        vals=labels[factor]; xs=[int(pl+i*(pr-pl)/max(1,len(vals)-1)) for i in range(len(vals))]
        for xx,label in zip(xs,vals): d.text((xx-24,pb+8),label,fill="#333")
        for method,color in colors.items():
            ys=[]
            for value in vals:
                if factor=="blur":
                    match=next(r for r in rows if r["method"]==method and r["factor"]==factor and abs(float(r["strength"])-float(value))<1e-8)
                elif factor=="blur_location":
                    if value=="B sigma=2":
                        match=next(r for r in rows if r["method"]==method and r["factor"]=="blur" and abs(float(r["strength"])-2)<1e-8)
                    else:
                        match=next(r for r in rows if r["method"]==method and r["factor"]==factor and r["condition"]==value)
                else:
                    match=next(r for r in rows if r["method"]==method and r["factor"]==factor and str(r["strength"])==value)
                ys.append(float(match["f1"]))
            pts=[(xx,int(pb-y*(pb-pt))) for xx,y in zip(xs,ys)]
            if len(pts)>1: d.line(pts,fill=color,width=3)
            for xx,yy in pts: d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=color)
        d.text((pl,pb+34),"F1",fill="#555")
    for i,(m,color) in enumerate(colors.items()):
        xx=1000+i*190; d.line((xx,913,xx+35,913),fill=color,width=4); d.text((xx+42,905),m,fill="#222")
    im.save(OUT/"robustness_f1.png")


def main():
    if not torch.backends.mps.is_available(): raise RuntimeError("MPS required")
    device=torch.device("mps"); samples=list(load_samples("test"))
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT/"models/tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True)
    predictions={m:{} for m in METHODS}; positive={m:0 for m in METHODS}
    npx=len(samples)*1024*1024
    for i,s in enumerate(samples,1):
        preds={METHODS[0]:basic_prediction(s,BASIC_THRESHOLD),
            METHODS[1]:improved_prediction(s,IMPROVED_BLUR,IMPROVED_MORPH),
            METHODS[2]:tinycd_predict(model,s,device)}
        for m,p in preds.items(): predictions[m][s.name]=p; positive[m]+=int(p.sum())
        if i%16==0 or i==len(samples): print(f"Null prediction inference {i}/{len(samples)}",flush=True)
    mapping=object_null(samples,predictions)
    fractions=[{"method":m,"predicted_change_pixels":positive[m],"total_pixels":npx,
        "predicted_change_fraction":positive[m]/npx,"derangement_seed":SEED}
        for m in METHODS]
    write_csv(OUT/"predicted_change_fraction_test.csv",fractions)
    print("Predicted change fraction:",fractions,flush=True)
    blur_rows=blur_breakdown(samples,model,device)
    all_rows=update_robust_csv(blur_rows); chart(all_rows)
    from regenerate_summary_charts import main as regenerate_summary_charts
    regenerate_summary_charts()
    print("Derangement pairs",len(mapping),"| robustness rows",len(all_rows),flush=True)


if __name__=="__main__": main()
