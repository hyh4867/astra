"""Object completeness and perturbation robustness on the fixed LEVIR-CD test split."""
from __future__ import annotations

import csv
from pathlib import Path
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

METHODS = ("OpenCV basic", "OpenCV improved", "TinyCD")
BASIC_THRESHOLD = 105
IMPROVED_BLUR = 3
IMPROVED_MORPH = 5
TINYCD_CUTOFF = 0.5
PIXEL_M2 = 0.25
Z95 = 1.959963984540054
AREA_EDGES = (0, 100, 250, 500, 1000, 2500, 5000, float("inf"))
AREA_NAMES = ("<100", "100-250", "250-500", "500-1000", "1000-2500", "2500-5000", ">=5000")
PURITY_EDGES = AREA_EDGES
PURITY_NAMES = AREA_NAMES
CROP = 8
ROBUST_SEED = 42
ROBUST_N = 32


def components(mask: np.ndarray):
    # 8-connectivity: diagonal touching pixels belong to the same object.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    areas = stats[1:, cv2.CC_STAT_AREA].astype(np.int64)
    return count, labels, areas


def area_bin(area: int, names=AREA_NAMES):
    for i, edge in enumerate(AREA_EDGES[1:]):
        if area < edge:
            return i
    return len(names)-1


def wilson(successes: int, total: int):
    if not total:
        return "", ""
    p = successes/total
    den = 1+Z95**2/total
    center = (p+Z95**2/(2*total))/den
    half = Z95*np.sqrt(p*(1-p)/total+Z95**2/(4*total**2))/den
    return max(0.0,center-half), min(1.0,center+half)


def pred_and_objects(pred: np.ndarray, gt: np.ndarray):
    ng, glabel, garea = components(gt)
    npred, plabel, parea = components(pred)
    # bincount maps overlapping predicted pixels/GT pixels to each component id.
    gt_hits = np.bincount(glabel[pred], minlength=ng)
    pred_hits = np.bincount(plabel[gt], minlength=npred)
    return garea, gt_hits[1:ng], parea, pred_hits[1:npred]


def wilson_and_completeness(test_rows, objects, methods=METHODS):
    out=[]
    thresholds=(0.10,0.30,0.50)
    for method in methods:
        for bi,name in enumerate(AREA_NAMES):
            gt_objs=[o for o in objects[method]["gt"] if area_bin(o["area"])==bi]
            for fraction in thresholds:
                detected=sum(o["overlap"] >= fraction*o["area"] for o in gt_objs)
                n=len(gt_objs); lo,hi=wilson(detected,n)
                out.append({"method":method,"gt_area_bin_px":name,
                    "area_min_px":AREA_EDGES[bi],"area_max_px_exclusive":AREA_EDGES[bi+1],
                    "area_min_m2":AREA_EDGES[bi]*PIXEL_M2 if AREA_EDGES[bi] else 0,
                    "area_max_m2_exclusive":AREA_EDGES[bi+1]*PIXEL_M2 if np.isfinite(AREA_EDGES[bi+1]) else "",
                    "detection_coverage_threshold":fraction,"n_gt_objects":n,
                    "n_detected":detected,"completeness":detected/n if n else "",
                    "wilson95_low":lo,"wilson95_high":hi})
    return out


def purity_by_size(objects):
    out=[]
    for method in METHODS:
        for bi,name in enumerate(PURITY_NAMES):
            preds=[o for o in objects[method]["pred"] if area_bin(o["area"])==bi]
            vals=[o["purity"] for o in preds]
            false=sum(o["purity"]<0.10 for o in preds)
            out.append({"method":method,"pred_area_bin_px":name,
                "area_min_px":AREA_EDGES[bi],"area_max_px_exclusive":AREA_EDGES[bi+1],
                "area_min_m2":AREA_EDGES[bi]*PIXEL_M2 if AREA_EDGES[bi] else 0,
                "area_max_m2_exclusive":AREA_EDGES[bi+1]*PIXEL_M2 if np.isfinite(AREA_EDGES[bi+1]) else "",
                "n_pred_objects":len(preds),"mean_object_purity":float(np.mean(vals)) if vals else "",
                "median_object_purity":float(np.median(vals)) if vals else "",
                "false_alarm_objects_overlap_lt_10pct":false,
                "false_alarm_fraction":false/len(preds) if preds else ""})
    return out


def completeness50(completeness_rows, method):
    rows=[r for r in completeness_rows if r["method"]==method and r["detection_coverage_threshold"]==0.30 and r["n_gt_objects"]>0]
    points=[]
    max_area=max((o["area"] for o in all_objects[method]["gt"]),default=AREA_EDGES[-2]+1)
    for r in rows:
        bi=AREA_NAMES.index(r["gt_area_bin_px"])
        lo=max(1,AREA_EDGES[bi]); hi=AREA_EDGES[bi+1]
        if not np.isfinite(hi): hi=max(lo,max_area)
        if hi<=lo: hi=lo+1
        center=np.sqrt(lo*hi)
        points.append((center,float(r["completeness"])))
    points.sort()
    if not points:
        return "", "no GT objects"
    if points[0][1]>=0.5:
        return points[0][0], "already >=50% in smallest populated bin"
    for (x0,y0),(x1,y1) in zip(points,points[1:]):
        if y0<0.5<=y1:
            if y1==y0: return x1,""
            lx0,lx1=np.log10(x0),np.log10(x1)
            x=10**(lx0+(0.5-y0)*(lx1-lx0)/(y1-y0))
            return x,"log-area interpolation"
    return "", "not reached within populated bins"


def completeness_png(rows):
    methods=METHODS; colors={methods[0]:"#2878b5",methods[1]:"#ef8354",methods[2]:"#3b9b65"}
    width,height=1100,700; im=Image.new("RGB",(width,height),"white"); d=ImageDraw.Draw(im)
    left,right,top,bottom=105,1040,75,560
    d.text((20,18),"GT object completeness vs area (30% pixel coverage)",fill="#111")
    d.text((30,top),"1.0",fill="#333"); d.text((30,bottom-8),"0.0",fill="#333")
    d.line((left,top,left,bottom),fill="#333",width=2); d.line((left,bottom,right,bottom),fill="#333",width=2)
    xs=[10,100,1000,10000,100000]
    xmap=lambda x:left+(np.log10(max(1,x))-1)/(5-1)*(right-left)
    ymap=lambda y:bottom-y*(bottom-top)
    for x in xs:
        px=int(xmap(x)); d.line((px,top,px,bottom),fill="#dddddd",width=1); d.text((px-20,bottom+12),f"{x:g}",fill="#333")
    for y in (0,.25,.5,.75,1):
        py=int(ymap(y)); d.line((left,py,right,py),fill="#eeeeee",width=1)
    for method in methods:
        vals=[r for r in rows if r["method"]==method and r["detection_coverage_threshold"]==0.30 and r["n_gt_objects"]>0]
        vals.sort(key=lambda r:AREA_NAMES.index(r["gt_area_bin_px"]))
        pts=[]
        # The final bin is open-ended; use the largest observed GT object area
        # as its plotting boundary instead of infinity.
        maxa=max((o["area"] for o in all_objects[method]["gt"]),default=10000)
        for r in vals:
            i=AREA_NAMES.index(r["gt_area_bin_px"]); lo=max(1,AREA_EDGES[i]); hi=AREA_EDGES[i+1]
            if not np.isfinite(hi): hi=max(lo,float(maxa))
            if hi<=lo: hi=lo+1
            x=float(np.sqrt(lo*hi)); y=float(r["completeness"]); low=float(r["wilson95_low"]); high=float(r["wilson95_high"])
            px=int(xmap(x)); py=int(ymap(y)); p0=int(ymap(high)); p1=int(ymap(low))
            d.line((px,p0,px,p1),fill=colors[method],width=2); d.line((px-5,p0,px+5,p0),fill=colors[method],width=2); d.line((px-5,p1,px+5,p1),fill=colors[method],width=2)
            pts.append((px,py))
        if len(pts)>1: d.line(pts,fill=colors[method],width=4)
        for px,py in pts: d.ellipse((px-5,py-5,px+5,py+5),fill=colors[method])
    d.text((left,bottom+42),"GT object area (pixels, log scale; 1 px = 0.25 m²)",fill="#222")
    for i,m in enumerate(methods):
        y=620+i*22; d.line((720,y+7,760,y+7),fill=colors[m],width=4); d.text((770,y),m,fill="#222")
    im.save(OUT/"completeness_vs_size.png")


def object_analysis(samples, model, device):
    from evaluate_benchmark import basic_prediction, improved_prediction, tinycd_predict
    global all_objects
    all_objects={m:{"gt":[],"pred":[]} for m in METHODS}
    base_masks={}; total_times={m:0.0 for m in METHODS}
    for index,sample in enumerate(samples,1):
        start=time.perf_counter(); basic=basic_prediction(sample, BASIC_THRESHOLD); total_times["OpenCV basic"]+=time.perf_counter()-start
        start=time.perf_counter(); improved=improved_prediction(sample, IMPROVED_BLUR, IMPROVED_MORPH); total_times["OpenCV improved"]+=time.perf_counter()-start
        start=time.perf_counter(); tiny=tinycd_predict(model,sample,device); total_times["TinyCD"]+=time.perf_counter()-start
        pred_map={"OpenCV basic":basic,"OpenCV improved":improved,"TinyCD":tiny}
        for method,pred in pred_map.items():
            ga,gh,pa,ph=pred_and_objects(pred,sample.truth)
            all_objects[method]["gt"].extend({"area":int(a),"overlap":int(h)} for a,h in zip(ga,gh))
            all_objects[method]["pred"].extend({"area":int(a),"overlap":int(h),"purity":float(h/a)} for a,h in zip(pa,ph))
        base_masks[sample.name]=pred_map
        if index%16==0 or index==len(samples): print(f"Object test: {index}/{len(samples)}",flush=True)
    comp=wilson_and_completeness(None,all_objects)
    purity=purity_by_size(all_objects)
    comp_rows=[r for r in comp if r["detection_coverage_threshold"]==0.30]
    area_km2=len(samples)*1024*1024*PIXEL_M2/1_000_000
    summary=[]
    for method in METHODS:
        fp=sum(o["purity"]<0.10 for o in all_objects[method]["pred"])
        estimate,note=completeness50(comp,method)
        summary.append({"method":method,"n_gt_objects":len(all_objects[method]["gt"]),
            "n_pred_objects":len(all_objects[method]["pred"]),"false_alarm_objects":fp,
            "test_area_km2":area_km2,"false_alarms_per_km2":fp/area_km2,
            "completeness50_object_px":estimate,"completeness50_note":note,
            "runtime_seconds":round(total_times[method],3)})
    return comp,purity,summary,base_masks


def save_csv(path, rows):
    if not rows: return
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n"); w.writeheader(); w.writerows(rows)


def perturb(image, kind, strength, rng):
    if kind=="shift":
        mat=np.float32([[1,0,strength],[0,1,strength]])
        return cv2.warpAffine(image,mat,(image.shape[1],image.shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
    if kind=="gain": return np.clip(image.astype(np.float32)*strength,0,255).astype(np.uint8)
    if kind=="gamma": return np.clip((image.astype(np.float32)/255.0)**strength*255,0,255).astype(np.uint8)
    if kind=="blur": return cv2.GaussianBlur(image,(0,0),sigmaX=float(strength),sigmaY=float(strength))
    if kind=="noise": return np.clip(image.astype(np.float32)+rng.normal(0,float(strength),image.shape),0,255).astype(np.uint8)
    raise ValueError(kind)


def binary_metrics(pred,gt):
    p=pred.astype(bool); t=gt.astype(bool)
    tp=int(np.count_nonzero(p&t)); fp=int(np.count_nonzero(p&~t)); fn=int(np.count_nonzero(~p&t))
    precision=tp/(tp+fp) if tp+fp else 0.0; recall=tp/(tp+fn) if tp+fn else 0.0
    return {"precision":precision,"recall":recall,"f1":2*precision*recall/(precision+recall) if precision+recall else 0.0,"fp_pixels":fp}


def robustness_chart(rows):
    factors=["control","shift","gain","gamma","blur","noise"]
    labels={"control":["0"],"shift":["1","2","3","5"],"gain":["0.8","1.2"],"gamma":["0.7","1.4"],"blur":["1","2"],"noise":["5","10"]}
    colors={"OpenCV basic":"#2878b5","OpenCV improved":"#ef8354","TinyCD":"#3b9b65"}
    width,height=1700,980; im=Image.new("RGB",(width,height),"white"); d=ImageDraw.Draw(im)
    d.text((25,15),"Robustness on fixed 32-image subset; 8 px border cropped",fill="#111")
    panel_w,panel_h=320,390; xstart,ystart=28,65; xgap,ygap=16,22
    order={"control":None,"shift":[1,2,3,5],"gain":[.8,1.2],"gamma":[.7,1.4],"blur":[1,2],"noise":[5,10]}
    titles={"control":"Control","shift":"Shift dx=dy (px)","gain":"Gain","gamma":"Gamma","blur":"Gaussian sigma","noise":"Noise sigma"}
    for pi,factor in enumerate(factors):
        col=pi%5; row=pi//5; x0=xstart+col*(panel_w+xgap); y0=ystart+row*(panel_h+ygap)
        plot_l,plot_r=x0+42,x0+panel_w-12; plot_t,plot_b=y0+45,y0+panel_h-72
        d.text((x0+8,y0+8),titles[factor],fill="#111")
        for yv in (0,.25,.5,.75,1):
            yy=int(plot_b-yv*(plot_b-plot_t)); d.line((plot_l,yy,plot_r,yy),fill="#ddd",width=1); d.text((x0+2,yy-7),f"{yv:.2g}",fill="#333")
        d.line((plot_l,plot_t,plot_l,plot_b),fill="#333",width=2); d.line((plot_l,plot_b,plot_r,plot_b),fill="#333",width=2)
        xs=labels[factor]; xloc=[int(plot_l+i*(plot_r-plot_l)/max(1,len(xs)-1)) for i in range(len(xs))]
        for x,label in zip(xloc,xs): d.text((x-20,plot_b+8),label,fill="#333")
        for method,color in colors.items():
            vals=[]
            for value in xs:
                rowdata=next(r for r in rows if r["method"]==method and r["factor"]==factor and str(r["strength"])==value)
                vals.append(float(rowdata["f1"]))
            pts=[(x,int(plot_b-v*(plot_b-plot_t))) for x,v in zip(xloc,vals)]
            if len(pts)>1:d.line(pts,fill=color,width=3)
            for px,py in pts:d.ellipse((px-4,py-4,px+4,py+4),fill=color)
        d.text((plot_l,plot_b+34),"F1",fill="#555")
    for i,(method,color) in enumerate(colors.items()):
        x=1000+i*190; y=905; d.line((x,y+8,x+35,y+8),fill=color,width=4); d.text((x+42,y),method,fill="#222")
    im.save(OUT/"robustness_f1.png")


def robustness(samples, model, device, base_masks):
    from evaluate_benchmark import Sample, basic_prediction, improved_prediction, tinycd_predict
    names=sorted(s.name for s in samples)
    rng_select=np.random.default_rng(ROBUST_SEED)
    selected=list(rng_select.choice(names,size=ROBUST_N,replace=False))
    (ROOT/"data"/"manifests").mkdir(parents=True,exist_ok=True)
    (ROOT/"data"/"manifests"/"robustness_test_seed42.txt").write_text("\n".join(selected)+"\n")
    byname={s.name:s for s in samples}
    conditions=[("control","control",0)]+[("shift","shift",x) for x in (1,2,3,5)]+[("gain","gain",x) for x in (.8,1.2)]+[("gamma","gamma",x) for x in (.7,1.4)]+[("blur","blur",x) for x in (1,2)]+[("noise","noise",x) for x in (5,10)]
    results=[]; total=len(conditions)
    for ci,(factor,kind,strength) in enumerate(conditions,1):
        perturbed=[]
        for ii,name in enumerate(selected):
            s=byname[name]
            if kind=="control": b=s.after
            else: b=perturb(s.after,kind,strength,np.random.default_rng(ROBUST_SEED+ci*1000+ii))
            perturbed.append(Sample(s.name,s.before,b,s.truth,s.gray_diff))
        preds={m:[] for m in METHODS}; runtimes={m:0.0 for m in METHODS}
        for i,(s,orig_name) in enumerate(zip(perturbed,selected),1):
            if kind=="control":
                t=time.perf_counter(); preds["OpenCV basic"].append(basic_prediction(s,BASIC_THRESHOLD)[CROP:-CROP,CROP:-CROP]); runtimes["OpenCV basic"]+=time.perf_counter()-t
                t=time.perf_counter(); preds["OpenCV improved"].append(improved_prediction(s,IMPROVED_BLUR,IMPROVED_MORPH)[CROP:-CROP,CROP:-CROP]); runtimes["OpenCV improved"]+=time.perf_counter()-t
                t=time.perf_counter(); tm=tinycd_predict(model,s,device); runtimes["TinyCD"]+=time.perf_counter()-t; preds["TinyCD"].append(tm[CROP:-CROP,CROP:-CROP])
            else:
                t=time.perf_counter(); preds["OpenCV basic"].append(basic_prediction(s,BASIC_THRESHOLD)[CROP:-CROP,CROP:-CROP]); runtimes["OpenCV basic"]+=time.perf_counter()-t
                t=time.perf_counter(); preds["OpenCV improved"].append(improved_prediction(s,IMPROVED_BLUR,IMPROVED_MORPH)[CROP:-CROP,CROP:-CROP]); runtimes["OpenCV improved"]+=time.perf_counter()-t
                t=time.perf_counter(); tm=tinycd_predict(model,s,device); runtimes["TinyCD"]+=time.perf_counter()-t; preds["TinyCD"].append(tm[CROP:-CROP,CROP:-CROP])
            if i%16==0 or i==ROBUST_N: print(f"Robustness {factor}={strength}: {i}/{ROBUST_N}",flush=True)
        for method in METHODS:
            sums={k:0 for k in ("tp","fp","fn")}
            for s,pred in zip(perturbed,preds[method]):
                gt=s.truth[CROP:-CROP,CROP:-CROP]; p=pred.astype(bool); t=gt.astype(bool)
                sums["tp"]+=int(np.count_nonzero(p&t)); sums["fp"]+=int(np.count_nonzero(p&~t)); sums["fn"]+=int(np.count_nonzero(~p&t))
            p=sums["tp"]/(sums["tp"]+sums["fp"]) if sums["tp"]+sums["fp"] else 0
            r=sums["tp"]/(sums["tp"]+sums["fn"]) if sums["tp"]+sums["fn"] else 0
            results.append({"factor":factor,"perturbation":kind,"strength":strength,"condition":f"{kind}={strength}","method":method,
                "precision":p,"recall":r,"f1":2*p*r/(p+r) if p+r else 0,"fp_pixels":sums["fp"],
                "runtime_seconds":round(runtimes[method],3),"images":ROBUST_N,"crop_border_px":CROP,"seed":ROBUST_SEED,"device":str(device)})
        print(f"Robustness condition complete {ci}/{total}: {factor}={strength}",flush=True)
    save_csv(OUT/"robustness_test_seed42.csv",results)
    robustness_chart(results)
    return selected,results


def main():
    from evaluate_benchmark import load_samples, ChangeClassifier
    import torch
    OUT.mkdir(exist_ok=True)
    samples=list(load_samples("test"))
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"platform.machine()={__import__('platform').machine()}; MPS available={torch.backends.mps.is_available()}; device={device}",flush=True)
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT/"models"/"tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True)
    comp,purity,summary,base_masks=object_analysis(samples,model,device)
    save_csv(OUT/"object_completeness.csv",comp)
    save_csv(OUT/"object_purity_by_size.csv",purity)
    save_csv(OUT/"object_metrics_summary.csv",summary)
    completeness_png(comp)
    # Use the shared phone-readable renderer with Wilson confidence bands.
    from regenerate_summary_charts import main as regenerate_summary_charts
    regenerate_summary_charts()
    selected,robust_rows=robustness(samples,model,device,base_masks)
    print("Object summary:")
    for r in summary: print(r)
    print("Robustness list saved:",ROOT/"data"/"manifests"/"robustness_test_seed42.txt")
    print("Robustness CSV/PNG saved",len(robust_rows),"rows; subset",len(selected))


if __name__=="__main__":
    main()
