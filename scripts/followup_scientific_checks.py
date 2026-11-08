"""Follow-up LEVIR-CD object-isolation, FP-size and MPS parity checks."""
from __future__ import annotations

import csv
import platform
import time

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw

from evaluate_benchmark import (
    OUT, ROOT, ChangeClassifier, basic_prediction, improved_prediction,
    load_samples, tinycd_predict,
)
from scientific_evaluation import (
    AREA_EDGES, AREA_NAMES, METHODS, PIXEL_M2, TINYCD_CUTOFF,
    BASIC_THRESHOLD, IMPROVED_BLUR, IMPROVED_MORPH, area_bin, wilson,
)

ISOLATION_RADIUS = 10
ISOLATION_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21))


def gt_objects(truth: np.ndarray):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(
        truth.astype(np.uint8), connectivity=8
    )
    isolated = np.zeros(n, dtype=bool)
    areas = stats[:, cv2.CC_STAT_AREA]
    for label_id in range(1, n):
        x, y, w, h = stats[label_id, :4]
        x0, y0 = max(0, x-ISOLATION_RADIUS), max(0, y-ISOLATION_RADIUS)
        x1 = min(labels.shape[1], x+w+ISOLATION_RADIUS)
        y1 = min(labels.shape[0], y+h+ISOLATION_RADIUS)
        roi = labels[y0:y1, x0:x1]
        own = (roi == label_id).astype(np.uint8)
        dilated = cv2.dilate(own, ISOLATION_KERNEL)
        isolated[label_id] = not np.any((dilated > 0) & (roi > 0) & (roi != label_id))
    return labels, areas[1:].astype(np.int64), isolated[1:]


def connected(mask):
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    return n, labels, stats[1:, cv2.CC_STAT_AREA].astype(np.int64)


def save_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys(), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def log_intersection(rows, method):
    data = [r for r in rows if r["method"] == method and r["n_objects"]]
    data.sort(key=lambda r: AREA_NAMES.index(r["gt_area_bin_px"]))
    points = []
    for r in data:
        i = AREA_NAMES.index(r["gt_area_bin_px"])
        lo = max(1, AREA_EDGES[i])
        hi = AREA_EDGES[i+1]
        if not np.isfinite(hi):
            hi = max(lo+1, r["max_observed_area_px"])
        x = np.sqrt(lo*hi)
        points.append((float(x), float(r["completeness"])))
    if not points or points[0][1] >= .5:
        return "", "below measured range (<100 px)" if points else "no isolated objects"
    for (x0,y0),(x1,y1) in zip(points, points[1:]):
        if y0 < .5 <= y1:
            lx0,lx1=np.log10(x0),np.log10(x1)
            return 10**(lx0+(.5-y0)*(lx1-lx0)/(y1-y0)), "log-area interpolation"
    return "", "not reached within measured bins"


def completeness_chart(rows):
    colors = {METHODS[0]:"#2878b5", METHODS[1]:"#ef8354", METHODS[2]:"#3b9b65"}
    width,height=1100,700
    im=Image.new("RGB",(width,height),"white"); d=ImageDraw.Draw(im)
    left,right,top,bottom=105,1040,75,560
    d.text((20,18),"Isolated GT object completeness (30% coverage; 10 px isolation)",fill="#111")
    xmap=lambda x:left+(np.log10(max(1,x))-1)/4*(right-left)
    ymap=lambda y:bottom-y*(bottom-top)
    for x in (10,100,1000,10000,100000):
        xx=int(xmap(x)); d.line((xx,top,xx,bottom),fill="#ddd"); d.text((xx-20,bottom+12),str(x),fill="#333")
    for y in (0,.25,.5,.75,1):
        yy=int(ymap(y)); d.line((left,yy,right,yy),fill="#eee"); d.text((30,yy-7),f"{y:.2g}",fill="#333")
    d.line((left,top,left,bottom),fill="#333",width=2); d.line((left,bottom,right,bottom),fill="#333",width=2)
    for method,color in colors.items():
        selected=[r for r in rows if r["method"]==method and r["coverage_threshold"]==.3 and r["n_objects"]]
        selected.sort(key=lambda r: AREA_NAMES.index(r["gt_area_bin_px"]))
        pts=[]
        for r in selected:
            i=AREA_NAMES.index(r["gt_area_bin_px"]); lo=max(1,AREA_EDGES[i]); hi=AREA_EDGES[i+1]
            if not np.isfinite(hi): hi=max(lo+1,r["max_observed_area_px"])
            xx=int(xmap(np.sqrt(lo*hi))); yy=int(ymap(r["completeness"]))
            y0,y1=int(ymap(r["wilson95_high"])),int(ymap(r["wilson95_low"]))
            d.line((xx,y0,xx,y1),fill=color,width=2); d.line((xx-5,y0,xx+5,y0),fill=color,width=2); d.line((xx-5,y1,xx+5,y1),fill=color,width=2)
            pts.append((xx,yy))
        if len(pts)>1: d.line(pts,fill=color,width=4)
        for xx,yy in pts: d.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color)
    d.text((left,bottom+42),"GT object area (px, log scale; 1 px = 0.25 m²)",fill="#222")
    for i,(method,color) in enumerate(colors.items()):
        yy=620+i*22; d.line((720,yy+7,760,yy+7),fill=color,width=4); d.text((770,yy),method,fill="#222")
    im.save(OUT/"isolated_completeness_vs_size.png")


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this requested run must use MPS")
    device=torch.device("mps")
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(ROOT/"models"/"tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True)
    isolated_objects={m:[] for m in METHODS}
    pred_objects={m:[] for m in METHODS}
    confusion={k:0 for k in ("tp","fp","fn","tn")}
    runtime=0.0
    samples=list(load_samples("test"))
    for i,sample in enumerate(samples,1):
        t=time.perf_counter(); tiny=tinycd_predict(model,sample,device); runtime+=time.perf_counter()-t
        preds={METHODS[0]:basic_prediction(sample,BASIC_THRESHOLD),
               METHODS[1]:improved_prediction(sample,IMPROVED_BLUR,IMPROVED_MORPH),
               METHODS[2]:tiny}
        truth=sample.truth
        p=tiny.astype(bool); g=truth.astype(bool)
        confusion["tp"]+=int(np.count_nonzero(p&g)); confusion["fp"]+=int(np.count_nonzero(p&~g))
        confusion["fn"]+=int(np.count_nonzero(~p&g)); confusion["tn"]+=int(np.count_nonzero(~p&~g))
        glabel,gareas,gisolated=gt_objects(truth)
        for method,pred in preds.items():
            hits=np.bincount(glabel[pred],minlength=int(glabel.max())+1)
            for j,(area,iso) in enumerate(zip(gareas, gisolated),1):
                rec={"area":int(area),"overlap":int(hits[j])}
                if iso: isolated_objects[method].append(rec)
            n,plabel,pareas=connected(pred)
            pred_hits=np.bincount(plabel[truth],minlength=n)[1:]
            pred_objects[method].extend({"area":int(a),"purity":float(h/a)} for a,h in zip(pareas,pred_hits))
        if i%16==0 or i==len(samples): print(f"MPS test + isolated objects {i}/{len(samples)}",flush=True)

    iso_rows=[]
    for method,objects in isolated_objects.items():
        max_area=max((o["area"] for o in objects),default=5001)
        for bi,name in enumerate(AREA_NAMES):
            obs=[o for o in objects if area_bin(o["area"])==bi]
            for threshold in (.10,.30,.50):
                det=sum(o["overlap"]>=threshold*o["area"] for o in obs)
                low,high=wilson(det,len(obs))
                iso_rows.append({"method":method,"gt_area_bin_px":name,"area_min_px":AREA_EDGES[bi],
                    "area_max_px_exclusive":AREA_EDGES[bi+1],"area_min_m2":AREA_EDGES[bi]*PIXEL_M2,
                    "area_max_m2_exclusive":AREA_EDGES[bi+1]*PIXEL_M2 if np.isfinite(AREA_EDGES[bi+1]) else "",
                    "coverage_threshold":threshold,"n_objects":len(obs),"n_detected":det,
                    "completeness":det/len(obs) if obs else "","wilson95_low":low,"wilson95_high":high,
                    "isolation_definition":"21x21 ellipse dilation (10 px radius), no other 8-connected GT component",
                    "max_observed_area_px":max_area})
    save_csv(OUT/"isolated_object_completeness.csv",iso_rows)
    summaries=[]
    for method in METHODS:
        rows=[r for r in iso_rows if r["method"]==method and r["coverage_threshold"]==.3]
        estimate,note=log_intersection(rows,method)
        summaries.append({"method":method,"n_isolated_gt_objects":len(isolated_objects[method]),
            "completeness50_object_px":estimate,"completeness50_note":note,
            "false_alarm_objects_all_sizes":sum(o["purity"]<.10 for o in pred_objects[method]),
            "false_alarms_per_km2_all_sizes":sum(o["purity"]<.10 for o in pred_objects[method])/33.554432,
            "false_alarm_objects_area_ge20_px":sum(o["area"]>=20 and o["purity"]<.10 for o in pred_objects[method]),
            "false_alarms_per_km2_area_ge20_px":sum(o["area"]>=20 and o["purity"]<.10 for o in pred_objects[method])/33.554432})
    save_csv(OUT/"isolated_object_summary.csv",summaries)
    completeness_chart(iso_rows)
    from regenerate_summary_charts import main as regenerate_summary_charts
    regenerate_summary_charts()

    cpu=list(csv.DictReader((OUT/"benchmark_test.csv").open()))
    cpu_tiny=next(r for r in cpu if r["method"]=="TinyCD")
    row={"device":"mps","platform_machine":platform.machine(),"images":len(samples),
        "tp":confusion["tp"],"fp":confusion["fp"],"fn":confusion["fn"],"tn":confusion["tn"],
        "runtime_seconds":round(runtime,3),"threshold":TINYCD_CUTOFF,"tile_px":256,
        "cpu_tp":int(cpu_tiny["tp"]),"cpu_fp":int(cpu_tiny["fp"]),"cpu_fn":int(cpu_tiny["fn"]),
        "cpu_runtime_seconds":float(cpu_tiny["runtime_seconds"]),
        "delta_tp_vs_cpu":confusion["tp"]-int(cpu_tiny["tp"]),
        "delta_fp_vs_cpu":confusion["fp"]-int(cpu_tiny["fp"]),
        "delta_fn_vs_cpu":confusion["fn"]-int(cpu_tiny["fn"]),
        "pixel_count_assert":sum(confusion.values())==len(samples)*1024*1024}
    save_csv(OUT/"tinycd_test_mps_comparison.csv",[row])
    print("TinyCD MPS vs CPU:",row,flush=True)
    print("Isolated summaries:",summaries,flush=True)


if __name__=="__main__": main()
