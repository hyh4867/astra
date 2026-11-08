"""One-pass clean TEST, object-null, robustness, and order-swap comparison."""
from __future__ import annotations

import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch

from evaluate_benchmark import Sample, load_samples, tinycd_predict
from followup_scientific_checks import gt_objects
from inference import load_tinycd_model, tinycd_probability
from scientific_evaluation import AREA_NAMES, area_bin, perturb, wilson

OUT = ROOT / "outputs/diagnosis_round"
TEST_SUBSET = ROOT / "data/manifests/robustness_test_seed42.txt"
NULL_MAP = ROOT / "data/manifests/object_null_derangement_seed42.csv"
CHECKPOINTS = {
    "Original": ROOT / "models/tinycd_levir_best.pth",
    "CONTROL": ROOT / "models/finetuned/tinycd_control_best.pth",
    "ROBUST": ROOT / "models/finetuned/tinycd_robust_best.pth",
}
CROP = 8
SEED = 42


def metrics(pred, truth):
    p, t = pred.astype(bool), truth.astype(bool)
    tp = int(np.count_nonzero(p & t))
    fp = int(np.count_nonzero(p & ~t))
    fn = int(np.count_nonzero(~p & t))
    tn = int(np.count_nonzero(~p & ~t))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    iou = tp / (tp + fp + fn) if tp + fp + fn else 0.0
    assert tp + fp + fn + tn == truth.size
    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "pixels": truth.size,
            "precision": precision, "recall": recall, "f1": f1, "iou": iou}


def read_csv(path):
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def clean_test(samples, model, device):
    masks = {}
    rows = []
    total = {k: 0 for k in ("tp", "fp", "fn", "tn")}
    runtime = 0.0
    for index, sample in enumerate(samples, 1):
        started = time.perf_counter()
        probability = tinycd_probability(model, sample.before, sample.after, device)
        elapsed = time.perf_counter() - started
        runtime += elapsed
        pred = probability >= 0.5
        masks[sample.name] = pred
        row = metrics(pred, sample.truth)
        total.update({k: total[k] + row[k] for k in total})
        if not np.any(sample.truth):
            row.update({"precision": "", "recall": "", "f1": "", "iou": ""})
        row["gt_change_pixels"] = int(np.count_nonzero(sample.truth))
        row["pred_change_pixels"] = int(np.count_nonzero(pred))
        rows.append({"filename": sample.name, **row, "runtime_seconds": elapsed})
        if index % 16 == 0 or index == len(samples):
            print(f"Clean test {index}/{len(samples)}", flush=True)
    summary = metrics_from_counts(total)
    assert summary["pixels"] == len(samples) * 1024 * 1024
    summary["runtime_seconds"] = runtime
    summary["no_change_gt_images"] = sum(not np.any(s.truth) for s in samples)
    return summary, rows, masks


def metrics_from_counts(c):
    p = c["tp"] / (c["tp"] + c["fp"]) if c["tp"] + c["fp"] else 0.0
    r = c["tp"] / (c["tp"] + c["fn"]) if c["tp"] + c["fn"] else 0.0
    return {**c, "pixels": sum(c[k] for k in ("tp", "fp", "fn", "tn")),
            "precision": p, "recall": r,
            "f1": 2*p*r/(p+r) if p+r else 0.0,
            "iou": c["tp"]/(c["tp"]+c["fp"]+c["fn"]) if c["tp"]+c["fp"]+c["fn"] else 0.0}


def object_completeness(samples, masks):
    with NULL_MAP.open(newline="") as f:
        null_map = {r["gt_filename"]: r["prediction_filename"] for r in csv.DictReader(f)}
    if set(null_map) != set(masks):
        raise ValueError("Seed-42 null map does not cover the test predictions.")
    # This function receives one model's masks; compute overall and size-binned
    # observed and deranged-null completeness for all and isolated components.
    aggregate = {scope: {"n": 0, "real": 0, "chance": 0, "bins": {b: {"n": 0, "real": 0, "chance": 0} for b in AREA_NAMES}}
                 for scope in ("all", "isolated")}
    per_image = []
    for sample in samples:
        labels, areas, isolated = gt_objects(sample.truth)
        real_hits = np.bincount(labels[masks[sample.name]], minlength=int(labels.max()) + 1)
        chance_mask = masks[null_map[sample.name]]
        chance_hits = np.bincount(labels[chance_mask], minlength=int(labels.max()) + 1)
        for label_id, area in enumerate(areas, start=1):
            bname = AREA_NAMES[area_bin(int(area))]
            scopes = ["all"] + (["isolated"] if isolated[label_id-1] else [])
            for scope in scopes:
                is_real = real_hits[label_id] >= 0.30 * area
                is_chance = chance_hits[label_id] >= 0.30 * area
                ag = aggregate[scope]; bg = ag["bins"][bname]
                ag["n"] += 1; ag["real"] += int(is_real); ag["chance"] += int(is_chance)
                bg["n"] += 1; bg["real"] += int(is_real); bg["chance"] += int(is_chance)
        per_image.append({"filename": sample.name, "gt_objects": int(len(areas)),
                          "isolated_gt_objects": int(isolated.sum())})

    summary_rows, bin_rows = [], []
    for scope, vals in aggregate.items():
        n=vals["n"]; real=vals["real"]/n if n else 0.0; chance=vals["chance"]/n if n else 0.0
        skill=(real-chance)/(1-chance) if chance<1 else ""
        summary_rows.append({"scope":scope,"n_objects":n,"real_detected":vals["real"],
            "chance_detected":vals["chance"],"real_completeness":real,"chance_completeness":chance,
            "object_skill":skill,"coverage_threshold":0.30,"null_derangement_seed":SEED})
        for bname, counts in vals["bins"].items():
            n=counts["n"]; real_lo,real_hi=wilson(counts["real"],n); chance_lo,chance_hi=wilson(counts["chance"],n)
            bin_rows.append({"scope":scope,"gt_area_bin_px":bname,"n_objects":n,
                "real_detected":counts["real"],"real_completeness":counts["real"]/n if n else "",
                "real_wilson95_low":real_lo,"real_wilson95_high":real_hi,
                "chance_detected":counts["chance"],"chance_completeness":counts["chance"]/n if n else "",
                "chance_wilson95_low":chance_lo,"chance_wilson95_high":chance_hi,
                "coverage_threshold":0.30,"null_derangement_seed":SEED})
    return summary_rows, bin_rows, per_image


def robustness_conditions(samples, selected_names):
    byname = {s.name: s for s in samples}
    conds = [("control", "unchanged", None)]
    conds += [("shift", f"{v}px", ("shift", v)) for v in (1, 2, 3, 5)]
    conds += [("gain", str(v), ("gain", v)) for v in (0.8, 1.2)]
    conds += [("gamma", str(v), ("gamma", v)) for v in (0.7, 1.4)]
    conds += [("blur_B", f"sigma={v}", ("blur_B", v)) for v in (1, 2)]
    conds += [("blur_A", "sigma=2", ("blur_A", 2))]
    conds += [("blur_A+B", "sigma=2", ("blur_AB", 2))]
    conds += [("noise", str(v), ("noise", v)) for v in (5, 10)]
    result = []
    for ci, (factor, strength_label, action) in enumerate(conds):
        items=[]
        for ii, name in enumerate(selected_names):
            s=byname[name]; a=s.before; b=s.after
            if action:
                kind,value=action; rng=np.random.default_rng(SEED+ci*1000+ii)
                if kind in {"shift","gain","gamma","noise"}: b=perturb(b,kind,value,rng)
                elif kind=="blur_B": b=perturb(b,"blur",value,rng)
                elif kind=="blur_A": a=perturb(a,"blur",value,rng)
                elif kind=="blur_AB":
                    a=perturb(a,"blur",value,rng); b=perturb(b,"blur",value,rng)
            items.append((name,a,b,byname[name].truth))
        result.append((factor,strength_label,items))
    return result


def robustness_run(conditions, model, device):
    rows=[]
    for ci,(factor,strength,items) in enumerate(conditions,1):
        total={k:0 for k in ("tp","fp","fn","tn")}; runtime=0.0
        for name,a,b,truth in items:
            gt=truth[CROP:-CROP,CROP:-CROP]
            sample=Sample(name,a,b,truth,np.zeros_like(truth,dtype=np.uint8))
            started=time.perf_counter(); pred=tinycd_predict(model,sample,device)[CROP:-CROP,CROP:-CROP]
            runtime+=time.perf_counter()-started
            c=metrics(pred,gt)
            for key in total: total[key]+=c[key]
        row=metrics_from_counts(total)
        rows.append({"condition":factor,"strength":strength,**row,"runtime_seconds":runtime,
            "images":len(items),"crop_border_px":CROP,"seed":SEED,"device":str(device)})
        print(f"Robustness {ci}/{len(conditions)} {factor}={strength}: F1={row['f1']:.4f}",flush=True)
    return rows


def swap_run(samples, selected_names, model, device):
    byname={s.name:s for s in samples}; total={k:0 for k in ("tp","fp","fn","tn")}; runtime=0.0
    for name in selected_names:
        s=byname[name]; started=time.perf_counter()
        reverse=Sample(name,s.after,s.before,s.truth,s.gray_diff)
        pred=tinycd_predict(model,reverse,device)[CROP:-CROP,CROP:-CROP]; runtime+=time.perf_counter()-started
        c=metrics(pred,s.truth[CROP:-CROP,CROP:-CROP])
        for k in total: total[k]+=c[k]
    return {"input_order":"B,A","images":len(selected_names),**metrics_from_counts(total),"runtime_seconds":runtime,"device":str(device)}


def robustness_png(rows):
    groups=[("control",["unchanged"]),("shift",["1px","2px","3px","5px"]),
        ("gain",["0.8","1.2"]),("gamma",["0.7","1.4"]),
        ("blur_B",["sigma=1","sigma=2"]),("blur placement",["B sigma=2","A sigma=2","A+B sigma=2"]),
        ("noise",["5","10"])]
    colors={"Original":"#2864a5","CONTROL":"#d47a23","ROBUST":"#2f8b57"}
    width,height=1800,1120; canvas=Image.new("RGB",(width,height),"white");d=ImageDraw.Draw(canvas)
    font_path=Path("/System/Library/Fonts/Supplemental/Arial.ttf")
    bold_path=Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")
    def font(size,bold=False):
        path=bold_path if bold else font_path
        return ImageFont.truetype(str(path),size) if path.exists() else ImageFont.load_default()
    title_font,panel_font,tick_font,legend_font=font(27,True),font(22,True),font(17),font(18,True)
    d.text((25,15),"Fine-tuned TinyCD robustness · seed-42 test subset · 8 px crop",fill="#111",font=title_font)
    panel_w,panel_h=565,300; left0,top0=25,60; gap_x,gap_y=25,35
    for gi,(title,labels) in enumerate(groups):
        col=gi%3;row=gi//3;x0=left0+col*(panel_w+gap_x);y0=top0+row*(panel_h+gap_y)
        pl,pr,pt,pb=x0+55,x0+panel_w-15,y0+42,y0+panel_h-55
        d.text((x0+8,y0+6),title,fill="#111",font=panel_font)
        for y in (0,.25,.5,.75,1):
            yy=int(pb-y*(pb-pt));d.line((pl,yy,pr,yy),fill="#ddd");d.text((x0+4,yy-10),f"{y:.2g}",fill="#333",font=tick_font)
        d.line((pl,pt,pl,pb),fill="#333",width=2);d.line((pl,pb,pr,pb),fill="#333",width=2)
        xs=[int(pl+i*(pr-pl)/max(1,len(labels)-1)) for i in range(len(labels))]
        for xx,label in zip(xs,labels):
            label_w=d.textbbox((0,0),label,font=tick_font)[2]
            label_x=max(pl,min(xx-label_w//2,pr-label_w))
            d.text((label_x,pb+8),label,fill="#333",font=tick_font)
        for method,color in colors.items():
            vals=[]
            for label in labels:
                candidate=[r for r in rows if r["method"]==method and r["condition"]==title and r["strength"]==label]
                if title=="blur placement":
                    factor,label2=label.split(" sigma=");candidate=[r for r in rows if r["method"]==method and r["condition"]==f"blur_{factor}" and r["strength"]==f"sigma={label2}"]
                vals.append(float(candidate[0]["f1"]) if candidate else 0.0)
            pts=[(xx,int(pb-v*(pb-pt))) for xx,v in zip(xs,vals)]
            if len(pts)>1:d.line(pts,fill=color,width=3)
            for xx,yy in pts:d.ellipse((xx-4,yy-4,xx+4,yy+4),fill=color)
    for i,(method,color) in enumerate(colors.items()):
        x=630+i*200;y=1078;d.line((x,y+13,x+42,y+13),fill=color,width=5);d.text((x+50,y),method,fill="#222",font=legend_font)
    canvas.save(OUT/"robustness_f1_advanced.png")


def main():
    if not torch.backends.mps.is_available():
        raise RuntimeError("MPS is unavailable; this requested evaluation requires MPS.")
    OUT.mkdir(parents=True,exist_ok=True)
    samples=list(load_samples("test"))
    if len(samples)!=128:
        raise ValueError(f"Expected the full 128-image test split, found {len(samples)}.")
    selected_names=sorted(TEST_SUBSET.read_text().splitlines())
    if len(selected_names)!=32 or not set(selected_names).issubset({s.name for s in samples}):
        raise ValueError("Expected the fixed seed-42 list of 32 test images.")
    conditions=robustness_conditions(samples,selected_names)
    full_rows=[]; per_image_rows=[]; object_rows=[]; object_bin_rows=[]; swap_rows=[]; robust_rows=[]
    for method,checkpoint in CHECKPOINTS.items():
        if not checkpoint.is_file(): raise FileNotFoundError(checkpoint)
        model,device=load_tinycd_model(checkpoint)
        clean,per_image,masks=clean_test(samples,model,device)
        full_rows.append({"method":method,**clean,"threshold":0.5,"split":"TEST","images":len(samples),"device":str(device)})
        per_image_rows.extend({"method":method,**row} for row in per_image)
        objects,bins,_=object_completeness(samples,masks)
        object_rows.extend({"method":method,**r} for r in objects)
        object_bin_rows.extend({"method":method,**r} for r in bins)
        robust_rows.extend({"method":method,**r} for r in robustness_run(conditions,model,device))
        swap_rows.append({"method":method,**swap_run(samples,selected_names,model,device)})
        del model
    write_csv(OUT/"advanced_test_summary.csv",full_rows)
    write_csv(OUT/"advanced_test_per_image.csv",per_image_rows)
    write_csv(OUT/"advanced_object_completeness_skill.csv",object_rows)
    write_csv(OUT/"advanced_object_completeness_bins.csv",object_bin_rows)
    write_csv(OUT/"advanced_robustness.csv",robust_rows)
    write_csv(OUT/"advanced_swap_test.csv",swap_rows)
    robustness_png(robust_rows)
    print("Unified advanced evaluation complete. All files remain local in outputs/diagnosis_round/.",flush=True)


if __name__ == "__main__":
    main()
