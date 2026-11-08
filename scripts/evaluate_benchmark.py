"""Reproducible LEVIR-CD benchmark: OpenCV baselines and pretrained TinyCD."""
from __future__ import annotations

import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image, ImageDraw
from torchvision.transforms import Normalize

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "vendor" / "tinycd"))
from models.change_classifier import ChangeClassifier  # noqa: E402

DATA = ROOT / "data" / "benchmark"
WEIGHTS = ROOT / "models" / "tinycd_levir_best.pth"
OUT = ROOT / "outputs"
IMAGE_PIXELS = 1024 * 1024
NORMALIZE = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
VAL_NOTE = "not held-out (checkpoint may have been selected on val)"


@dataclass
class Sample:
    name: str
    before: np.ndarray
    after: np.ndarray
    truth: np.ndarray
    gray_diff: np.ndarray


def load_samples(split: str):
    base = DATA / split
    names = sorted(p.name for p in (base / "A").glob("*.png"))
    if not names:
        raise FileNotFoundError(f"No images in {base / 'A'}")
    for name in names:
        paths = [base / sub / name for sub in ("A", "B", "label")]
        if not all(p.is_file() for p in paths):
            raise FileNotFoundError(f"A/B/label missing for {split}/{name}")
        before = np.asarray(Image.open(paths[0]).convert("RGB"))
        after = np.asarray(Image.open(paths[1]).convert("RGB"))
        raw_label = np.asarray(Image.open(paths[2]).convert("L"))
        if before.shape != (1024, 1024, 3) or after.shape != before.shape or raw_label.shape != before.shape[:2]:
            raise ValueError(f"Unexpected shape in {split}/{name}: {before.shape}, {after.shape}, {raw_label.shape}")
        # Observed labels are {0,156,254,255}; >127 is explicit and equivalent to >0.
        truth = raw_label > 127
        gray_diff = cv2.absdiff(cv2.cvtColor(before, cv2.COLOR_RGB2GRAY), cv2.cvtColor(after, cv2.COLOR_RGB2GRAY))
        yield Sample(name, before, after, truth, gray_diff)


def confusion(pred: np.ndarray, truth: np.ndarray):
    p, t = pred.astype(bool), truth.astype(bool)
    return {
        "tp": int(np.count_nonzero(p & t)),
        "fp": int(np.count_nonzero(p & ~t)),
        "fn": int(np.count_nonzero(~p & t)),
        "tn": int(np.count_nonzero(~p & ~t)),
    }


def summarize(counts: dict[str, int]):
    tp, fp, fn, tn = (counts[k] for k in ("tp", "fp", "fn", "tn"))
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {**counts, "precision": p, "recall": r,
            "f1": 2*p*r/(p+r) if p+r else 0.0,
            "iou": tp/(tp+fp+fn) if tp+fp+fn else 0.0,
            "pixels": tp+fp+fn+tn}


def check_total(counts: dict[str, int], sample_count: int, method: str):
    expected = sample_count * IMAGE_PIXELS
    got = sum(counts[k] for k in ("tp", "fp", "fn", "tn"))
    assert got == expected, f"{method}: confusion total {got} != expected {expected}"


def per_image_row(method: str, sample: Sample, pred: np.ndarray):
    c = confusion(pred, sample.truth)
    check_total(c, 1, f"{method}/{sample.name}")
    positive_gt = c["tp"] + c["fn"]
    positive_pred = c["tp"] + c["fp"]
    if positive_gt == 0:
        return {"method": method, "filename": sample.name, **c,
                "precision": "", "recall": "", "f1": "", "iou": "",
                "status": "no_ground_truth_change; excluded from metric median/worst ranking"}
    p = c["tp"] / positive_pred if positive_pred else None
    r = c["tp"] / positive_gt
    f1 = 2*p*r/(p+r) if p is not None and p+r else 0.0
    iou = c["tp"] / (c["tp"] + c["fp"] + c["fn"])
    status = "ok" if p is not None else "no_predicted_change; precision undefined"
    return {"method": method, "filename": sample.name, **c,
            "precision": "" if p is None else p, "recall": r, "f1": f1, "iou": iou, "status": status}


def aggregate(method: str, preds: list[np.ndarray], samples: list[Sample], runtime: float,
              *, note="", device="", threshold="", parameters=""):
    totals = {k: 0 for k in ("tp", "fp", "fn", "tn")}
    rows = []
    for sample, pred in zip(samples, preds):
        c = confusion(pred, sample.truth)
        for key in totals: totals[key] += c[key]
        rows.append(per_image_row(method, sample, pred))
    check_total(totals, len(samples), method)
    return {"method": method, **summarize(totals), "runtime_seconds": round(runtime, 3),
            "device": device, "threshold": threshold, "parameters": parameters,
            "interpretation": note}, rows


def find_threshold(samples: list[Sample]):
    pos = np.zeros(256, dtype=np.int64); neg = np.zeros(256, dtype=np.int64)
    for s in samples:
        pos += np.bincount(s.gray_diff[s.truth], minlength=256)
        neg += np.bincount(s.gray_diff[~s.truth], minlength=256)
    best = None
    for threshold in range(1, 256):
        tp, fp = int(pos[threshold:].sum()), int(neg[threshold:].sum())
        fn, tn = int(pos.sum()-tp), int(neg.sum()-fp)
        score = summarize({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
        if best is None or score["f1"] > best[0]["f1"]:
            best = (score, threshold)
    return best[1]


def basic_predictions(samples: list[Sample], threshold: int):
    return [basic_prediction(s, threshold) for s in samples]


def basic_prediction(sample: Sample, threshold: int):
    before=cv2.cvtColor(sample.before,cv2.COLOR_RGB2GRAY)
    after=cv2.cvtColor(sample.after,cv2.COLOR_RGB2GRAY)
    return cv2.absdiff(before,after) >= threshold


def improved_prediction(sample: Sample, blur: int, morph: int):
    a = cv2.GaussianBlur(cv2.cvtColor(sample.before, cv2.COLOR_RGB2GRAY), (blur, blur), 0)
    b = cv2.GaussianBlur(cv2.cvtColor(sample.after, cv2.COLOR_RGB2GRAY), (blur, blur), 0)
    diff = cv2.absdiff(a, b)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    if morph:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (morph, morph))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask > 0


def tune_improved(samples: list[Sample]):
    best = None
    for blur in (3, 5, 7):
        for morph in (0, 3, 5):
            preds = [improved_prediction(s, blur, morph) for s in samples]
            total = {k: 0 for k in ("tp", "fp", "fn", "tn")}
            for s, p in zip(samples, preds):
                for k, v in confusion(p, s.truth).items(): total[k] += v
            score = summarize(total)
            if best is None or score["f1"] > best[0]["f1"]:
                best = (score, blur, morph)
    return best[1], best[2]


def tinycd_predict(model, sample: Sample, device: torch.device, batch_size=16):
    coords = [(y, x) for y in range(0, 1024, 256) for x in range(0, 1024, 256)]
    pred = np.zeros((1024, 1024), dtype=bool)
    with torch.inference_mode():
        for start in range(0, len(coords), batch_size):
            group = coords[start:start+batch_size]
            a = torch.stack([torch.from_numpy(sample.before[y:y+256, x:x+256].copy()).permute(2,0,1).float()/255 for y,x in group])
            b = torch.stack([torch.from_numpy(sample.after[y:y+256, x:x+256].copy()).permute(2,0,1).float()/255 for y,x in group])
            prob = model(NORMALIZE(a).to(device), NORMALIZE(b).to(device))[:,0].cpu().numpy()
            for (y,x), tile in zip(group, prob): pred[y:y+256, x:x+256] = tile >= 0.5
    return pred


def metric_png(rows):
    names = [r["method"] for r in rows]
    colors = ["#5b8ff9", "#61dDAA", "#f6bd16", "#e8684a"]
    canvas = Image.new("RGB", (1120, 520), "white"); draw = ImageDraw.Draw(canvas)
    draw.text((25, 18), "LEVIR-CD test metrics (pixel-micro)", fill="#172b4d")
    keys = ("precision", "recall", "f1", "iou")
    for j, key in enumerate(keys):
        x0 = 230 + j*215; draw.text((x0, 58), key.upper(), fill="#222")
        for i, row in enumerate(rows):
            y=105+i*92; width=int(150*row[key])
            draw.rectangle((x0,y,x0+150,y+24),fill="#eeeeee")
            draw.rectangle((x0,y,x0+width,y+24),fill=colors[i])
            draw.text((x0,y+31),f"{row[key]:.3f}",fill="#222")
    for i,name in enumerate(names): draw.text((20,110+i*92),name[:27],fill="#222")
    canvas.save(OUT/"benchmark_test.png")


def write_visual(sample: Sample, pred_by_method: dict[str,np.ndarray], tag: str):
    panels=[sample.before,sample.after,np.repeat(sample.truth[...,None].astype(np.uint8)*255,3,axis=2),
            np.repeat(pred_by_method["OpenCV improved"][...,None].astype(np.uint8)*255,3,axis=2),
            np.repeat(pred_by_method["TinyCD"][...,None].astype(np.uint8)*255,3,axis=2)]
    labels=["A / earlier","B / later","Ground truth","OpenCV improved","TinyCD"]
    sheet=Image.new("RGB",(1250,440),"white"); d=ImageDraw.Draw(sheet)
    for i,(im,label) in enumerate(zip(panels,labels)):
        x=10+i*248; d.text((x,8),label,fill="#202020"); sheet.paste(Image.fromarray(im).resize((238,400)),(x,30))
    sheet.save(OUT/f"test_{tag}_{sample.name.removesuffix('.png')}.png")


def write_per_image_report(rows):
    path=OUT/"benchmark_test_per_image.csv"
    fields=["method","filename","tp","fp","fn","tn","precision","recall","f1","iou","status"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)
    lines=["Per-image metrics: no-ground-truth-change images are excluded from metric summaries; see status and FP counts."]
    methods=sorted(set(r["method"] for r in rows))
    for method in methods:
        group=[r for r in rows if r["method"]==method]
        valid=[r for r in group if isinstance(r["f1"],(float,int))]
        excluded=len(group)-len(valid)
        lines.append(f"\n{method}: valid={len(valid)}, excluded_no_gt_change={excluded}")
        for key in ("precision","recall","f1","iou"):
            vals=sorted(float(r[key]) for r in valid if isinstance(r[key],(float,int)))
            if vals:
                n=len(vals); median=(vals[(n-1)//2]+vals[n//2])/2
                lines.append(f"median {key}={median:.6f}")
        worst=sorted(valid,key=lambda r:float(r["f1"]))[:5]
        lines.append("worst 5 by F1: "+", ".join(f"{r['filename']} ({float(r['f1']):.4f})" for r in worst))
    (OUT/"benchmark_test_per_image_summary.txt").write_text("\n".join(lines)+"\n")


def save_csv(path: Path, rows):
    fields=["method","interpretation","precision","recall","f1","iou","tp","fp","fn","tn","pixels","runtime_seconds","device","threshold","parameters"]
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n"); w.writeheader(); w.writerows(rows)


def run():
    OUT.mkdir(exist_ok=True)
    val=list(load_samples("val")); test=list(load_samples("test"))

    # Tune every selectable OpenCV parameter on validation only.
    val_start=time.perf_counter(); basic_threshold=find_threshold(val); basic_tune_s=time.perf_counter()-val_start
    improved_blur,improved_morph=tune_improved(val)
    print(f"Validation tuning: basic threshold={basic_threshold}; improved Gaussian={improved_blur}, morphology={improved_morph}",flush=True)

    val_rows=[]
    t=time.perf_counter(); p=basic_predictions(val,basic_threshold); elapsed=time.perf_counter()-t
    val_rows.append(aggregate("OpenCV basic",p,val,elapsed,threshold=basic_threshold,parameters="RGB2GRAY absdiff" )[0])
    t=time.perf_counter(); p=[improved_prediction(s,improved_blur,improved_morph) for s in val]; elapsed=time.perf_counter()-t
    val_rows.append(aggregate("OpenCV improved",p,val,elapsed,parameters=f"Gaussian={improved_blur}; Otsu; open+close={improved_morph}" )[0])
    device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model=ChangeClassifier(pretrained=False).eval().to(device)
    model.load_state_dict(torch.load(WEIGHTS,map_location="cpu",weights_only=True),strict=True)
    t=time.perf_counter(); vp=[]
    for i,s in enumerate(val,1):
        vp.append(tinycd_predict(model,s,device))
        if i%8==0 or i==len(val): print(f"TinyCD validation: {i}/{len(val)}",flush=True)
    elapsed=time.perf_counter()-t
    val_rows.append(aggregate("TinyCD",vp,val,elapsed,note=VAL_NOTE,device=str(device),parameters="0.5 cutoff; 256px tiles")[0])
    save_csv(OUT/"benchmark_val.csv",val_rows)

    # Held-out test: fixed validation parameters, plus explicitly test-tuned oracle.
    test_rows=[]; per_rows=[]; masks_by_name={}
    for method, fn, threshold, params in [
        ("OpenCV basic",lambda s:s.gray_diff>=basic_threshold,basic_threshold,"RGB2GRAY absdiff"),
        ("OpenCV improved",lambda s:improved_prediction(s,improved_blur,improved_morph),"",f"Gaussian={improved_blur}; Otsu; open+close={improved_morph}"),
    ]:
        t=time.perf_counter(); preds=[fn(s) for s in test]; elapsed=time.perf_counter()-t
        summary,rows=aggregate(method,preds,test,elapsed,threshold=threshold,parameters=params)
        test_rows.append(summary); per_rows.extend(rows)
        masks_by_name[method]=preds

    t=time.perf_counter(); oracle_threshold=find_threshold(test); oracle_preds=basic_predictions(test,oracle_threshold); elapsed=time.perf_counter()-t
    summary,rows=aggregate("OpenCV oracle",oracle_preds,test,elapsed,note="oracle upper bound, test-tuned, NOT a fair result (this threshold family only)",threshold=oracle_threshold,parameters="RGB2GRAY absdiff; test sweep")
    test_rows.append(summary); per_rows.extend(rows); masks_by_name["OpenCV oracle"]=oracle_preds

    t=time.perf_counter(); tiny_preds=[]
    for i,s in enumerate(test,1):
        tiny_preds.append(tinycd_predict(model,s,device))
        if i%8==0 or i==len(test): print(f"TinyCD test: {i}/{len(test)}",flush=True)
    elapsed=time.perf_counter()-t
    summary,rows=aggregate("TinyCD",tiny_preds,test,elapsed,note="held-out test; checkpoint provenance in PROVENANCE.md",device=str(device),parameters="0.5 cutoff; 256px tiles")
    test_rows.append(summary); per_rows.extend(rows); masks_by_name["TinyCD"]=tiny_preds
    save_csv(OUT/"benchmark_test.csv",test_rows)
    metric_png(test_rows)
    write_per_image_report(per_rows)

    # Select visual examples by TinyCD F1 on changed-GT images. These remain local.
    tiny_rows=[r for r in rows if isinstance(r["f1"],(float,int))]
    # rows currently contains TinyCD rows only; recompute its per-image metrics.
    tiny_rows=[per_image_row("TinyCD",s,p) for s,p in zip(test,tiny_preds)]
    ranked=[i for i,r in enumerate(tiny_rows) if r["status"] != "no_ground_truth_change; excluded from metric median/worst ranking"]
    ranked.sort(key=lambda i:float(tiny_rows[i]["f1"]))
    bad=ranked[:3]; good=ranked[-3:][::-1]
    for tag,indices in (("good",good),("bad",bad)):
        for rank,index in enumerate(indices,1):
            preds={name:arr[index] for name,arr in masks_by_name.items()}
            write_visual(test[index],preds,f"{tag}_{rank}")

    print(f"VAL threshold={basic_threshold}; improved blur={improved_blur}, morph={improved_morph}; TinyCD={device}")
    print("TEST rows:")
    for r in test_rows:
        print(f"{r['method']}: P={r['precision']:.4f} R={r['recall']:.4f} F1={r['f1']:.4f} IoU={r['iou']:.4f}; runtime={r['runtime_seconds']:.2f}s")
    print(f"Basic validation threshold search seconds={basic_tune_s:.3f}")


if __name__=="__main__":
    run()
