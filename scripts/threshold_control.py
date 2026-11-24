"""Reproduce frozen controls, select thresholds on validation, then score saved test probabilities."""
import argparse
import csv
import json
import subprocess
import time

import numpy as np

from direction_protocol import ROOT, SEEDS, weights
from direction_resources import wait_for_memory
from study_direction import class_mask, match
from study_io import MANIFEST, counts, digest, dump, images, load_tinycd_model, manifest, pair, probability, restore, write

OLD = ROOT / "outputs/direction"
OUT = ROOT / "outputs/direction_control"
CACHE = ROOT / "data/raw/direction_control"
KEYS = ("tp", "fp", "fn", "new_hits", "new_pixels", "disappeared_hits", "disappeared_pixels")
CELLS = {"F50": ("forward", 50), "M30": ("max", 30), "F30": ("forward", 30), "M50": ("max", 50)}
GRID = tuple(range(10, 91, 5))


def read(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def guard():
    wait_for_memory(enforce_direction_deadline=False)


def source():
    frozen = json.loads((OLD / "frozen.json").read_text())
    hashes = {m["seed"]: m["weights_sha256"] for m in frozen["models"] if m["model"] == "A"}
    assert set(hashes) == set(SEEDS)
    for s in SEEDS:
        assert digest(weights("A", s)) == hashes[s]
    for split in ("validation", "test"):
        assert digest(MANIFEST / f"{split}.csv") == frozen["manifests"][split]
    assert digest(ROOT/"data/manifests/test_filenames.txt") == frozen["levir_test_manifest"]
    return dict(weights=hashes, previous_config=digest(OLD/"frozen.json"),
                previous_counts=digest(OLD/"fotbcd_counts.csv"), manifests=frozen["manifests"],
                levir_manifest=frozen["levir_test_manifest"])


def infer(model, device, a, b, shape):
    # probability() transfers results to CPU before returning.
    start = time.perf_counter()
    f = restore(probability(model, device, a, b), shape)
    mid = time.perf_counter()
    r = restore(probability(model, device, b, a), shape)
    maximum = np.maximum(f, r)
    return f, maximum, mid-start, time.perf_counter()-mid


def binary(mask, truth, targets):
    row = counts(mask, truth)
    for k, g in targets.items():
        row[k+"_hits"] = int((mask & g).sum())
        row[k+"_pixels"] = int(g.sum())
    return row


def targets_for(rec):
    return {k:class_mask(rec, idx) for k,idx in (("new",3),("disappeared",2))}


def reproduce():
    guard()
    assert not git("status","--porcelain")
    stamp = dict(source=source(), commit=git("rev-parse","HEAD"), git_dirty=False)
    names = [r["image_id"] for r in manifest("test")]
    polygons = match("test")
    assert len(names) == len(set(names)) == len(polygons) == 979 and set(names) == set(polygons)
    old = {(int(r["seed"]),r["family"],r["name"]):r for r in read(OLD/"fotbcd_counts.csv") if r["family"] in ("A","A+T")}
    assert len(old)==len(names)*6
    OUT.mkdir(parents=True,exist_ok=True)
    stamps = CACHE/"cache_source.json"
    if stamps.exists():
        assert json.loads(stamps.read_text())==json.loads(json.dumps(stamp))
    else:
        dump(stamps,stamp)
    summaries, times = [], []
    for seed in SEEDS:
        guard()
        model, device = load_tinycd_model(weights("A", seed))
        folder=CACHE/str(seed)
        folder.mkdir(exist_ok=True)
        for name in names:
            a,b,truth = pair("test",name)
            zones = targets_for(polygons[name])
            path = folder/f"{name}.npy"
            timing = folder/f"{name}.json"
            if path.exists() and timing.exists():
                f,m = np.load(path, allow_pickle=False)
                times.append(json.loads(timing.read_text()))
            else:
                a,b=images(a,b,"half")
                f,m,ft,rt=infer(model,device,a,b,truth.shape)
                row=dict(seed=seed,name=name,forward_s=ft,reverse_and_max_s=rt)
                tmp=path.with_suffix(".tmp.npy")
                np.save(tmp,np.stack([f,m]),allow_pickle=False)
                tmp.replace(path)
                dump(timing,row)
                times.append(row)
            for family,mask in (("A",f>=.5),("A+T",m>=.3)):
                measured=binary(mask,truth,zones)
                expected=old[seed,family,name]
                mismatch={k:[int(expected[k]),v] for k,v in measured.items() if v!=int(expected[k])}
                if mismatch:
                    dump(OUT/"reproduction_failure.json",dict(seed=seed,name=name,family=family,mismatch=mismatch,**stamp))
                    raise ValueError(f"Reproduction mismatch: {seed} {family} {name} {mismatch}")
        for family in ("A","A+T"):
            summaries.append(dict(seed=seed,family=family,matched=len(names),fields=list(KEYS)))
        print("Exact reproduction",seed,len(names),"pairs x 2",flush=True)
        del model
    write(OUT/"fotbcd_inference_time.csv",times)
    dump(OUT/"reproduction.json",dict(**stamp,checks=summaries))


def pr(tp,fp,fn):
    return dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)


def choose(means):
    base=next(r for r in means if (r["mode"],r["pct"])==("forward",50))
    tuned=max((r for r in means if r["mode"]=="forward" and r["pct"] in (30,40,50)),
              key=lambda r:(r["f1"],r["pct"]))
    eligible=[r for r in means if r["mode"]=="max" and r["pct"]>=30 and r["precision"]>=base["precision"]]
    matched=max(eligible,key=lambda r:(r["recall"],r["precision"],r["pct"]),default=None)
    return tuned,matched,base


def validate():
    repro=json.loads((OUT/"reproduction.json").read_text())
    assert len(repro["checks"])==6 and all(r["matched"]==979 for r in repro["checks"])
    assert repro["source"] == json.loads(json.dumps(source()))
    names=[r["image_id"] for r in manifest("validation")]
    assert len(names)==len(set(names))==984
    # A cumulative histogram yields exact threshold counts including equality.
    thresholds=np.array(GRID,dtype=np.float32)/100
    rows=[]
    for seed in SEEDS:
        guard()
        model,device=load_tinycd_model(weights("A",seed))
        hist=np.zeros((2,2,len(GRID)+1),dtype=np.int64)
        for name in names:
            a,b,g=pair("validation",name)
            a,b=images(a,b,"half")
            f,m,_,_=infer(model,device,a,b,g.shape)
            for i,p in enumerate((f,m)):
                bins=np.searchsorted(thresholds,p,side="right")
                hist[i,0]+=np.bincount(bins[g],minlength=len(GRID)+1)
                hist[i,1]+=np.bincount(bins[~g],minlength=len(GRID)+1)
        for i,mode in enumerate(("forward","max")):
            above=hist[i,:,::-1].cumsum(axis=1)[:,::-1]
            positives=int(hist[i,0].sum())
            for j,t in enumerate(GRID,1):
                tp,fp=map(int,above[:,j]); fn=positives-tp
                rows.append(dict(seed=seed,mode=mode,pct=t,tp=tp,fp=fp,fn=fn,**pr(tp,fp,fn)))
        del model
        print("Validation curve",seed,flush=True)
    write(OUT/"validation_pr.csv",rows)
    means=[]
    for mode in ("forward","max"):
        for t in GRID:
            g=[r for r in rows if (r["mode"],r["pct"])==(mode,t)]
            means.append(dict(mode=mode,pct=t,**{k:sum(r[k] for r in g)/len(g) for k in ("precision","recall","f1")}))
    tuned,matched,base=choose(means)
    policies=dict(CELLS, A_tuned=("forward",tuned["pct"]))
    if matched: policies["A_T_P"]=("max",matched["pct"])
    write(OUT/"validation_mean.csv",means)
    dump(OUT/"frozen.json",dict(policies=policies,baseline=base,tuned=tuned,matched=matched,
         previous_source=repro["source"],validation_sha256=digest(OUT/"validation_pr.csv"),
         reproduction_sha256=digest(OUT/"reproduction.json")))
    print("Selected",policies,flush=True)


def score():
    guard()
    freeze=OUT/"frozen.json"
    assert not git("status","--porcelain")
    git("ls-files","--error-unmatch",str(freeze.relative_to(ROOT)))
    config=json.loads(freeze.read_text())
    assert config["previous_source"]==json.loads(json.dumps(source()))
    assert digest(OUT/"reproduction.json")==config["reproduction_sha256"]
    stamp=dict(commit=git("rev-parse","HEAD"),config_sha256=digest(freeze),git_dirty=False)
    french=[r["image_id"] for r in manifest("test")]
    polygons=match("test")
    levir=(ROOT/"data/manifests/test_filenames.txt").read_text().split()
    assert len(set(levir))==len(levir)==128
    for data in ("fotbcd","levir"):
        rows=[]
        for seed in SEEDS:
            guard()
            if data=="levir":
                model,device=load_tinycd_model(weights("A",seed))
                def samples():
                    from evaluate_benchmark import load_samples
                    for s in load_samples("test",levir):
                        f,m,_,_=infer(model,device,s.before,s.after,s.truth.shape)
                        yield s.name,f,m,s.truth,{}
            else:
                def samples():
                    for n in french:
                        f,m=np.load(CACHE/str(seed)/f"{n}.npy",allow_pickle=False)
                        _,_,g=pair("test",n)
                        yield n,f,m,g,targets_for(polygons[n])
            for n,f,m,g,zones in samples():
                per={}
                for policy,(mode,t) in config["policies"].items():
                    key=(mode,t)
                    if key not in per:
                        per[key]=binary((f if mode=="forward" else m)>=t/100,g,zones)
                    rows.append(dict(policy=policy,seed=seed,name=n,**per[key]))
            if data=="levir": del model
            print("Scored",data,seed,flush=True)
        write(CACHE/f"{data}_counts.csv",rows)
    dump(CACHE/"final_run.json",stamp)


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("stage",choices=("reproduce","validate","score"))
    args=parser.parse_args()
    {"reproduce":reproduce,"validate":validate,"score":score}[args.stage]()
