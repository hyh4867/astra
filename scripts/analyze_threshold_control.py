"""Matched-cell contrasts, paired bootstrap, path decomposition and final verification."""
import csv
import json
from pathlib import Path
from shutil import copyfile

import numpy as np

from direction_analyze import calculate
from direction_protocol import ROOT, SEEDS
from study_io import dump, write
from verify_threshold_control import verify

OUT=ROOT/"outputs/direction_control"
CACHE=ROOT/"data/raw/direction_control"
KEYS=("tp","fp","fn","new_hits","new_pixels","disappeared_hits","disappeared_pixels")
METRICS=("f1","precision","recall","new_recall","disappeared_recall")
CONTRASTS={
    "fusion_at_50": {"M50":1,"F50":-1},
    "fusion_at_30": {"M30":1,"F30":-1},
    "threshold_forward": {"F30":1,"F50":-1},
    "threshold_max": {"M30":1,"M50":-1},
    "interaction": {"M30":1,"F30":-1,"M50":-1,"F50":1},
    "joint_change": {"M30":1,"F50":-1},
    "primary": {"M30":1,"A_tuned":-1},
    "matched_vs_A": {"A_T_P":1,"F50":-1},
}


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def linear(values, weights):
    return sum(c*values[k] for k,c in weights.items())


def main():
    config=json.loads((OUT/"frozen.json").read_text())
    policies=config["policies"]
    scores,means,effects,boot=[],[],[],[]
    for dataset in ("fotbcd","levir"):
        path=OUT/f"{dataset}_counts.csv"
        copyfile(CACHE/path.name,path)
        rows=read(path)
        fields=KEYS if dataset=="fotbcd" else KEYS[:3]
        metrics=METRICS if dataset=="fotbcd" else METRICS[:3]
        names=sorted({r["name"] for r in rows})
        size=len(names)
        assert size==(979 if dataset=="fotbcd" else 128)
        idx=np.random.default_rng(42).integers(size,size=(1000,size))
        ave,points,draws={},{},{}
        for p in policies:
            vals=[]
            for seed in SEEDS:
                group=sorted((r for r in rows if (r["policy"],r["seed"])==(p,str(seed))),key=lambda r:r["name"])
                assert [r["name"] for r in group]==names
                x=np.array([[int(r[k]) for k in fields] for r in group],dtype=np.int64)
                sums=x.sum(axis=0)
                v=calculate(sums)
                vals.append(v)
                row=dict(dataset=dataset,policy=p,seed=seed,pairs=size)
                row.update(zip(fields,map(int,sums)))
                row.update(zip(metrics,map(float,v)))
                scores.append(row)
                if seed==42:
                    points[p]=v
                    draws[p]=calculate(x[idx].sum(axis=1))
            ave[p]=np.array(vals).mean(axis=0)
            means.append(dict(dataset=dataset,policy=p,**dict(zip(metrics,map(float,ave[p])))))
        for name,weights in CONTRASTS.items():
            if not weights.keys()<=ave.keys():
                continue
            obs=linear(points,weights)
            lo,hi=np.quantile(linear(draws,weights),[.025,.975],axis=0)
            for metric,m,p,l,h in zip(metrics,linear(ave,weights),obs,lo,hi):
                effects.append(dict(dataset=dataset,contrast=name,metric=metric,mean_difference=m))
                boot.append(dict(dataset=dataset,contrast=name,metric=metric,seed=42,draws=1000,
                                 delta=p,lower95=l,upper95=h))
    write(OUT/"scores.csv",scores)
    write(OUT/"means.csv",means)
    write(OUT/"contrasts.csv",effects)
    write(OUT/"bootstrap.csv",boot)
    copyfile(CACHE/"final_run.json",OUT/"final_run.json")
    dump(OUT/"independent_f1.json",verify())
    def effect(name):
        return next(r["mean_difference"] for r in effects if (r["dataset"],r["contrast"],r["metric"])==("fotbcd",name,"disappeared_recall"))
    joint=effect("joint_change")
    path_fractions={name:effect(name)/joint if joint else None for name in
                    ("fusion_at_50","threshold_max","threshold_forward","fusion_at_30")}
    dump(OUT/"path_fractions.json",dict(total_disappeared_recall_gain=joint,**path_fractions))
    t=read(OUT/"fotbcd_inference_time.csv")
    ft=sum(float(r["forward_s"]) for r in t)
    rt=sum(float(r["reverse_and_max_s"]) for r in t)
    dump(OUT/"timing.json",dict(pair_seed_runs=len(t),forward_seconds=ft,reverse_plus_max_seconds=rt,
                                 both_seconds=ft+rt,ratio=(ft+rt)/ft))
    print("Independent F1 verification complete",flush=True)


if __name__=="__main__":
    main()
