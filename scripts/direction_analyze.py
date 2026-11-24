"""Aggregate the frozen run, bootstrap paired seed-42 differences, and verify counts."""
import csv
import json
from shutil import copyfile

import numpy as np

from direction_protocol import CACHE, FAMILIES, OUTPUT, ROOT, SEEDS
from study_io import dump, write
from verify_direction import verify

PIXEL = ("tp","fp","fn")
DIRECTION = ("new_hits","new_pixels","disappeared_hits","disappeared_pixels")
METRICS = ("f1","precision","recall","new_recall","disappeared_recall")


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def calculate(sums):
    tp, fp, fn = sums[...,0], sums[...,1], sums[...,2]
    result = [2*tp/(2*tp+fp+fn),tp/(tp+fp),tp/(tp+fn)]
    if sums.shape[-1] == 7:
        result += [sums[...,3]/sums[...,4],sums[...,5]/sums[...,6]]
    return np.stack(result, axis=-1)


def summarize():
    src = CACHE / "locked_test"
    stamp = json.loads((src/"provenance.json").read_text())
    copyfile(src/"provenance.json", OUTPUT/"run.json")
    scores, summaries, bootstrap, baseline_comparison = [], [], [], []
    for dataset in ("fotbcd","levir"):
        file = f"{dataset}_counts.csv"
        copyfile(src/file, OUTPUT/file)
        records = read(src/file)
        fields = PIXEL + (DIRECTION if dataset=="fotbcd" else ())
        matrices = {}
        names = sorted({r["name"] for r in records})
        for family in FAMILIES:
            seed_values = []
            for seed in SEEDS:
                group = [r for r in records if r["family"]==family and int(r["seed"])==seed]
                group.sort(key=lambda r: r["name"])
                assert [r["name"] for r in group] == names
                array = np.array([[int(r[k]) for k in fields] for r in group],dtype=np.int64)
                if seed == 42:
                    matrices[family] = array
                sums = array.sum(axis=0)
                values = calculate(sums)
                seed_values.append(values)
                item = dict(dataset=dataset,family=family,seed=seed,pairs=len(group),**stamp)
                item.update({k:int(v) for k,v in zip(fields,sums)})
                item.update({k:float(v) for k,v in zip(METRICS,values)})
                scores.append(item)
            for i,k in enumerate(METRICS[:len(values)]):
                v = np.array(seed_values)[:,i]
                summaries.append(dict(dataset=dataset, family=family, metric=k,
                    mean=float(v.mean()), minimum=float(v.min()), maximum=float(v.max()), span=float(np.ptp(v))))
        prior = ROOT / "outputs/generalization"
        prior_file = prior / ("final/adapted/counts.csv" if dataset=="fotbcd" else "levir_adapted.csv")
        history = {r["name"]:r for r in read(prior_file)}
        assert set(history)==set(names)
        if dataset=="fotbcd":
            directions={r["name"]:r for r in read(prior/"direction_test.csv") if r["method"]=="adapted"}
            assert set(directions)==set(names)
            for n in names:
                for k in DIRECTION:
                    history[n][k]=directions[n][k.replace("disappeared","demolished")]
        old = np.array([[int(history[n][k]) for k in fields] for n in names])
        delta = matrices["A"] - old
        baseline_comparison.append(dict(dataset=dataset, pairs=len(names),
            pairs_with_any_difference=int(np.any(delta!=0,axis=1).sum()),
            total_count_differences=dict(zip(fields,[int(x) for x in delta.sum(axis=0)])),
            previous_f1=float(calculate(old.sum(axis=0))[0]),
            current_f1=float(calculate(matrices["A"].sum(axis=0))[0])))
        if dataset == "fotbcd":
            rng=np.random.default_rng(42)
            ids=rng.integers(len(names),size=(1000,len(names)))
            base = calculate(matrices["A"][ids].sum(axis=1))
            for family in FAMILIES:
                if family=="A":
                    continue
                diff=calculate(matrices[family][ids].sum(axis=1))-base
                low,high=np.quantile(diff,[.025,.975],axis=0)
                points=calculate(matrices[family].sum(axis=0))-calculate(matrices["A"].sum(axis=0))
                for i,k in enumerate(METRICS):
                    bootstrap.append(dict(comparison=family+" minus A",metric=k,delta=float(points[i]),
                                    lower95=float(low[i]),upper95=float(high[i]),seed=42,draws=1000))
    # LEVIR rows have no direction labels; retain the FOTBCD table's columns.
    write(OUTPUT/"scores.csv",scores)
    write(OUTPUT/"family_summary.csv",summaries)
    write(OUTPUT/"bootstrap_seed42.csv",bootstrap)
    dump(OUTPUT/"baseline_reproduction.json",baseline_comparison)
    dump(OUTPUT/"independent_f1.json",verify())
    print("All final F1 checks passed",flush=True)


if __name__ == "__main__":
    summarize()
