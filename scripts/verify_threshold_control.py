"""Independent, standard-library-only verification of the final F1 tables."""
import csv
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/direction_control"


def read(path):
    with path.open() as f:
        return list(csv.DictReader(f))


def verify():
    import json
    scores=read(OUT/"scores.csv")
    policies=json.loads((OUT/"frozen.json").read_text())["policies"]
    result=[]
    for dataset in ("fotbcd","levir"):
        names=({r["image_id"] for r in read(ROOT/"data/manifests/fotbcd/test.csv")}
               if dataset=="fotbcd" else set((ROOT/"data/manifests/test_filenames.txt").read_text().split()))
        assert len(names)==(979 if dataset=="fotbcd" else 128)
        rows=read(OUT/f"{dataset}_counts.csv")
        assert len(rows)==len(names)*len(policies)*3
        for policy in policies:
            for seed in ("42","1042","2042"):
                group=[r for r in rows if r["policy"]==policy and r["seed"]==seed]
                assert len(group)==len(names) and {r["name"] for r in group}==names
                tp,fp,fn=(sum(int(r[k]) for r in group) for k in ("tp","fp","fn"))
                f1=2*tp/(2*tp+fp+fn)
                matches=[r for r in scores if (r["dataset"],r["policy"],r["seed"])==(dataset,policy,seed)]
                assert len(matches)==1
                assert (tp,fp,fn)==tuple(int(matches[0][k]) for k in ("tp","fp","fn"))
                diff=abs(f1-float(matches[0]["f1"]))
                assert diff<=1e-12
                result.append(dict(dataset=dataset,policy=policy,seed=seed,pairs=len(group),tp=tp,fp=fp,fn=fn,
                                   f1=f1,absolute_difference=diff))
    assert len(scores)==len(result)
    return result


if __name__=="__main__":
    import json
    print(json.dumps(verify(),indent=2))
