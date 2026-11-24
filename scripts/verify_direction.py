"""Independently check all 24 final micro-F1 results; Python standard library only."""
import csv
import json
import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/direction"


def verify(folder=OUT):
    with (folder/"scores.csv").open() as f:
        scores = list(csv.DictReader(f))
    configs = {(family, str(seed)) for family in ("A", "S", "A+T", "S+T") for seed in (42,1042,2042)}
    assert len(scores) == 24
    checks = []
    for split, size in (("fotbcd",979), ("levir",128)):
        if split == "fotbcd":
            with (ROOT/"data/manifests/fotbcd/test.csv").open() as f:
                expected = {r["image_id"] for r in csv.DictReader(f)}
        else:
            expected = set((ROOT/"data/manifests/test_filenames.txt").read_text().split())
        assert len(expected) == size
        with (folder / f"{split}_counts.csv").open() as f:
            records = list(csv.DictReader(f))
        assert len(records) == size * len(configs)
        selected = [v for v in scores if v["dataset"] == split]
        assert {(r["family"], r["seed"]) for r in selected} == configs
        for score in selected:
            rows = [r for r in records if (r["family"], r["seed"]) == (score["family"], score["seed"])]
            assert len(rows) == size and {r["name"] for r in rows} == expected
            tp, fp, fn = [sum(int(r[k]) for r in rows) for k in ("tp","fp","fn")]
            value = 2 * tp / (2 * tp + fp + fn)
            assert (tp,fp,fn) == tuple(int(score[k]) for k in ("tp","fp","fn"))
            delta = abs(value - float(score["f1"]))
            assert math.isclose(value, float(score["f1"]), abs_tol=1e-12, rel_tol=0)
            checks.append(dict(dataset=split, family=score["family"],seed=int(score["seed"]),pairs=size,
                               tp=tp,fp=fp,fn=fn,recomputed_f1=value,absolute_difference=delta))
    return checks


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
