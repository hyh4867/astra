"""Recompute micro-F1 from individual integer counts using only the standard library."""
import csv
import json
from math import isclose
from pathlib import Path


def verify(folder):
    with (folder/'counts.csv').open() as f:
        rows=list(csv.DictReader(f))
    expected=json.loads((folder/'metrics.json').read_text())
    tp,fp,fn=(sum(int(r[key]) for r in rows) for key in ('tp','fp','fn'))
    f1=2*tp/(2*tp+fp+fn)
    assert len(set(r['name'] for r in rows))==len(rows)==979
    assert all(expected[k]==v for k,v in [('tp',tp),('fp',fp),('fn',fn)])
    assert isclose(f1,expected['f1'],rel_tol=0,abs_tol=1e-12)
    return dict(tp=tp,fp=fp,fn=fn,recomputed=f1,difference=f1-expected['f1'],pairs=len(rows))


if __name__=='__main__':
    import sys
    for folder in sys.argv[1:]:
        print(folder,verify(Path(folder)))
