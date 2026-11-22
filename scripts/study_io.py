"""Per-pair files in a private working directory for the preregistered study."""
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image

from fetch_generalization import ROOT, RAW
from freeze_generalization import MANIFEST, rank, write
sys.path.insert(0,str(ROOT))
from inference import tinycd_probability, load_tinycd_model

WORK = RAW / 'work'
ORIGINAL = ROOT / 'models/tinycd_levir_best.pth'


def digest(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def dump(path, obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(obj,sort_keys=True,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def manifest(split):
    source=RAW/f'{split}_metadata.csv'
    audit=json.loads((MANIFEST/'audit.json').read_text())[split]
    if digest(source)!=audit['metadata_sha256'] or digest(MANIFEST/f'{split}.csv')!=audit['sha256']:
        raise ValueError(f'Frozen metadata mismatch for {split}')
    with source.open() as f:
        return list(csv.DictReader(f))


def extract(split):
    import pyarrow.parquet as pq
    from collections import defaultdict
    wanted=defaultdict(dict)
    dest=WORK/split
    dest.mkdir(parents=True,exist_ok=True)
    for r in manifest(split):
        wanted[r['shard']][int(r['row'])]=r['image_id']
    for shard,indices in sorted(wanted.items()):
        missing={i:name for i,name in indices.items() if not (dest/name/'label').is_file()}
        if not missing:
            continue
        offset=0
        for batch in pq.ParquetFile(RAW/shard).iter_batches(batch_size=4):
            for i in range(batch.num_rows):
                name=missing.get(offset+i)
                if name is None:
                    continue
                row=batch.slice(i,1).to_pylist()[0]
                assert row['image_id']==name and name.isdecimal()
                folder=dest/name
                folder.mkdir(exist_ok=True)
                for col in ['image_before','image_after','label']:
                    part=folder/(col+'.part')
                    part.write_bytes(row[col]['bytes'])
                    part.replace(folder/col)
            offset+=batch.num_rows


def pair(split,name):
    p=WORK/split/name
    a,b=[np.array(Image.open(p/k).convert('RGB')) for k in ('image_before','image_after')]
    label=np.array(Image.open(p/'label').convert('L'))
    assert a.shape==b.shape==(512,512,3) and label.shape==(512,512) and np.isin(label,[0,255]).all()
    return a,b,label>0


def images(a,b,grid):
    if grid=='native':
        return a,b
    return tuple(cv2.resize(im,(205,205),interpolation=cv2.INTER_AREA) for im in (a,b))


def probability(model,device,a,b):
    return tinycd_probability(model,a,b,device,batch_size=4)


def restore(prob,shape):
    if prob.shape==shape:
        return prob
    return cv2.resize(prob,(shape[1],shape[0]),interpolation=cv2.INTER_LINEAR)


def counts(pred,gt):
    hit=int(np.count_nonzero(pred & gt))
    return dict(tp=hit,fp=int(pred.sum())-hit,fn=int(gt.sum())-hit)


def metrics(rows):
    tp,fp,fn=(sum(int(r[k]) for r in rows) for k in ('tp','fp','fn'))
    def ratio(a,b):
        return a/b if b else None
    return dict(tp=tp,fp=fp,fn=fn,precision=ratio(tp,tp+fp),recall=ratio(tp,tp+fn),
                f1=ratio(2*tp,2*tp+fp+fn),iou=ratio(tp,tp+fp+fn))
