"""Audit split identity and positive-area spatial overlap without decoding images."""
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from fetch_generalization import RAW, ROOT

MANIFEST = ROOT / 'data/manifests/fotbcd'
COLS = ['image_id','dep','epsg','year_before','year_after','patch_bounds']


def rank(name):
    return hashlib.sha256(('42/'+name).encode()).hexdigest(), name


def write(path, rows):
    with path.open('w',newline='') as f:
        w = csv.DictWriter(f, rows[0], lineterminator='\n')
        w.writeheader()
        w.writerows(rows)


def bounds(row):
    b=json.loads(row['patch_bounds'])
    return [round(b[k],3) for k in ['minx','miny','maxx','maxy']]


def cells(box):
    x,y,u,v=box
    for i in range(math.floor(x/128),math.ceil(u/128)):
        for j in range(math.floor(y/128),math.ceil(v/128)):
            yield i,j


def freeze():
    import pyarrow.parquet as pq
    MANIFEST.mkdir(exist_ok=True)
    previous_ids=set()
    occupied=defaultdict(list)
    audit={}
    for split,n in [('test',979),('validation',984),('train',25908)]:
        rows=[]
        for path in sorted(RAW.glob(f'{split}-*.parquet')):
            for i,row in enumerate(pq.read_table(path,columns=COLS).to_pylist()):
                rows.append(dict(row, shard=path.name, row=i))
        assert len(rows)==n and len({r['image_id'] for r in rows})==n,(split,len(rows))
        collisions=sum(r['image_id'] in previous_ids for r in rows)
        kept=[]
        excluded=[]
        for row in rows:
            assert row['epsg']==2154
            x,y,u,v=box=bounds(row)
            overlap=any(min(u,b[2])>max(x,b[0]) and min(v,b[3])>max(y,b[1])
                        for cell in cells(box) for b in occupied[cell])
            clash=row['image_id'] in previous_ids
            if overlap:
                excluded.append(dict(image_id=row['image_id'],spatial=overlap,id_collision=clash))
            else:
                kept.append(row)
        for row in kept:
            box=bounds(row)
            for cell in cells(box):
                occupied[cell].append(box)
            previous_ids.add(row['image_id'])
        if split=='train':
            kept=sorted(kept,key=lambda r:rank(r['image_id']))[:1024]
        kept.sort(key=lambda r:r['image_id'])
        target=MANIFEST/f'{split}.csv'
        enriched=RAW/f'{split}_metadata.csv'
        write(enriched,kept)
        write(target,[{k:r[k] for k in ('image_id','shard','row')} for r in kept])
        if excluded:
            write(MANIFEST/f'{split}_excluded.csv',excluded)
        audit[split]=dict(source=n,split_local_id_reuse=collisions,excluded=len(excluded),eligible=n-len(excluded),used=len(kept),
                          metadata_sha256=hashlib.sha256(enriched.read_bytes()).hexdigest(),
                          sha256=hashlib.sha256(target.read_bytes()).hexdigest())
    (MANIFEST/'audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    print(audit)


if __name__=='__main__':
    freeze()
