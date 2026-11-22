"""Match polygons by metadata and measure directional recall on saved predictions."""
import json
import tarfile

import cv2
import numpy as np
from rasterio.features import rasterize
from affine import Affine

from PIL import Image
from study_io import (
    RAW, WORK, manifest, write,
)


def footprint(dep,epsg,y1,y2,box):
    return str(dep),int(epsg),int(y1),int(y2),tuple(round(float(box[k]),3) for k in ('minx','miny','maxx','maxy'))


def match(split):
    release_split='val' if split=='validation' else 'test'
    rows=manifest(split)
    wanted={footprint(r['dep'],r['epsg'],r['year_before'],r['year_after'],json.loads(r['patch_bounds'])):r['image_id'] for r in rows}
    assert len(wanted)==len(rows)
    found={}
    with tarfile.open(RAW/f'{release_split}-000000.tar') as tar:
        for member in tar:
            if not member.name.endswith('.json'): continue
            rec=json.load(tar.extractfile(member)); im=rec['image']
            key=footprint(im['dep'],im['epsg'],im['before_year'],im['after_year'],im['patch_bounds'])
            if key not in wanted: continue
            n=wanted[key]
            if n in found: raise ValueError(f'Ambiguous footprint {n}')
            assert (im['width'],im['height'])==(512,512)
            assert {c['id']:c['name'] for c in rec['categories']}=={1:'UNCHANGED',2:'DEMOLISHED',3:'NEW'}
            found[n]=rec
    return found


def class_mask(rec,category):
    shapes=[]
    for ann in rec['annotations']:
        if ann['category_id']!=category:continue
        for polygon in ann['segmentation']:
            xy=np.reshape(polygon,(-1,2)).tolist()
            if len(xy)>=3:shapes.append((dict(type='Polygon',coordinates=[xy]),1))
    if not shapes:return np.zeros((512,512),bool)
    return rasterize(shapes,out_shape=(512,512),transform=Affine.identity(),dtype='uint8').astype(bool)


def direction(split,folders):
    records=match(split)
    total=len(manifest(split))
    if split=='test' and len(records)!=total:
        return dict(split=split,matched=len(records),total=total,available=False)
    rows=[]
    for n,rec in sorted(records.items()):
        gt=np.array(Image.open(WORK/split/n/'label').convert('L'))>0
        new=class_mask(rec,3)
        dem=class_mask(rec,2)
        for method,folder in folders.items():
            pred=cv2.imread(str(folder/(n+'.png')),cv2.IMREAD_GRAYSCALE)>0
            r=dict(name=n,method=method,polygon_union=int((new|dem).sum()),
                   binary_gt=int(gt.sum()),intersection=int((gt&(new|dem)).sum()),
                   both=int((dem&new).sum()))
            for c,m in [('new',new),('demolished',dem)]:
                r[c+'_pixels']=int(m.sum())
                r[c+'_hits']=int((pred&m).sum())
            rows.append(r)
    write(WORK/f'direction_{split}.csv',rows)
    result={}
    for method in folders:
        g=[r for r in rows if r['method']==method]
        result[method]={}
        for c in ['new','demolished']:
            hits=sum(r[c+'_hits'] for r in g)
            pixels=sum(r[c+'_pixels'] for r in g)
            result[method][c]=dict(hits=hits,pixels=pixels,recall=hits/pixels if pixels else None)
    return dict(split=split,matched=len(records),total=total,available=True,methods=result)
