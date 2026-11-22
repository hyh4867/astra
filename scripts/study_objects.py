"""Independent-object diagnostics on native FOTBCD masks, including a deranged null."""
from collections import defaultdict
import cv2
import numpy as np

from study_io import counts,metrics,manifest,pair,write
from final_analysis_additions import derangement


def objects(mask,other):
    n,labels,stats,_=cv2.connectedComponentsWithStats(mask.astype('uint8'),8)
    return stats[1:,cv2.CC_STAT_AREA], np.bincount(labels[other],minlength=n)[1:]


def object_report(split,folder):
    rows=manifest(split)
    names=[r['image_id'] for r in rows]
    mapping=derangement(sorted(names))
    area=np.zeros(7,int)
    found=np.zeros_like(area)
    null=np.zeros_like(area)
    fp_all=fp_large=0
    per_pair=[]
    def pred(n):
        return cv2.imread(str(folder/(n+'.png')),cv2.IMREAD_GRAYSCALE)>0
    for r in rows:
        n=r['image_id']
        _,_,gt=pair(split,n)
        mask=pred(n)
        areas,hits=objects(gt,mask)
        _,chance=objects(gt,pred(mapping[n]))
        bins=np.searchsorted([25,62.5,125,250,625,1250],areas*0.04,side='right')
        area+=np.bincount(bins,minlength=7)
        found+=np.bincount(bins[hits>=.30*areas],minlength=7)
        null+=np.bincount(bins[chance>=.30*areas],minlength=7)
        pa,ph=objects(mask,gt)
        false=ph<.1*pa
        fp_all+=int(false.sum())
        fp_large+=int(np.count_nonzero(false & (pa*.04>=5)))
        per_pair.append(dict(name=n,dep=r['dep'],year_before=r['year_before'],year_after=r['year_after'],**counts(mask,gt)))
    def binrow(i,a,f,z):
        c=f/a if a else None
        n=z/a if a else None
        return dict(bin=i,objects=int(a),found=int(f),null_found=int(z),completeness=c,null=n,
                    skill=(c-n)/(1-n) if a and n<1 else None)
    bins=[binrow(i,a,f,z) for i,(a,f,z) in enumerate(zip(area,found,null))]
    km2=len(names)*512**2*.04/1e6
    return per_pair,dict(**metrics(per_pair),bins=bins,all_objects=binrow('all',area.sum(),found.sum(),null.sum()),
                         km2=km2,false_alarms=fp_all,false_per_km2=fp_all/km2,false_ge5m2_per_km2=fp_large/km2)
