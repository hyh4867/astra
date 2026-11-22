"""Dev-only morphology, region, time and representative errors on the native grid."""
from collections import defaultdict
import numpy as np
import cv2
from study_io import (
    WORK, counts, manifest, metrics, pair, write,
)


def diagnose():
    dest=WORK/'baseline_validation'
    groups=defaultdict(list)
    shapes=defaultdict(lambda:[0,0])
    candidates=defaultdict(list)
    for r in manifest('validation'):
        n=r['image_id']
        _,_,gt=pair('validation',n)
        mask=cv2.imread(str(dest/(n+'.png')),cv2.IMREAD_GRAYSCALE)>0
        c=counts(mask,gt)
        groups['department',r['dep']].append(c)
        groups['year_pair',r['year_before']+'-'+r['year_after']].append(c)
        _,labels,stats,_=cv2.connectedComponentsWithStats(gt.astype('uint8'),connectivity=8)
        hits=np.bincount(labels[mask],minlength=len(stats))
        for i,(_,_,w,h,a) in enumerate(stats[1:],1):
            aspect=max(w,h)/min(w,h)
            key=('elongated' if aspect>=2 else 'compact','under_25m2' if a*.04<25 else 'at_least_25m2')
            shapes[key][0]+=1
            shapes[key][1]+=int(hits[i]>=.3*a)
        for k in ['fp','fn']:candidates[k].append((c[k],n))
    write(WORK/'dev_strata.csv',[dict(factor=k[0],group=k[1],pairs=len(v),**metrics(v)) for k,v in sorted(groups.items())])
    write(WORK/'dev_shape.csv',[dict(aspect=k[0],area=k[1],objects=v[0],found=v[1],completeness=v[1]/v[0]) for k,v in sorted(shapes.items())])
    write(WORK/'dev_errors.csv',[dict(error=k,image_id=n,pixels=count) for k,v in candidates.items() for count,n in sorted(v,reverse=True)[:3]])


if __name__=='__main__':diagnose()
