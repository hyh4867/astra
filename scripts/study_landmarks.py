"""Independent keypoint translation checks on preregistered external validation IDs.

Create inspection sheets and apply explicit point exclusions before interpreting
a natural displacement. Nothing here reads a prediction or a change mask.
"""
from collections import defaultdict
from PIL import ImageDraw
import numpy as np
import cv2
from PIL import Image
from study_io import (
    WORK, dump, manifest, rank, write,
)


def differences(x,y):
    return np.linalg.norm(x-y,axis=1)


def fit(d):
    if len(d)==0:
        return np.zeros(2)
    # A one-match translation is a minimal RANSAC hypothesis. For this bounded
    # sample, score every hypothesis deterministically, then refit by median.
    support=np.array([(differences(d,v)<=1.5).sum() for v in d])
    best=d[support.argmax()]
    return np.median(d[differences(d,best)<=1.5],axis=0)


def correspondences(a,b):
    detector=cv2.SIFT_create(nfeatures=700)
    ka,da=detector.detectAndCompute(cv2.cvtColor(a,cv2.COLOR_RGB2GRAY),None)
    kb,db=detector.detectAndCompute(cv2.cvtColor(b,cv2.COLOR_RGB2GRAY),None)
    if da is None or db is None or min(len(da),len(db))<2:return []
    matcher=cv2.BFMatcher()
    def matches(p,q):
        return {m.queryIdx:m.trainIdx for m,k in matcher.knnMatch(p,q,k=2) if m.distance<.75*k.distance}
    forward,reverse=matches(da,db),matches(db,da)
    result=[]
    for i,j in sorted(forward.items(),key=lambda m:ka[m[0]].response,reverse=True):
        if reverse.get(j)!=i:continue
        x,y=ka[i].pt
        u,v=kb[j].pt
        if any(np.hypot(x-r['x'],y-r['y'])<10 or np.hypot(u-r['u'],v-r['v'])<10 for r in result):continue
        result.append(dict(point=len(result),x=x,y=y,u=u,v=v,quadrant=int(x>=256)+2*int(y>=256)))
    return result


def assess(points):
    if len(points)<12:return dict(pass_geometry=False,reason='fewer_than_12_matches',matches=len(points))
    q=defaultdict(list)
    for r in sorted(points,key=lambda r:(r['x'],r['y'])):q[r['quadrant']].append(r)
    tr=[r for group in q.values() for r in group[::2]]
    va=[r for group in q.values() for r in group[1::2]]
    delta=lambda rs:np.array([[r['u']-r['x'],r['v']-r['y']] for r in rs])
    train,valid=delta(tr),delta(va)
    model=fit(train)
    d=delta(points)
    err=differences(d,model)
    inliers=[r for r,e in zip(points,err) if e<=1.5]
    half_errors=[]
    for axis,center in [('x',256),('y',256)]:
        sides=[[r for r in tr if (r[axis]>=center)==side] for side in (True,False)]
        half_errors.append(float(np.linalg.norm(fit(delta(sides[0]))-fit(delta(sides[1])))) if all(sides) else 1e6)
    med=float(np.median(err))
    val=float(np.median(differences(valid,model)))
    ratio=len(inliers)/len(points)
    quadrant=len({r['quadrant'] for r in inliers})
    passed=len(inliers)>=12 and ratio>=.7 and quadrant>=3 and med<=1 and val<=1 and max(half_errors)<=1
    return dict(pass_geometry=passed,dx=float(model[0]),dy=float(model[1]),length=float(np.linalg.norm(model)),
                matches=len(points),inliers=len(inliers),inlier_fraction=ratio,quadrants=quadrant,
                residual_px=med,held_out_residual_px=val,half_difference=max(half_errors))


def survey():
    dest=WORK/'landmarks'
    dest.mkdir(exist_ok=True)
    evidence=[]
    rows=[]
    for r in sorted(manifest('validation'),key=lambda r:rank(r['image_id']))[:24]:
        n=r['image_id']
        ims=[np.array(Image.open(WORK/'validation'/n/k).convert('RGB')) for k in ('image_before','image_after')]
        pts=correspondences(*ims)
        for p in pts:evidence.append(dict(name=n,**p))
        rows.append(dict(name=n,**assess(pts)))
        boards=[Image.fromarray(im) for im in ims]
        for board,(kx,ky) in zip(boards,[('x','y'),('u','v')]):
            draw=ImageDraw.Draw(board)
            for p in pts:
                x,y=p[kx],p[ky]
                draw.ellipse((x-2,y-2,x+2,y+2),outline='lime')
                draw.text((x+2,y),str(p['point']),fill='yellow',stroke_width=1,stroke_fill='black')
        canvas=Image.new('RGB',(1024,512))
        canvas.paste(boards[0]);canvas.paste(boards[1],(512,0))
        canvas.save(dest/(n+'.jpg'),quality=90)
    if evidence:write(dest/'points.csv',evidence)
    dump(dest/'survey.json',rows)
    print('geometry and size passing',[(r['name'],r['length']) for r in rows if r.get('pass_geometry') and r['length']>=2],flush=True)


if __name__=='__main__':survey()
