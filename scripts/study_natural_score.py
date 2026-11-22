"""Score verified stable-ground translations only when the preregistered count is met."""
import csv
import json
import numpy as np
from study_io import WORK,counts,dump,load_tinycd_model,pair,probability,write
from study_registration import external_move
from study_landmarks import assess
from registration import shift_image,shift_valid_mask


def natural_score():
    root=WORK/'landmarks'
    survey=json.loads((root/'survey.json').read_text())
    passing=[r for r in survey if r['pass_geometry'] and r['length']>=2]
    status=dict(screened=len(survey),preliminary_qualified=len(passing))
    if len(passing)<5:
        dump(root/'result.json',dict(status,evaluated=False,reason='fewer than five geometrically qualified pairs before ground-feature review'))
        return
    # Only reviewed, numbered, stable-ground points may support a reference.
    with (root/'reviewed_ground_points.csv').open() as f:
        include={(r['name'],int(r['point'])) for r in csv.DictReader(f)}
    with (root/'points.csv').open() as f: all_points=list(csv.DictReader(f))
    accepted=[]
    for pre in passing:
        pts=[{k:int(p[k]) if k in ('point','quadrant') else float(p[k]) for k in p if k!='name'}
             for p in all_points if p['name']==pre['name'] and (p['name'],int(p['point'])) in include]
        ref=assess(pts)
        if ref.get('pass_geometry') and ref['length']>=2:
            accepted.append(dict(name=pre['name'],**ref))
    status['ground_qualified']=len(accepted)
    if len(accepted)<5:
        dump(root/'result.json',dict(status,evaluated=False,reason='fewer than five stable-ground references'))
        return
    policy=json.loads((WORK/'registration_choice.json').read_text())['selected']
    model,device=load_tinycd_model()
    results=[]
    for ref in accepted:
        a,b,g=pair('validation',ref['name'])
        move=external_move(a,b,'native',policy)
        valid=shift_valid_mask(g.shape,*move)
        for condition,delta in [('uncorrected',(0.,0.)),('selected',move)]:
            after=shift_image(b,*delta) if any(delta) else b
            pred=probability(model,device,a,after)>=.5
            residual=float(np.hypot(ref['dx']+delta[0],ref['dy']+delta[1]))
            results.append(dict(name=ref['name'],condition=condition,dx=delta[0],dy=delta[1],
                                residual_px=residual,residual_m=.2*residual,**counts(pred[valid],g[valid])))
    write(root/'natural_scores.csv',results)
    dump(root/'result.json',dict(status,evaluated=True,policy=policy,accepted=accepted))


if __name__=='__main__':natural_score()
