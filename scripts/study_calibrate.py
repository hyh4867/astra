"""Validation-only search and metadata-stratified domain-shift diagnostics."""
from collections import defaultdict
from study_registration import external_move
from registration import shift_image,shift_valid_mask
import json
from study_io import (
    WORK, counts, dump, images, load_tinycd_model, manifest, metrics, pair, probability, restore, write,
)
from study_objects import object_report
from study_direction import direction


def transform(a,b,grid,registration):
    a,b=images(a,b,grid)
    dx,dy=external_move(a,b,grid,registration)
    if dx or dy:
        return a,shift_image(b,dx,dy),shift_valid_mask(b.shape[:2],dx,dy)
    return a,b,None


def predict(model,device,a,b,size,valid):
    prob=restore(probability(model,device,a,b),size)
    if valid is not None:
        support=restore(valid.astype('float32'),size)
        prob[support<1.-1e-6]=0
    return prob


def calibrate():
    policy=json.loads((WORK/'registration_choice.json').read_text())['selected']
    policies=list(dict.fromkeys(['off',policy]))
    model,device=load_tinycd_model()
    rows=[]
    reverse=[]
    dest=WORK/'calibration'
    dest.mkdir(exist_ok=True)
    for r in manifest('validation'):
        n=r['image_id']
        file=dest/(n+'.json')
        if file.exists():
            saved=json.loads(file.read_text())
            assert saved['policy']==policy
            rows.extend(saved['rows'])
            reverse.append(saved['reverse'])
            continue
        a,b,g=pair('validation',n)
        per=[]
        for grid in ['native','half']:
            for reg in policies:
                x,y,valid=transform(a,b,grid,reg)
                prob=predict(model,device,x,y,g.shape,valid)
                for t in [.3,.5,.7]:
                    per.append(dict(name=n,grid=grid,reg=reg,cutoff=t,**counts(prob>=t,g)))
        rev=dict(name=n,**counts(probability(model,device,b,a)>=.5,g))
        dump(file,dict(policy=policy,rows=per,reverse=rev))
        rows.extend(per)
        reverse.append(rev)
    write(WORK/'calibration.csv',rows)
    write(WORK/'reverse_validation.csv',reverse)
    groups=defaultdict(list)
    for r in rows:groups[r['reg'],r['grid'],r['cutoff']].append(r)
    table=[dict(reg=k[0],grid=k[1],cutoff=k[2],**metrics(v)) for k,v in groups.items()]
    winner=min(table,key=lambda r:(-r['f1'],r['reg']!='off',r['grid']!='native',r['cutoff']!=.5,r['cutoff']))
    write(WORK/'calibration_summary.csv',table)
    dump(WORK/'calibration_choice.json',{k:winner[k] for k in ('reg','grid','cutoff')})
    dump(WORK/'direction_validation.json',direction('validation',{'original':WORK/'baseline_validation'}))
    print('chosen',winner,flush=True)


if __name__=='__main__':
    calibrate()
