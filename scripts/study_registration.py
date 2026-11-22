"""Select translation refinements on validation and compare on common real pixels."""
import argparse
from collections import defaultdict
from dataclasses import replace
import math

import numpy as np
import json
from study_io import (
    WORK, counts, dump, load_tinycd_model, metrics, probability, write,
)
from evaluate_benchmark import load_samples
from registration import DEFAULT_CONFIG, estimate_shift,shift_image

SHIFTS=[(0,0),(3,3),(5,5),(-4,7)]
BASE='centroid_4'
CANDIDATES=[BASE]+[f'{alg}_{t}' for alg in ('dft','ecc') for t in (4,5,6)]


def movement(est,alg,threshold,scale=1):
    if est['reasons'] or np.hypot(est['dx'],est['dy'])<threshold*scale:
        return 0.,0.
    dx,dy=-est['dx'],-est['dy']
    return (float(round(dx)),float(round(dy))) if alg=='centroid' else (dx,dy)


def external_move(a,b,grid,policy):
    if policy=='off':
        return (0.,0.)
    alg,t=policy.split('_')
    scale=2.5 if grid=='native' else 205/204.8
    # Keep search range, confidence distances and displacement gates in metres.
    cfg=replace(DEFAULT_CONFIG,refinement=alg,max_auto_px=12*scale,
                search_px=round(16*scale),tile_tol_px=scale)
    return movement(estimate_shift(a,b,cfg),alg,float(t),scale)


def evaluate(split,policies,provenance=None):
    model,device=load_tinycd_model()
    dest=WORK/f'reg_{split}'
    dest.mkdir(exist_ok=True)
    records=[]
    for sample in load_samples(split):
        cache=dest/(sample.name+'.json')
        if cache.exists():
            item=json.loads(cache.read_text())
            if item['policies']!=policies or item['provenance']!=provenance:
                raise ValueError('Registration checkpoint configuration mismatch')
            records.extend(item['rows'])
            continue
        pair_rows=[]
        clean_est={}
        for sx,sy in SHIFTS:
            a=sample.before
            b=shift_image(sample.after,sx,sy) if sx or sy else sample.after
            estimates={alg:estimate_shift(a,b,replace(DEFAULT_CONFIG,refinement=alg))
                       for alg in sorted({p.split('_')[0] for p in policies if p!='off'})}
            moves={'raw':(0,0),'oracle':(-sx,-sy)}
            for policy in policies:
                if policy=='off':
                    moves[policy]=(0,0)
                    continue
                alg,t=policy.split('_')
                moves[policy]=movement(estimates[alg],alg,float(t))
            def interval(k,delta,size):
                displacements=[v for m in moves.values() for v in (m[k],m[k]+delta)]
                return max(8,math.ceil(max(displacements))),size+min(-8,math.floor(min(displacements)))
            x0,x1=interval(0,sx,b.shape[1])
            y0,y1=interval(1,sy,b.shape[0])
            scores={}
            for policy,m in moves.items():
                if m not in scores:
                    adj=shift_image(b,*m) if any(m) else b
                    mask=probability(model,device,a,adj)>=.5
                    scores[m]=counts(mask[y0:y1,x0:x1],sample.truth[y0:y1,x0:x1])
                e=estimates.get(policy.split('_')[0])
                err=rel=None
                if e:
                    v=np.array([e['dx'],e['dy']])
                    if (sx,sy)==(0,0): clean_est[policy]=v
                    err=float(np.linalg.norm(v-[sx,sy]))
                    rel=float(np.linalg.norm(v-[sx,sy]-clean_est[policy]))
                pair_rows.append(dict(name=sample.name,sx=sx,sy=sy,policy=policy,
                    error=err,relative=rel,held=bool(e and e['reasons']),move_x=m[0],move_y=m[1],
                    x0=x0,x1=x1,y0=y0,y1=y1,**scores[m]))
        dump(cache,dict(rows=pair_rows,policies=policies,provenance=provenance))
        records.extend(pair_rows)
        print(split,sample.name,flush=True)
    if provenance:
        for row in records:
            row.update(provenance)
    write(WORK/f'reg_{split}.csv',records)
    return records


def summarize(rows):
    groups=defaultdict(list)
    for r in rows: groups[r['sx'],r['sy'],r['policy']].append(r)
    summary=[]
    for (sx,sy,p),items in groups.items():
        measured=[r for r in items if r['error'] is not None]
        base=metrics(groups[sx,sy,'raw'])['f1']
        ideal=metrics(groups[sx,sy,'oracle'])['f1']
        m=metrics(items)
        summary.append(dict(sx=sx,sy=sy,policy=p,pairs=len(items),held=sum(r['held'] for r in items),
            moved=sum(bool(r['move_x'] or r['move_y']) for r in items),
            median_error=float(np.median([r['error'] for r in measured])) if measured else None,
            median_relative=float(np.median([r['relative'] for r in measured])) if measured else None,
            recovery=(m['f1']-base)/(ideal-base) if ideal>base else None,**m))
    return summary


def select(summary):
    clean=next(r['f1'] for r in summary if r['policy']=='raw' and r['sx']==0)
    scores=[]
    for p in CANDIDATES:
        clean_r=next(r for r in summary if r['policy']==p and r['sx']==0)
        shifted=[r for r in summary if r['policy']==p and r['sx']!=0]
        scores.append(dict(policy=p,eligible=clean-clean_r['f1']<=.002,clean_loss=clean-clean_r['f1'],
                     recovery=float(np.mean([r['recovery'] for r in shifted])),
                     relative=float(np.mean([r['median_relative'] for r in shifted]))))
    def key(r):
        alg,t=r['policy'].split('_')
        return (-r['recovery'],r['relative'],['centroid','dft','ecc'].index(alg),-int(t))
    valid=[r for r in scores if r['eligible']]
    chosen=min(valid,key=key)['policy'] if valid else 'off'
    candidates=[r for r in scores if not r['policy']==BASE]
    best=min([r for r in candidates if r['eligible']] or candidates,key=key)['policy']
    dump(WORK/'registration_choice.json',dict(selected=chosen,best_new=best,scores=scores))
    print('selected',chosen,best,flush=True)


if __name__=='__main__':
    summary=summarize(evaluate('val',CANDIDATES))
    write(WORK/'reg_val_summary.csv',summary)
    select(summary)
