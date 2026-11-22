"""One locked execution per method with recoverable, hash-bound prediction masks."""
import subprocess
import csv
import cv2
import hashlib
import json
from study_io import (
    ORIGINAL, ROOT, WORK, dump, extract, load_tinycd_model, manifest, pair, write,
)
from study_lock import config
from study_calibrate import transform,predict
from study_direction import direction
from study_objects import object_report
from study_registration import BASE,evaluate,summarize
from verify_generalization import verify
from reproduce_generalization import evaluate as levir_regression


def git(*args):
    return subprocess.check_output(['git',*args],cwd=ROOT,text=True).strip()


def final():
    path=ROOT/'docs/generalization-final-config.json'
    frozen=json.loads(path.read_text())
    assert frozen==config() and not git('status','--porcelain','--untracked-files=normal')
    sha=hashlib.sha256(json.dumps(frozen,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    provenance=dict(commit=git('rev-parse','HEAD'),config_sha256=sha,git_dirty=False,
                    test_manifest_sha256=frozen['manifests']['test'])
    dest=WORK/'final'
    dest.mkdir(exist_ok=True)
    stamp=dest/'provenance.json'
    if stamp.exists() and json.loads(stamp.read_text())!=provenance:raise ValueError('Final checkpoint belongs to another configuration')
    dump(stamp,provenance)
    extract('test')
    folders={}
    for name in ('original','calibrated','adapted'):
        conf=frozen['methods'][name]
        folder=dest/name
        folder.mkdir(exist_ok=True)
        folders[name]=folder
        rows=manifest('test')
        model,device=load_tinycd_model(WORK/'adapted.pth' if name=='adapted' else ORIGINAL)
        for idx,r in enumerate(rows):
            n=r['image_id']; out=folder/(n+'.png')
            if out.exists():continue
            a,b,g=pair('test',n)
            a,b,v=transform(a,b,conf['grid'],conf['reg'])
            mask=predict(model,device,a,b,g.shape,v)>=conf['cutoff']
            tmp=out.with_suffix('.tmp.png')
            assert cv2.imwrite(str(tmp),mask.astype('uint8')*255)
            tmp.replace(out)
            if idx%100==0:print(name,idx,'/',len(rows),flush=True)
        del model
        per,summary=object_report('test',folder)
        for r in per:r.update(provenance)
        write(folder/'counts.csv',per)
        dump(folder/'metrics.json',dict(summary,**provenance))
        dump(folder/'verification.json',dict(verify(folder),**provenance))
    directional=direction('test',folders)
    dump(dest/'direction.json',dict(directional,**provenance))
    if not directional['available']:print('Use the validation directional analysis',flush=True)
    levir_regression(WORK/'adapted.pth','adapted')
    for csvname in ['direction_test.csv','levir_adapted.csv']:
        file=WORK/csvname
        if file.exists():
            with file.open() as f:
                records=[dict(r,**provenance) for r in csv.DictReader(f)]
            write(file,records)
    c=frozen['registration']
    policies=list(dict.fromkeys([BASE,c['best_new'],c['selected']]))
    summary=summarize(evaluate('test',policies,provenance))
    for r in summary:r.update(provenance)
    write(WORK/'reg_test_summary.csv',summary)
    print('Final run complete',provenance,flush=True)


if __name__=='__main__':final()
