"""Frozen baseline with full-object scoring before the development sweeps."""
import cv2
from study_io import (
    WORK, dump, extract, load_tinycd_model, manifest, pair, probability, write,
)
from study_objects import object_report


def baseline():
    extract('validation')
    model,device=load_tinycd_model()
    dest=WORK/'baseline_validation'
    dest.mkdir(exist_ok=True)
    for r in manifest('validation'):
        n=r['image_id']
        path=dest/(n+'.png')
        if path.is_file():
            continue
        a,b,_=pair('validation',n)
        mask=probability(model,device,a,b)>=.5
        tmp=dest/(n+'.tmp.png')
        assert cv2.imwrite(str(tmp),mask.astype('uint8')*255)
        tmp.replace(path)
    rows,summary=object_report('validation',dest)
    write(WORK/'baseline_validation.csv',rows)
    dump(WORK/'baseline_validation.json',summary)
    print('Baseline',summary['f1'],flush=True)


if __name__=='__main__':
    baseline()
