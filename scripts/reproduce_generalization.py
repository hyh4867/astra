"""Fresh full-frame LEVIR test predictions; separately score the frozen adapted model."""
import sys
from study_io import (
    ORIGINAL, WORK, counts, digest, load_tinycd_model, metrics, write,
)
from evaluate_benchmark import load_samples,tinycd_predict


def evaluate(weights,label):
    model,device=load_tinycd_model(weights)
    rows=[]
    for sample in load_samples('test'):
        pred=tinycd_predict(model,sample,device,batch_size=4)
        rows.append(dict(name=sample.name,**counts(pred,sample.truth)))
    assert len(rows)==128
    WORK.mkdir(exist_ok=True,parents=True)
    write(WORK/f'levir_{label}.csv',rows)
    result=metrics(rows)
    print(label,result,flush=True)
    if label=='original':
        assert round(result['f1'],4)==0.9107


if __name__=='__main__':
    if len(sys.argv)>1:
        evaluate(WORK/'adapted.pth','adapted')
    else:
        assert digest(ORIGINAL)=='743645b87199c96bd0c3271ae5df6f0918b3f1d84618dfb9396843ca08ee4a3e'
        evaluate(ORIGINAL,'original')
