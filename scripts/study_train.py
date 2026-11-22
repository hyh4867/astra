"""One fixed external fine-tune, resume from the most recent completed epoch."""
import random
import torch
import torch.nn.functional as F

import numpy as np
import cv2
from study_io import (
    WORK, counts, extract, images, load_tinycd_model, manifest, metrics, pair, probability, restore, write,
)
from inference import pad_to_tile_multiple,IMAGENET_MEAN,IMAGENET_STD


def train_pair(name,flip):
    a,b,g=pair('train',name)
    a,b=images(a,b,'half')
    y=cv2.resize(g.astype('uint8'),(205,205),interpolation=cv2.INTER_NEAREST)
    if flip:
        a,b,y=a[:,::-1],b[:,::-1],y[:,::-1]
    def tensor(im):
        padded,_=pad_to_tile_multiple(im)
        x=(padded.astype('float32')/255.-IMAGENET_MEAN.reshape(3))/IMAGENET_STD.reshape(3)
        return torch.from_numpy(x.transpose(2,0,1).copy())
    return tensor(a),tensor(b),torch.from_numpy(y.astype('float32').copy())


def train():
    extract('train')
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    model,device=load_tinycd_model()
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.0001)
    path=WORK/'epoch_state.pth'
    start=1
    history=[]
    if path.exists():
        state=torch.load(path,map_location='cpu',weights_only=True)
        model.load_state_dict(state['model'])
        opt.load_state_dict(state['optimizer'])
        history=state['history']
        start=len(history)+1
    names=[r['image_id'] for r in manifest('train')]
    for epoch in range(start,4):
        torch.manual_seed(42+epoch)
        rng=random.Random(42+epoch)
        order=rng.sample(names,len(names))
        model.train()
        losses=[]
        for idx in range(0,len(order),2):
            group=[train_pair(n,rng.random()<.5) for n in order[idx:idx+2]]
            a,b,y=[torch.stack(t).to(device) for t in zip(*group)]
            opt.zero_grad(set_to_none=True)
            prob=model(a,b)[:,0,:205,:205]
            loss=F.binary_cross_entropy(prob,y)
            loss.backward()
            opt.step()
            losses.append(float(loss.detach().cpu()))
            if idx%128==0: print('epoch',epoch,idx,'/',len(order),flush=True)
        model.eval()
        val=[]
        for r in manifest('validation'):
            a,b,g=pair('validation',r['image_id'])
            a,b=images(a,b,'half')
            val.append(counts(restore(probability(model,device,a,b),g.shape)>=.5,g))
        result=dict(epoch=epoch,loss=float(np.mean(losses)),**metrics(val))
        if not history or result['f1']>max(h['f1'] for h in history):
            torch.save(model.state_dict(),WORK/'adapted.tmp')
            (WORK/'adapted.tmp').replace(WORK/'adapted.pth')
        history.append(result)
        torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),history=history),path.with_suffix('.tmp'))
        path.with_suffix('.tmp').replace(path)
        write(WORK/'training.csv',history)
        print(result,flush=True)


if __name__=='__main__':
    train()
