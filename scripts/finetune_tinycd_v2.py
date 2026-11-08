"""BatchNorm-frozen, pre-registered 8-epoch TinyCD v2 training."""
from __future__ import annotations
import argparse, csv, random, sys, time
from pathlib import Path
import cv2, numpy as np, torch
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(ROOT/"vendor/tinycd"))
from models.change_classifier import ChangeClassifier
from evaluate_benchmark import Sample, load_samples, tinycd_predict
from finetune_tinycd_robust import TRAIN, VAL, BASE, LOG, PATCH, upstream_photometric, paired_geometry, normalize_batch, metric_counts

def item(name, index, epoch, seed, robust):
    rng=np.random.default_rng(seed+epoch*1_000_003+index)
    a=np.asarray(Image.open(TRAIN/"A"/name).convert("RGB")); b=np.asarray(Image.open(TRAIN/"B"/name).convert("RGB"))
    y=(np.asarray(Image.open(TRAIN/"label"/name).convert("L"))>127).astype(np.uint8)
    a,b,y=paired_geometry(a,b,y,rng); a,b=upstream_photometric(a,rng),upstream_photometric(b,rng)
    yy=int(rng.integers(0,a.shape[0]-PATCH+1)); xx=int(rng.integers(0,a.shape[1]-PATCH+1))
    a,b,y=a[yy:yy+PATCH,xx:xx+PATCH],b[yy:yy+PATCH,xx:xx+PATCH],y[yy:yy+PATCH,xx:xx+PATCH]
    if robust:
        for image_name in (0,1):
            if rng.random()<0.5:
                sigma=float(rng.uniform(0,2.5))
                if sigma>0:
                    if image_name==0: a=cv2.GaussianBlur(a,(0,0),sigmaX=sigma)
                    else: b=cv2.GaussianBlur(b,(0,0),sigmaX=sigma)
        if rng.random()<0.5: a,b=b,a
    return a,b,y.astype(np.float32)

def evaluate(model,device,blur_sigma=0):
    model.eval()
    names=sorted(p.name for p in (VAL/"A").glob("*.png")); tp=fp=fn=0
    with torch.inference_mode():
        for name in names:
            a=np.asarray(Image.open(VAL/"A"/name).convert("RGB")); b=np.asarray(Image.open(VAL/"B"/name).convert("RGB"))
            y=np.asarray(Image.open(VAL/"label"/name).convert("L"))>127
            if blur_sigma: b=cv2.GaussianBlur(b,(0,0),sigmaX=blur_sigma)
            sample=Sample(name,a,b,y,np.zeros_like(y,dtype=np.uint8)); pred=tinycd_predict(model,sample,device)
            tp+=int((pred&y).sum()); fp+=int((pred&~y).sum()); fn+=int((~pred&y).sum())
    return metric_counts(tp,fp,fn)

def freeze_bn(model):
    for m in model.modules():
        if isinstance(m,torch.nn.modules.batchnorm._BatchNorm):
            m.eval()
            for p in m.parameters(): p.requires_grad_(False)

def resolve_device(requested_epochs, force_cpu=False):
    """Probe host MPS twice; if unavailable, use the specified 5-epoch CPU fallback."""
    if force_cpu:
        print("MPS training retry exited 134; using explicit CPU fallback for 5 epochs", flush=True)
        return torch.device("cpu"), 5
    for attempt in range(2):
        available = torch.backends.mps.is_available()
        print(f"mps_probe_{attempt + 1}={available}", flush=True)
        if available:
            assert torch.backends.mps.is_available(), "MPS availability changed during initialization"
            return torch.device("mps"), requested_epochs
        if attempt == 0:
            time.sleep(3)
    print("MPS unavailable after retry; falling back to CPU for 5 epochs", flush=True)
    return torch.device("cpu"), 5

def main():
    p=argparse.ArgumentParser(); p.add_argument("--mode",choices=["control","robust"],required=True); p.add_argument("--seed",type=int,default=42); p.add_argument("--batch-size",type=int,default=1); p.add_argument("--epochs",type=int,default=8); p.add_argument("--smoke-steps",type=int,default=0); p.add_argument("--force-cpu",action="store_true"); args=p.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device,actual_epochs=resolve_device(args.epochs,args.force_cpu); names=sorted(p.name for p in (TRAIN/"A").glob("*.png"))
    if not names: raise RuntimeError(f"No training pairs found under {TRAIN / 'A'}")
    model=ChangeClassifier(pretrained=False); model.load_state_dict(torch.load(BASE,map_location="cpu",weights_only=True),strict=True); model.to(device)
    freeze_bn(model); optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=1e-4,weight_decay=0.009449677083344786)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=actual_epochs); loss_fn=torch.nn.BCELoss()
    if args.smoke_steps:
        if args.smoke_steps < 1: raise ValueError("--smoke-steps must be positive")
        model.train(); freeze_bn(model); start=time.perf_counter()
        for step in range(args.smoke_steps):
            index=step % len(names); batch=[item(names[index],index,0,args.seed,args.mode=="robust")]
            xa=normalize_batch([x[0] for x in batch]).to(device); xb=normalize_batch([x[1] for x in batch]).to(device); y=torch.from_numpy(np.stack([x[2] for x in batch])).to(device).float()
            optimizer.zero_grad(set_to_none=True); prob=model(xa,xb).squeeze(1); loss=loss_fn(prob,y); loss.backward(); optimizer.step()
            if (step+1)%10==0: print(f"smoke_step={step+1}/{args.smoke_steps} loss={float(loss.detach().cpu()):.5f}",flush=True)
        clean=evaluate(model,device)
        print(f"smoke_complete device={device} steps={args.smoke_steps} clean_val_f1={clean['f1']:.6f} seconds={time.perf_counter()-start:.1f} batch_size=1 num_workers=0",flush=True)
        return
    LOG.mkdir(parents=True,exist_ok=True); out=ROOT/f"models/finetuned/tinycd_{args.mode}_v2_seed{args.seed}_best.pth"
    epoch_dir=ROOT/f"models/finetuned/tinycd_{args.mode}_v2_seed{args.seed}_epochs"; epoch_dir.mkdir(parents=True,exist_ok=True)
    log=LOG/f"{args.mode}_v2_seed{args.seed}_epochs.csv"; best=-1
    with log.open("a",newline="") as f:
        if f.tell()==0:
            w=csv.DictWriter(f,fieldnames=["epoch","train_loss","clean_val_f1","blurB_sigma2_val_f1","epoch_seconds","batch_size","num_workers","seed","device","actual_epochs"]);w.writeheader()
        else:
            w=csv.DictWriter(f,fieldnames=["epoch","train_loss","clean_val_f1","blurB_sigma2_val_f1","epoch_seconds","batch_size","num_workers","seed","device","actual_epochs"])
        for ep in range(actual_epochs):
            start=time.perf_counter(); model.train(); freeze_bn(model); order=np.random.default_rng(args.seed+ep).permutation(len(names)); losses=[]
            for st in range(0,len(order),args.batch_size):
                ids=order[st:st+args.batch_size]; batch=[item(names[int(i)],int(i),ep,args.seed,args.mode=="robust") for i in ids]
                xa=normalize_batch([x[0] for x in batch]).to(device); xb=normalize_batch([x[1] for x in batch]).to(device); y=torch.from_numpy(np.stack([x[2] for x in batch])).to(device).float()
                optimizer.zero_grad(set_to_none=True); prob=model(xa,xb).squeeze(1); loss=loss_fn(prob,y); loss.backward(); optimizer.step(); losses.append(float(loss.detach().cpu()))
                steps_done=st+len(ids)
                if steps_done%50==0 or steps_done==len(order): print(f"{args.mode}-v2 seed={args.seed} epoch={ep+1}/{actual_epochs} train_steps={steps_done}/{len(order)}",flush=True)
            clean=evaluate(model,device); blur=evaluate(model,device,2.0); elapsed=time.perf_counter()-start
            row={"epoch":ep+1,"train_loss":float(np.mean(losses)),"clean_val_f1":clean["f1"],"blurB_sigma2_val_f1":blur["f1"],"epoch_seconds":elapsed,"batch_size":args.batch_size,"num_workers":0,"seed":args.seed,"device":str(device),"actual_epochs":actual_epochs};w.writerow(row);f.flush()
            print(f"{args.mode}-v2 seed={args.seed} epoch={ep+1}/{actual_epochs} cleanVAL={clean['f1']:.5f} blurB2VAL={blur['f1']:.5f} secs={elapsed:.1f} device={device} batch=1 num_workers=0",flush=True)
            torch.save(model.state_dict(),epoch_dir/f"epoch_{ep+1:02d}.pth")
            if clean["f1"]>best: best=clean["f1"];torch.save(model.state_dict(),out)
            scheduler.step()
    print(f"best_clean_val_f1={best:.6f}; checkpoint={out}; device={device}; actual_epochs={actual_epochs}",flush=True)
if __name__=="__main__": main()
