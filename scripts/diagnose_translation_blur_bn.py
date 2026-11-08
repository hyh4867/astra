"""Step 1/2 diagnosis: frame alignment, object behavior, probabilities and BN."""
import argparse,csv,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from PIL import Image,ImageDraw
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/"outputs/diagnosis_round";OUT.mkdir(parents=True,exist_ok=True)
from evaluate_benchmark import load_samples,Sample,tinycd_predict,ChangeClassifier,NORMALIZE
from inference import load_tinycd_model
from finetune_tinycd_robust import make_train_item,TRAIN,VAL,normalize_batch
from followup_scientific_checks import gt_objects
from evaluate_advanced_models import metrics,metrics_from_counts,CROP,TEST_SUBSET,CHECKPOINTS

def crop(x):return x[CROP:-CROP,CROP:-CROP]
def moved_truth(y,d):
    m=np.float32([[1,0,d],[0,1,d]])
    return cv2.warpAffine(y.astype(np.uint8),m,(y.shape[1],y.shape[0]),flags=cv2.INTER_NEAREST,borderMode=cv2.BORDER_REFLECT_101)>0
def obj_stats(pred,truth):
    labels,areas,isolated=gt_objects(truth)
    hits=np.bincount(labels[pred],minlength=int(labels.max())+1)
    n=int(len(areas));det=int(sum(hits[i+1]>=.30*a for i,a in enumerate(areas)))
    pc,pl,ps,_=cv2.connectedComponentsWithStats(pred.astype(np.uint8),8)
    overlap=np.bincount(pl[truth],minlength=pc); pa=ps[1:,cv2.CC_STAT_AREA]
    alarms=int(sum(pa[i]>=20 and overlap[i+1]/pa[i]<.1 for i in range(len(pa))))
    km2=32*1008*1008*.25/1e6
    return {"gt_objects":n,"detected_30pct":det,"completeness":det/n if n else "","false_alarm_objects_ge20":alarms,"false_alarms_per_km2":alarms/km2,"isolated_objects":int(isolated.sum())}
def save(path,rows):
    with path.open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def predict(model,s,a,b,device=torch.device("mps")):return tinycd_predict(model,Sample(s.name,a,b,s.truth,s.gray_diff),device)
def probability_legacy_tiles(model,a,b,device):
    coords=[(y,x) for y in range(0,1024,256) for x in range(0,1024,256)];out=np.zeros((1024,1024),np.float32)
    with torch.inference_mode():
        for start in range(0,len(coords),16):
            group=coords[start:start+16]
            xa=torch.stack([torch.from_numpy(a[y:y+256,x:x+256].copy()).permute(2,0,1).float()/255 for y,x in group])
            xb=torch.stack([torch.from_numpy(b[y:y+256,x:x+256].copy()).permute(2,0,1).float()/255 for y,x in group])
            prob=model(NORMALIZE(xa).to(device),NORMALIZE(xb).to(device))[:,0].cpu().numpy()
            for (y,x),tile in zip(group,prob):out[y:y+256,x:x+256]=tile
    return out

def step1(samples,subset):
    by={s.name:s for s in samples};sel=[by[n] for n in subset];rows=[]
    conds=[("control",0),("shift",1),("shift",2),("shift",3),("shift",5),("blur_B",2),("blur_A",2),("blur_AB",2)]
    for method,ckpt in CHECKPOINTS.items():
        model,dev=load_tinycd_model(ckpt)
        cache={}
        for kind,v in conds:
            totals={k:0 for k in ("tp","fp","fn","tn")};btot={k:0 for k in totals};o_real=[];o_b=[]
            for s in sel:
                a,b=s.before,s.after
                if kind=="shift": b=cv2.warpAffine(b,np.float32([[1,0,v],[0,1,v]]),(1024,1024),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_REFLECT_101)
                elif kind=="blur_B": b=cv2.GaussianBlur(b,(0,0),sigmaX=v)
                elif kind=="blur_A": a=cv2.GaussianBlur(a,(0,0),sigmaX=v)
                elif kind=="blur_AB": a=cv2.GaussianBlur(a,(0,0),sigmaX=v);b=cv2.GaussianBlur(b,(0,0),sigmaX=v)
                pred=crop(predict(model,s,a,b)); gt=crop(s.truth); c=metrics(pred,gt)
                for k in totals:totals[k]+=c[k]
                truth_b=moved_truth(s.truth,v) if kind=="shift" else s.truth
                cb=metrics(pred,crop(truth_b))
                for k in btot:btot[k]+=cb[k]
                o_real.append(obj_stats(pred,gt));o_b.append(obj_stats(pred,crop(truth_b)))
            raw=metrics_from_counts(totals);bf=metrics_from_counts(btot)
            def avg(key,arr):
                if key=="completeness":
                    n=sum(x["gt_objects"] for x in arr);return sum(x["detected_30pct"] for x in arr)/n if n else ""
                return float(np.mean([x[key] for x in arr]))
            rows.append({"method":method,"condition":kind,"strength":v,"tp":raw["tp"],"fp":raw["fp"],"fn":raw["fn"],"f1_Aframe":raw["f1"],"bframe_tp":bf["tp"],"bframe_fp":bf["fp"],"bframe_fn":bf["fn"],"bframe_precision":bf["precision"],"bframe_recall":bf["recall"],"bframe_f1":bf["f1"],"object_completeness_Aframe":avg("completeness",o_real),"false_alarms_km2_Aframe":avg("false_alarms_per_km2",o_real),"object_completeness_Bframe":avg("completeness",o_b),"false_alarms_km2_Bframe":avg("false_alarms_per_km2",o_b)})
            print(f"Step1 {method} {kind} {v} F1={raw['f1']:.4f} B-frame={bf['f1']:.4f}",flush=True)
        del model
    # Add delta counts relative to unperturbed control per model.
    for method in CHECKPOINTS:
        base=next(r for r in rows if r["method"]==method and r["condition"]=="control")
        for r in rows:
            if r["method"]==method:r["delta_fp_vs_control"]=r["fp"]-base["fp"];r["delta_fn_vs_control"]=r["fn"]-base["fn"]
    save(OUT/"translation_object_diagnostics.csv",rows)

def prob_curve(samples,subset):
    by={s.name:s for s in samples};sel=[by[n] for n in subset];rows=[]
    for method,ckpt in CHECKPOINTS.items():
        model,dev=load_tinycd_model(ckpt)
        for sigma in (0,1,1.5,1.75,2):
            tp=fp=fn=0;pos_sum=neg_sum=0.0;pos_count=neg_count=0
            for s in sel:
                b=s.after if sigma==0 else cv2.GaussianBlur(s.after,(0,0),sigmaX=sigma)
                prob=probability_legacy_tiles(model,s.before,b,dev);pred=prob>=.5;gt=s.truth
                tp+=int((pred&gt).sum());fp+=int((pred&~gt).sum());fn+=int((~pred&gt).sum())
                pos_sum+=float(prob[gt].sum());pos_count+=int(gt.sum());neg_sum+=float(prob[~gt].sum());neg_count+=int((~gt).sum())
            m=metrics_from_counts({"tp":tp,"fp":fp,"fn":fn,"tn":0})
            rows.append({"method":method,"B_blur_sigma":sigma,"precision":m["precision"],"recall":m["recall"],"f1":m["f1"],"fp":fp,"mean_probability_GT_change":pos_sum/pos_count,"mean_probability_background":neg_sum/neg_count,"inference_pipeline":"legacy full-frame tiles; normalize CPU before device transfer","device":str(dev)})
            print(f"Step2 probability {method} sigma={sigma}: F1={m['f1']:.4f} means={pos_sum/pos_count:.4f}/{neg_sum/neg_count:.4f}",flush=True)
        del model
    save(OUT/"blur_probability_diagnostics.csv",rows)

def examples():
    from finetune_tinycd_robust import make_train_item
    names=sorted(p.name for p in (TRAIN/"A").glob("*.png"))[:8];sheet=Image.new("RGB",(1200,8*290),"white");d=ImageDraw.Draw(sheet)
    for i,n in enumerate(names):
        a,b,y=make_train_item(n,i,3,True); row=Image.new("RGB",(360,256));
        trip=np.concatenate([a,b,np.repeat((y[...,None]*255).astype(np.uint8),3,axis=2)],axis=1)
        sheet.paste(Image.fromarray(trip).resize((1080,256)),(100,i*290+24));d.text((5,i*290+24),n,fill="black")
    sheet.save(OUT/"training_augmentation_samples.png")

def bn_deltas():
    models={}
    for name,ckpt in CHECKPOINTS.items():models[name]=torch.load(ckpt,map_location="cpu",weights_only=True)
    rows=[]
    keys=sorted(k for k in models["Original"] if k.endswith("running_mean"))
    for k in keys:
        row={"layer":k}
        for name in ("CONTROL","ROBUST"):
            row[f"{name}_mean_abs_delta_vs_Original"]=float((models[name][k]-models["Original"][k]).abs().mean())
            vk=k.replace("running_mean","running_var")
            row[f"{name}_var_abs_delta_vs_Original"]=float((models[name][vk]-models["Original"][vk]).abs().mean())
        rows.append(row)
    if rows:save(OUT/"batchnorm_checkpoint_deltas.csv",rows)

def bn_test_metrics(model, device, split, blur_sigma=0.0, subset=None):
    from evaluate_benchmark import DATA
    if split=="val":
        names=sorted(p.name for p in ((DATA/"val")/"A").glob("*.png")); crop_score=False
    else:
        names=subset; crop_score=True
    sums={k:0 for k in ("tp","fp","fn","tn")}
    with torch.inference_mode():
        for name in names:
            base=DATA/split
            a=np.asarray(Image.open(base/"A"/name).convert("RGB")); b=np.asarray(Image.open(base/"B"/name).convert("RGB")); y=np.asarray(Image.open(base/"label"/name).convert("L"))>127
            if blur_sigma:b=cv2.GaussianBlur(b,(0,0),sigmaX=blur_sigma)
            pred=predict(model,Sample(name,a,b,y,np.zeros_like(y,dtype=np.uint8)),a,b,device)
            if crop_score: pred= crop(pred); y=crop(y)
            c=metrics(pred,y)
            for k in sums:sums[k]+=c[k]
    return metrics_from_counts(sums)

def bn_interventions(subset):
    from evaluate_benchmark import DATA
    from finetune_tinycd_robust import TRAIN
    device=torch.device("cpu"); original_state=torch.load(CHECKPOINTS["Original"],map_location="cpu",weights_only=True)
    all_names=sorted(p.name for p in (TRAIN/"A").glob("*.png"));rng=np.random.default_rng(42)
    names=list(rng.choice(all_names,size=500,replace=True));items=[]
    for i,n in enumerate(names):
        a=np.asarray(Image.open(TRAIN/"A"/n).convert("RGB"));b=np.asarray(Image.open(TRAIN/"B"/n).convert("RGB")); yy=int(rng.integers(0,a.shape[0]-256+1));xx=int(rng.integers(0,a.shape[1]-256+1));items.append((a[yy:yy+256,xx:xx+256],b[yy:yy+256,xx:xx+256]))
    rows=[]
    for intervention in ("recalibrated_500_clean_crops","original_BN_stats"):
        state=torch.load(CHECKPOINTS["ROBUST"],map_location="cpu",weights_only=True)
        model=ChangeClassifier(pretrained=False).eval();model.load_state_dict(state,strict=True); bns=[m for m in model.modules() if isinstance(m,torch.nn.modules.batchnorm._BatchNorm)]
        if intervention=="recalibrated_500_clean_crops":
            for bn in bns:bn.reset_running_stats();bn.momentum=None;bn.train()
            model.to(device)
            with torch.no_grad():
                for start in range(0,500,8):
                    chunk=items[start:start+8];xa=normalize_batch([x[0] for x in chunk]).to(device);xb=normalize_batch([x[1] for x in chunk]).to(device);model(xa,xb)
            for bn in bns:bn.eval()
        else:
            for k in list(state):
                if k.endswith(("running_mean","running_var","num_batches_tracked")) and k in original_state:state[k]=original_state[k]
            model.load_state_dict(state,strict=True);model.eval().to(device)
        val=bn_test_metrics(model,device,"val")
        test_blur=bn_test_metrics(model,device,"test",2.0,subset)
        rows.append({"intervention":intervention,"val_clean_precision":val["precision"],"val_clean_recall":val["recall"],"val_clean_f1":val["f1"],"test32_Bsigma2_precision":test_blur["precision"],"test32_Bsigma2_recall":test_blur["recall"],"test32_Bsigma2_f1":test_blur["f1"]})
        print(f"BN {intervention} ({device} fallback): val={val['f1']:.4f}, B2={test_blur['f1']:.4f}",flush=True);del model
    save(OUT/"bn_intervention_metrics.csv",rows)

def main():
    parser=argparse.ArgumentParser();parser.add_argument("--bn-only",action="store_true");parser.add_argument("--prob-only",action="store_true");args=parser.parse_args()
    if not torch.backends.mps.is_available() and not (args.bn_only or args.prob_only):raise RuntimeError("MPS required for the primary evaluation; explicit diagnostic CPU fallbacks are allowed")
    subset=sorted(TEST_SUBSET.read_text().splitlines())
    if args.bn_only:bn_interventions(subset);return
    samples=list(load_samples("test"));(OUT/"subset_seed42_sorted.txt").write_text("\n".join(subset)+"\n")
    if args.prob_only:prob_curve(samples,subset);return
    examples();bn_deltas();step1(samples,subset);prob_curve(samples,subset);bn_interventions(subset)
if __name__=="__main__":main()
