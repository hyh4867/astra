"""NAIP/LEVIR input audit, equivalent blur statistics, and 0.5m scale test."""
import csv,json,sys,time
from pathlib import Path
import cv2,numpy as np,torch
from rasterio.warp import transform as warp_transform
from PIL import Image,ImageDraw,ImageFont
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT));OUT=ROOT/"outputs/naip_demo";OUT.mkdir(parents=True,exist_ok=True)
from evaluate_benchmark import load_samples,Sample,tinycd_predict
from inference import tinycd_probability,make_overlay,probability_heatmap
from geospatial import align_pair,read_image_bytes,object_features
from rasterio import Affine

PAIR=ROOT/"data/demo_naip/celina_town_2020_2022_overlap"
def stats(x):
    gray=cv2.cvtColor(x,cv2.COLOR_RGB2GRAY).astype(np.float32);var=float(gray.var());lap=float(cv2.Laplacian(gray,cv2.CV_32F).var())
    fft=np.fft.rfft2(gray-gray.mean());power=np.abs(fft)**2;h,w=gray.shape;fy=np.fft.fftfreq(h)[:,None];fx=np.fft.rfftfreq(w)[None,:];radius=np.sqrt(fx*fx+fy*fy);hf=float(power[radius>=.25].sum()/max(power.sum(),1e-12))
    return {"laplacian_variance":lap,"image_variance":var,"lap_over_image_variance":lap/max(var,1e-9),"high_frequency_power_fraction_f_gt_0p25":hf}
def summarize_input(name,x):
    print(f"INPUT {name}: dtype={x.dtype} order=RGB shape={x.shape} min={x.min(axis=(0,1)).tolist()} max={x.max(axis=(0,1)).tolist()} mean={x.mean(axis=(0,1)).round(2).tolist()}",flush=True)
def make_sheet(a,b,p,tag):
    overlay=make_overlay(b,p>=.5);panels=[("A · before",a),("B · after",b),("Probability · 0–1",probability_heatmap(p)),("Overlay",overlay)]
    w,h=512,640;im=Image.new("RGB",(w*4,h+58),"white");d=ImageDraw.Draw(im);d.text((10,8),"QUALITATIVE ONLY · NAIP · NO GROUND TRUTH",fill="#9b1c1c")
    for i,(label,x) in enumerate(panels):
        d.text((i*w+8,36),label,fill="#10243a");tile=Image.fromarray(x);tile.thumbnail((w,h),Image.Resampling.LANCZOS);im.paste(tile,(i*w+(w-tile.width)//2,58+(h-tile.height)//2))
    im.save(OUT/f"figure_{tag}.png")
def main():
    meta=json.loads((PAIR/"metadata.json").read_text());a,ga=read_image_bytes((PAIR/"A_before_2020.tif").read_bytes(),"A_before_2020.tif");b,gb=read_image_bytes((PAIR/"B_after_2022.tif").read_bytes(),"B_after_2022.tif");a,b,geo,gsd=align_pair(a,ga,b,gb)
    summarize_input("NAIP A",a);summarize_input("NAIP B",b)
    samples=list(load_samples("test"));select={n for n in (ROOT/"data/manifests/robustness_test_seed42.txt").read_text().splitlines()};sub=[s for s in samples if s.name in select]
    summarize_input("LEVIR A",sub[0].before);summarize_input("LEVIR B",sub[0].after)
    rows=[];lev_stats={}
    for sigma in (0,1,1.5,1.75,2,2.5,3):
        vals=[]
        for s in sub:
            for img in (s.before,s.after):
                blur=img if sigma==0 else cv2.GaussianBlur(img,(0,0),sigmaX=sigma)
                resized=cv2.resize(blur,None,fx=.5/.6,fy=.5/.6,interpolation=cv2.INTER_LANCZOS4)
                vals.append(stats(resized))
        lev_stats[sigma]={k:float(np.mean([v[k] for v in vals])) for k in vals[0]}
        rows.append({"source":"LEVIR test 32 pairs, native 0.5m then sigma blur and resample to 0.6m","sigma_px":sigma,**lev_stats[sigma]})
    naip_stats={"A":stats(a),"B":stats(b)}
    for date,v in naip_stats.items():rows.append({"source":f"NAIP {date}","sigma_px":"NA","date":date,**v})
    target=np.mean([naip_stats["A"]["lap_over_image_variance"],naip_stats["B"]["lap_over_image_variance"]]);equiv=min(lev_stats,key=lambda s:abs(lev_stats[s]["lap_over_image_variance"]-target))
    with (OUT/"naip_sharpness_equivalence.csv").open("w",newline="") as f:
        fields=list(dict.fromkeys(k for row in rows for k in row));w=csv.DictWriter(f,fieldnames=fields,extrasaction="ignore");w.writeheader();w.writerows(rows)
    print(f"NAIP mean contrast-normalized sharpness={target:.6g}; nearest LEVIR blur sigma={equiv}px ({lev_stats[equiv]['lap_over_image_variance']:.6g})",flush=True)
    if not torch.backends.mps.is_available():device=torch.device("cpu")
    else:device=torch.device("cpu") # host MPS inference has exited 134; force CPU for this local qualitative analysis.
    from models.change_classifier import ChangeClassifier
    sys.path.insert(0,str(ROOT/"vendor/tinycd"));model=ChangeClassifier(pretrained=False).eval();model.load_state_dict(torch.load(ROOT/"models/tinycd_levir_best.pth",map_location="cpu",weights_only=True),strict=True);model.to(device)
    started=time.perf_counter();p06=tinycd_probability(model,a,b,device);dt06=time.perf_counter()-started
    mask06=p06>=.5;rows06,_=object_features(mask06,p06,geo,gsd);make_sheet(a,b,p06,"original_0p6m")
    # Select the largest detected object centroid for a review inset; center fallback if empty.
    h,w=a.shape[:2];cx=w//2;cy=h//2
    if rows06:
        top=max(rows06,key=lambda r:r["area_m2"]);xx,yy=warp_transform("EPSG:4326",geo["crs"],[top["centroid_lon"]],[top["centroid_lat"]]);cx,cy=map(int,~geo["transform"]*(xx[0],yy[0]))
    x0=max(0,min(w-512,cx-256));y0=max(0,min(h-512,cy-256));
    inset=Image.new("RGB",(1536,560),"white");d=ImageDraw.Draw(inset)
    for i,(label,img) in enumerate((("A zoom",a),("B zoom",b),("Overlay zoom",make_overlay(b,mask06)))):
        d.text((i*512+8,4),label,fill="#10243a");inset.paste(Image.fromarray(img[y0:y0+512,x0:x0+512]).resize((512,512),Image.Resampling.LANCZOS),(i*512,36))
    d.text((8,540),f"QUALITATIVE ONLY · zoom 512x512 at x={x0}, y={y0}; centered on largest detected component/fallback",fill="#9b1c1c");inset.save(OUT/"figure_original_0p6m_inset.png")
    # Lanczos upsample from 0.6m to 0.5m GSD.
    aa=cv2.resize(a,None,fx=1.2,fy=1.2,interpolation=cv2.INTER_LANCZOS4);bb=cv2.resize(b,None,fx=1.2,fy=1.2,interpolation=cv2.INTER_LANCZOS4)
    geo05=dict(geo);geo05["transform"]=geo["transform"]*Affine.scale(.5/.6);geo05["gsd_m"]=.5;geo05["pixel_area_m2"]=.25
    started=time.perf_counter();p05=tinycd_probability(model,aa,bb,device);dt05=time.perf_counter()-started;mask05=p05>=.5;rows05,_=object_features(mask05,p05,geo05,.5);make_sheet(aa,bb,p05,"original_0p5m")
    result={"qualitative_only":True,"ground_truth":False,"MPS_attempt":False,"inference_device":str(device),"input_rgb_uint8":True,"gsd_original_m":float(gsd),"gsd_resampled_m":0.5,"blur_equivalent_sigma_px_LEVIR":float(equiv),"models":{"Original":{"0.6m":{"objects":len(rows06),"area_m2":sum(r["area_m2"] for r in rows06),"runtime_s":dt06},"0.5m":{"objects":len(rows05),"area_m2":sum(r["area_m2"] for r in rows05),"runtime_s":dt05}}},"v2_model":"not available: MPS training did not start"}
    (OUT/"naip_diagnosis_summary.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2),flush=True)
if __name__=="__main__":main()
