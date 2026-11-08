"""Regenerate phone-readable object and null plots with Wilson confidence bands."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageColor, ImageDraw, ImageFont

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"scripts"))
from evaluate_benchmark import load_samples
from followup_scientific_checks import gt_objects

OUT=ROOT/"outputs"
METHODS=["OpenCV basic","OpenCV improved","TinyCD"]
BINS=["<100","100-250","250-500","500-1000","1000-2500","2500-5000",">=5000"]
EDGES=[0,100,250,500,1000,2500,5000,float("inf")]
COLORS={METHODS[0]:"#2878b5",METHODS[1]:"#ef8354",METHODS[2]:"#3b9b65"}
FONT_DIR=Path("/System/Library/Fonts/Supplemental")


def font(size,bold=False):
    path=FONT_DIR/("Arial Bold.ttf" if bold else "Arial.ttf")
    try: return ImageFont.truetype(str(path),size)
    except OSError: return ImageFont.load_default(size=size)


def rows_from(path):
    with path.open(newline="") as f: return list(csv.DictReader(f))


def save_rows(path,rows):
    with path.open("w",newline="") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def median_gt_areas():
    all_areas={name:[] for name in BINS}
    isolated_areas={name:[] for name in BINS}
    for sample in load_samples("test"):
        _,areas,isolated=gt_objects(sample.truth)
        for area,is_iso in zip(areas,isolated):
            idx=np.searchsorted(EDGES,int(area),side="right")-1
            name=BINS[idx]
            all_areas[name].append(int(area))
            if is_iso: isolated_areas[name].append(int(area))
    return {
        "all":{k:float(np.median(v)) for k,v in all_areas.items() if v},
        "isolated":{k:float(np.median(v)) for k,v in isolated_areas.items() if v},
    }


def draw_band(draw,points,low_points,color,alpha=38):
    rgb=ImageColor.getrgb(color)
    polygon=[(int(x),int(y)) for x,y in points+list(reversed(low_points))]
    if len(polygon)>=3: draw.polygon(polygon,fill=(*rgb,alpha))


def completeness_plot(rows,isolated=False,medians=None,null_rows=None):
    w,h=1600,1000
    img=Image.new("RGBA",(w,h),(255,255,255,255))
    draw=ImageDraw.Draw(img)
    title="Isolated GT object completeness with Wilson 95% CI" if isolated else "GT object completeness with Wilson 95% CI"
    draw.text((36,22),title,font=font(32,True),fill="#101820")
    if isolated:
        draw.text((36,62),"Dashed: deranged-null; OpenCV improved is near chance overall (43.5% real vs 35.3% null).",font=font(17),fill="#303840")
    left,right,top,bottom=150,1510,100,810
    xmap=lambda x:left+(np.log10(max(10,float(x)))-1)/(np.log10(30000)-1)*(right-left)
    ymap=lambda y:bottom-float(y)*(bottom-top)
    centers=[(medians or {}).get("isolated" if isolated else "all",{}).get(b) for b in BINS]
    selected_rows=[r for r in rows if (r["coverage_threshold"] if isolated else r["detection_coverage_threshold"]) and float(r["coverage_threshold"] if isolated else r["detection_coverage_threshold"])==.3]
    band_layer=Image.new("RGBA",img.size,(0,0,0,0)); band_draw=ImageDraw.Draw(band_layer)
    series={}
    for method,color in COLORS.items():
        vals=[]
        for bi,bin_name in enumerate(BINS):
            row=next((r for r in selected_rows if r["method"]==method and r["gt_area_bin_px"]==bin_name and int(r["n_objects"] if isolated else r["n_gt_objects"])>0),None)
            if not row or centers[bi] is None: continue
            xx=xmap(centers[bi]); value=float(row["completeness"])
            lo=float(row["wilson95_low"]); hi=float(row["wilson95_high"])
            vals.append((xx,ymap(value),ymap(hi),ymap(lo)))
        series[method]=vals
        if vals:
            draw_band(band_draw,[(v[0],v[2]) for v in vals],[(v[0],v[3]) for v in vals],color,42)
    img=Image.alpha_composite(img,band_layer); draw=ImageDraw.Draw(img)
    for x in (10,100,1000,10000,30000):
        xx=int(xmap(x)); draw.line((xx,top,xx,bottom),fill="#dedede",width=2)
        draw.text((xx-32,bottom+16),f"{x:,}",font=font(19),fill="#303840")
    for y in (0,.25,.5,.75,1):
        yy=int(ymap(y)); draw.line((left,yy,right,yy),fill="#d7dce0",width=2)
        draw.text((75,yy-12),f"{y:.2f}",font=font(19),fill="#303840")
    draw.line((left,top,left,bottom),fill="#252b30",width=3); draw.line((left,bottom,right,bottom),fill="#252b30",width=3)
    for method,color in COLORS.items():
        vals=series[method]; points=[(int(x),int(y)) for x,y,_,_ in vals]
        if len(points)>1: draw.line(points,fill=color,width=5,joint="curve")
        for x,y,hi,lo in vals:
            xx,yy=int(x),int(y); draw.line((xx,int(hi),xx,int(lo)),fill=color,width=3)
            draw.ellipse((xx-7,yy-7,xx+7,yy+7),fill=color)
    if isolated and null_rows:
        for method,color in COLORS.items():
            vals=[]
            for bi,bin_name in enumerate(BINS):
                if centers[bi] is None: continue
                row=next((r for r in null_rows if r["gt_subset"]=="isolated" and r["method"]==method and r["gt_area_bin_px"]==bin_name),None)
                if row is None or not row["null_completeness"]: continue
                vals.append((xmap(centers[bi]),ymap(row["null_completeness"])))
            for a,b in zip(vals,vals[1:]):
                dx,dy=b[0]-a[0],b[1]-a[1]; dist=max(1,int(np.hypot(dx,dy)))
                for start in range(0,dist,16):
                    end=min(start+8,dist)
                    p0=(int(a[0]+dx*start/dist),int(a[1]+dy*start/dist)); p1=(int(a[0]+dx*end/dist),int(a[1]+dy*end/dist))
                    draw.line((*p0,*p1),fill=color,width=4)
    axis="GT object area (pixels, log scale; median area per bin; 1 px = 0.25 m²)"
    draw.text((left,bottom+56),axis,font=font(22,True),fill="#202830")
    for i,(method,color) in enumerate(COLORS.items()):
        xx=left+i*410; yy=930
        draw.line((xx,yy+12,xx+50,yy+12),fill=color,width=6)
        draw.text((xx+64,yy),method,font=font(22),fill="#202830")
    path=OUT/("isolated_completeness_vs_size.png" if isolated else "completeness_vs_size.png")
    img.convert("RGB").save(path)


def null_plot(rows):
    w,h=1700,1200; img=Image.new("RGBA",(w,h),(255,255,255,255))
    draw=ImageDraw.Draw(img)
    draw.text((36,20),"Real vs deranged-mask null completeness · Wilson 95% CI",font=font(32,True),fill="#101820")
    left,right=170,1645; panel_h=390; centers=list(range(len(BINS)))
    xmap=lambda i:left+i*(right-left)/(len(BINS)-1)
    panels=(("all","All GT objects"),("isolated","10 px isolated GT objects"))
    for pi,(subset,label) in enumerate(panels):
        top=92+pi*510; bottom=top+panel_h
        draw.text((36,top+8),label,font=font(24,True),fill="#202830")
        panel_rows=[r for r in rows if r["gt_subset"]==subset]
        layer=Image.new("RGBA",img.size,(0,0,0,0)); ld=ImageDraw.Draw(layer)
        series={}
        for method,color in COLORS.items():
            for kind in ("real","null"):
                vals=[]
                for bi,bin_name in enumerate(BINS):
                    row=next(r for r in panel_rows if r["method"]==method and r["gt_area_bin_px"]==bin_name)
                    key="real_completeness_30pct" if kind=="real" else "null_completeness"
                    if row[key]=="": continue
                    lo_key="real_wilson95_low" if kind=="real" else "null_wilson95_low"
                    hi_key="real_wilson95_high" if kind=="real" else "null_wilson95_high"
                    vals.append((xmap(bi),bottom-float(row[key])*panel_h,
                        bottom-float(row[hi_key])*panel_h,bottom-float(row[lo_key])*panel_h))
                series[(method,kind)]=vals
                alpha=34 if kind=="real" else 20
                if vals: draw_band(ld,[(v[0],v[2]) for v in vals],[(v[0],v[3]) for v in vals],color,alpha)
        img=Image.alpha_composite(img,layer); draw=ImageDraw.Draw(img)
        for y in (0,.25,.5,.75,1):
            yy=int(bottom-y*panel_h); draw.line((left,yy,right,yy),fill="#d9dee2",width=2)
            draw.text((108,yy-12),f"{y:.2f}",font=font(19),fill="#303840")
        draw.line((left,top,left,bottom),fill="#252b30",width=3); draw.line((left,bottom,right,bottom),fill="#252b30",width=3)
        for i,bin_name in enumerate(BINS):
            xx=int(xmap(i)); draw.text((xx-45,bottom+12),bin_name,font=font(17),fill="#303840")
        for (method,kind),vals in series.items():
            color=COLORS[method]; points=[(int(x),int(y)) for x,y,_,_ in vals]
            if kind=="real" and len(points)>1: draw.line(points,fill=color,width=5,joint="curve")
            elif kind=="null":
                for a,b in zip(points,points[1:]):
                    dx,dy=b[0]-a[0],b[1]-a[1]; dist=max(1,int(np.hypot(dx,dy)))
                    for start in range(0,dist,18):
                        end=min(start+9,dist)
                        p0=(int(a[0]+dx*start/dist),int(a[1]+dy*start/dist)); p1=(int(a[0]+dx*end/dist),int(a[1]+dy*end/dist))
                        draw.line((*p0,*p1),fill=color,width=4)
            for x,y,_,_ in vals:
                xx,yy=int(x),int(y); draw.ellipse((xx-5,yy-5,xx+5,yy+5),fill=color)
    y=1140; x=210
    for i,(method,color) in enumerate(COLORS.items()):
        xx=x+i*360; draw.line((xx,y+12,xx+42,y+12),fill=color,width=5)
        draw.text((xx+52,y),method,font=font(20),fill="#202830")
    xx=1320; draw.line((xx,y+12,xx+42,y+12),fill="#333",width=5); draw.text((xx+54,y),"real (solid)",font=font(19),fill="#202830")
    for start in range(xx+205,xx+247,12): draw.line((start,y+12,min(start+6,xx+247),y+12),fill="#333",width=5)
    draw.text((xx+259,y),"null (dashed)",font=font(19),fill="#202830")
    img.convert("RGB").save(OUT/"object_null_completeness.png")


def main():
    null_rows=rows_from(OUT/"object_null_completeness_seed42.csv")
    medians=median_gt_areas()
    completeness_plot(rows_from(OUT/"object_completeness.csv"),False,medians,null_rows)
    completeness_plot(rows_from(OUT/"isolated_object_completeness.csv"),True,medians,null_rows)
    null_plot(null_rows)
    print("Updated charts; median x positions from test GT; x-axis 10–30,000 px")


if __name__=="__main__": main()
