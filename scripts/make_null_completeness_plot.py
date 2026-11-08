"""Render the deranged-mask null comparison from local summary CSV."""
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
CSV=OUT/"object_null_completeness_seed42.csv"
BINS=["<100","100-250","250-500","500-1000","1000-2500","2500-5000",">=5000"]
METHODS=["OpenCV basic","OpenCV improved","TinyCD"]
COLORS={METHODS[0]:"#2878b5",METHODS[1]:"#ef8354",METHODS[2]:"#3b9b65"}

rows=list(csv.DictReader(CSV.open()))
im=Image.new("RGB",(1500,950),"white"); d=ImageDraw.Draw(im)
d.text((30,18),"Real vs fixed deranged-mask null object completeness (30% coverage)",fill="#111")
x0,x1=130,1370; panel_h=350
for panel_i,subset in enumerate(("all","isolated")):
    top=75+panel_i*425; bottom=top+panel_h
    d.text((30,top+8),"All GT objects" if subset=="all" else "10 px isolated GT objects",fill="#111")
    for yv in (0,.25,.5,.75,1):
        yy=int(bottom-yv*panel_h); d.line((x0,yy,x1,yy),fill="#e5e5e5"); d.text((75,yy-7),f"{yv:.2g}",fill="#333")
    d.line((x0,top,x0,bottom),fill="#333",width=2); d.line((x0,bottom,x1,bottom),fill="#333",width=2)
    xs=[int(x0+i*(x1-x0)/(len(BINS)-1)) for i in range(len(BINS))]
    for xx,label in zip(xs,BINS): d.text((xx-27,bottom+10),label,fill="#333")
    for method,color in COLORS.items():
        for kind,offset,solid in (("real",-5,True),("null",5,False)):
            selected=[next(r for r in rows if r["gt_subset"]==subset and r["method"]==method and r["gt_area_bin_px"]==b) for b in BINS]
            key="real_completeness_30pct" if kind=="real" else "null_completeness"
            pts=[]
            for xx,r in zip(xs,selected):
                if r[key]=="": continue
                y=int(bottom-float(r[key])*panel_h); x=xx+offset
                if kind=="null":
                    lo=int(bottom-float(r["null_wilson95_high"])*panel_h); hi=int(bottom-float(r["null_wilson95_low"])*panel_h)
                    d.line((x,lo,x,hi),fill=color,width=2)
                pts.append((x,y))
            if len(pts)>1:
                for a,b in zip(pts,pts[1:]):
                    if solid: d.line((*a,*b),fill=color,width=3)
                    else:
                        dx,dy=b[0]-a[0],b[1]-a[1]; dist=max(1,int(np.hypot(dx,dy)))
                        for start in range(0,dist,12):
                            end=min(start+6,dist)
                            q0=(int(a[0]+dx*start/dist),int(a[1]+dy*start/dist)); q1=(int(a[0]+dx*end/dist),int(a[1]+dy*end/dist))
                            d.line((*q0,*q1),fill=color,width=3)
            for x,y in pts: d.ellipse((x-4,y-4,x+4,y+4),fill=color)
    d.text((x0,bottom+40),"GT component area bin (pixels)",fill="#333")
for i,(method,color) in enumerate(COLORS.items()):
    y=920; x=410+i*270; d.line((x,y+7,x+32,y+7),fill=color,width=4); d.text((x+40,y),method,fill="#222")
d.line((1200,927,1240,927),fill="#555",width=3); d.text((1248,920),"solid real / dashed null",fill="#333")
im.save(OUT/"object_null_completeness.png")
