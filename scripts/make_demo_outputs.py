from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from PIL import Image, ImageDraw
from detector import detect_change, score

out = Path(__file__).resolve().parents[1] / "outputs"; out.mkdir(exist_ok=True)
a = Image.new("RGB", (480, 320), (96, 128, 94)); d = ImageDraw.Draw(a)
d.rectangle((0, 210, 480, 320), fill=(142, 132, 105)); d.rectangle((35, 40, 160, 165), fill=(78, 116, 77)); d.rectangle((300, 60, 430, 170), fill=(83, 120, 82))
b = a.copy(); ImageDraw.Draw(b).rectangle((185, 105, 276, 220), fill=(184, 74, 62))
truth = np.zeros((320,480),dtype=np.uint8); truth[105:221,185:277]=255
arr_a, arr_b = np.array(a), np.array(b); mask, overlay = detect_change(arr_a,arr_b,threshold=20,blur=1)
for name, im in [("before.png",a),("after.png",b),("change_mask.png",Image.fromarray(mask)),("overlay.png",Image.fromarray(overlay))]: im.save(out/name)
canvas=Image.new("RGB",(960,320),(245,245,245)); canvas.paste(a,(0,0)); canvas.paste(Image.fromarray(overlay),(480,0)); canvas.save(out/"side_by_side.png")
metrics=score(mask,truth)
with open(out/"metrics.txt","w") as f: f.write("Synthetic OpenCV demo metrics\n"+"\n".join(f"{k}: {v:.4f}" for k,v in metrics.items() if isinstance(v,float))+"\n")
