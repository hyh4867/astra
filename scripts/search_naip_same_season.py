"""Metadata-only STAC search for fall NAIP acquisitions over the Celina window."""
import json
from pathlib import Path
from pystac_client import Client
ROOT=Path(__file__).resolve().parents[1]
api=Client.open("https://planetarycomputer.microsoft.com/api/stac/v1")
bbox=[-96.80,33.31,-96.76,33.35]
rows=[]
for year in (2020,2022):
    start=f"{year}-09-01T00:00:00Z";end=f"{year}-12-31T23:59:59Z"
    items=list(api.search(collections=["naip"],bbox=bbox,datetime=f"{start}/{end}",limit=100).items())
    rows.extend({"year":year,"id":x.id,"datetime":x.datetime.isoformat() if x.datetime else None,"gsd":x.properties.get("gsd"),"bbox":x.bbox,"assets":list(x.assets)} for x in items)
out=ROOT/"outputs/naip_demo/naip_same_season_stac_search.json";out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(rows,indent=2)+"\n")
print(json.dumps({"counts":{str(y):sum(r["year"]==y for r in rows) for y in (2020,2022)},"items":rows[:12],"saved":str(out)},indent=2),flush=True)
