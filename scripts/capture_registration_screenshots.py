"""Capture the alignment-check flow of the running Streamlit app (headless Chrome).

Start the app and Chrome first:
    streamlit run app.py --server.headless true --server.port 8511
    "$CHROME_BIN" --headless=new \\
        --remote-debugging-port=9223 --user-data-dir=<temp dir> about:blank
Then: python scripts/capture_registration_screenshots.py [frames_dir]

Shots use the public-domain NAIP demo and the synthetic repeated-pattern pair
only (no LEVIR imagery). With frames_dir, viewport frames of a scroll through
each state are also written for a short screen recording.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
import sys
from pathlib import Path
from urllib.request import urlopen

import websockets

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "registration" / "screens"
APP = os.environ.get("ASTRA_APP_URL", "http://127.0.0.1:8511")
DEBUG = os.environ.get("ASTRA_CHROME_DEBUG", "http://127.0.0.1:9223/json")
SHOTS = [
    ("naip_1_as_is", "/?method=tinycd", "TinyCD device"),
    ("naip_2_shift_corrected", "/?method=tinycd&reg=auto&shift_dx=6&shift_dy=-4", "TinyCD device"),
    ("naip_3_shift_small_not_applied", "/?method=tinycd&reg=auto&shift_dx=2&shift_dy=2", "TinyCD device"),
    ("naip_4_shift_out_of_range_review", "/?method=tinycd&reg=auto&shift_dx=10&shift_dy=-8", "TinyCD device"),
    ("synthetic_repeated_pattern_review", "/?demo=1&source=repeat", "Fixed OpenCV settings"),
]
WIDTH, VIEW_H, TALL_H = 1280, 800, 5200


async def main():
    frames_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    OUT.mkdir(parents=True, exist_ok=True)
    with urlopen(DEBUG, timeout=10) as response:
        target = next(t for t in json.loads(response.read()) if t.get("type") == "page")
    async with websockets.connect(target["webSocketDebuggerUrl"], max_size=None) as ws:
        serial = 0

        async def cdp(method, params=None):
            nonlocal serial
            serial += 1; ident = serial
            await ws.send(json.dumps({"id": ident, "method": method, "params": params or {}}))
            while True:
                message = json.loads(await ws.recv())
                if message.get("id") == ident:
                    return message.get("result", {})

        async def evaluate(expression):
            result = await cdp("Runtime.evaluate", {"expression": expression, "returnByValue": True})
            return result.get("result", {}).get("value")

        async def shot(path, fmt="jpeg"):
            params = {"format": fmt, "fromSurface": True}
            if fmt == "jpeg":
                params["quality"] = 82
            result = await cdp("Page.captureScreenshot", params)
            path.write_bytes(base64.b64decode(result["data"]))

        await cdp("Page.enable")
        frame = 0
        for name, query, ready_text in SHOTS:
            await cdp("Emulation.setDeviceMetricsOverride", {"width": WIDTH, "height": TALL_H, "deviceScaleFactor": 1, "mobile": False})
            await cdp("Page.navigate", {"url": APP + query})
            for _ in range(120):
                await asyncio.sleep(1)
                state = await evaluate(f"document.body.innerText.includes({json.dumps(ready_text)}) && (document.body.innerText.includes('객체 목록') || document.body.innerText.includes('검토용 후보 목록'))")
                if state:
                    break
            else:
                body = await evaluate("document.body.innerText")
                raise TimeoutError(f"{name}: {str(body)[-2000:]}")
            await asyncio.sleep(4)
            notices = await evaluate("Array.from(document.querySelectorAll('[data-testid=stAlert]'), e => e.innerText)")
            print(name, notices, flush=True)
            await shot(OUT / f"{name}.jpg")
            print("saved", OUT / f"{name}.jpg", flush=True)
            if frames_dir:
                frames_dir.mkdir(parents=True, exist_ok=True)
                await cdp("Emulation.setDeviceMetricsOverride", {"width": WIDTH, "height": VIEW_H, "deviceScaleFactor": 1, "mobile": False})
                await asyncio.sleep(2)
                scroller = "(document.querySelector('[data-testid=stMain]')||document.scrollingElement)"
                top = await evaluate(f"(()=>{{const h=[...document.querySelectorAll('h3')].find(x=>x.innerText.includes('Alignment check'));const s={scroller};return h?h.getBoundingClientRect().top+s.scrollTop-90:0}})()") or 0
                for step in range(22):
                    await evaluate(f"{scroller}.scrollTo(0,{int(top) + step * 75})")
                    await asyncio.sleep(0.25)
                    for _ in range(6 if step in (0, 21) else 2):   # hold longer on the first and last view
                        await shot(frames_dir / f"frame_{frame:05d}.png", "png"); frame += 1


if __name__ == "__main__":
    asyncio.run(main())
