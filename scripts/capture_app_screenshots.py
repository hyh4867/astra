"""Capture the local NAIP qualitative demo tab screenshots using headless Chrome."""
from __future__ import annotations

import asyncio
import base64
import json
import os
from pathlib import Path
from urllib.request import urlopen

import websockets

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs"
DEBUG="http://127.0.0.1:9222/json"
APP_URL=os.environ.get("ASTRA_APP_URL","http://127.0.0.1:8501/?method=tinycd")


async def main():
    with urlopen(DEBUG,timeout=10) as response:
        targets=json.loads(response.read())
    target=next(t for t in targets if t.get("type")=="page")
    async with websockets.connect(target["webSocketDebuggerUrl"],max_size=None) as ws:
        serial=0
        async def cdp(method,params=None):
            nonlocal serial
            serial+=1; ident=serial
            await ws.send(json.dumps({"id":ident,"method":method,"params":params or {}}))
            while True:
                message=json.loads(await ws.recv())
                if message.get("id")==ident: return message.get("result",{})
        await cdp("Page.enable")
        await cdp("Emulation.setDeviceMetricsOverride",{"width":1440,"height":1500,"deviceScaleFactor":1,"mobile":False})
        await cdp("Page.navigate",{"url":APP_URL})
        for _ in range(90):
            await asyncio.sleep(1)
            state=await cdp("Runtime.evaluate",{"expression":"document.querySelector('[data-testid=stApp]') && document.querySelectorAll('[role=tab]').length","returnByValue":True})
            if state.get("result",{}).get("value",0)>=3: break
        else: raise TimeoutError("Streamlit tabs did not render")
        for _ in range(90):
            await asyncio.sleep(1)
            state=await cdp("Runtime.evaluate",{"expression":"document.body.innerText.includes('TinyCD device') || document.body.innerText.includes('TinyCD inference failed')","returnByValue":True})
            if state.get("result",{}).get("value"): break
        else: raise TimeoutError("TinyCD synthetic demo did not finish")

        async def save(filename):
            await asyncio.sleep(1)
            result=await cdp("Page.captureScreenshot",{"format":"png","fromSurface":True})
            (OUT/filename).write_bytes(base64.b64decode(result["data"]))

        await save("app_analyze.png")
        for label,filename in (("Results","app_results.png"),("Limitations","app_limitations.png")):
            expression=f"(()=>{{const t=[...document.querySelectorAll('[role=tab]')].find(x=>x.innerText.trim()==={json.dumps(label)});if(!t)return false;t.click();return true}})()"
            result=await cdp("Runtime.evaluate",{"expression":expression,"returnByValue":True})
            if not result.get("result",{}).get("value"): raise RuntimeError(f"Tab not found: {label}")
            await save(filename)
        print("Saved synthetic-only screenshots:",*[OUT/f"app_{x}.png" for x in ("analyze","results","limitations")],sep="\n")


if __name__=="__main__": asyncio.run(main())
