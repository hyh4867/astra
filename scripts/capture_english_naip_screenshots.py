"""Capture the NAIP Analyze, Results, and Limitations tabs with local headless Chromium."""
from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    from playwright.async_api import async_playwright
except ModuleNotFoundError:
    async_playwright = None

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"
LOG = ROOT / "logs/app_screenshots_streamlit.log"
PORT = 8501
URL = f"http://127.0.0.1:{PORT}"


async def start_server(hide_map: bool):
    env = os.environ.copy()
    env.pop("ASTRA_TEST_MODE", None)
    env["ASTRA_FORCE_CPU"] = "1"
    if hide_map:
        env["ASTRA_SCREENSHOT_HIDE_MAP"] = "1"
    else:
        env.pop("ASTRA_SCREENSHOT_HIDE_MAP", None)
    env["PLAYWRIGHT_BROWSERS_PATH"] = str(ROOT / ".cache/ms-playwright")
    LOG.parent.mkdir(parents=True, exist_ok=True)
    log_file = LOG.open("a")
    proc = await asyncio.create_subprocess_exec(
        sys.executable, "-m", "streamlit", "run", str(ROOT / "app.py"),
        "--server.headless=true", f"--server.port={PORT}", "--server.address=127.0.0.1",
        "--browser.gatherUsageStats=false", cwd=ROOT, env=env,
        stdout=log_file, stderr=asyncio.subprocess.STDOUT, start_new_session=True,
    )
    for _ in range(120):
        if proc.returncode is not None:
            log_file.close()
            raise RuntimeError(f"Streamlit exited early with status {proc.returncode}; see {LOG}")
        try:
            with urlopen(URL + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    return proc, log_file
        except (URLError, TimeoutError, OSError):
            await asyncio.sleep(1)
    proc.terminate(); await proc.wait(); log_file.close()
    raise TimeoutError(f"Streamlit health check timed out; see {LOG}")


async def capture_with_chrome_cdp() -> None:
    """Use the installed Chrome and Python websockets when Playwright is unavailable."""
    import websockets

    chrome_bin = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
    if not chrome_bin.exists():
        raise FileNotFoundError(f"Chrome executable not found: {chrome_bin}")
    profile = ROOT / ".cache/round3-chrome-cdp"
    profile.mkdir(parents=True, exist_ok=True)
    chrome_log = (ROOT / "logs/chrome_cdp.log").open("a")
    proc = subprocess.Popen(
        [str(chrome_bin), "--headless=new", "--disable-gpu", "--no-sandbox",
         "--disable-dev-shm-usage", "--no-first-run", "--no-default-browser-check",
         "--remote-debugging-address=127.0.0.1", "--remote-debugging-port=9225",
         "--force-device-scale-factor=1", "--window-size=1440,1800",
         f"--user-data-dir={profile}", "about:blank"],
        stdout=chrome_log, stderr=subprocess.STDOUT, start_new_session=True,
    )
    try:
        targets = None
        for _ in range(30):
            if proc.poll() is not None:
                raise RuntimeError(f"Headless Chrome exited with {proc.returncode}; see logs/chrome_cdp.log")
            try:
                with urlopen("http://127.0.0.1:9225/json/list", timeout=2) as response:
                    targets = json.loads(response.read())
                break
            except (URLError, TimeoutError, OSError):
                await asyncio.sleep(1)
        if not targets:
            raise TimeoutError("Chrome DevTools endpoint did not start")
        target = next((x for x in targets if x.get("type") == "page"), targets[0])
        ws_url = target["webSocketDebuggerUrl"]
        async with websockets.connect(ws_url, max_size=32 * 1024 * 1024) as ws:
            command_id = 0

            async def cdp(method: str, params: dict | None = None, timeout: float = 30):
                nonlocal command_id
                command_id += 1
                this_id = command_id
                await ws.send(json.dumps({"id": this_id, "method": method, "params": params or {}}))
                while True:
                    message = json.loads(await asyncio.wait_for(ws.recv(), timeout=timeout))
                    if message.get("id") == this_id:
                        if "error" in message:
                            raise RuntimeError(f"CDP {method}: {message['error']}")
                        return message.get("result", {})

            async def evaluate(expression: str, timeout: float = 30):
                result = await cdp("Runtime.evaluate", {"expression": expression,
                    "returnByValue": True, "awaitPromise": True}, timeout)
                if "exceptionDetails" in result:
                    raise RuntimeError(str(result["exceptionDetails"]))
                return result.get("result", {}).get("value")

            await cdp("Page.enable")
            await cdp("Runtime.enable")
            await cdp("Page.setDeviceMetricsOverride", {"width": 1440, "height": 1800,
                 "deviceScaleFactor": 1, "mobile": False})
            await cdp("Page.navigate", {"url": URL})
            state = None
            for _ in range(1200):
                await asyncio.sleep(1)
                state = await evaluate("({text: document.body?.innerText || '', tabs: Array.from(document.querySelectorAll('[role=tab]')).map(x => x.innerText.trim())})")
                if state and "Qualitative demo only" in state["text"] and (
                    "TinyCD device:" in state["text"] or "TinyCD inference failed" in state["text"]
                ):
                    break
            else:
                raise TimeoutError("Streamlit did not finish rendering/inference")
            await asyncio.sleep(5)
            OUT.mkdir(parents=True, exist_ok=True)

            async def screenshot(filename: str) -> None:
                result = await cdp("Page.captureScreenshot", {"format": "png", "fromSurface": True,
                                    "captureBeyondViewport": False}, timeout=60)
                (OUT / filename).write_bytes(base64.b64decode(result["data"]))

            await screenshot("app_analyze_en.png")
            for tab_name, filename in (("Results", "app_results_en.png"),
                                       ("Limitations", "app_limitations_en.png")):
                clicked = await evaluate(
                    "(() => { const el = Array.from(document.querySelectorAll('[role=tab]')).find(x => x.innerText.trim() === "
                    + json.dumps(tab_name) + "); if (!el) return false; el.click(); return true; })()"
                )
                if not clicked:
                    raise RuntimeError(f"Could not find Streamlit tab {tab_name}; saw {state['tabs']}")
                await asyncio.sleep(2)
                await screenshot(filename)
            await evaluate("(() => { const el = Array.from(document.querySelectorAll('[role=tab]')).find(x => x.innerText.trim() === 'Analyze'); if (el) el.click(); return !!el; })()")
            await asyncio.sleep(2)
            changed = await evaluate("(async () => { const box = Array.from(document.querySelectorAll('[data-testid=stSelectbox]')).find(x => x.innerText.includes('TinyCD model')); const combo = box && box.querySelector('[role=combobox]'); if (!combo) return false; combo.click(); await new Promise(r => setTimeout(r, 500)); const option = Array.from(document.querySelectorAll('[role=option]')).find(x => x.innerText.includes('ROBUST-v2 (blur/order-robust fine-tune)')); if (!option) return false; option.click(); return true; })()")
            if not changed:
                raise RuntimeError("Could not select ROBUST-v2 in the Analyze tab")
            await asyncio.sleep(5000)
            await screenshot("app_analyze_en_robust.png")
            print("Saved English NAIP screenshots via Chrome CDP:",
                  *(OUT / n for n in ("app_analyze_en.png", "app_results_en.png", "app_limitations_en.png", "app_analyze_en_robust.png")), flush=True)
    finally:
        if proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill(); proc.wait()
        chrome_log.close()


async def capture_with_native_webkit() -> None:
    """Fallback to macOS WebKit when Chromium cannot start in the remote session."""
    script = ROOT / "scripts/capture_screenshots_webkit.swift"
    proc = await asyncio.create_subprocess_exec(
        "swift", str(script), str(OUT), cwd=ROOT,
        stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT,
    )
    try:
        output, _ = await asyncio.wait_for(proc.communicate(), timeout=300)
    except asyncio.TimeoutError:
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
        raise TimeoutError("Native WebKit did not finish capturing screenshots within 5 minutes")
    if output:
        print(output.decode("utf-8", errors="replace").strip(), flush=True)
    if proc.returncode:
        raise RuntimeError(f"Native WebKit screenshot process exited {proc.returncode}")


async def capture_attempt(hide_map: bool) -> None:
    server, log_file = await start_server(hide_map)
    browser = None
    try:
        if async_playwright is None:
            try:
                await capture_with_chrome_cdp()
            except Exception as exc:
                print(f"Chromium unavailable ({type(exc).__name__}: {exc}); trying macOS WebKit", flush=True)
                await capture_with_native_webkit()
            return
        async with async_playwright() as playwright:
            browser = await playwright.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
            context = await browser.new_context(viewport={"width": 1440, "height": 1800}, device_scale_factor=1)

            async def block_external(route):
                url = route.request.url
                if url.startswith(("http://127.0.0.1", "http://localhost", "data:", "blob:")):
                    await route.continue_()
                else:
                    await route.abort()

            await context.route("**/*", block_external)
            page = await context.new_page()
            await page.goto(URL, wait_until="domcontentloaded", timeout=120_000)
            await page.get_by_text("Qualitative demo only", exact=False).wait_for(timeout=120_000)
            await page.wait_for_function(
                "document.body.innerText.includes('TinyCD device:') || document.body.innerText.includes('TinyCD inference failed')",
                timeout=20 * 60_000,
            )
            await page.wait_for_timeout(5000)
            OUT.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(OUT / "app_analyze_en.png"), full_page=False)
            for tab_name, filename in (("Results", "app_results_en.png"), ("Limitations", "app_limitations_en.png")):
                await page.get_by_role("tab", name=tab_name).click()
                await page.wait_for_timeout(1000)
                await page.screenshot(path=str(OUT / filename), full_page=False)
            await page.get_by_role("tab", name="Analyze").click()
            changed = await page.evaluate("(async () => { const box = Array.from(document.querySelectorAll('[data-testid=stSelectbox]')).find(x => x.innerText.includes('TinyCD model')); const combo = box && box.querySelector('[role=combobox]'); if (!combo) return false; combo.click(); await new Promise(r => setTimeout(r, 500)); const option = Array.from(document.querySelectorAll('[role=option]')).find(x => x.innerText.includes('ROBUST-v2 (blur/order-robust fine-tune)')); if (!option) return false; option.click(); return true; })()")
            if not changed:
                raise RuntimeError("Could not select ROBUST-v2 in the Analyze tab")
            await page.wait_for_timeout(5000)
            await page.screenshot(path=str(OUT / "app_analyze_en_robust.png"), full_page=False)
            print("Saved English NAIP screenshots:", *(OUT / n for n in ("app_analyze_en.png", "app_results_en.png", "app_limitations_en.png", "app_analyze_en_robust.png")), flush=True)
    finally:
        if browser is not None:
            try: await browser.close()
            except Exception: pass
        if server.returncode is None:
            server.terminate()
            try: await asyncio.wait_for(server.wait(), timeout=10)
            except asyncio.TimeoutError:
                server.kill(); await server.wait()
        log_file.close()


async def main() -> None:
    try:
        await capture_attempt(False)
    except Exception as exc:
        print(f"Map-enabled screenshot attempt failed: {type(exc).__name__}: {exc}", flush=True)
        await capture_attempt(True)


if __name__ == "__main__":
    asyncio.run(main())
