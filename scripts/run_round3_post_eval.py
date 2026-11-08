"""After model evaluation, create the NAIP candidate sheet and English app screenshots."""
from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs"
EVAL_STATUS = LOG / "round3_evaluation_status.json"
RESULT = LOG / "round3_post_eval_status.json"
STATUS = ROOT / "STATUS.md"


def run_step(name: str, argv: list[str], env: dict[str, str]) -> dict:
    stdout_path = LOG / f"{name}.stdout.log"
    stderr_path = LOG / f"{name}.stderr.log"
    with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
        proc = subprocess.run(["/usr/bin/caffeinate", "-dimsu", *argv], cwd=ROOT,
                              env=env, stdout=stdout, stderr=stderr, check=False)
    return {"step": name, "returncode": proc.returncode,
            "stdout": str(stdout_path.relative_to(ROOT)), "stderr": str(stderr_path.relative_to(ROOT))}


def main() -> int:
    LOG.mkdir(exist_ok=True)
    while not EVAL_STATUS.exists():
        time.sleep(30)
    env = os.environ.copy()
    env.update({"PYTHONUNBUFFERED": "1", "PYTORCH_ENABLE_MPS_FALLBACK": "1",
                "TORCH_HOME": str(ROOT / ".torch"), "XDG_CACHE_HOME": str(ROOT / ".cache"),
                "PIP_CACHE_DIR": str(ROOT / ".cache/pip"),
                "PLAYWRIGHT_BROWSERS_PATH": str(ROOT / ".cache/ms-playwright")})
    steps = []
    # A failed STAC lookup is recorded, then local screenshot capture still proceeds.
    steps.append(run_step("round3_naip_candidates", [sys.executable, "-u", str(ROOT / "scripts/search_naip_construction_candidates.py")], env))
    if importlib.util.find_spec("playwright") is None:
        steps.append(run_step("playwright_install", [sys.executable, "-m", "pip", "install", "--no-cache-dir", "playwright"], env))
    browser_cache = ROOT / ".cache/ms-playwright"
    if not any(browser_cache.glob("chromium-*/chrome-mac/Chromium.app/Contents/MacOS/Chromium")):
        steps.append(run_step("playwright_chromium_install", [sys.executable, "-m", "playwright", "install", "chromium"], env))
    steps.append(run_step("round3_app_screenshots", [sys.executable, "-u", str(ROOT / "scripts/capture_english_naip_screenshots.py")], env))
    RESULT.write_text(json.dumps({"finished_local": datetime.now().astimezone().isoformat(timespec="seconds"), "steps": steps}, indent=2) + "\n")
    with STATUS.open("a") as f:
        f.write("\n\n## Round 3 post-evaluation automation\n\n")
        for row in steps:
            state = "completed" if row["returncode"] == 0 else f"failed (exit {row['returncode']})"
            f.write(f"- {row['step']}: {state}; stdout `{row['stdout']}`, stderr `{row['stderr']}`.\n")
        f.write("- No PDF was rebuilt. Candidate selection uses the declared RGB transition rule; TinyCD is not run on candidate windows.\n")
    return 0 if all(row["returncode"] == 0 for row in steps) else 1


if __name__ == "__main__":
    raise SystemExit(main())
