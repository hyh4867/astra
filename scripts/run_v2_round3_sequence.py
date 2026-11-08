"""Run CONTROL-v2 then ROBUST-v2 sequentially on the CPU fallback."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs"
TRAINER = ROOT / "scripts/finetune_tinycd_v2.py"


def stamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def main() -> int:
    LOG.mkdir(exist_ok=True)
    env = os.environ.copy()
    env.update({
        "PYTORCH_ENABLE_MPS_FALLBACK": "1",
        "PYTHONUNBUFFERED": "1",
        "TORCH_HOME": str(ROOT / ".torch"),
        "XDG_CACHE_HOME": str(ROOT / ".cache"),
    })
    results = []
    for mode in ("control", "robust"):
        started = stamp()
        stdout_path = LOG / f"{mode}_v2_seed42.stdout.log"
        stderr_path = LOG / f"{mode}_v2_seed42.stderr.log"
        cmd = [
            "/usr/bin/caffeinate", "-dimsu", sys.executable, "-u", str(TRAINER),
            "--mode", mode, "--seed", "42", "--batch-size", "1", "--epochs", "8", "--force-cpu",
        ]
        print(f"start {mode}-v2 at {started}; CPU fallback, batch=1, num_workers=0", flush=True)
        with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
            proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
        result = {
            "mode": mode, "started_utc": started, "finished_utc": stamp(),
            "returncode": proc.returncode, "stdout": str(stdout_path.relative_to(ROOT)),
            "stderr": str(stderr_path.relative_to(ROOT)),
        }
        results.append(result)
        (LOG / "v2_sequence_status.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"finish {mode}-v2 returncode={proc.returncode}; see {stdout_path.name}", flush=True)
    return 0 if all(row["returncode"] == 0 for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
