"""Wait for the seed-42 pair; repeat seeds 43/44 only if it finishes under 3h."""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOG = ROOT / "logs"
SEQUENCE = LOG / "v2_sequence_status.json"
TRAINER = ROOT / "scripts/finetune_tinycd_v2.py"
WATCH = LOG / "v2_repeat_seeds_watch.log"


def append(text: str) -> None:
    with WATCH.open("a") as f:
        f.write(f"{datetime.now().astimezone().isoformat(timespec='seconds')} {text}\n")


def main() -> int:
    LOG.mkdir(exist_ok=True)
    append("waiting for seed-42 CONTROL and ROBUST sequence")
    while True:
        if SEQUENCE.exists():
            try:
                rows = json.loads(SEQUENCE.read_text())
            except (json.JSONDecodeError, OSError):
                rows = []
            if len(rows) >= 2 and {r.get("mode") for r in rows[:2]} >= {"control", "robust"}:
                break
        time.sleep(30)
    start = datetime.fromisoformat(rows[0]["started_utc"])
    finish = datetime.fromisoformat(rows[1]["finished_utc"])
    elapsed = (finish - start).total_seconds()
    if elapsed >= 3 * 3600 or any(r.get("returncode") != 0 for r in rows[:2]):
        append(f"additional seeds skipped; initial_elapsed_seconds={elapsed:.1f}; initial_success={all(r.get('returncode') == 0 for r in rows[:2])}")
        return 0
    append(f"initial pair finished in {elapsed:.1f}s; launching seeds 43 and 44 sequentially")
    env = os.environ.copy()
    env.update({"PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1",
                "TORCH_HOME": str(ROOT / ".torch"), "XDG_CACHE_HOME": str(ROOT / ".cache")})
    results = []
    for seed in (43, 44):
        for mode in ("control", "robust"):
            stdout_path = LOG / f"{mode}_v2_seed{seed}.stdout.log"
            stderr_path = LOG / f"{mode}_v2_seed{seed}.stderr.log"
            cmd = ["/usr/bin/caffeinate", "-dimsu", sys.executable, "-u", str(TRAINER),
                   "--mode", mode, "--seed", str(seed), "--batch-size", "1", "--epochs", "8", "--force-cpu"]
            append(f"start {mode}-v2 seed={seed}; CPU fallback; batch=1")
            with stdout_path.open("w") as stdout, stderr_path.open("w") as stderr:
                proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
            row = {"mode": mode, "seed": seed, "returncode": proc.returncode,
                   "stdout": str(stdout_path.relative_to(ROOT)), "stderr": str(stderr_path.relative_to(ROOT))}
            results.append(row)
            (LOG / "v2_repeat_seeds_status.json").write_text(json.dumps(results, indent=2) + "\n")
            append(f"finish {mode}-v2 seed={seed}; returncode={proc.returncode}")
    return 0 if all(r["returncode"] == 0 for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
