"""Wait for all selected v2 runs, then run the shared CPU evaluation once."""
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
INITIAL = LOG / "v2_sequence_status.json"
REPEATS = LOG / "v2_repeat_seeds_status.json"
EVALUATOR = ROOT / "scripts/evaluate_round3_models.py"
STATUS = LOG / "round3_evaluation_status.json"


def read_rows(path: Path) -> list[dict]:
    try: return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError): return []


def main() -> int:
    LOG.mkdir(exist_ok=True)
    print("waiting for the seed-42 training pair", flush=True)
    while True:
        initial = read_rows(INITIAL)
        if len(initial) >= 2: break
        time.sleep(30)
    initial_start = datetime.fromisoformat(initial[0]["started_utc"])
    initial_end = datetime.fromisoformat(initial[1]["finished_utc"])
    fast_pair = (initial_end-initial_start).total_seconds() < 3 * 3600 and all(r.get("returncode") == 0 for r in initial[:2])
    if fast_pair:
        print("initial pair finished under 3h; waiting for conditional seed 43/44 repeats", flush=True)
        while True:
            repeats = read_rows(REPEATS)
            if len(repeats) >= 4: break
            time.sleep(30)
    env = os.environ.copy()
    env.update({"PYTORCH_ENABLE_MPS_FALLBACK": "1", "PYTHONUNBUFFERED": "1",
                "TORCH_HOME": str(ROOT / ".torch"), "XDG_CACHE_HOME": str(ROOT / ".cache")})
    out_path, err_path = LOG / "round3_evaluation.stdout.log", LOG / "round3_evaluation.stderr.log"
    print("starting shared Round 3 CPU evaluation", flush=True)
    with out_path.open("w") as stdout, err_path.open("w") as stderr:
        proc = subprocess.run(["/usr/bin/caffeinate", "-dimsu", sys.executable, "-u", str(EVALUATOR)],
                              cwd=ROOT, env=env, stdout=stdout, stderr=stderr, check=False)
    result = {"returncode": proc.returncode, "stdout": str(out_path.relative_to(ROOT)),
              "stderr": str(err_path.relative_to(ROOT)), "finished_local": datetime.now().astimezone().isoformat(timespec="seconds")}
    STATUS.write_text(json.dumps(result, indent=2) + "\n")
    print(f"evaluation exit={proc.returncode}; see {out_path.name}", flush=True)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
