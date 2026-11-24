"""Wait for memory before a long stage, without acquiring a shared process lock."""
import argparse
from datetime import datetime
import re
import subprocess
import time
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
CUTOFF = datetime(2026, 11, 23, 4, 30, tzinfo=KST)
DEADLINE = datetime(2026, 11, 23, 6, 0, tzinfo=KST)


def wait_for_memory(training=False, *, enforce_direction_deadline=False):
    while True:
        now = datetime.now(KST)
        if enforce_direction_deadline and now >= (CUTOFF if training else DEADLINE):
            raise SystemExit("Stage start deadline reached")
        swap = subprocess.check_output(["sysctl", "-n", "vm.swapusage"], text=True)
        pressure = subprocess.check_output(["memory_pressure", "-Q"], text=True)
        amount, unit = re.search(r"used\s*=\s*([\d.]+)([BKMGTP])", swap).groups()
        gib = float(amount) * 1024 ** ("BKMGTP".index(unit) - 3)
        free = int(re.search(r"free percentage:\s*(\d+)%", pressure)[1])
        print(f"{now.isoformat(timespec='seconds')} swap_used={gib:.3f} GiB free={free}%", flush=True)
        if gib <= 3 and free >= 10:
            return
        time.sleep(300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--training", action="store_true")
    wait_for_memory(parser.parse_args().training)
