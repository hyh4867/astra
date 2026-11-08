#!/bin/zsh
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTORCH_ENABLE_MPS_FALLBACK=1
./.venv/bin/python -u scripts/finetune_tinycd_robust.py --mode control --epochs 20 --stop-after 10 --batch-size 1 --seed 42
./.venv/bin/python -u scripts/finetune_tinycd_robust.py --mode robust --epochs 20 --stop-after 10 --batch-size 1 --seed 42
echo "Both 20-epoch runs completed at $(date '+%Y-%m-%d %H:%M:%S %Z')" 
