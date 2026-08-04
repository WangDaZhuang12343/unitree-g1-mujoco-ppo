#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-g1

python3 train.py \
  --steps 5000000 \
  --envs 6 \
  --min-speed 0.0 \
  --max-speed 0.0 \
  --run-dir runs/g1_stand
