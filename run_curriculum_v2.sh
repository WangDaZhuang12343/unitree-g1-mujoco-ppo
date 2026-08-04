#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR=/tmp/matplotlib-g1

STAND_MODEL="runs/g1_stand/best/best_model.zip"
COMMON=(
  --envs 6
  --min-speed 0.0
  --learning-rate 0.0001
  --ent-coef 0.001
  --action-scale-factor 0.5
  --gait-clock
  --gait-reference
  --early-stop-successes 3
)

test -f "$STAND_MODEL"

python3 train.py "${COMMON[@]}" \
  --steps 1000000 \
  --max-speed 0.10 \
  --eval-speed 0.10 \
  --resume "$STAND_MODEL" \
  --reset-action-std 0.3 \
  --initialize-command-inputs \
  --run-dir runs/g1_walk_010

python3 train.py "${COMMON[@]}" \
  --steps 1500000 \
  --max-speed 0.25 \
  --eval-speed 0.25 \
  --resume runs/g1_walk_010/best/best_model.zip \
  --run-dir runs/g1_walk_025

python3 train.py "${COMMON[@]}" \
  --steps 2500000 \
  --max-speed 0.45 \
  --eval-speed 0.45 \
  --resume runs/g1_walk_025/best/best_model.zip \
  --run-dir runs/g1_walk_045

python3 train.py "${COMMON[@]}" \
  --steps 5000000 \
  --max-speed 0.70 \
  --eval-speed 0.70 \
  --resume runs/g1_walk_045/best/best_model.zip \
  --run-dir runs/g1_walk
