#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f build/libort_bridge.so ]]; then
  ./build.sh
fi

exec python3 navigation_sim.py \
  --duration 30 \
  --goal-x 5.0 \
  --obstacle-x 2.5 \
  --obstacle-height 0.15 \
  --obstacle-width 0.8 \
  --viewer \
  --output runs/navigation/viewer.csv
