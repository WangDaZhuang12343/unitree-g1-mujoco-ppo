#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HEIGHT="${1:-0.02}"
cd "$ROOT"

if [[ ! -f build/libort_bridge.so ]]; then
  ./build.sh
fi

exec python3 simulate.py \
  --duration 12 \
  --vx 0.45 \
  --terrain bar \
  --obstacle-height "$HEIGHT" \
  --viewer \
  --output "runs/viewer_bar_${HEIGHT}m.csv"
