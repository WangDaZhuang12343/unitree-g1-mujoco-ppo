#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TERRAIN="${1:-rough}"
LEVEL="${2:-0.02}"
cd "$ROOT"

case "$TERRAIN" in
  ramp)
    EXTRA=(--slope-angle "$LEVEL" --obstacle-x 1.8)
    ;;
  step|stairs)
    EXTRA=(--obstacle-height "$LEVEL")
    ;;
  rough)
    EXTRA=(--roughness "$LEVEL" --obstacle-x 0.8)
    ;;
  *)
    echo "用法：$0 {ramp|step|stairs|rough} 数值" >&2
    exit 2
    ;;
esac

exec python3 simulate.py \
  --terrain "$TERRAIN" \
  --vx 0.45 \
  --duration 20 \
  --viewer \
  --output "runs/viewer_${TERRAIN}_${LEVEL}.csv" \
  "${EXTRA[@]}"
