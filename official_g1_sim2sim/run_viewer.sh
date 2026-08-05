#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f build/libort_bridge.so ]]; then
  ./build.sh
fi

exec python3 simulate.py --duration 120 --vx 0.45 --viewer --output runs/viewer_0.45.csv
