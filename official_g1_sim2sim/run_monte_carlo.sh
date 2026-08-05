#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f build/libort_bridge.so ]]; then
  ./build.sh
fi

exec nice -n 10 python3 benchmark_monte_carlo.py \
  --runs 100 \
  --resume \
  --output runs/navigation_monte_carlo
