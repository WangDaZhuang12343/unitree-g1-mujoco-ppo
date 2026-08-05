#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

if [[ ! -f build/libort_bridge.so ]]; then
  ./build.sh
fi

SCENE="${1:-single_obstacle}"
exec python3 navigation_debug_viewer.py --scene "$SCENE"
