#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${UNITREE_RL_LAB_ROOT:-}" ]]; then
  RL_LAB="$UNITREE_RL_LAB_ROOT"
elif [[ -d "$ROOT/../unitree_rl_lab" ]]; then
  RL_LAB="$ROOT/../unitree_rl_lab"
elif [[ -d "$ROOT/../../unitree_rl_lab" ]]; then
  RL_LAB="$ROOT/../../unitree_rl_lab"
else
  echo "找不到 unitree_rl_lab，请设置 UNITREE_RL_LAB_ROOT" >&2
  exit 1
fi

ORT="$RL_LAB/deploy/thirdparty/onnxruntime-linux-x64-1.22.0"

mkdir -p "$ROOT/build"
g++ -std=c++17 -O2 -fPIC -shared "$ROOT/ort_bridge.cpp" \
  -I"$ORT/include" \
  "$ORT/lib/libonnxruntime.so.1.22.0" \
  -Wl,-rpath,"$ORT/lib" \
  -o "$ROOT/build/libort_bridge.so"

echo "已生成 $ROOT/build/libort_bridge.so"
