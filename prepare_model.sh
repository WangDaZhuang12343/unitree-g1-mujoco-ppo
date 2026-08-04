#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
UPSTREAM="$ROOT/third_party/unitree_mujoco"
G1_MODEL="$UPSTREAM/unitree_robots/g1"

if [[ ! -d "$UPSTREAM/.git" ]]; then
  mkdir -p "$ROOT/third_party"
  git clone --depth 1 https://github.com/unitreerobotics/unitree_mujoco.git "$UPSTREAM"
fi

install -m 0644 "$ROOT/models/g1_29dof_pd_v3.xml" "$G1_MODEL/g1_29dof_pd_v3.xml"
install -m 0644 "$ROOT/models/scene_walk_pd_v3.xml" "$G1_MODEL/scene_walk_pd_v3.xml"

printf 'G1 模型已准备：%s\n' "$G1_MODEL/scene_walk_pd_v3.xml"
