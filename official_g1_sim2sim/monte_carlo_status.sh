#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKPOINT="$ROOT/runs/navigation_monte_carlo/checkpoint.csv"
REPORT="$ROOT/runs/navigation_monte_carlo/navigation_report.md"

if [[ -f "$CHECKPOINT" ]]; then
  COMPLETED="$(awk 'END { print NR > 0 ? NR - 1 : 0 }' "$CHECKPOINT")"
else
  COMPLETED=0
fi

echo "Monte Carlo 进度：${COMPLETED}/100"
if systemctl --user --quiet is-active g1-navigation-monte-carlo.service; then
  echo "运行状态：运行中（低优先级）"
else
  echo "运行状态：未运行"
fi

if [[ -f "$REPORT" ]]; then
  echo
  sed -n '1,12p' "$REPORT"
fi
