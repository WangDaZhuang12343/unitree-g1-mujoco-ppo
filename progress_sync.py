from __future__ import annotations

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
START = "<!-- AUTO_STATUS_START -->"
END = "<!-- AUTO_STATUS_END -->"


def last_value(text: str, name: str, pattern: str = r"[-+0-9.eE]+") -> str | None:
    matches = re.findall(rf"\|\s+{re.escape(name)}\s+\|\s+({pattern})\s+\|", text)
    return matches[-1] if matches else None


def evaluation_summary(run_dir: Path) -> tuple[int, float, float] | None:
    path = run_dir / "eval/evaluations.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return (
            int(data["timesteps"][-1]),
            float(data["results"][-1].mean()),
            float(data["ep_lengths"][-1].mean()),
        )


def build_status() -> str:
    v2_log = HERE / "runs/curriculum_v2.log"
    old_log = HERE / "runs/curriculum.log"
    log_path = (
        v2_log
        if v2_log.exists() and (not old_log.exists() or v2_log.stat().st_mtime >= old_log.stat().st_mtime)
        else old_log
    )
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    stage_candidates = [
        ("g1_walk", "行走 0～0.70m/s（阶段 2.4/2）"),
        ("g1_walk_045", "行走 0～0.45m/s（阶段 2.3/2）"),
        ("g1_walk_025", "行走 0～0.25m/s（阶段 2.2/2）"),
        ("g1_walk_010", "行走 0～0.10m/s（阶段 2.1/2）"),
    ]
    selected = next(
        ((name, label) for name, label in stage_candidates if (HERE / "runs" / name / "run_config.json").exists()),
        None,
    )
    if selected:
        run_dir = HERE / "runs" / selected[0]
        stage = selected[1]
        config = json.loads((run_dir / "run_config.json").read_text())
        target = int(config.get("target_timesteps", config.get("steps", 0)))
    else:
        run_dir = HERE / "runs/g1_stand"
        stand_finished = (run_dir / "final_model.zip").exists()
        stage = "站立已完成，V2 待启动" if stand_finished else "站立（阶段 1/2）"
        target = 5_000_000
    steps = last_value(text, "total_timesteps", r"\d+") or "尚无记录"
    fps = last_value(text, "fps", r"\d+") or "尚无记录"
    reward = last_value(text, "ep_rew_mean") or "尚无记录"
    length = last_value(text, "ep_len_mean") or "尚无记录"
    evaluation = evaluation_summary(run_dir)
    timestamp = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

    lines = [
        START,
        "## 实时状态",
        "",
        f"- 自动同步时间：{timestamp}",
        f"- 当前阶段：{stage}",
        f"- 最近训练步数：{steps} / {target}",
        f"- 最近吞吐：{fps} steps/s",
        f"- Rollout 平均奖励：{reward}",
        f"- Rollout 平均回合长度：{length} 个控制步",
    ]
    if evaluation:
        eval_steps, eval_reward, eval_length = evaluation
        lines.extend(
            [
                f"- 最近独立评估：{eval_steps} 步",
                f"- 评估平均奖励：{eval_reward:.2f}",
                f"- 评估平均长度：{eval_length:.2f} 个控制步（{eval_length / 50:.2f} 秒）",
            ]
        )
    else:
        lines.append("- 最近独立评估：尚未生成")
    lines.append(END)
    return "\n".join(lines)


def update() -> None:
    path = HERE / "progress.md"
    document = path.read_text()
    replacement = build_status()
    updated, count = re.subn(
        rf"{re.escape(START)}.*?{re.escape(END)}", replacement, document, flags=re.DOTALL
    )
    if count != 1:
        raise RuntimeError("progress.md must contain exactly one automatic status block")
    temporary = path.with_suffix(".md.tmp")
    temporary.write_text(updated)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize live G1 training metrics")
    parser.add_argument("--watch", action="store_true")
    parser.add_argument("--interval", type=float, default=600.0)
    args = parser.parse_args()
    while True:
        update()
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
