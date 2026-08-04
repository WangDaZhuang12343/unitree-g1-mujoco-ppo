from __future__ import annotations

import re
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, send_file


HERE = Path(__file__).resolve().parent
LOG_PATH = HERE / "runs/curriculum.log"
app = Flask(__name__)


def _number(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _training_history(text: str) -> list[dict[str, float]]:
    history: list[dict[str, float]] = []
    for block in re.split(r"\n-{10,}\n", text):
        values = dict(re.findall(r"\|\s+([a-zA-Z0-9_]+)\s+\|\s+([-+0-9.eE]+)\s+\|", block))
        step = _number(values.get("total_timesteps"))
        reward = _number(values.get("ep_rew_mean"))
        length = _number(values.get("ep_len_mean"))
        if step is None or (reward is None and length is None):
            continue
        item = {"step": step}
        if reward is not None:
            item["reward"] = reward
        if length is not None:
            item["length"] = length
        fps = _number(values.get("fps"))
        if fps is not None:
            item["fps"] = fps
        if history and history[-1]["step"] == step:
            history[-1].update(item)
        else:
            history.append(item)
    return history[-120:]


def _evaluations(run_dir: Path) -> list[dict[str, float]]:
    path = run_dir / "eval/evaluations.npz"
    if not path.exists():
        return []
    with np.load(path) as data:
        return [
            {
                "step": int(step),
                "reward": float(rewards.mean()),
                "reward_std": float(rewards.std()),
                "length": float(lengths.mean()),
            }
            for step, rewards, lengths in zip(
                data["timesteps"], data["results"], data["ep_lengths"]
            )
        ]


def _status() -> dict:
    v2_log = HERE / "runs/curriculum_v2.log"
    log_path = (
        v2_log
        if v2_log.exists() and (not LOG_PATH.exists() or v2_log.stat().st_mtime >= LOG_PATH.stat().st_mtime)
        else LOG_PATH
    )
    text = log_path.read_text(errors="replace") if log_path.exists() else ""
    history = _training_history(text)
    stages = [
        ("g1_walk", "行走训练 · 0～0.70m/s", 4),
        ("g1_walk_045", "行走训练 · 0～0.45m/s", 3),
        ("g1_walk_025", "行走训练 · 0～0.25m/s", 2),
        ("g1_walk_010", "行走训练 · 0～0.10m/s", 1),
    ]
    selected = next(
        ((name, label, substage) for name, label, substage in stages if (HERE / "runs" / name / "run_config.json").exists()),
        None,
    )
    walk_started = selected is not None
    run_dir = HERE / "runs" / selected[0] if selected else HERE / "runs/g1_stand"
    evaluations = _evaluations(run_dir)
    latest = history[-1] if history else {}
    step = int(latest.get("step", 0))
    config_path = run_dir / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    target = int(config.get("target_timesteps", 5_000_000))
    start = int(config.get("starting_timesteps", 0))
    log_age = time.time() - log_path.stat().st_mtime if log_path.exists() else None
    fps = latest.get("fps")
    remaining = max(target - step, 0)
    eta_seconds = remaining / fps if fps and fps > 0 else None
    best_eval = max(evaluations, key=lambda item: item["reward"], default=None)
    best_length = max((item["length"] for item in evaluations), default=None)
    best_model = run_dir / "best/best_model.zip"
    return {
        "active": log_age is not None and log_age < 120,
        "log_age_seconds": log_age,
        "updated_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "stage": selected[1] if selected else "站立训练",
        "stage_number": 2 if walk_started else 1,
        "step": step,
        "stage_start": start,
        "target": target,
        "progress": (step - start) / (target - start) if target > start else 0,
        "eta_seconds": eta_seconds,
        "fps": fps,
        "rollout_reward": latest.get("reward"),
        "rollout_length": latest.get("length"),
        "latest_eval": evaluations[-1] if evaluations else None,
        "best_eval": best_eval,
        "best_eval_length": best_length,
        "best_model_exists": best_model.exists(),
        "best_model_updated": (
            datetime.fromtimestamp(best_model.stat().st_mtime).astimezone().isoformat(timespec="seconds")
            if best_model.exists()
            else None
        ),
        "history": history,
        "evaluations": evaluations[-40:],
    }


@app.get("/")
def index():
    return send_file(HERE / "dashboard.html")


@app.get("/api/status")
def api_status():
    return jsonify(_status())


@app.get("/health")
def health():
    return {"ok": True}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8765, debug=False)
