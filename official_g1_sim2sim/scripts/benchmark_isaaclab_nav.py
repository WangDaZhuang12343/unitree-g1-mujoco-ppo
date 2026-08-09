"""Evaluate the frozen 11-scenario contract in Isaac Lab without changing training terrain."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.enable()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--scenes", default="all", help="all or comma-separated scenario names")
parser.add_argument("--seed", type=int, default=7)
parser.add_argument("--planner", choices=("dwa", "onnx"), default="dwa")
parser.add_argument("--policy_onnx", type=Path)
parser.add_argument("--duration", type=float, help="shorten every scenario for smoke testing")
parser.add_argument("--output", type=Path, default=Path("runs/isaaclab_navigation_benchmark"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import numpy as np
import torch

import isaaclab_nav
from isaaclab_nav.benchmark import make_benchmark_terrain_cfg
from isaaclab_nav.env_cfg import G1VisualNavigationEnvCfg
from navigation.scenarios import get_scenario, scenario_names


def _select_scenes(value: str) -> tuple[str, ...]:
    available = scenario_names()
    selected = available if value == "all" else tuple(x.strip() for x in value.split(",") if x.strip())
    unknown = set(selected) - set(available)
    if unknown:
        raise ValueError(f"unknown scenarios: {', '.join(sorted(unknown))}")
    if not selected:
        raise ValueError("at least one scenario is required")
    return selected


class OnnxUpperPolicy:
    def __init__(self, path: Path):
        if not path or not path.is_file():
            raise FileNotFoundError("--planner onnx requires an exported --policy_onnx file")
        import onnxruntime as ort

        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.input_name = self.session.get_inputs()[0].name

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        value = self.session.run(None, {self.input_name: observation.detach().cpu().numpy()})[0]
        return torch.as_tensor(value, device=observation.device, dtype=torch.float32)


def _write_report(output: Path, rows: list[dict[str, object]], planner: str) -> None:
    output.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    completed = [row for row in rows if row["status"] == "completed"]
    static_count = sum(row["status"] != "unsupported_dynamic_perception" for row in rows)
    successes = sum(bool(row["success"]) for row in completed)
    lines = [
        "# Isaac Lab G1 Navigation Benchmark",
        "",
        f"- Planner: `{planner}`",
        f"- Completed static scenarios: {len(completed)}/{static_count}",
        f"- Static success rate: {successes}/{len(completed) if completed else 0}",
        "- `dynamic_obstacle` is not scored: Isaac Lab 2.3 RayCaster only sees the static terrain mesh; "
        "scoring a moving rigid body as sensed would be invalid.",
        "",
        "| Scenario | Status | Success | Survived | Collision | Time (s) | Final distance (m) | Min clearance (m) |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['scenario']} | {row['status']} | {int(bool(row['success']))} | "
            f"{int(bool(row['survived']))} | {row['collision_count']} | "
            f"{float(row['elapsed_sim_s']):.2f} | "
            f"{float(row['final_distance_m']):.3f} | {float(row['min_clearance_m']):.3f} |"
        )
    lines += [
        "",
        "Success uses the frozen contract: within 0.30 m, survived, and zero collisions.",
    ]
    (output / "isaaclab_navigation_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    requested = _select_scenes(args.scenes)
    static_scenes = tuple(name for name in requested if not get_scenario(name, args.seed).dynamic)
    if not static_scenes:
        rows = [{
            "scenario": name, "status": "unsupported_dynamic_perception", "success": False,
            "reached": False, "survived": False, "collision_count": 0,
            "elapsed_sim_s": 0.0, "final_distance_m": float("nan"),
            "min_clearance_m": float("nan"),
        } for name in requested]
        _write_report(args.output, rows, args.planner)
        return

    cfg = G1VisualNavigationEnvCfg()
    cfg.scene.num_envs = len(static_scenes)
    cfg.sim.device = args.device
    cfg.terrain.terrain_generator = make_benchmark_terrain_cfg(static_scenes, args.seed)
    cfg.terrain.max_init_terrain_level = 0
    cfg.benchmark_scenarios = static_scenes
    cfg.benchmark_seed = args.seed
    cfg.enable_dwa_fallback = args.planner == "dwa"
    cfg.episode_length_s = max(
        args.duration or get_scenario(name, args.seed).duration for name in static_scenes
    ) + cfg.decimation * cfg.sim.dt
    cfg.seed = args.seed
    env = gym.make(isaaclab_nav.TASK_ID, cfg=cfg)
    observation, _ = env.reset(seed=args.seed)
    raw = env.unwrapped
    terrain_types = raw._terrain.terrain_types.detach().cpu().tolist()
    assigned = [static_scenes[index] for index in terrain_types]
    if assigned != list(static_scenes):
        raise RuntimeError(f"scenario-to-environment mapping changed: {assigned}")

    policy = OnnxUpperPolicy(args.policy_onnx) if args.planner == "onnx" else None
    active = torch.ones(len(static_scenes), dtype=torch.bool, device=raw.device)
    done_reason = torch.zeros(len(static_scenes), dtype=torch.int8, device=raw.device)
    collision_count = torch.zeros(len(static_scenes), dtype=torch.int32, device=raw.device)
    min_clearance = torch.full((len(static_scenes),), float("inf"), device=raw.device)
    last_distance = torch.full((len(static_scenes),), float("inf"), device=raw.device)
    elapsed_steps = torch.zeros(len(static_scenes), dtype=torch.int32, device=raw.device)
    durations = torch.tensor(
        [args.duration or get_scenario(name, args.seed).duration for name in static_scenes],
        device=raw.device,
    )
    start = time.perf_counter()
    while active.any() and simulation_app.is_running():
        if policy:
            actions = policy(observation["policy"])
        else:
            actions = torch.full((len(static_scenes), 3), torch.nan, device=raw.device)
        actions[~active] = 0.0
        observation, _, terminated, truncated, extras = env.step(actions)
        _, _, clearance, _, _ = raw._perception()
        min_clearance[active] = torch.minimum(min_clearance[active], clearance[active])
        last_distance[active] = extras["goal_distance"][active]
        elapsed_steps[active] += 1
        collision_count = torch.maximum(collision_count, extras["collision_count"])
        time_limit = elapsed_steps.float() * raw.step_dt >= durations
        newly_done = active & (terminated | truncated | time_limit)
        done_reason[newly_done] = extras["termination_reason"][newly_done]
        done_reason[newly_done & time_limit & (done_reason == 0)] = 4
        active[newly_done] = False

    wall_s = time.perf_counter() - start
    rows = []
    for env_id, name in enumerate(static_scenes):
        reason = int(done_reason[env_id])
        collisions = int(collision_count[env_id])
        reached = reason == 1 or float(last_distance[env_id]) <= cfg.success_radius
        survived = reason not in (2, 3, 5)
        rows.append({
            "scenario": name,
            "status": "completed",
            "success": reached and survived and collisions == 0,
            "reached": reached,
            "survived": survived,
            "collision_count": collisions,
            "elapsed_sim_s": float(elapsed_steps[env_id]) * raw.step_dt,
            "final_distance_m": float(last_distance[env_id]),
            "min_clearance_m": float(min_clearance[env_id]),
        })
    for name in requested:
        if get_scenario(name, args.seed).dynamic:
            rows.append({
                "scenario": name, "status": "unsupported_dynamic_perception", "success": False,
                "reached": False, "survived": False, "collision_count": 0,
                "elapsed_sim_s": 0.0, "final_distance_m": float("nan"),
                "min_clearance_m": float("nan"),
            })
    order = {name: index for index, name in enumerate(requested)}
    rows.sort(key=lambda row: order[str(row["scenario"])])
    _write_report(args.output, rows, args.planner)
    print("ISAACLAB_BENCHMARK_OK", {"rows": len(rows), "wall_s": round(wall_s, 2), "output": str(args.output)})
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
