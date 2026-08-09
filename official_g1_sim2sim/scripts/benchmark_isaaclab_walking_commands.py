"""Measure frozen Walking ONNX stability over upper-layer velocity commands on flat terrain."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.enable()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--output", type=Path, default=Path("runs/isaaclab_walking_commands"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_nav
import isaaclab.terrains as terrain_gen
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_nav.env_cfg import G1VisualNavigationEnvCfg
from isaaclab_nav.pretraining import physical_command_to_normalized_action


COMMANDS = (
    ("stand", 0.0, 0.0, 0.0),
    ("forward_025", 0.25, 0.0, 0.0),
    ("forward_045", 0.45, 0.0, 0.0),
    ("left", 0.25, 0.10, 0.0),
    ("right", 0.25, -0.10, 0.0),
    ("turn_left", 0.25, 0.0, 0.20),
    ("turn_right", 0.25, 0.0, -0.20),
    ("left_turn", 0.25, 0.10, 0.20),
    ("right_turn", 0.25, -0.10, -0.20),
)


def _flat_terrain() -> TerrainGeneratorCfg:
    return TerrainGeneratorCfg(
        seed=23,
        size=(20.0, 20.0),
        border_width=2.0,
        num_rows=1,
        num_cols=1,
        sub_terrains={
            "plane": terrain_gen.MeshPlaneTerrainCfg(
                proportion=1.0,
                flat_patch_sampling={
                    "goal": terrain_gen.FlatPatchSamplingCfg(
                        num_patches=16,
                        patch_radius=0.45,
                        x_range=(5.0, 8.0),
                        y_range=(-2.0, 2.0),
                        z_range=(-0.05, 0.05),
                        max_height_diff=0.01,
                    )
                },
            )
        },
    )


def main() -> None:
    if args.duration <= 0.0:
        raise ValueError("duration must be positive")
    cfg = G1VisualNavigationEnvCfg()
    cfg.scene.num_envs = len(COMMANDS)
    cfg.sim.device = args.device
    cfg.terrain.terrain_generator = _flat_terrain()
    cfg.terrain.max_init_terrain_level = 0
    cfg.episode_length_s = args.duration + cfg.decimation * cfg.sim.dt
    cfg.enable_dwa_fallback = False
    env = gym.make(isaaclab_nav.TASK_ID, cfg=cfg)
    env.reset(seed=23)
    raw = env.unwrapped
    # Keep the navigation success condition out of this locomotion-only test.
    raw._goal_pos_w[:] = raw._terrain.env_origins + raw._goal_pos_w.new_tensor((50.0, 50.0, 0.05))
    _, distance = raw._goal_body()
    raw._previous_goal_distance[:] = distance

    physical = torch.tensor([values[1:] for values in COMMANDS], device=raw.device)
    command_action = physical_command_to_normalized_action(physical)
    stand_action = physical_command_to_normalized_action(torch.zeros_like(physical))
    active = torch.ones(len(COMMANDS), dtype=torch.bool, device=raw.device)
    reason = torch.zeros(len(COMMANDS), dtype=torch.int8, device=raw.device)
    elapsed = torch.zeros(len(COMMANDS), dtype=torch.int32, device=raw.device)
    max_tilt = torch.zeros(len(COMMANDS), device=raw.device)
    final_displacement = torch.zeros(len(COMMANDS), device=raw.device)
    start_pos = raw._robot.data.root_pos_w.clone()
    body_id = torch.full((len(COMMANDS),), -1, dtype=torch.int32, device=raw.device)
    max_steps = int(round(args.duration / raw.step_dt))
    for _ in range(max_steps):
        if not active.any() or not simulation_app.is_running():
            break
        position_before = raw._robot.data.root_pos_w.clone()
        roll, pitch, _ = euler_xyz_from_quat(raw._robot.data.root_quat_w)
        max_tilt[active] = torch.maximum(
            max_tilt[active], torch.maximum(roll.abs(), pitch.abs())[active]
        )
        action = torch.where(active[:, None], command_action, stand_action)
        _, _, terminated, truncated, extras = env.step(action)
        elapsed[active] += 1
        newly_done = active & (terminated | truncated)
        reason[newly_done] = extras["termination_reason"][newly_done]
        body_id[newly_done] = extras["collision_body_id"][newly_done]
        final_displacement[newly_done] = torch.linalg.norm(
            position_before[newly_done, :2] - start_pos[newly_done, :2], dim=1
        )
        active[newly_done] = False
    final_displacement[active] = torch.linalg.norm(
        raw._robot.data.root_pos_w[active, :2] - start_pos[active, :2], dim=1
    )
    reason[active] = 4

    reason_names = {0: "none", 1: "success", 2: "collision", 3: "fall", 4: "timeout", 5: "invalid"}
    body_names = raw._contact_sensor.body_names
    rows = []
    for index, (name, vx, vy, omega) in enumerate(COMMANDS):
        collision_index = int(body_id[index])
        rows.append({
            "command": name,
            "vx": vx,
            "vy": vy,
            "omega": omega,
            "survived": int(reason[index]) == 4,
            "termination_reason": reason_names.get(int(reason[index]), f"unknown_{int(reason[index])}"),
            "elapsed_sim_s": float(elapsed[index]) * raw.step_dt,
            "planar_displacement_m": float(final_displacement[index]),
            "max_tilt_rad": float(max_tilt[index]),
            "collision_body": (
                body_names[collision_index]
                if int(reason[index]) == 2 and 0 <= collision_index < len(body_names)
                else "none"
            ),
        })

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Isaac Lab Frozen Walking Command Matrix",
        "",
        "| Command | vx | vy | omega | Survived | End reason | Time (s) | Displacement (m) | Max tilt (rad) |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['command']} | {row['vx']:.2f} | {row['vy']:.2f} | {row['omega']:.2f} | "
            f"{int(row['survived'])} | {row['termination_reason']} | {row['elapsed_sim_s']:.2f} | "
            f"{row['planar_displacement_m']:.3f} | {row['max_tilt_rad']:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("ISAACLAB_WALKING_COMMANDS_OK", {"rows": len(rows), "output": str(output)})
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
