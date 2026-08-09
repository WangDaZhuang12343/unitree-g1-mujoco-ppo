"""Run the frozen G1 Walking ONNX inside Unitree's official ManagerBasedRLEnv."""

from __future__ import annotations

import argparse
import csv
import faulthandler
import importlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.enable()

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--duration", type=float, default=12.0)
parser.add_argument("--warmup", type=float, default=2.0)
parser.add_argument("--seed", type=int, default=23)
parser.add_argument(
    "--walking_policy",
    type=Path,
    default=Path(
        "/home/qc/qc/project/unitree_rl_lab/deploy/robots/g1_29dof/config/policy/velocity/v0/"
        "exported/policy.onnx"
    ),
)
parser.add_argument(
    "--official_usd",
    type=Path,
    default=Path(
        "/home/qc/qc/project/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/"
        "g1_29dof_rev_1_0.usd"
    ),
)
parser.add_argument(
    "--clean_inference",
    action="store_true",
    help="Disable observation corruption, random events, and curricula for an adapter-only comparison.",
)
parser.add_argument("--output", type=Path, default=Path("runs/official_manager_walking_commands"))
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import unitree_rl_lab.tasks  # noqa: F401 - registers the official task
from g1_nav.walking_commands import COMMANDS
from isaaclab_nav.walking import FrozenWalkingPolicy

RobotPlayEnvCfg = importlib.import_module(
    "unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg"
).RobotPlayEnvCfg


def _termination_reason(env, index: int, truncated: torch.Tensor) -> str:
    if bool(truncated[index]):
        return "timeout"
    for name in env.termination_manager.active_terms:
        if bool(env.termination_manager.get_term(name)[index]):
            return name
    return "terminated"


def main() -> None:
    if args.duration <= 0.0 or args.warmup < 0.0 or args.warmup >= args.duration:
        raise ValueError("duration must be positive and warmup must be in [0, duration)")
    walking_policy = args.walking_policy.expanduser().resolve()
    official_usd = args.official_usd.expanduser().resolve()
    if not walking_policy.is_file():
        raise FileNotFoundError(walking_policy)
    if not official_usd.is_file():
        raise FileNotFoundError(official_usd)

    cfg = RobotPlayEnvCfg()
    cfg.scene.num_envs = len(COMMANDS)
    cfg.sim.device = args.device
    cfg.scene.robot.spawn.usd_path = str(official_usd)
    cfg.scene.terrain.terrain_generator.num_rows = 1
    cfg.scene.terrain.terrain_generator.num_cols = 1
    cfg.scene.terrain.terrain_generator.curriculum = False
    cfg.scene.terrain.max_init_terrain_level = 0
    cfg.episode_length_s = args.duration + cfg.decimation * cfg.sim.dt
    cfg.commands.base_velocity.debug_vis = False
    if args.clean_inference:
        cfg.observations.policy.enable_corruption = False
        cfg.observations.critic.enable_corruption = False
        cfg.events.physics_material = None
        cfg.events.add_base_mass = None
        cfg.events.base_external_force_torque = None
        cfg.events.push_robot = None
        cfg.events.reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        cfg.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        cfg.curriculum.terrain_levels = None
        cfg.curriculum.lin_vel_cmd_levels = None

    env = gym.make("Unitree-G1-29dof-Velocity", cfg=cfg)
    raw = env.unwrapped
    observation, _ = env.reset(seed=args.seed)
    physical = torch.tensor([values[1:] for values in COMMANDS], device=raw.device)
    command_term = raw.command_manager.get_term("base_velocity")
    command_term.vel_command_b[:] = physical
    # Remove the random command inserted during reset from the five-frame history.
    raw.observation_manager.reset()
    observation = raw.observation_manager.compute(update_history=True)
    if observation["policy"].shape != (len(COMMANDS), 480):
        raise RuntimeError(f"unexpected official policy observation shape: {observation['policy'].shape}")

    policy = FrozenWalkingPolicy(walking_policy)
    active = torch.ones(len(COMMANDS), dtype=torch.bool, device=raw.device)
    elapsed = torch.zeros(len(COMMANDS), dtype=torch.int32, device=raw.device)
    reason = ["none"] * len(COMMANDS)
    start_pos = raw.scene["robot"].data.root_pos_w.clone()
    final_displacement = torch.zeros(len(COMMANDS), device=raw.device)
    velocity_sum = torch.zeros_like(physical)
    squared_error_sum = torch.zeros_like(physical)
    samples = torch.zeros(len(COMMANDS), dtype=torch.int32, device=raw.device)
    max_steps = int(round(args.duration / raw.step_dt))
    warmup_steps = int(round(args.warmup / raw.step_dt))

    for _ in range(max_steps):
        if not active.any() or not simulation_app.is_running():
            break
        command_term.vel_command_b[:] = torch.where(active[:, None], physical, 0.0)
        actions = policy(observation["policy"])
        actions[~active] = 0.0
        position_before = raw.scene["robot"].data.root_pos_w.clone()
        observation, _, terminated, truncated, _ = env.step(actions)
        elapsed[active] += 1
        newly_done = active & (terminated | truncated)
        measured = torch.stack(
            (
                raw.scene["robot"].data.root_lin_vel_b[:, 0],
                raw.scene["robot"].data.root_lin_vel_b[:, 1],
                raw.scene["robot"].data.root_ang_vel_b[:, 2],
            ),
            dim=1,
        )
        sample = active & ~newly_done & (elapsed > warmup_steps)
        velocity_sum[sample] += measured[sample]
        squared_error_sum[sample] += torch.square(measured[sample] - physical[sample])
        samples[sample] += 1
        for index in newly_done.nonzero(as_tuple=False).flatten().tolist():
            reason[index] = _termination_reason(raw, index, truncated)
        final_displacement[newly_done] = torch.linalg.norm(
            position_before[newly_done, :2] - start_pos[newly_done, :2], dim=1
        )
        active[newly_done] = False

    final_displacement[active] = torch.linalg.norm(
        raw.scene["robot"].data.root_pos_w[active, :2] - start_pos[active, :2], dim=1
    )
    for index in active.nonzero(as_tuple=False).flatten().tolist():
        reason[index] = "duration_complete"

    denominator = samples.clamp(min=1).float()[:, None]
    mean_velocity = velocity_sum / denominator
    rmse = torch.sqrt(squared_error_sum / denominator)
    rows = []
    for index, (name, vx, vy, omega) in enumerate(COMMANDS):
        rows.append(
            {
                "command": name,
                "vx": vx,
                "vy": vy,
                "omega": omega,
                "survived": reason[index] == "duration_complete",
                "termination_reason": reason[index],
                "elapsed_sim_s": float(elapsed[index]) * raw.step_dt,
                "planar_displacement_m": float(final_displacement[index]),
                "mean_vx": float(mean_velocity[index, 0]),
                "mean_vy": float(mean_velocity[index, 1]),
                "mean_omega": float(mean_velocity[index, 2]),
                "rmse_vx": float(rmse[index, 0]),
                "rmse_vy": float(rmse[index, 1]),
                "rmse_omega": float(rmse[index, 2]),
                "tracking_samples": int(samples[index]),
            }
        )

    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Official ManagerBased Frozen Walking Command Matrix",
        "",
        f"- Clean inference: `{args.clean_inference}`",
        f"- Walking policy: `{walking_policy}`",
        f"- Official USD: `{official_usd}`",
        "",
        "| Command | vx | vy | omega | Survived | End reason | Time (s) | Mean vx | Mean vy | Mean omega | RMSE norm |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rmse_norm = (row["rmse_vx"] ** 2 + row["rmse_vy"] ** 2 + row["rmse_omega"] ** 2) ** 0.5
        lines.append(
            f"| {row['command']} | {row['vx']:.2f} | {row['vy']:.2f} | {row['omega']:.2f} | "
            f"{int(row['survived'])} | {row['termination_reason']} | {row['elapsed_sim_s']:.2f} | "
            f"{row['mean_vx']:.3f} | {row['mean_vy']:.3f} | {row['mean_omega']:.3f} | {rmse_norm:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(
        "OFFICIAL_MANAGER_WALKING_OK",
        {"rows": len(rows), "survived": sum(row["survived"] for row in rows), "output": str(output)},
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
