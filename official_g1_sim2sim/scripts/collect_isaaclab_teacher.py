"""Collect exact 483-D Isaac observations labeled by the frozen DWA teacher."""

from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.enable()

from isaaclab.app import AppLauncher
from isaaclab_nav.pretraining import physical_command_to_normalized_action
from isaaclab_nav.teacher import ParallelDwaTeacher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=20_000)
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--dwa_workers", type=int, default=8)
    parser.add_argument("--terrain", choices=("random", "benchmark"), default="random")
    parser.add_argument("--rollout_policy_onnx", type=Path)
    parser.add_argument("--teacher_rollout_probability", type=float, default=1.0)
    parser.add_argument("--output", type=Path, default=Path("datasets/isaaclab_dwa_teacher_v1.npz"))
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


def run_collection(args: argparse.Namespace, simulation_app) -> None:
    # Isaac imports must follow AppLauncher creation. Keeping them inside this
    # function also prevents spawned CPU label workers from launching Kit.
    import gymnasium as gym

    import isaaclab_nav
    from isaaclab.utils.math import quat_apply_inverse, subtract_frame_transforms
    from isaaclab_nav.benchmark import make_benchmark_terrain_cfg
    from isaaclab_nav.env_cfg import G1VisualNavigationEnvCfg
    from navigation.scenarios import get_scenario, scenario_names

    if args.samples <= 0 or args.num_envs <= 0 or args.dwa_workers <= 0:
        raise ValueError("samples, num_envs, and dwa_workers must be positive")
    if not 0.0 <= args.teacher_rollout_probability <= 1.0:
        raise ValueError("teacher_rollout_probability must be in [0, 1]")
    if args.rollout_policy_onnx is None and args.teacher_rollout_probability != 1.0:
        raise ValueError("a mixed rollout requires --rollout_policy_onnx")
    cfg = G1VisualNavigationEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.seed = args.seed
    cfg.enable_dwa_fallback = False
    if args.terrain == "benchmark":
        static_scenes = tuple(name for name in scenario_names() if not get_scenario(name).dynamic)
        if args.num_envs % len(static_scenes):
            raise ValueError(f"benchmark num_envs must be a multiple of {len(static_scenes)}")
        cfg.terrain.terrain_generator = make_benchmark_terrain_cfg(static_scenes, args.seed)
        cfg.terrain.max_init_terrain_level = 0
        cfg.benchmark_scenarios = static_scenes
        cfg.benchmark_seed = args.seed

    env = gym.make(isaaclab_nav.TASK_ID, cfg=cfg)
    observation, _ = env.reset(seed=args.seed)
    raw = env.unwrapped
    teacher = ParallelDwaTeacher(args.num_envs, args.dwa_workers)
    rollout_session = None
    rollout_input_name = None
    if args.rollout_policy_onnx is not None:
        if not args.rollout_policy_onnx.is_file():
            raise FileNotFoundError(args.rollout_policy_onnx)
        import onnxruntime as ort

        rollout_session = ort.InferenceSession(
            str(args.rollout_policy_onnx), providers=["CPUExecutionProvider"]
        )
        rollout_input_name = rollout_session.get_inputs()[0].name
    observations: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    trajectory_ids: list[np.ndarray] = []
    current_trajectory = torch.arange(args.num_envs, dtype=torch.int64, device=raw.device)
    next_trajectory = args.num_envs
    collected = 0
    while collected < args.samples and simulation_app.is_running():
        hits = raw._forward_scanner.data.ray_hits_w
        relative = hits - raw._robot.data.root_pos_w[:, None, :]
        quat = raw._robot.data.root_quat_w[:, None, :].expand(-1, relative.shape[1], -1)
        points_body = quat_apply_inverse(quat, relative)
        goal_body, _ = subtract_frame_transforms(
            raw._robot.data.root_pos_w, raw._robot.data.root_quat_w, raw._goal_pos_w
        )
        ground_z_body = raw._terrain.env_origins[:, 2] - raw._robot.data.root_pos_w[:, 2]
        command = teacher.plan(points_body, ground_z_body, goal_body)
        action = physical_command_to_normalized_action(command)
        rollout_action = action
        if rollout_session is not None:
            learner = rollout_session.run(
                None,
                {rollout_input_name: observation["policy"].detach().cpu().numpy()},
            )[0]
            if learner.shape != (args.num_envs, 3) or not np.isfinite(learner).all():
                raise ValueError(f"rollout policy returned invalid action shape or values: {learner.shape}")
            learner_action = torch.as_tensor(
                learner, dtype=torch.float32, device=raw.device
            ).clamp(-1.0, 1.0)
            teacher_mask = torch.rand(args.num_envs, device=raw.device) < args.teacher_rollout_probability
            rollout_action = torch.where(teacher_mask[:, None], action, learner_action)
        take = min(args.num_envs, args.samples - collected)
        observations.append(observation["policy"][:take].detach().cpu().numpy())
        actions.append(action[:take].detach().cpu().numpy())
        trajectory_ids.append(current_trajectory[:take].detach().cpu().numpy())
        collected += take
        observation, _, terminated, truncated, _ = env.step(rollout_action)
        reset_ids = torch.nonzero(terminated | truncated, as_tuple=False).flatten()
        for env_id in reset_ids.detach().cpu().tolist():
            current_trajectory[env_id] = next_trajectory
            next_trajectory += 1
        teacher.reset(reset_ids)
        if collected % max(args.num_envs, 1000) < args.num_envs:
            print(f"COLLECTED {collected}/{args.samples}")

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    observation_array = np.concatenate(observations).astype(np.float32, copy=False)
    action_array = np.concatenate(actions).astype(np.float32, copy=False)
    trajectory_array = np.concatenate(trajectory_ids).astype(np.int64, copy=False)
    np.savez_compressed(
        output,
        observation=observation_array,
        action=action_array,
        trajectory_id=trajectory_array,
        seed=np.asarray(args.seed, dtype=np.int64),
        terrain=np.asarray(args.terrain),
        rollout_policy=np.asarray(
            str(args.rollout_policy_onnx.resolve()) if args.rollout_policy_onnx else "dwa_teacher"
        ),
        teacher_rollout_probability=np.asarray(args.teacher_rollout_probability, dtype=np.float32),
        walking_actuator_profile=np.asarray(cfg.walking_actuator_profile),
        format=np.asarray("g1_isaaclab_dwa_teacher_v2"),
    )
    print(
        "ISAACLAB_TEACHER_DATA_OK",
        {"samples": len(observation_array), "trajectories": int(np.unique(trajectory_array).size),
         "observation": observation_array.shape, "action": action_array.shape,
         "rollout_policy": "learner_mixed" if rollout_session is not None else "dwa_teacher",
         "walking_actuator_profile": cfg.walking_actuator_profile,
         "teacher_rollout_probability": args.teacher_rollout_probability,
         "output": str(output)},
    )
    teacher.close()
    env.close()


def main() -> None:
    args = parse_args()
    launcher = AppLauncher(args)
    simulation_app = launcher.app
    try:
        run_collection(args, simulation_app)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()


if __name__ == "__main__":
    main()
