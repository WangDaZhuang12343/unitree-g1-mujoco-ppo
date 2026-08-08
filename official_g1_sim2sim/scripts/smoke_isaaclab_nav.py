"""Run a short GPU/PhysX contract check for the G1 navigation environment."""

from __future__ import annotations

import argparse
import faulthandler
import sys
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
faulthandler.enable()
faulthandler.dump_traceback_later(45, repeat=False)

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=32)
parser.add_argument("--steps", type=int, default=20)
parser.add_argument("--enable_dwa_fallback", action="store_true")
parser.add_argument("--video", action="store_true")
parser.add_argument("--video_folder", type=str, default="/tmp/g1_isaaclab_video")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args)
simulation_app = launcher.app

import gymnasium as gym
import torch

import isaaclab_nav
from isaaclab_nav.env_cfg import G1VisualNavigationEnvCfg


def main() -> None:
    cfg = G1VisualNavigationEnvCfg()
    cfg.scene.num_envs = args.num_envs
    cfg.sim.device = args.device
    cfg.enable_dwa_fallback = args.enable_dwa_fallback
    cfg.viewer.origin_type = "asset_root"
    cfg.viewer.env_index = 0
    cfg.viewer.asset_name = "robot"
    cfg.viewer.eye = (3.0, 3.0, 1.8)
    cfg.viewer.lookat = (0.5, 0.0, 0.7)
    cfg.viewer.resolution = (960, 540)
    env = gym.make(
        isaaclab_nav.TASK_ID,
        cfg=cfg,
        render_mode="rgb_array" if args.video else None,
    )
    if args.video:
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_folder,
            step_trigger=lambda step: step == 0,
            video_length=args.steps,
            disable_logger=True,
        )
    observation, _ = env.reset(seed=17)
    assert observation["policy"].shape == (args.num_envs, 483)
    assert torch.isfinite(observation["policy"]).all()
    termination_count = 0
    fallback_count = 0
    for index in range(args.steps):
        action = torch.zeros((args.num_envs, 3), device=env.unwrapped.device)
        action[:, 0] = 0.3
        if args.enable_dwa_fallback and index == 0:
            action[0] = torch.nan
        observation, reward, terminated, truncated, extras = env.step(action)
        assert observation["policy"].shape == (args.num_envs, 483)
        assert reward.shape == terminated.shape == truncated.shape == (args.num_envs,)
        assert torch.isfinite(observation["policy"]).all() and torch.isfinite(reward).all()
        assert extras["collision_count"].shape == (args.num_envs,)
        assert extras["termination_reason"].shape == (args.num_envs,)
        termination_count += int(terminated.sum())
        fallback_count += int(extras["fallback_used"].sum())
    if args.enable_dwa_fallback:
        assert fallback_count >= 1
    print(
        "ISAACLAB_NAV_SMOKE_OK",
        {
            "num_envs": args.num_envs,
            "steps": args.steps,
            "terminations": termination_count,
            "fallback_uses": fallback_count,
            "device": str(env.unwrapped.device),
            "observation": tuple(observation["policy"].shape),
            "walking_action": tuple(env.unwrapped._walking_action.shape),
        },
    )
    env.close()


if __name__ == "__main__":
    try:
        main()
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
