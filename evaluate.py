from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-g1")

from stable_baselines3 import PPO

from env import G1WalkEnv


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained G1 PPO policy")
    parser.add_argument("model", type=Path)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--speed", type=float)
    args = parser.parse_args()
    config_path = args.model.resolve().parent.parent / "run_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    env = G1WalkEnv(
        render_mode="human" if args.viewer else None,
        command_speed=(config.get("min_speed", 0.0), config.get("max_speed", 0.7)),
        fixed_command_speed=args.speed,
        gait_clock=config.get("gait_clock", False),
        gait_reference=config.get("gait_reference", False),
        action_scale_factor=config.get("action_scale_factor", 1.0),
    )
    model = PPO.load(args.model, device="cpu")
    try:
        for episode in range(args.episodes):
            obs, info = env.reset(seed=10_000 + episode)
            total_reward = 0.0
            terminated = truncated = False
            while not (terminated or truncated):
                started = time.perf_counter()
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                if args.viewer:
                    time.sleep(max(0.0, env.control_dt - (time.perf_counter() - started)))
            print(
                f"episode={episode + 1} reward={total_reward:.1f} "
                f"vx={info['x_velocity']:.3f} command={info['command_x']:.3f} "
                f"mean_error={info['mean_velocity_error']:.3f} "
                f"height={info['height']:.3f} upright={info['upright']:.3f} "
                f"success={info.get('is_success', False)}"
            )
    finally:
        env.close()


if __name__ == "__main__":
    main()
