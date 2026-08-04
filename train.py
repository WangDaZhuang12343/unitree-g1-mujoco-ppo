from __future__ import annotations

import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-g1")

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import SubprocVecEnv

from env import G1WalkEnv


HERE = Path(__file__).resolve().parent


class StopOnConsecutiveSuccess(BaseCallback):
    """Stop after the evaluation success-rate target is met repeatedly."""

    def __init__(self, consecutive: int = 3, threshold: float = 0.8) -> None:
        super().__init__(verbose=1)
        self.consecutive = consecutive
        self.threshold = threshold
        self.hits = 0

    def _on_step(self) -> bool:
        results = getattr(self.parent, "_is_success_buffer", [])
        rate = float(np.mean(results)) if results else 0.0
        self.hits = self.hits + 1 if rate >= self.threshold else 0
        if self.verbose:
            print(
                f"课程成功率={rate:.0%}，连续达标={self.hits}/{self.consecutive}"
            )
        return self.hits < self.consecutive


def configure_resumed_model(model: PPO, args: argparse.Namespace) -> None:
    model.learning_rate = args.learning_rate
    model.lr_schedule = FloatSchedule(args.learning_rate)
    model.ent_coef = args.ent_coef
    for group in model.policy.optimizer.param_groups:
        group["lr"] = args.learning_rate
    if args.reset_action_std is not None:
        with torch.no_grad():
            model.policy.log_std.fill_(float(np.log(args.reset_action_std)))
    if args.initialize_command_inputs:
        with torch.no_grad():
            networks = (
                model.policy.mlp_extractor.policy_net,
                model.policy.mlp_extractor.value_net,
            )
            for network in networks:
                first_linear = next(layer for layer in network if isinstance(layer, torch.nn.Linear))
                first_linear.weight[:, -3:].zero_()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a CPU PPO walking policy for Unitree G1")
    parser.add_argument("--steps", type=int, default=10_000_000)
    parser.add_argument("--envs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run-dir", type=Path, default=HERE / "runs/g1_walk")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-freq", type=int, default=250_000)
    parser.add_argument("--min-speed", type=float, default=0.0)
    parser.add_argument("--max-speed", type=float, default=0.7)
    parser.add_argument("--eval-speed", type=float)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--ent-coef", type=float, default=0.005)
    parser.add_argument("--initial-action-std", type=float, default=1.0)
    parser.add_argument("--reset-action-std", type=float)
    parser.add_argument("--action-scale-factor", type=float, default=1.0)
    parser.add_argument("--gait-clock", action="store_true")
    parser.add_argument("--gait-reference", action="store_true")
    parser.add_argument("--initialize-command-inputs", action="store_true")
    parser.add_argument("--early-stop-successes", type=int, default=0)
    parser.add_argument("--eval-freq", type=int, default=100_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    vec_cls = SubprocVecEnv if args.envs > 1 else None
    vec_kwargs = {"start_method": "fork"} if args.envs > 1 else None
    env_kwargs = {
        "command_speed": (args.min_speed, args.max_speed),
        "gait_clock": args.gait_clock,
        "gait_reference": args.gait_reference,
        "action_scale_factor": args.action_scale_factor,
    }
    train_env = make_vec_env(
        G1WalkEnv,
        n_envs=args.envs,
        seed=args.seed,
        vec_env_cls=vec_cls,
        vec_env_kwargs=vec_kwargs,
        env_kwargs=env_kwargs,
    )
    eval_kwargs = {**env_kwargs, "fixed_command_speed": args.eval_speed}
    eval_env = make_vec_env(G1WalkEnv, n_envs=1, seed=args.seed + 10_000, env_kwargs=eval_kwargs)

    if args.resume:
        model = PPO.load(args.resume, env=train_env, device="cpu")
        configure_resumed_model(model, args)
    else:
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=args.learning_rate,
            n_steps=1024,
            batch_size=512,
            n_epochs=5,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=args.ent_coef,
            policy_kwargs={
                "net_arch": {"pi": [256, 256], "vf": [256, 256]},
                "log_std_init": float(np.log(args.initial_action_std)),
            },
            tensorboard_log=(
                str(args.run_dir / "tensorboard")
                if importlib.util.find_spec("tensorboard") is not None
                else None
            ),
            seed=args.seed,
            device="cpu",
            verbose=1,
        )
    config["starting_timesteps"] = model.num_timesteps
    config["target_timesteps"] = model.num_timesteps + args.steps
    (args.run_dir / "run_config.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2) + "\n"
    )

    checkpoint = CheckpointCallback(
        save_freq=max(args.checkpoint_freq // args.envs, 1),
        save_path=str(args.run_dir / "checkpoints"),
        name_prefix="g1_ppo",
    )
    stop_callback = (
        StopOnConsecutiveSuccess(consecutive=args.early_stop_successes)
        if args.early_stop_successes > 0
        else None
    )
    evaluation = EvalCallback(
        eval_env,
        best_model_save_path=str(args.run_dir / "best"),
        log_path=str(args.run_dir / "eval"),
        eval_freq=max(args.eval_freq // args.envs, 1),
        n_eval_episodes=5,
        deterministic=True,
        callback_after_eval=stop_callback,
    )
    try:
        model.learn(args.steps, callback=[checkpoint, evaluation], progress_bar=False, reset_num_timesteps=not bool(args.resume))
        model.save(args.run_dir / "final_model")
    finally:
        train_env.close()
        eval_env.close()


if __name__ == "__main__":
    main()
