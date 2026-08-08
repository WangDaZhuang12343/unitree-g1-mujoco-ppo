"""Register this repository's task and delegate to Isaac Lab's RSL-RL trainer."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_wrapper_args() -> tuple[Path | None, dict[str, float]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--init_checkpoint", type=Path)
    parser.add_argument("--finetune_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--finetune_entropy_coef", type=float, default=0.003)
    parser.add_argument("--finetune_clip_param", type=float, default=0.15)
    parser.add_argument("--finetune_desired_kl", type=float, default=0.005)
    parsed, remaining = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]
    settings = {
        "learning_rate": parsed.finetune_learning_rate,
        "entropy_coef": parsed.finetune_entropy_coef,
        "clip_param": parsed.finetune_clip_param,
        "desired_kl": parsed.finetune_desired_kl,
    }
    if any(value <= 0.0 for value in settings.values()):
        raise ValueError("finetune PPO settings must be positive")
    return parsed.init_checkpoint, settings


init_checkpoint, finetune_settings = _parse_wrapper_args()

if init_checkpoint is not None:
    from rsl_rl.runners import OnPolicyRunner

    from isaaclab_nav.pretraining import load_actor_initialization

    original_runner_init = OnPolicyRunner.__init__

    def _runner_init_with_actor(self, *args, **kwargs):
        original_runner_init(self, *args, **kwargs)
        metadata = load_actor_initialization(self.alg.policy, init_checkpoint)
        for name, value in finetune_settings.items():
            setattr(self.alg, name, value)
        for parameter_group in self.alg.optimizer.param_groups:
            parameter_group["lr"] = finetune_settings["learning_rate"]
        print(f"[INFO] Initialized PPO actor only from {init_checkpoint}: {metadata}")
        print(f"[INFO] PPO finetune settings: {finetune_settings}")

    OnPolicyRunner.__init__ = _runner_init_with_actor

import isaaclab_nav  # noqa: F401 -- registration must happen before Hydra resolves the task


repository_root = Path(__file__).resolve().parents[2]
isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", repository_root.parent / "IsaacLab"))
trainer = isaaclab_root / "scripts/reinforcement_learning/rsl_rl/train.py"
if not trainer.is_file():
    raise FileNotFoundError(f"Set ISAACLAB_PATH to the Isaac Lab checkout; missing {trainer}")
sys.path.insert(0, str(trainer.parent))
runpy.run_path(str(trainer), run_name="__main__")
