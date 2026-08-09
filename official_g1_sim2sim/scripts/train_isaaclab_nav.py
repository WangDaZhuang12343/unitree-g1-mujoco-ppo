"""Register this repository's task and delegate to Isaac Lab's RSL-RL trainer."""

from __future__ import annotations

import argparse
import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _parse_wrapper_args() -> tuple[Path | None, dict[str, float], list[Path], float, int]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--init_checkpoint", type=Path)
    parser.add_argument("--finetune_learning_rate", type=float, default=1.0e-4)
    parser.add_argument("--finetune_entropy_coef", type=float, default=0.003)
    parser.add_argument("--finetune_clip_param", type=float, default=0.15)
    parser.add_argument("--finetune_desired_kl", type=float, default=0.005)
    parser.add_argument("--teacher_datasets", type=Path, nargs="+", default=[])
    parser.add_argument("--bc_anchor_coef", type=float, default=0.0)
    parser.add_argument("--bc_anchor_batch_size", type=int, default=512)
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
    if parsed.bc_anchor_coef < 0.0 or parsed.bc_anchor_batch_size <= 0:
        raise ValueError("BC anchor coefficient must be non-negative and batch size positive")
    if bool(parsed.teacher_datasets) != (parsed.bc_anchor_coef > 0.0):
        raise ValueError("--teacher_datasets and a positive --bc_anchor_coef must be used together")
    return (
        parsed.init_checkpoint,
        settings,
        parsed.teacher_datasets,
        parsed.bc_anchor_coef,
        parsed.bc_anchor_batch_size,
    )


(
    init_checkpoint,
    finetune_settings,
    teacher_datasets,
    bc_anchor_coef,
    bc_anchor_batch_size,
) = _parse_wrapper_args()

if init_checkpoint is not None:
    from rsl_rl.runners import OnPolicyRunner

    import torch
    from torch import nn

    from isaaclab_nav.pretraining import load_actor_initialization, load_teacher_datasets
    from isaaclab_nav.walking_compatibility import POLICY_TRAINING_PROFILE

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
        if teacher_datasets:
            teacher_observation, teacher_action, _ = load_teacher_datasets(
                teacher_datasets, expected_actuator_profile=POLICY_TRAINING_PROFILE
            )
            update_losses: list[float] = []

            def _add_teacher_gradient(_optimizer, _args, _kwargs):
                ids = torch.randint(len(teacher_observation), (bc_anchor_batch_size,))
                observation = teacher_observation[ids].to(self.device)
                target = teacher_action[ids].to(self.device)
                normalized = self.alg.policy.actor_obs_normalizer(observation)
                loss = nn.functional.smooth_l1_loss(self.alg.policy.actor(normalized), target)
                (bc_anchor_coef * loss).backward()
                nn.utils.clip_grad_norm_(self.alg.policy.actor.parameters(), self.alg.max_grad_norm)
                update_losses.append(float(loss.detach()))

            self.alg.optimizer.register_step_pre_hook(_add_teacher_gradient)
            original_update = self.alg.update

            def _update_with_teacher_anchor():
                update_losses.clear()
                losses = original_update()
                losses["bc_anchor"] = sum(update_losses) / max(1, len(update_losses))
                return losses

            self.alg.update = _update_with_teacher_anchor
            print(
                "[INFO] PPO BC anchor enabled: "
                f"datasets={teacher_datasets}, samples={len(teacher_observation)}, "
                f"coefficient={bc_anchor_coef}, batch_size={bc_anchor_batch_size}"
            )

    OnPolicyRunner.__init__ = _runner_init_with_actor

import isaaclab_nav  # noqa: F401 -- registration must happen before Hydra resolves the task


repository_root = Path(__file__).resolve().parents[2]
isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", repository_root.parent / "IsaacLab"))
trainer = isaaclab_root / "scripts/reinforcement_learning/rsl_rl/train.py"
if not trainer.is_file():
    raise FileNotFoundError(f"Set ISAACLAB_PATH to the Isaac Lab checkout; missing {trainer}")
sys.path.insert(0, str(trainer.parent))
runpy.run_path(str(trainer), run_name="__main__")
