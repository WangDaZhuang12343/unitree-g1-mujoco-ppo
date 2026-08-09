"""Behavior-clone the current 483→3 RSL-RL actor from DWA teacher datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab_nav.pretraining import (
    ACTOR_INPUT_DIM,
    ACTOR_OUTPUT_DIM,
    NORMALIZER_EPS,
    NavigationActor,
    NormalizedNavigationActor,
    load_teacher_datasets,
    make_actor_checkpoint,
    observation_moments,
)
from isaaclab_nav.walking_compatibility import POLICY_TRAINING_PROFILE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("datasets", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/g1_visual_navigation/dwa_bc_v1/model_init.pt"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--learning_rate", type=float, default=3.0e-4)
    parser.add_argument("--validation_fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs <= 0 or args.batch_size <= 0 or not 0.0 < args.validation_fraction < 1.0:
        raise ValueError("invalid training arguments")
    torch.manual_seed(args.seed)
    observation, action, trajectory_id = load_teacher_datasets(
        args.datasets, expected_actuator_profile=POLICY_TRAINING_PROFILE
    )
    trajectories = torch.unique(trajectory_id)
    if len(trajectories) < 2:
        raise ValueError("at least two independent trajectories are required")
    trajectories = trajectories[torch.randperm(len(trajectories))]
    validation_trajectory_count = max(1, int(len(trajectories) * args.validation_fraction))
    validation_trajectories = trajectories[:validation_trajectory_count]
    validation_mask = torch.isin(trajectory_id, validation_trajectories)
    validation_ids = torch.nonzero(validation_mask, as_tuple=False).flatten()
    training_ids = torch.nonzero(~validation_mask, as_tuple=False).flatten()
    validation_count = len(validation_ids)
    if not len(training_ids):
        raise ValueError("dataset is too small for the requested validation split")
    mean, std = observation_moments(observation[training_ids])
    normalized = (observation - mean) / (std + NORMALIZER_EPS)

    device = torch.device(args.device)
    actor = NavigationActor().to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=args.learning_rate)
    loss_fn = nn.SmoothL1Loss()
    generator = torch.Generator().manual_seed(args.seed)
    train_dataset = torch.utils.data.TensorDataset(normalized[training_ids], action[training_ids])
    loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, generator=generator
    )
    for epoch in range(args.epochs):
        actor.train()
        total_loss = 0.0
        for batch_observation, batch_action in loader:
            prediction = actor(batch_observation.to(device))
            loss = loss_fn(prediction, batch_action.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(actor.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * len(batch_observation)
        if epoch == 0 or (epoch + 1) % 5 == 0 or epoch + 1 == args.epochs:
            print(f"epoch={epoch + 1} train_loss={total_loss / len(training_ids):.6f}")

    actor.eval()
    with torch.inference_mode():
        prediction = actor(normalized[validation_ids].to(device)).cpu()
    validation_mae = float((prediction - action[validation_ids]).abs().mean())
    checkpoint = make_actor_checkpoint(
        actor,
        mean,
        std,
        sample_count=len(training_ids),
        validation_mae=validation_mae,
    )
    checkpoint["metadata"]["walking_actuator_profile"] = POLICY_TRAINING_PROFILE
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, output)
    onnx_output = output.with_suffix(".onnx")
    deployment_actor = NormalizedNavigationActor(actor.cpu(), mean, std).eval()
    torch.onnx.export(
        deployment_actor,
        torch.zeros(1, ACTOR_INPUT_DIM),
        onnx_output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    report = {
        **checkpoint["metadata"],
        "datasets": [str(path.resolve()) for path in args.datasets],
        "validation_count": validation_count,
        "training_trajectories": len(trajectories) - validation_trajectory_count,
        "validation_trajectories": validation_trajectory_count,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "checkpoint": str(output),
        "onnx": str(onnx_output),
    }
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("ISAACLAB_ACTOR_PRETRAIN_OK", report)


if __name__ == "__main__":
    main()
