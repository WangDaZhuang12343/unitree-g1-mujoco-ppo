"""Export a deterministic 483→3 actor from an RSL-RL PPO checkpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from isaaclab_nav.pretraining import ACTOR_INPUT_DIM, actor_from_rsl_rl_checkpoint
from isaaclab_nav.walking_compatibility import POLICY_TRAINING_PROFILE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--walking_actuator_profile", default=POLICY_TRAINING_PROFILE)
    args = parser.parse_args()
    if args.output.suffix != ".onnx":
        raise ValueError("--output must end in .onnx")
    actor, metadata = actor_from_rsl_rl_checkpoint(args.checkpoint)
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        actor,
        torch.zeros(1, ACTOR_INPUT_DIM),
        output,
        input_names=["observation"],
        output_names=["action"],
        dynamic_axes={"observation": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    metadata["onnx"] = str(output)
    metadata["walking_actuator_profile"] = args.walking_actuator_profile
    output.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("ISAACLAB_ACTOR_EXPORT_OK", metadata)


if __name__ == "__main__":
    main()
