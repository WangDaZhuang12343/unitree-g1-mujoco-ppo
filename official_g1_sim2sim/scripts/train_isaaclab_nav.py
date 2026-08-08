"""Register this repository's task and delegate to Isaac Lab's RSL-RL trainer."""

from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import isaaclab_nav  # noqa: F401 -- registration must happen before Hydra resolves the task


repository_root = Path(__file__).resolve().parents[2]
isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", repository_root.parent / "IsaacLab"))
trainer = isaaclab_root / "scripts/reinforcement_learning/rsl_rl/train.py"
if not trainer.is_file():
    raise FileNotFoundError(f"Set ISAACLAB_PATH to the Isaac Lab checkout; missing {trainer}")
sys.path.insert(0, str(trainer.parent))
runpy.run_path(str(trainer), run_name="__main__")
