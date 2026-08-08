"""Immutable Unitree walking-policy inference adapter."""

from __future__ import annotations

from pathlib import Path
from time import perf_counter

import numpy as np
import torch


class FrozenWalkingPolicy:
    """Run the unmodified fixed-batch Unitree ONNX policy for many environments.

    The released graph has a static ``[1, 480]`` input, so batching is performed
    by repeated calls without rewriting the ONNX graph.  This preserves the
    policy artifact byte-for-byte and makes the CPU/GPU transfer explicit.
    """

    observation_dim = 480
    action_dim = 29

    def __init__(self, model_path: str | Path) -> None:
        try:
            import onnxruntime as ort
        except ImportError as error:  # pragma: no cover - environment error
            raise RuntimeError("onnxruntime is required for the frozen Walking Policy") from error

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.is_file():
            raise FileNotFoundError(self.model_path)
        self.session = ort.InferenceSession(str(self.model_path), providers=["CPUExecutionProvider"])
        model_input = self.session.get_inputs()[0]
        model_output = self.session.get_outputs()[0]
        if model_input.shape != [1, self.observation_dim]:
            raise ValueError(f"Walking Policy input changed: {model_input.shape}")
        if model_output.shape != [1, self.action_dim]:
            raise ValueError(f"Walking Policy output changed: {model_output.shape}")
        self.input_name = model_input.name
        self.output_name = model_output.name
        self.last_inference_seconds = 0.0

    def __call__(self, observation: torch.Tensor) -> torch.Tensor:
        if observation.ndim != 2 or observation.shape[1] != self.observation_dim:
            raise ValueError(f"Walking observation must have shape (N, {self.observation_dim})")
        if not torch.isfinite(observation).all():
            raise ValueError("Walking observation contains non-finite values")
        host = observation.detach().to(device="cpu", dtype=torch.float32).numpy()
        start = perf_counter()
        actions = np.concatenate(
            [
                self.session.run([self.output_name], {self.input_name: row[None]})[0]
                for row in host
            ],
            axis=0,
        )
        self.last_inference_seconds = perf_counter() - start
        if actions.shape != (observation.shape[0], self.action_dim) or not np.isfinite(actions).all():
            raise RuntimeError("Walking Policy returned invalid actions")
        return torch.from_numpy(actions).to(device=observation.device)
