"""Unitree 官方 G1 速度策略的观测、动作和关节顺序合同。"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Mapping

import numpy as np


OBSERVATION_DIMS = {
    "base_ang_vel": 3,
    "projected_gravity": 3,
    "velocity_commands": 3,
    "joint_pos_rel": 29,
    "joint_vel_rel": 29,
    "last_action": 29,
}


class ObservationHistory:
    """复现官方 use_gym_history=false 的按观测项历史拼接。"""

    def __init__(self, lengths: Mapping[str, int], initial: Mapping[str, np.ndarray]) -> None:
        self.buffers = {
            name: deque([value.copy() for _ in range(lengths[name])], maxlen=lengths[name])
            for name, value in initial.items()
        }

    def append(self, values: Mapping[str, np.ndarray]) -> None:
        if values.keys() != self.buffers.keys():
            raise ValueError("观测项与初始化时不一致")
        for name, value in values.items():
            self.buffers[name].append(np.asarray(value, dtype=np.float32).copy())

    def flatten(self, order: list[str]) -> np.ndarray:
        return np.concatenate([frame for name in order for frame in self.buffers[name]]).astype(np.float32)


@dataclass(frozen=True)
class PolicyContract:
    observation_order: tuple[str, ...]
    history_lengths: dict[str, int]
    observation_slices: dict[str, slice]
    joint_map: np.ndarray
    action_scale: np.ndarray
    action_offset: np.ndarray
    command_ranges: np.ndarray
    step_dt: float

    @classmethod
    def from_config(cls, config: dict) -> "PolicyContract":
        order = tuple(config["observations"].keys())
        if tuple(OBSERVATION_DIMS) != order:
            raise ValueError(f"官方观测顺序发生变化：{order}")

        history_lengths = {
            name: int(config["observations"][name].get("history_length", 1)) for name in order
        }
        slices: dict[str, slice] = {}
        cursor = 0
        for name in order:
            width = OBSERVATION_DIMS[name] * history_lengths[name]
            slices[name] = slice(cursor, cursor + width)
            cursor += width
        if cursor != 480:
            raise ValueError(f"策略观测应为480维，配置计算得到{cursor}维")

        joint_map = np.asarray(config["joint_ids_map"], dtype=np.int32)
        if joint_map.shape != (29,) or set(joint_map.tolist()) != set(range(29)):
            raise ValueError("joint_ids_map 必须是0～28的完整排列")

        action = config["actions"]["JointPositionAction"]
        action_scale = np.asarray(action["scale"], dtype=np.float64)
        action_offset = np.asarray(action["offset"], dtype=np.float64)
        if action_scale.shape != (29,) or action_offset.shape != (29,):
            raise ValueError("动作缩放和偏置必须均为29维")

        ranges = config["commands"]["base_velocity"]["ranges"]
        command_ranges = np.asarray(
            [ranges["lin_vel_x"], ranges["lin_vel_y"], ranges["ang_vel_z"]], dtype=np.float32
        )
        return cls(
            observation_order=order,
            history_lengths=history_lengths,
            observation_slices=slices,
            joint_map=joint_map,
            action_scale=action_scale,
            action_offset=action_offset,
            command_ranges=command_ranges,
            step_dt=float(config["step_dt"]),
        )

    @property
    def command_component_indices(self) -> dict[str, tuple[int, ...]]:
        start = self.observation_slices["velocity_commands"].start
        count = self.history_lengths["velocity_commands"]
        return {
            "vx": tuple(start + frame * 3 for frame in range(count)),
            "vy": tuple(start + frame * 3 + 1 for frame in range(count)),
            "omega": tuple(start + frame * 3 + 2 for frame in range(count)),
        }

    def clip_command(self, command: np.ndarray) -> np.ndarray:
        command = np.asarray(command, dtype=np.float32)
        if command.shape != (3,):
            raise ValueError("速度指令必须是(vx, vy, omega)三维")
        return np.clip(command, self.command_ranges[:, 0], self.command_ranges[:, 1])

    def process_action(self, raw_action: np.ndarray) -> np.ndarray:
        raw_action = np.asarray(raw_action, dtype=np.float64)
        if raw_action.shape != (29,):
            raise ValueError("策略动作必须是29维")
        return raw_action * self.action_scale + self.action_offset

    def policy_to_motor(self, policy_values: np.ndarray) -> np.ndarray:
        policy_values = np.asarray(policy_values)
        if policy_values.shape != (29,):
            raise ValueError("策略关节向量必须是29维")
        motor_values = np.empty_like(policy_values)
        motor_values[self.joint_map] = policy_values
        return motor_values
