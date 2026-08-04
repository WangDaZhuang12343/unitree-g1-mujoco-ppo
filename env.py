from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import gymnasium as gym
import mujoco
import numpy as np


HERE = Path(__file__).resolve().parent


def default_model_path() -> Path:
    configured = os.environ.get("UNITREE_MUJOCO_ROOT")
    roots = [
        Path(configured).expanduser() if configured else None,
        HERE / "third_party/unitree_mujoco",
        HERE.parent / "unitree_mujoco",
    ]
    for root in roots:
        if root is None:
            continue
        candidate = root / "unitree_robots/g1/scene_walk_pd_v3.xml"
        if candidate.is_file():
            return candidate.resolve()
    return (HERE / "third_party/unitree_mujoco/unitree_robots/g1/scene_walk_pd_v3.xml").resolve()


DEFAULT_MODEL = default_model_path()


class G1WalkEnv(gym.Env[np.ndarray, np.ndarray]):
    """Velocity-tracking G1 task using position actuators.

    The policy controls the 12 leg and 3 waist joints. Arms and wrists are held
    at a relaxed pose, reducing the initial CPU training problem to 15 actions.
    """

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path = DEFAULT_MODEL,
        render_mode: str | None = None,
        frame_skip: int = 10,
        episode_seconds: float = 20.0,
        command_speed: tuple[float, float] = (0.0, 0.7),
        fixed_command_speed: float | None = None,
        gait_clock: bool = False,
        gait_reference: bool = False,
        action_scale_factor: float = 1.0,
    ) -> None:
        super().__init__()
        resolved_model = Path(model_path).resolve()
        if not resolved_model.is_file():
            raise FileNotFoundError(
                f"未找到 G1 MuJoCo 模型：{resolved_model}。"
                "请先运行 ./prepare_model.sh，或设置 UNITREE_MUJOCO_ROOT。"
            )
        self.model_path = str(resolved_model)
        self.model = mujoco.MjModel.from_xml_path(self.model_path)
        self.data = mujoco.MjData(self.model)
        self.render_mode = render_mode
        self.frame_skip = frame_skip
        self.control_dt = self.model.opt.timestep * frame_skip
        self.max_steps = round(episode_seconds / self.control_dt)
        self.command_speed = command_speed
        self.fixed_command_speed = fixed_command_speed
        self.gait_clock = gait_clock
        self.gait_reference = gait_reference
        self.action_scale_factor = action_scale_factor
        self.gait_frequency = 1.4
        self.pelvis_id = self._body_id("pelvis")
        self.left_foot_id = self._body_id("left_ankle_roll_link")
        self.right_foot_id = self._body_id("right_ankle_roll_link")

        names = [
            "left_hip_pitch", "left_hip_roll", "left_hip_yaw", "left_knee",
            "left_ankle_pitch", "left_ankle_roll", "right_hip_pitch",
            "right_hip_roll", "right_hip_yaw", "right_knee",
            "right_ankle_pitch", "right_ankle_roll", "waist_yaw",
            "waist_roll", "waist_pitch",
        ]
        self.actuator_ids = np.asarray([self._actuator_id(n) for n in names])
        self.joint_ids = self.model.actuator_trnid[self.actuator_ids, 0]
        self.qpos_ids = self.model.jnt_qposadr[self.joint_ids]
        self.qvel_ids = self.model.jnt_dofadr[self.joint_ids]

        # A small knee bend makes exploration less brittle than locked legs.
        self.default_pose = np.zeros(self.model.nu, dtype=np.float64)
        self.default_pose[[3, 9]] = 0.12
        self.default_pose[[4, 10]] = -0.06
        self.default_pose[[15, 22]] = 0.15
        self.default_pose[[18, 25]] = -0.20
        self.action_scale = np.asarray(
            [0.35, 0.20, 0.20, 0.45, 0.25, 0.15] * 2
            + [0.25, 0.18, 0.18], dtype=np.float64
        )
        self.ctrl_low = self.model.actuator_ctrlrange[:, 0]
        self.ctrl_high = self.model.actuator_ctrlrange[:, 1]

        obs_size = 3 + 3 + 3 + 15 + 15 + 15 + 3
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (obs_size,), np.float32)
        self.action_space = gym.spaces.Box(-1.0, 1.0, (15,), np.float32)
        self.last_action = np.zeros(15, dtype=np.float64)
        self.command = np.zeros(3, dtype=np.float64)
        self.phase = 0.0
        self.foot_reference_z = np.zeros(2, dtype=np.float64)
        self.velocity_error_sum = 0.0
        self.steps = 0
        self.viewer = None
        self.renderer = None

    def _body_id(self, name: str) -> int:
        result = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if result < 0:
            raise ValueError(f"Body not found in model: {name}")
        return result

    def _actuator_id(self, name: str) -> int:
        result = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
        if result < 0:
            raise ValueError(f"Actuator not found in model: {name}")
        return result

    def _orientation(self) -> tuple[np.ndarray, float]:
        rotation = self.data.xmat[self.pelvis_id].reshape(3, 3)
        projected_gravity = rotation.T @ np.asarray([0.0, 0.0, -1.0])
        return projected_gravity, float(rotation[2, 2])

    def _get_obs(self) -> np.ndarray:
        gravity, _ = self._orientation()
        joint_pos = self.data.qpos[self.qpos_ids] - self.default_pose[self.actuator_ids]
        joint_vel = self.data.qvel[self.qvel_ids] * 0.1
        command_obs = self.command.copy()
        if self.gait_clock:
            angle = 2.0 * np.pi * self.phase
            command_obs[:] = (self.command[0], np.sin(angle), np.cos(angle))
        obs = np.concatenate(
            (
                self.data.qvel[:3] * 0.5,
                self.data.qvel[3:6] * 0.25,
                gravity,
                joint_pos,
                joint_vel,
                self.last_action,
                command_obs,
            )
        )
        return obs.astype(np.float32)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[self.qpos_ids] = self.default_pose[self.actuator_ids]
        self.data.qpos[self.qpos_ids] += self.np_random.uniform(-0.015, 0.015, 15)
        self.data.qvel[:] = self.np_random.normal(0.0, 0.01, self.model.nv)
        self.data.ctrl[:] = np.clip(self.default_pose, self.ctrl_low, self.ctrl_high)
        speed = (
            self.fixed_command_speed
            if self.fixed_command_speed is not None
            else self.np_random.uniform(*self.command_speed)
        )
        self.command[:] = (speed, 0.0, 0.0)
        self.phase = float(self.np_random.uniform()) if self.gait_clock else 0.0
        self.last_action.fill(0.0)
        self.velocity_error_sum = 0.0
        self.steps = 0
        mujoco.mj_forward(self.model, self.data)
        self.foot_reference_z[:] = (
            self.data.xpos[self.left_foot_id, 2],
            self.data.xpos[self.right_foot_id, 2],
        )
        return self._get_obs(), {"command_x": float(self.command[0])}

    def step(self, action: np.ndarray):
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        target = self.default_pose.copy()
        phase_angle = 2.0 * np.pi * self.phase
        phase_sine = np.sin(phase_angle)
        speed_scale = np.sqrt(np.clip(self.command[0] / 0.7, 0.0, 1.0))
        left_swing = max(phase_sine, 0.0)
        right_swing = max(-phase_sine, 0.0)
        if self.gait_reference:
            gait = np.zeros(15, dtype=np.float64)
            gait[0] = -0.18 * speed_scale * phase_sine
            gait[6] = 0.18 * speed_scale * phase_sine
            gait[3] = 0.25 * speed_scale * left_swing
            gait[9] = 0.25 * speed_scale * right_swing
            gait[4] = -0.12 * speed_scale * left_swing
            gait[10] = -0.12 * speed_scale * right_swing
            target[self.actuator_ids] += gait
        target[self.actuator_ids] += action * self.action_scale * self.action_scale_factor
        self.data.ctrl[:] = np.clip(target, self.ctrl_low, self.ctrl_high)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self.steps += 1

        gravity, upright = self._orientation()
        vx, vy = self.data.qvel[0], self.data.qvel[1]
        wz = self.data.qvel[5]
        height = self.data.xpos[self.pelvis_id, 2]
        velocity_tracking = np.exp(-((vx - self.command[0]) / 0.15) ** 2)
        lateral_tracking = np.exp(-4.0 * vy**2)
        yaw_tracking = np.exp(-2.0 * (wz - self.command[2]) ** 2)
        upright_reward = np.clip((upright - 0.7) / 0.3, 0.0, 1.0)
        height_reward = np.exp(-20.0 * (height - 0.76) ** 2)
        action_rate = np.mean((action - self.last_action) ** 2)
        action_magnitude = np.mean(action**2)
        joint_speed = np.mean((self.data.qvel[self.qvel_ids] * 0.1) ** 2)
        foot_tracking = 0.0
        if self.gait_clock and self.command[0] > 0.02:
            clearance = 0.08 * speed_scale
            desired_left = self.foot_reference_z[0] + clearance * left_swing
            desired_right = self.foot_reference_z[1] + clearance * right_swing
            left_error = self.data.xpos[self.left_foot_id, 2] - desired_left
            right_error = self.data.xpos[self.right_foot_id, 2] - desired_right
            foot_tracking = 0.5 * (
                np.exp(-200.0 * left_error**2) + np.exp(-200.0 * right_error**2)
            )
        reward = (
            1.5 * velocity_tracking
            + 0.4 * lateral_tracking
            + 0.2 * yaw_tracking
            + 0.8 * upright_reward
            + 0.4 * height_reward
            + 0.35 * foot_tracking
            - 0.05 * action_rate
            - 0.01 * action_magnitude
            - 0.01 * joint_speed
        )
        self.last_action[:] = action
        self.velocity_error_sum += abs(vx - self.command[0])

        unhealthy = bool(height < 0.50 or upright < 0.55 or not np.isfinite(self.data.qpos).all())
        truncated = self.steps >= self.max_steps
        if unhealthy:
            reward -= 10.0
        mean_velocity_error = self.velocity_error_sum / self.steps
        info = {
            "x_velocity": float(vx),
            "command_x": float(self.command[0]),
            "height": float(height),
            "upright": upright,
            "reward_tracking": float(velocity_tracking),
            "mean_velocity_error": float(mean_velocity_error),
        }
        if unhealthy or truncated:
            info["is_success"] = bool(
                truncated and mean_velocity_error < 0.06 and upright > 0.85
            )
        self.phase = (self.phase + self.gait_frequency * self.control_dt) % 1.0
        if self.render_mode == "human":
            self.render()
        return self._get_obs(), float(reward), unhealthy, truncated, info

    def render(self):
        if self.render_mode == "human":
            if self.viewer is None:
                import mujoco.viewer

                self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
            self.viewer.sync()
            return None
        if self.render_mode == "rgb_array":
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.model, height=480, width=640)
            self.renderer.update_scene(self.data)
            return self.renderer.render()
        return None

    def close(self) -> None:
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
