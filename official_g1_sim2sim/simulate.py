#!/usr/bin/env python3
"""Unitree 官方 G1 速度策略的纯 MuJoCo 验证器，不初始化 DDS。"""

from __future__ import annotations

import argparse
import ctypes
import csv
import math
import os
from pathlib import Path
import time

import mujoco
import numpy as np
import yaml

from g1_nav.policy_contract import ObservationHistory, PolicyContract


ROOT = Path(__file__).resolve().parent


def find_repository(name: str, environment_variable: str) -> Path:
    candidates = []
    configured = os.environ.get(environment_variable)
    if configured:
        candidates.append(Path(configured).expanduser())
    candidates.extend([ROOT.parent / name, ROOT.parent.parent / name])
    for candidate in candidates:
        if candidate.is_dir():
            return candidate.resolve()
    locations = "、".join(str(path) for path in candidates)
    raise FileNotFoundError(f"找不到 {name}，已检查：{locations}；也可设置 {environment_variable}")


UNITREE_RL_LAB_ROOT = find_repository("unitree_rl_lab", "UNITREE_RL_LAB_ROOT")
UNITREE_MUJOCO_ROOT = find_repository("unitree_mujoco", "UNITREE_MUJOCO_ROOT")
POLICY_DIR = UNITREE_RL_LAB_ROOT / "deploy/robots/g1_29dof/config/policy/velocity/v0"
MODEL_PATH = POLICY_DIR / "exported/policy.onnx"
CONFIG_PATH = POLICY_DIR / "params/deploy.yaml"
SCENE_PATH = UNITREE_MUJOCO_ROOT / "unitree_robots/g1/scene.xml"
BRIDGE_PATH = ROOT / "build/libort_bridge.so"


class OrtRunner:
    def __init__(self, model_path: Path) -> None:
        if not BRIDGE_PATH.exists():
            raise RuntimeError("推理桥接库不存在，请先运行 ./build.sh")
        self.lib = ctypes.CDLL(str(BRIDGE_PATH))
        self.lib.ort_runner_create.argtypes = [ctypes.c_char_p]
        self.lib.ort_runner_create.restype = ctypes.c_void_p
        self.lib.ort_runner_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ort_runner_input_size.argtypes = [ctypes.c_void_p]
        self.lib.ort_runner_input_size.restype = ctypes.c_int
        self.lib.ort_runner_output_size.argtypes = [ctypes.c_void_p]
        self.lib.ort_runner_output_size.restype = ctypes.c_int
        self.lib.ort_runner_input_name.argtypes = [ctypes.c_void_p]
        self.lib.ort_runner_input_name.restype = ctypes.c_char_p
        self.lib.ort_runner_output_name.argtypes = [ctypes.c_void_p]
        self.lib.ort_runner_output_name.restype = ctypes.c_char_p
        self.lib.ort_runner_run.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
        ]
        self.lib.ort_runner_run.restype = ctypes.c_int
        self.lib.ort_runner_last_error.restype = ctypes.c_char_p
        self.handle = self.lib.ort_runner_create(str(model_path).encode())
        if not self.handle:
            self._raise_error()
        self.input_size = self.lib.ort_runner_input_size(self.handle)
        self.output_size = self.lib.ort_runner_output_size(self.handle)
        self.input_name = self.lib.ort_runner_input_name(self.handle).decode()
        self.output_name = self.lib.ort_runner_output_name(self.handle).decode()

    def _raise_error(self) -> None:
        message = self.lib.ort_runner_last_error().decode() or "ONNX Runtime 未知错误"
        raise RuntimeError(message)

    def run(self, observation: np.ndarray) -> np.ndarray:
        observation = np.ascontiguousarray(observation, dtype=np.float32)
        output = np.empty(self.output_size, dtype=np.float32)
        code = self.lib.ort_runner_run(
            self.handle,
            observation.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            observation.size,
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            output.size,
        )
        if code != 0:
            self._raise_error()
        return output

    def close(self) -> None:
        if getattr(self, "handle", None):
            self.lib.ort_runner_destroy(self.handle)
            self.handle = None


def projected_gravity(quaternion_wxyz: np.ndarray) -> np.ndarray:
    matrix = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(matrix, quaternion_wxyz)
    return matrix.reshape(3, 3).T @ np.array([0.0, 0.0, -1.0])


def quaternion_euler(quaternion_wxyz: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = quaternion_wxyz
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch_sine = np.clip(2.0 * (w * y - z * x), -1.0, 1.0)
    pitch = math.asin(float(pitch_sine))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return roll, pitch, yaw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=20.0, help="仿真时长（秒）")
    parser.add_argument("--vx", type=float, default=0.0, help="前向速度指令（米/秒）")
    parser.add_argument("--vy", type=float, default=0.0, help="侧向速度指令（米/秒）")
    parser.add_argument("--yaw", type=float, default=0.0, help="偏航角速度指令（弧度/秒）")
    parser.add_argument(
        "--terrain",
        choices=("flat", "official", "bar", "step", "stairs", "ramp", "rough"),
        default="flat",
        help="平地、官方场景、横杆、台阶、楼梯、斜坡或随机起伏",
    )
    parser.add_argument("--obstacle-height", type=float, default=0.04, help="横杆/台阶高度或楼梯级高（米）")
    parser.add_argument("--obstacle-x", type=float, default=1.5, help="地形起点或中心位置 x（米）")
    parser.add_argument("--obstacle-width", type=float, default=1.8, help="横杆沿Y方向的总宽度（米）")
    parser.add_argument("--slope-angle", type=float, default=5.0, help="斜坡角度（度）")
    parser.add_argument("--roughness", type=float, default=0.02, help="随机起伏最大高度（米）")
    parser.add_argument("--terrain-seed", type=int, default=7, help="随机起伏种子")
    parser.add_argument("--viewer", action="store_true", help="打开 MuJoCo 窗口并按实时速度运行")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/latest.csv", help="指标 CSV")
    return parser.parse_args()


def build_model(args: argparse.Namespace) -> mujoco.MjModel:
    spec = mujoco.MjSpec.from_file(str(SCENE_PATH))

    def add_box(name: str, pos: list[float], size: list[float], rgba: list[float], quat=None) -> None:
        kwargs = {
            "name": f"course_{name}",
            "type": mujoco.mjtGeom.mjGEOM_BOX,
            "pos": pos,
            "size": size,
            "rgba": rgba,
            "friction": [1.0, 0.005, 0.0001],
        }
        if quat is not None:
            kwargs["quat"] = quat
        spec.worldbody.add_geom(**kwargs)

    if args.terrain in {"bar", "step", "stairs"} and not 0.0 < args.obstacle_height <= 0.30:
        raise ValueError("障碍高度必须在 0～0.30 米之间")
    if args.terrain == "bar":
        obstacle_width = float(getattr(args, "obstacle_width", 1.8))
        if not 0.1 <= obstacle_width <= 4.0:
            raise ValueError("横杆宽度必须在0.1～4.0米之间")
        spec.worldbody.add_geom(
            name="course_bar",
            type=mujoco.mjtGeom.mjGEOM_BOX,
            pos=[args.obstacle_x, 0.0, args.obstacle_height / 2.0],
            size=[0.06, obstacle_width / 2.0, args.obstacle_height / 2.0],
            rgba=[0.85, 0.22, 0.08, 1.0],
            friction=[1.0, 0.005, 0.0001],
        )
    elif args.terrain == "step":
        add_box(
            "step",
            [args.obstacle_x, 0.0, args.obstacle_height / 2.0],
            [0.40, 0.90, args.obstacle_height / 2.0],
            [0.12, 0.48, 0.82, 1.0],
        )
    elif args.terrain == "stairs":
        width = 0.30
        levels = [1, 2, 3, 3, 2, 1]
        for index, level in enumerate(levels):
            height = level * args.obstacle_height
            add_box(
                f"stair_{index}",
                [args.obstacle_x + (index + 0.5) * width, 0.0, height / 2.0],
                [width / 2.0, 0.90, height / 2.0],
                [0.18 + index * 0.05, 0.55, 0.30, 1.0],
            )
    elif args.terrain == "ramp":
        if not 0.0 < args.slope_angle <= 20.0:
            raise ValueError("斜坡角度必须在 0～20 度之间")
        angle = math.radians(args.slope_angle)
        half_length = 0.60
        half_thickness = 0.025
        # 上表面入口与地面齐平，两块斜板在坡顶连续衔接。
        center_z = half_length * math.sin(angle) - half_thickness * math.cos(angle)
        add_box(
            "ramp_up",
            [args.obstacle_x, 0.0, center_z],
            [half_length, 0.90, half_thickness],
            [0.20, 0.58, 0.78, 1.0],
            [math.cos(angle / 2.0), 0.0, -math.sin(angle / 2.0), 0.0],
        )
        add_box(
            "ramp_down",
            [args.obstacle_x + 2.0 * half_length * math.cos(angle), 0.0, center_z],
            [half_length, 0.90, half_thickness],
            [0.16, 0.50, 0.72, 1.0],
            [math.cos(angle / 2.0), 0.0, math.sin(angle / 2.0), 0.0],
        )
    elif args.terrain == "rough":
        if not 0.0 < args.roughness <= 0.15:
            raise ValueError("随机起伏高度必须在 0～0.15 米之间")
        rng = np.random.default_rng(args.terrain_seed)
        width = 0.30
        for index, height in enumerate(rng.uniform(0.002, args.roughness, size=10)):
            add_box(
                f"rough_{index}",
                [args.obstacle_x + (index + 0.5) * width, 0.0, float(height) / 2.0],
                [width / 2.0, 0.90, float(height) / 2.0],
                [0.42, 0.43 + index * 0.025, 0.22, 1.0],
            )

    for obstacle in getattr(args, "scene_obstacles", ()):
        if obstacle.height <= 0.0 or obstacle.size_x <= 0.0 or obstacle.size_y <= 0.0:
            raise ValueError(f"场景障碍物尺寸必须为正数：{obstacle.name}")
        add_box(
            obstacle.name,
            [obstacle.x, obstacle.y, obstacle.height / 2.0],
            [obstacle.size_x / 2.0, obstacle.size_y / 2.0, obstacle.height / 2.0],
            list(obstacle.rgba),
        )
    model = spec.compile()
    model.opt.timestep = 0.002

    if args.terrain != "official":
        floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        for geom_id in range(model.ngeom):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if (
                model.geom_bodyid[geom_id] == 0
                and geom_id != floor_id
                and not name.startswith("course_")
            ):
                model.geom_contype[geom_id] = 0
                model.geom_conaffinity[geom_id] = 0
                model.geom_rgba[geom_id, 3] = 0.0
    return model


def main() -> None:
    args = parse_args()
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    contract = PolicyContract.from_config(config)

    command = contract.clip_command(np.array([args.vx, args.vy, args.yaw], dtype=np.float32))
    joint_map = contract.joint_map
    default = np.asarray(config["default_joint_pos"], dtype=np.float64)
    stiffness = np.asarray(config["stiffness"], dtype=np.float64)
    damping = np.asarray(config["damping"], dtype=np.float64)
    observation_order = list(contract.observation_order)

    model = build_model(args)
    data = mujoco.MjData(model)
    data.qpos[7 + joint_map] = default
    mujoco.mj_forward(model, data)

    runner = OrtRunner(MODEL_PATH)
    if runner.output_size != 29:
        raise RuntimeError(f"策略输出应为 29，实际为 {runner.output_size}")

    last_action = np.zeros(29, dtype=np.float32)

    def observations() -> dict[str, np.ndarray]:
        policy_q = data.qpos[7 + joint_map]
        policy_dq = data.qvel[6 + joint_map]
        raw = {
            "base_ang_vel": np.asarray(data.sensor("imu_gyro").data, dtype=np.float32),
            "projected_gravity": projected_gravity(data.qpos[3:7]).astype(np.float32),
            "velocity_commands": command,
            "joint_pos_rel": (policy_q - default).astype(np.float32),
            "joint_vel_rel": policy_dq.astype(np.float32),
            "last_action": last_action,
        }
        return {
            name: value * np.asarray(config["observations"][name]["scale"], dtype=np.float32)
            for name, value in raw.items()
        }

    history = ObservationHistory(contract.history_lengths, observations())
    observation = history.flatten(observation_order)
    if observation.size != runner.input_size:
        raise RuntimeError(f"策略输入维度不匹配：模型 {runner.input_size}，观测 {observation.size}")

    target_policy = default.copy()
    control_steps = max(1, round(float(config["step_dt"]) / model.opt.timestep))
    total_steps = round(args.duration / model.opt.timestep)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[float]] = []
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.0, 0.0, 0.8]
        viewer.cam.distance = 3.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -15

    print(
        f"模型输入={runner.input_name}[{runner.input_size}] "
        f"输出={runner.output_name}[{runner.output_size}]，指令={command.tolist()}，地形={args.terrain}"
    )
    start_wall = time.perf_counter()
    try:
        for step in range(total_steps):
            if step % control_steps == 0:
                history.append(observations())
                observation = history.flatten(observation_order)
                last_action[:] = runner.run(observation)
                target_policy = contract.process_action(last_action)

            target_motor = contract.policy_to_motor(target_policy)
            torque = stiffness * (target_motor - data.qpos[7:36]) - damping * data.qvel[6:35]
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)

            if step % control_steps == 0:
                roll, pitch, yaw = quaternion_euler(data.qpos[3:7])
                rows.append(
                    [
                        data.time,
                        data.qpos[0],
                        data.qpos[1],
                        data.qpos[2],
                        data.qvel[0],
                        data.qvel[1],
                        float(projected_gravity(data.qpos[3:7])[2]),
                        roll,
                        pitch,
                        yaw,
                        float(np.max(np.abs(last_action))),
                    ]
                )
                if int(data.time * 10) % 20 == 0 and step % (control_steps * 10) == 0:
                    print(
                        f"t={data.time:5.1f}s x={data.qpos[0]:+.3f}m "
                        f"y={data.qpos[1]:+.3f}m vx={data.qvel[0]:+.3f}m/s "
                        f"偏航={math.degrees(yaw):+.1f}度 高度={data.qpos[2]:.3f}m"
                    )
            if viewer is not None:
                viewer.cam.lookat[0] = data.qpos[0]
                viewer.sync()
                deadline = start_wall + data.time
                if deadline > time.perf_counter():
                    time.sleep(deadline - time.perf_counter())
                if not viewer.is_running():
                    break
            if data.qpos[2] < 0.35:
                print(f"检测到跌倒，t={data.time:.2f}s，高度={data.qpos[2]:.3f}m")
                break
    finally:
        if viewer is not None:
            viewer.close()
        runner.close()

    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s",
                "x_m",
                "y_m",
                "height_m",
                "vx_m_s",
                "vy_m_s",
                "gravity_z",
                "roll_rad",
                "pitch_rad",
                "yaw_rad",
                "max_action",
            ]
        )
        writer.writerows(rows)

    elapsed = rows[-1][0] if rows else 0.0
    dx = rows[-1][1] - rows[0][1] if len(rows) > 1 else 0.0
    mean_vx = dx / elapsed if elapsed else 0.0
    print(f"完成：仿真 {elapsed:.2f}s，位移 {dx:.3f}m，平均速度 {mean_vx:.3f}m/s，结果 {args.output}")


if __name__ == "__main__":
    main()
