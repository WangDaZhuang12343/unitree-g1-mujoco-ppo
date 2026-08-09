"""Run the frozen Walking ONNX command envelope on the MuJoCo baseline asset."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from g1_nav.policy_contract import ObservationHistory, PolicyContract
from g1_nav.walking_commands import COMMANDS
from simulate import CONFIG_PATH, MODEL_PATH, build_model, projected_gravity, quaternion_euler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=12.0)
    parser.add_argument("--warmup", type=float, default=2.0)
    parser.add_argument("--output", type=Path, default=Path("runs/mujoco_walking_envelope"))
    return parser.parse_args()


def _flat_model() -> mujoco.MjModel:
    return build_model(
        SimpleNamespace(
            terrain="flat",
            obstacle_height=0.04,
            obstacle_x=1.5,
            obstacle_width=1.8,
            slope_angle=5.0,
            roughness=0.02,
            terrain_seed=7,
            scene_obstacles=(),
        )
    )


def _body_linear_velocity(data: mujoco.MjData) -> np.ndarray:
    rotation = np.empty(9, dtype=np.float64)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    return rotation.reshape(3, 3).T @ data.qvel[:3]


def main() -> None:
    args = parse_args()
    if args.duration <= 0.0 or args.warmup < 0.0 or args.warmup >= args.duration:
        raise ValueError("duration must be positive and warmup must be in [0, duration)")
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    contract = PolicyContract.from_config(config)
    default = np.asarray(config["default_joint_pos"], dtype=np.float64)
    stiffness = np.asarray(config["stiffness"], dtype=np.float64)
    damping = np.asarray(config["damping"], dtype=np.float64)
    model = _flat_model()
    control_steps = max(1, round(contract.step_dt / model.opt.timestep))
    total_steps = round(args.duration / model.opt.timestep)
    warmup_steps = round(args.warmup / model.opt.timestep)
    session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    rows: list[dict[str, object]] = []

    for name, vx, vy, omega in COMMANDS:
        data = mujoco.MjData(model)
        data.qpos[7 + contract.joint_map] = default
        mujoco.mj_forward(model, data)
        command = contract.clip_command(np.asarray((vx, vy, omega), dtype=np.float32))
        last_action = np.zeros(29, dtype=np.float32)

        def terms() -> dict[str, np.ndarray]:
            raw = {
                "base_ang_vel": np.asarray(data.sensor("imu_gyro").data, dtype=np.float32),
                "projected_gravity": projected_gravity(data.qpos[3:7]).astype(np.float32),
                "velocity_commands": command,
                "joint_pos_rel": (data.qpos[7 + contract.joint_map] - default).astype(np.float32),
                "joint_vel_rel": data.qvel[6 + contract.joint_map].astype(np.float32),
                "last_action": last_action,
            }
            return {
                key: value * np.asarray(config["observations"][key]["scale"], dtype=np.float32)
                for key, value in raw.items()
            }

        history = ObservationHistory(contract.history_lengths, terms())
        target_policy = default.copy()
        measured_sum = np.zeros(3, dtype=np.float64)
        squared_error_sum = np.zeros(3, dtype=np.float64)
        samples = 0
        reason = "timeout"
        completed_steps = 0
        max_tilt = 0.0
        start_xy = data.qpos[:2].copy()
        for step in range(total_steps):
            if step % control_steps == 0:
                history.append(terms())
                observation = history.flatten(list(contract.observation_order))[None]
                last_action[:] = session.run(None, {input_name: observation})[0][0]
                target_policy = contract.process_action(last_action)
            target_motor = contract.policy_to_motor(target_policy)
            torque = stiffness * (target_motor - data.qpos[7:36]) - damping * data.qvel[6:35]
            data.ctrl[:] = np.clip(
                torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1]
            )
            mujoco.mj_step(model, data)
            completed_steps = step + 1
            roll, pitch, _ = quaternion_euler(data.qpos[3:7])
            max_tilt = max(max_tilt, abs(roll), abs(pitch))
            if step >= warmup_steps:
                measured = np.asarray(
                    (*_body_linear_velocity(data)[:2], data.sensor("imu_gyro").data[2]),
                    dtype=np.float64,
                )
                measured_sum += measured
                squared_error_sum += np.square(measured - command)
                samples += 1
            if data.qpos[2] < 0.45 or abs(roll) > 0.8 or abs(pitch) > 0.8:
                reason = "fall"
                break
        denominator = max(samples, 1)
        mean = measured_sum / denominator
        rmse = np.sqrt(squared_error_sum / denominator)
        rows.append(
            {
                "command": name,
                "vx": vx,
                "vy": vy,
                "omega": omega,
                "survived": reason == "timeout",
                "termination_reason": reason,
                "elapsed_sim_s": completed_steps * model.opt.timestep,
                "planar_displacement_m": float(np.linalg.norm(data.qpos[:2] - start_xy)),
                "max_tilt_rad": max_tilt,
                "mean_vx": mean[0],
                "mean_vy": mean[1],
                "mean_omega": mean[2],
                "rmse_vx": rmse[0],
                "rmse_vy": rmse[1],
                "rmse_omega": rmse[2],
                "tracking_samples": samples,
            }
        )

    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# MuJoCo Frozen Walking Command Envelope",
        "",
        "| Command | vx | vy | omega | Survived | End reason | Time (s) | Mean vx | Mean vy | Mean omega | RMSE norm |",
        "|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        rmse_norm = math.sqrt(
            float(row["rmse_vx"]) ** 2
            + float(row["rmse_vy"]) ** 2
            + float(row["rmse_omega"]) ** 2
        )
        lines.append(
            f"| {row['command']} | {row['vx']:.2f} | {row['vy']:.2f} | {row['omega']:.2f} | "
            f"{int(row['survived'])} | {row['termination_reason']} | {row['elapsed_sim_s']:.2f} | "
            f"{row['mean_vx']:.3f} | {row['mean_vy']:.3f} | {row['mean_omega']:.3f} | "
            f"{rmse_norm:.3f} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("MUJOCO_WALKING_COMMANDS_OK", {"rows": len(rows), "output": str(output)})


if __name__ == "__main__":
    main()
