#!/usr/bin/env python3
"""相机、局部代价地图、DWA与官方G1策略的纯仿真导航闭环。"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from types import SimpleNamespace
import time

import mujoco
import numpy as np
import yaml

from g1_nav.l2_costmap import LocalCostMap
from g1_nav.l3_dwa import DWANavigator, PlanResult
from g1_nav.l4_camera import SimulatedDepthCamera
from g1_nav.l6_safety import SafetySystem
from g1_nav.policy_contract import ObservationHistory, PolicyContract
from simulate import (
    CONFIG_PATH,
    MODEL_PATH,
    ROOT,
    OrtRunner,
    build_model,
    projected_gravity,
    quaternion_euler,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--obstacle-x", type=float, default=2.5)
    parser.add_argument("--obstacle-height", type=float, default=0.15)
    parser.add_argument("--obstacle-width", type=float, default=0.8)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/navigation/latest.csv")
    return parser.parse_args()


def goal_in_body(data: mujoco.MjData, body_id: int, goal_world: np.ndarray) -> tuple[float, float]:
    rotation = data.xmat[body_id].reshape(3, 3)
    relative_world = np.array([goal_world[0] - data.xpos[body_id, 0], goal_world[1] - data.xpos[body_id, 1], 0.0])
    relative_body = rotation.T @ relative_world
    return float(relative_body[0]), float(relative_body[1])


def main() -> None:
    args = parse_args()
    terrain_args = SimpleNamespace(
        terrain="bar",
        obstacle_height=args.obstacle_height,
        obstacle_x=args.obstacle_x,
        obstacle_width=args.obstacle_width,
        slope_angle=5.0,
        roughness=0.02,
        terrain_seed=7,
    )
    with CONFIG_PATH.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    contract = PolicyContract.from_config(config)
    model = build_model(terrain_args)
    data = mujoco.MjData(model)
    default = np.asarray(config["default_joint_pos"], dtype=np.float64)
    stiffness = np.asarray(config["stiffness"], dtype=np.float64)
    damping = np.asarray(config["damping"], dtype=np.float64)
    data.qpos[7 + contract.joint_map] = default
    mujoco.mj_forward(model, data)

    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    obstacle_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "course_bar")
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
    camera = SimulatedDepthCamera()
    camera.isolate_world_geoms(model)
    costmap = LocalCostMap()
    navigator = DWANavigator()
    safety = SafetySystem()
    runner = OrtRunner(MODEL_PATH)
    goal_world = np.array([args.goal_x, args.goal_y], dtype=np.float64)
    command = np.zeros(3, dtype=np.float32)
    desired_plan = PlanResult(0.0, 0.0, 0.0, 0.0, np.zeros((1, 3), dtype=np.float32))
    last_action = np.zeros(29, dtype=np.float32)

    def observations() -> dict[str, np.ndarray]:
        raw = {
            "base_ang_vel": np.asarray(data.sensor("imu_gyro").data, dtype=np.float32),
            "projected_gravity": projected_gravity(data.qpos[3:7]).astype(np.float32),
            "velocity_commands": command,
            "joint_pos_rel": (data.qpos[7 + contract.joint_map] - default).astype(np.float32),
            "joint_vel_rel": data.qvel[6 + contract.joint_map].astype(np.float32),
            "last_action": last_action,
        }
        return {
            name: value * np.asarray(config["observations"][name]["scale"], dtype=np.float32)
            for name, value in raw.items()
        }

    history = ObservationHistory(contract.history_lengths, observations())
    target_policy = default.copy()
    physics_dt = float(model.opt.timestep)
    policy_interval = round(contract.step_dt / physics_dt)
    perception_interval = round(0.10 / physics_dt)
    total_steps = round(args.duration / physics_dt)
    rows = []
    collided = False
    reached = False
    viewer = None
    if args.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [1.5, 0.0, 0.7]
        viewer.cam.distance = 4.0
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20
    wall_start = time.perf_counter()

    try:
        for step in range(total_steps):
            if step % perception_interval == 0:
                frame = camera.capture(model, data, body_id)
                non_ground = frame.point_geom_ids != floor_id
                costmap.update(
                    frame.points_body[non_ground],
                    ground_z_body=-float(data.xpos[body_id, 2]),
                )
                desired_plan = navigator.plan(costmap, goal_in_body(data, body_id, goal_world))

            if step % policy_interval == 0:
                roll, pitch, yaw = quaternion_euler(data.qpos[3:7])
                command[:] = safety.update(
                    desired_plan.vx,
                    desired_plan.vy,
                    desired_plan.omega,
                    contract.step_dt,
                    float(data.qpos[2]),
                    roll,
                    pitch,
                )
                history.append(observations())
                observation = history.flatten(list(contract.observation_order))
                last_action[:] = runner.run(observation)
                target_policy = contract.process_action(last_action)

            target_motor = contract.policy_to_motor(target_policy)
            torque = stiffness * (target_motor - data.qpos[7:36]) - damping * data.qvel[6:35]
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)

            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                if obstacle_id in (contact.geom1, contact.geom2):
                    collided = True

            if step % policy_interval == 0:
                roll, pitch, yaw = quaternion_euler(data.qpos[3:7])
                distance = float(np.linalg.norm(goal_world - data.qpos[:2]))
                rows.append(
                    [
                        data.time, data.qpos[0], data.qpos[1], data.qpos[2], yaw,
                        command[0], command[1], command[2], desired_plan.vx, desired_plan.vy, desired_plan.omega,
                        desired_plan.score, distance, int(collided),
                    ]
                )
                if step % (policy_interval * 50) == 0:
                    print(
                        f"t={data.time:5.1f}s pos=({data.qpos[0]:+.2f},{data.qpos[1]:+.2f}) "
                        f"cmd=({command[0]:.2f},{command[1]:+.2f},{command[2]:+.2f}) "
                        f"plan=({desired_plan.vx:.2f},{desired_plan.vy:+.2f},{desired_plan.omega:+.2f}) "
                        f"goal={distance:.2f}m contact={collided}"
                    )
                if distance < 0.30:
                    reached = True
                    break
            if data.qpos[2] < 0.35 or safety.emergency_stopped:
                break
            if viewer is not None:
                viewer.cam.lookat[0] = data.qpos[0]
                viewer.cam.lookat[1] = data.qpos[1]
                viewer.sync()
                deadline = wall_start + data.time
                if deadline > time.perf_counter():
                    time.sleep(deadline - time.perf_counter())
                if not viewer.is_running():
                    break
    finally:
        runner.close()
        if viewer is not None:
            viewer.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "time_s", "x_m", "y_m", "height_m", "yaw_rad", "cmd_vx", "cmd_vy", "cmd_omega",
                "plan_vx", "plan_vy", "plan_omega", "plan_score", "goal_distance_m", "obstacle_contact",
            ]
        )
        writer.writerows(rows)
    final_distance = rows[-1][12] if rows else math.inf
    print(
        f"结果：reached={reached} collided={collided} final_distance={final_distance:.3f}m "
        f"survived={bool(rows and rows[-1][3] >= 0.35)} output={args.output}"
    )


if __name__ == "__main__":
    main()
