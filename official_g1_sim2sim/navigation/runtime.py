"""可被单场景和批量 Benchmark 共用的导航仿真运行时。"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from types import SimpleNamespace
import time
from typing import Callable

import mujoco
import numpy as np
import yaml

from camera import GroundSegmenter, SimulatedDepthCamera
from costmap import LocalCostMap
from g1_nav.l6_safety import SafetySystem
from g1_nav.policy_contract import ObservationHistory, PolicyContract
from planner import DWANavigator, PlanResult
from simulate import CONFIG_PATH, MODEL_PATH, OrtRunner, build_model, projected_gravity, quaternion_euler

from .run_log import write_navigation_log
from .scenarios import NavigationScenario, ObstacleBox


@dataclass(frozen=True)
class NavigationRunConfig:
    scenario: NavigationScenario
    duration: float | None = None
    output: Path | None = None
    viewer: bool = False
    realtime: bool = False
    run_id: str = "run_001"
    debug_callback: Callable[["NavigationDebugFrame"], bool | None] | None = None


@dataclass(frozen=True)
class NavigationRunResult:
    scenario: str
    run_id: str
    success: bool
    reached: bool
    survived: bool
    collision_count: int
    elapsed_sim_s: float
    elapsed_wall_s: float
    path_length_m: float
    min_clearance_m: float
    average_velocity_mps: float
    cpu_usage_percent: float
    fps: float
    depth_fps: float
    planning_fps: float
    average_planning_ms: float
    final_distance_m: float
    output: Path | None


@dataclass(frozen=True)
class NavigationDebugFrame:
    time_s: float
    depth: np.ndarray
    points_body: np.ndarray
    obstacle_points_body: np.ndarray
    ground_normal: tuple[float, float, float]
    ground_offset: float
    ground_inlier_ratio: float
    ground_plane_cached: bool
    obstacle_map: np.ndarray
    distance_field: np.ndarray
    map_extent: tuple[float, float, float, float]
    candidate_trajectories: np.ndarray
    candidate_valid: np.ndarray
    selected_trajectory: np.ndarray
    goal_body: tuple[float, float]
    robot_world_pose: tuple[float, float, float]
    command: tuple[float, float, float]
    planning_ms: float


def goal_in_body(data: mujoco.MjData, body_id: int, goal_world: np.ndarray) -> tuple[float, float]:
    rotation = data.xmat[body_id].reshape(3, 3)
    relative_world = np.array(
        [goal_world[0] - data.xpos[body_id, 0], goal_world[1] - data.xpos[body_id, 1], 0.0]
    )
    relative_body = rotation.T @ relative_world
    return float(relative_body[0]), float(relative_body[1])


def obstacle_clearance(
    robot_xy: np.ndarray, obstacles: tuple[ObstacleBox, ...], time_s: float, robot_radius: float = 0.22
) -> float:
    if not obstacles:
        return math.inf
    minimum = math.inf
    for obstacle in obstacles:
        center_x, center_y = obstacle.center_at(time_s)
        dx = max(abs(float(robot_xy[0]) - center_x) - obstacle.size_x / 2.0, 0.0)
        dy = max(abs(float(robot_xy[1]) - center_y) - obstacle.size_y / 2.0, 0.0)
        minimum = min(minimum, math.hypot(dx, dy) - robot_radius)
    return minimum


def _set_dynamic_obstacles(
    model: mujoco.MjModel, geom_ids: dict[str, int], obstacles: tuple[ObstacleBox, ...], time_s: float
) -> None:
    for obstacle in obstacles:
        if obstacle.velocity_y == 0.0:
            continue
        geom_id = geom_ids[obstacle.name]
        x, y = obstacle.center_at(time_s)
        model.geom_pos[geom_id, 0] = x
        model.geom_pos[geom_id, 1] = y


def run_navigation(
    run_config: NavigationRunConfig, policy_runner: OrtRunner | None = None
) -> NavigationRunResult:
    scenario = run_config.scenario
    duration = float(run_config.duration or scenario.duration)
    terrain_args = SimpleNamespace(
        terrain="flat",
        obstacle_height=0.15,
        obstacle_x=2.5,
        obstacle_width=0.8,
        slope_angle=5.0,
        roughness=0.02,
        terrain_seed=7,
        scene_obstacles=scenario.obstacles,
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
    obstacle_geom_ids = {
        obstacle.name: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"course_{obstacle.name}")
        for obstacle in scenario.obstacles
    }
    obstacle_geom_set = set(obstacle_geom_ids.values())
    if -1 in obstacle_geom_set:
        raise RuntimeError(f"场景 {scenario.name} 存在未创建的障碍物 geom")

    camera = SimulatedDepthCamera()
    camera.isolate_world_geoms(model)
    ground_segmenter = GroundSegmenter()
    costmap = LocalCostMap()
    navigator = DWANavigator()
    safety = SafetySystem()
    runner = policy_runner or OrtRunner(MODEL_PATH)
    owns_runner = policy_runner is None
    goal_world = np.asarray(scenario.goal, dtype=np.float64)
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
    total_steps = round(duration / physics_dt)
    rows: list[list[float]] = []
    positions: list[np.ndarray] = [data.qpos[:2].copy()]
    active_collisions: set[int] = set()
    collision_count = 0
    reached = False
    viewer = None
    perception_frames = 0
    planning_times: list[float] = []
    latest_planning_ms = 0.0
    latest_control_latency_ms = 0.0
    latest_plan_ready_at: float | None = None
    collision_events_since_log = 0
    min_clearance = obstacle_clearance(data.qpos[:2], scenario.obstacles, 0.0)
    completed_steps = 0
    wall_start = time.perf_counter()
    cpu_start = time.process_time()

    if run_config.viewer:
        from mujoco import viewer as mj_viewer

        viewer = mj_viewer.launch_passive(model, data)
        viewer.opt.geomgroup[1] = 1
        viewer.cam.lookat[:] = [1.5, 0.0, 0.7]
        viewer.cam.distance = 4.5
        viewer.cam.azimuth = 135
        viewer.cam.elevation = -20

    try:
        debug_running = True
        for step in range(total_steps):
            completed_steps = step + 1
            if scenario.dynamic:
                _set_dynamic_obstacles(model, obstacle_geom_ids, scenario.obstacles, float(data.time))
                mujoco.mj_forward(model, data)

            planning_ms = 0.0
            if step % perception_interval == 0:
                planning_start = time.perf_counter()
                frame = camera.capture(model, data, body_id)
                segmentation = ground_segmenter.segment(
                    frame.points_body,
                    projected_gravity(data.qpos[3:7]),
                )
                costmap.update(
                    segmentation.obstacle_points,
                    ground_plane=(segmentation.plane.normal, segmentation.plane.offset),
                )
                goal_body = goal_in_body(data, body_id, goal_world)
                desired_plan = navigator.plan(
                    costmap, goal_body, collect_debug=run_config.debug_callback is not None
                )
                planning_ms = (time.perf_counter() - planning_start) * 1000.0
                latest_planning_ms = planning_ms
                latest_plan_ready_at = time.perf_counter()
                planning_times.append(planning_ms)
                perception_frames += 1
                if run_config.debug_callback is not None:
                    planning_debug = navigator.last_debug
                    if planning_debug is None:
                        raise RuntimeError("调试模式未生成 DWA 候选轨迹")
                    _, _, debug_yaw = quaternion_euler(data.qpos[3:7])
                    debug_running = run_config.debug_callback(NavigationDebugFrame(
                        time_s=float(data.time),
                        depth=frame.depth.copy(),
                        points_body=frame.points_body.copy(),
                        obstacle_points_body=segmentation.obstacle_points.copy(),
                        ground_normal=tuple(float(value) for value in segmentation.plane.normal),
                        ground_offset=segmentation.plane.offset,
                        ground_inlier_ratio=segmentation.inlier_ratio,
                        ground_plane_cached=segmentation.used_cached_plane,
                        obstacle_map=costmap.obstacle_map.copy(),
                        distance_field=costmap.distance_field.copy(),
                        map_extent=(
                            -costmap.config.rear_range, costmap.config.front_range,
                            -costmap.config.side_range, costmap.config.side_range,
                        ),
                        candidate_trajectories=planning_debug.trajectories.copy(),
                        candidate_valid=planning_debug.valid.copy(),
                        selected_trajectory=desired_plan.trajectory.copy(),
                        goal_body=goal_body,
                        robot_world_pose=(float(data.qpos[0]), float(data.qpos[1]), debug_yaw),
                        command=(float(command[0]), float(command[1]), float(command[2])),
                        planning_ms=planning_ms,
                    )) is not False
                    if not debug_running:
                        break

            if step % policy_interval == 0:
                roll, pitch, yaw = quaternion_euler(data.qpos[3:7])
                command[:] = safety.update(
                    desired_plan.vx, desired_plan.vy, desired_plan.omega,
                    contract.step_dt, float(data.qpos[2]), roll, pitch,
                )
                if latest_plan_ready_at is not None:
                    latest_control_latency_ms = (
                        time.perf_counter() - latest_plan_ready_at
                    ) * 1000.0
                    latest_plan_ready_at = None
                history.append(observations())
                observation = history.flatten(list(contract.observation_order))
                last_action[:] = runner.run(observation)
                target_policy = contract.process_action(last_action)

            target_motor = contract.policy_to_motor(target_policy)
            torque = stiffness * (target_motor - data.qpos[7:36]) - damping * data.qvel[6:35]
            data.ctrl[:] = np.clip(torque, model.actuator_ctrlrange[:, 0], model.actuator_ctrlrange[:, 1])
            mujoco.mj_step(model, data)

            current_collisions: set[int] = set()
            for contact_index in range(data.ncon):
                contact = data.contact[contact_index]
                for geom_id in (contact.geom1, contact.geom2):
                    if geom_id in obstacle_geom_set:
                        current_collisions.add(geom_id)
            new_collision_events = len(current_collisions - active_collisions)
            collision_count += new_collision_events
            collision_events_since_log += new_collision_events
            active_collisions = current_collisions

            clearance = obstacle_clearance(data.qpos[:2], scenario.obstacles, float(data.time))
            min_clearance = min(min_clearance, clearance)
            if step % policy_interval == 0:
                _, _, yaw = quaternion_euler(data.qpos[3:7])
                distance = float(np.linalg.norm(goal_world - data.qpos[:2]))
                wall_elapsed = max(time.perf_counter() - wall_start, 1e-9)
                rotation_world_from_body = data.xmat[body_id].reshape(3, 3)
                velocity_body = rotation_world_from_body.T @ data.qvel[:3]
                positions.append(data.qpos[:2].copy())
                rows.append([
                    data.time, data.qpos[0], data.qpos[1], data.qpos[2], yaw,
                    command[0], command[1], command[2], desired_plan.vx, desired_plan.vy,
                    desired_plan.omega, velocity_body[0], velocity_body[1],
                    data.sensor("imu_gyro").data[2], desired_plan.score, distance, clearance,
                    perception_frames / wall_elapsed, len(planning_times) / wall_elapsed,
                    latest_planning_ms, latest_control_latency_ms,
                    collision_events_since_log, collision_count,
                ])
                collision_events_since_log = 0
                if distance < 0.30:
                    reached = True
                    break
            if data.qpos[2] < 0.35 or safety.emergency_stopped:
                break
            if viewer is not None:
                viewer.cam.lookat[:] = [data.qpos[0], data.qpos[1], max(0.6, data.qpos[2] * 0.65)]
                viewer.sync()
                if not viewer.is_running():
                    break
            if run_config.realtime:
                deadline = wall_start + data.time
                if deadline > time.perf_counter():
                    time.sleep(deadline - time.perf_counter())
    finally:
        if owns_runner:
            runner.close()
        if viewer is not None:
            viewer.close()

    elapsed_wall = max(time.perf_counter() - wall_start, 1e-9)
    elapsed_cpu = time.process_time() - cpu_start
    elapsed_sim = float(data.time)
    survived = bool(data.qpos[2] >= 0.35 and not safety.emergency_stopped)
    final_distance = float(np.linalg.norm(goal_world - data.qpos[:2]))
    path_length = float(
        sum(np.linalg.norm(second - first) for first, second in zip(positions, positions[1:]))
    )
    output = run_config.output
    if output is not None:
        write_navigation_log(output, rows)
    return NavigationRunResult(
        scenario=scenario.name,
        run_id=run_config.run_id,
        success=bool(reached and survived and collision_count == 0),
        reached=reached,
        survived=survived,
        collision_count=collision_count,
        elapsed_sim_s=elapsed_sim,
        elapsed_wall_s=elapsed_wall,
        path_length_m=path_length,
        min_clearance_m=float(min_clearance),
        average_velocity_mps=path_length / max(elapsed_sim, 1e-9),
        cpu_usage_percent=100.0 * elapsed_cpu / elapsed_wall,
        fps=completed_steps / elapsed_wall,
        depth_fps=perception_frames / elapsed_wall,
        planning_fps=len(planning_times) / elapsed_wall,
        average_planning_ms=float(np.mean(planning_times)) if planning_times else 0.0,
        final_distance_m=final_distance,
        output=output,
    )
