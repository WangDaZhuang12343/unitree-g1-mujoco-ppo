"""Direct Isaac Lab environment preserving the frozen G1 walking layer."""

from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import euler_xyz_from_quat, quat_apply_inverse, subtract_frame_transforms

from isaaclab_nav.contracts import BatchSafety, UpperObservationHistory
from isaaclab_nav.env_cfg import G1VisualNavigationEnvCfg
from isaaclab_nav.fallback import BatchedDwaFallback
from isaaclab_nav.perception import LocalDistanceFieldConfig, local_distance_field
from isaaclab_nav.walking import FrozenWalkingPolicy
from navigation.scenarios import get_scenario


class G1VisualNavigationEnv(DirectRLEnv):
    """Visual navigation policy → Safety → frozen ONNX → 29 joint targets."""

    cfg: G1VisualNavigationEnvCfg

    def __init__(self, cfg: G1VisualNavigationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._walking_policy = FrozenWalkingPolicy(cfg.walking_policy_path)
        self._safety = BatchSafety(self.num_envs, self.device)
        self._upper_history = UpperObservationHistory(self.num_envs, self.device)
        self._dwa_fallback = BatchedDwaFallback(self.num_envs)
        self._distance_field_cfg = LocalDistanceFieldConfig(
            rear_range=cfg.costmap_rear_range,
            front_range=cfg.costmap_front_range,
            side_range=cfg.costmap_side_range,
            inflation_radius=cfg.costmap_inflation_radius,
            max_clearance=cfg.distance_field_max_clearance,
        )

        self._desired_command = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_upper_action = torch.zeros_like(self._desired_command)
        self._walking_action = torch.zeros((self.num_envs, 29), device=self.device)
        self._walking_history = {
            "base_ang_vel": torch.zeros((self.num_envs, 5, 3), device=self.device),
            "projected_gravity": torch.zeros((self.num_envs, 5, 3), device=self.device),
            "velocity_commands": torch.zeros((self.num_envs, 5, 3), device=self.device),
            "joint_pos_rel": torch.zeros((self.num_envs, 5, 29), device=self.device),
            "joint_vel_rel": torch.zeros((self.num_envs, 5, 29), device=self.device),
            "last_action": torch.zeros((self.num_envs, 5, 29), device=self.device),
        }
        self._walking_tick = 0
        self._goal_pos_w = torch.zeros((self.num_envs, 3), device=self.device)
        self._previous_goal_distance = torch.zeros(self.num_envs, device=self.device)
        self._step_progress = torch.zeros(self.num_envs, device=self.device)
        self._success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._collision = torch.zeros_like(self._success)
        self._fall = torch.zeros_like(self._success)
        self._fallback_used = torch.zeros_like(self._success)
        self._invalid_action = torch.zeros_like(self._success)
        self._collision_count = torch.zeros(
            self.num_envs, dtype=torch.int32, device=self.device
        )
        self._termination_reason = torch.zeros(
            self.num_envs, dtype=torch.int8, device=self.device
        )
        self._undesired_body_ids, _ = self._contact_sensor.find_bodies(["(?!.*ankle.*).*"])
        self._undesired_body_ids_tensor = torch.tensor(
            self._undesired_body_ids, dtype=torch.long, device=self.device
        )
        self._collision_force = torch.zeros(self.num_envs, device=self.device)
        self._collision_body_id = torch.full(
            (self.num_envs,), -1, dtype=torch.int32, device=self.device
        )
        self._episode_sums = {
            name: torch.zeros(self.num_envs, device=self.device)
            for name in (
                "progress",
                "success",
                "collision",
                "fall",
                "clearance",
                "action_rate",
                "action_bound",
            )
        }

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot
        self._contact_sensor = ContactSensor(self.cfg.contact_sensor)
        self.scene.sensors["contact_sensor"] = self._contact_sensor
        self._forward_scanner = RayCaster(self.cfg.forward_scanner)
        self.scene.sensors["forward_scanner"] = self._forward_scanner

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        invalid = ~torch.isfinite(actions).all(dim=1)
        self._raw_upper_action = torch.nan_to_num(actions.clone())
        normalized = self._raw_upper_action.clamp(-1.0, 1.0)
        desired = torch.empty_like(normalized)
        desired[:, 0] = 0.225 * (normalized[:, 0] + 1.0)
        desired[:, 1] = 0.10 * normalized[:, 1]
        desired[:, 2] = 0.20 * normalized[:, 2]
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        _, _, clearance, _, _ = self._perception()
        close_obstacle = (
            clearance < self.cfg.dwa_fallback_clearance_trigger
        ) & (desired[:, 0] > 0.1)
        self._fallback_used = (invalid | close_obstacle) & self.cfg.enable_dwa_fallback
        self._invalid_action = invalid & ~self._fallback_used
        if self._fallback_used.any():
            hits = self._forward_scanner.data.ray_hits_w
            relative = hits - self._robot.data.root_pos_w[:, None, :]
            quat = self._robot.data.root_quat_w[:, None, :].expand(-1, relative.shape[1], -1)
            points_body = quat_apply_inverse(quat, relative)
            goal_body, _ = subtract_frame_transforms(
                self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._goal_pos_w
            )
            relative_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
            fallback = self._dwa_fallback.plan(
                points_body, -relative_height, goal_body, self._fallback_used
            )
            desired[self._fallback_used] = fallback[self._fallback_used]
        if not self.cfg.enable_dwa_fallback:
            desired[invalid] = torch.nan
        self._desired_command = self._safety.update(
            desired,
            self.step_dt,
            self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2],
            roll,
            pitch,
        )
        self._previous_upper_action, self._current_upper_action = (
            getattr(self, "_current_upper_action", torch.zeros_like(normalized)),
            normalized,
        )

    def _walking_terms(self) -> dict[str, torch.Tensor]:
        return {
            "base_ang_vel": self._robot.data.root_ang_vel_b * 0.2,
            "projected_gravity": self._robot.data.projected_gravity_b,
            "velocity_commands": self._desired_command,
            "joint_pos_rel": self._robot.data.joint_pos - self._robot.data.default_joint_pos,
            "joint_vel_rel": self._robot.data.joint_vel * 0.05,
            "last_action": self._walking_action,
        }

    def _fill_walking_history(self, env_ids: torch.Tensor) -> None:
        terms = self._walking_terms()
        for name, value in terms.items():
            self._walking_history[name][env_ids] = value[env_ids, None, :]

    def _append_walking_history(self) -> torch.Tensor:
        terms = self._walking_terms()
        for name, value in terms.items():
            history = self._walking_history[name]
            history[:, :-1] = history[:, 1:].clone()
            history[:, -1] = value
        observation = torch.cat(
            [
                self._walking_history[name].flatten(1)
                for name in (
                    "base_ang_vel",
                    "projected_gravity",
                    "velocity_commands",
                    "joint_pos_rel",
                    "joint_vel_rel",
                    "last_action",
                )
            ],
            dim=1,
        )
        if observation.shape != (self.num_envs, 480):
            raise RuntimeError(f"walking observation contract changed: {observation.shape}")
        return observation

    def _apply_action(self):
        if self._walking_tick % self.cfg.walking_decimation == 0:
            self._walking_action = self._walking_policy(self._append_walking_history())
        self._walking_tick += 1
        target = (
            self._robot.data.default_joint_pos
            + self.cfg.walking_action_scale * self._walking_action
        )
        self._robot.set_joint_position_target(target)

    def _perception(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        hits = self._forward_scanner.data.ray_hits_w
        valid = torch.isfinite(hits).all(dim=-1)
        relative = hits - self._robot.data.root_pos_w[:, None, :]
        quat = self._robot.data.root_quat_w[:, None, :].expand(-1, relative.shape[1], -1)
        points_body = quat_apply_inverse(quat, relative)
        height_w = hits[..., 2] - self._terrain.env_origins[:, None, 2]
        obstacle = (
            valid
            & (height_w >= self.cfg.obstacle_min_height)
            & (height_w <= self.cfg.obstacle_max_height)
            & (points_body[..., 0] >= -self.cfg.costmap_rear_range)
            & (points_body[..., 0] <= self.cfg.costmap_front_range)
            & (points_body[..., 1].abs() <= self.cfg.costmap_side_range)
        )
        distance_field, clearance = local_distance_field(
            points_body, obstacle, self._distance_field_cfg
        )
        ground = valid & (height_w.abs() <= self.cfg.ground_height_tolerance)
        valid_count = valid.sum(dim=1).clamp(min=1)
        ground_ratio = ground.sum(dim=1).float() / valid_count
        plane_ready = ground.sum(dim=1) >= 3
        return distance_field, valid, clearance, ground_ratio, plane_ready

    def _distance_field(self) -> tuple[torch.Tensor, torch.Tensor]:
        distance_field, valid, _, _, _ = self._perception()
        return distance_field, valid

    def _goal_body(self) -> tuple[torch.Tensor, torch.Tensor]:
        goal_b, _ = subtract_frame_transforms(
            self._robot.data.root_pos_w, self._robot.data.root_quat_w, self._goal_pos_w
        )
        distance = torch.linalg.norm(goal_b[:, :2], dim=1)
        goal = torch.stack(
            (
                goal_b[:, 0] / self.cfg.goal_range,
                goal_b[:, 1] / self.cfg.goal_range,
                distance / self.cfg.goal_range,
            ),
            dim=1,
        ).clamp(-1.0, 1.0)
        return goal, distance

    def _upper_frame(self) -> torch.Tensor:
        distance_field, valid, _, ground_ratio, plane_ready = self._perception()
        roll, pitch, _ = euler_xyz_from_quat(self._robot.data.root_quat_w)
        relative_height = self._robot.data.root_pos_w[:, 2] - self._terrain.env_origins[:, 2]
        safety_state = torch.stack((relative_height, roll, pitch), dim=1)
        perception_health = torch.stack(
            (
                valid.float().mean(dim=1),
                ground_ratio,
                plane_ready.float(),
            ),
            dim=1,
        )
        progress = torch.stack(
            (self._step_progress, self.episode_length_buf / self.max_episode_length), dim=1
        )
        return self._upper_history.make_frame(
            distance_field,
            self._robot.data.root_lin_vel_b,
            self._safety.command,
            self._robot.data.root_ang_vel_b,
            self._robot.data.projected_gravity_b,
            safety_state,
            perception_health,
            progress,
        )

    def _get_observations(self) -> dict[str, torch.Tensor]:
        frame = self._upper_frame()
        self._upper_history.append(frame)
        goal, _ = self._goal_body()
        return {"policy": self._upper_history.observation(goal)}

    def _contact_collision(self) -> torch.Tensor:
        forces = self._contact_sensor.data.net_forces_w_history
        magnitude = torch.linalg.norm(forces[:, :, self._undesired_body_ids], dim=-1)
        self._collision_force, flat_index = magnitude.flatten(1).max(dim=1)
        body_subset_index = flat_index % len(self._undesired_body_ids)
        self._collision_body_id = self._undesired_body_ids_tensor[body_subset_index].to(
            torch.int32
        )
        return self._collision_force > self.cfg.collision_force_threshold

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        _, distance = self._goal_body()
        self._collision = self._contact_collision()
        self._fall = self._safety.emergency_stopped & ~self._collision & ~self._invalid_action
        self._success = (
            (distance <= self.cfg.success_radius)
            & ~self._collision
            & ~self._fall
            & ~self._invalid_action
            & (self._collision_count == 0)
        )
        self._collision_count += self._collision.to(torch.int32)
        terminated = self._success | self._collision | self._fall | self._invalid_action
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        self._termination_reason.zero_()
        self._termination_reason[self._success] = 1
        self._termination_reason[self._collision] = 2
        self._termination_reason[self._fall] = 3
        self._termination_reason[time_out & ~terminated] = 4
        self._termination_reason[self._invalid_action] = 5
        self.extras["success"] = self._success.clone()
        self.extras["goal_distance"] = distance.clone()
        self.extras["fallback_used"] = self._fallback_used.clone()
        self.extras["collision_count"] = self._collision_count.clone()
        self.extras["collision_force"] = self._collision_force.clone()
        self.extras["collision_body_id"] = self._collision_body_id.clone()
        self.extras["termination_reason"] = self._termination_reason.clone()
        return terminated, time_out

    def _get_rewards(self) -> torch.Tensor:
        _, distance = self._goal_body()
        self._step_progress = self._previous_goal_distance - distance
        self._previous_goal_distance = distance
        _, _, clearance, _, _ = self._perception()
        parts = {
            "progress": self.cfg.progress_reward_scale * self._step_progress,
            "success": self.cfg.success_reward * self._success,
            "collision": self.cfg.collision_penalty * self._collision,
            "fall": self.cfg.fall_penalty * self._fall,
            "clearance": self.cfg.clearance_penalty_scale
            * (self.cfg.minimum_clearance - clearance).clamp(min=0.0),
            "action_rate": self.cfg.action_rate_penalty_scale
            * torch.square(self._current_upper_action - self._previous_upper_action).sum(dim=1),
            "action_bound": self.cfg.action_bound_penalty_scale
            * torch.square((self._raw_upper_action.abs() - 1.0).clamp(min=0.0)).sum(dim=1),
        }
        for name, value in parts.items():
            self._episode_sums[name] += value
        timeout = (self.episode_length_buf >= self.max_episode_length - 1) & ~self._success
        return torch.stack(tuple(parts.values())).sum(dim=0) + self.cfg.timeout_penalty * timeout

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        root_state = self._robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        count = len(env_ids)
        if self.cfg.benchmark_scenarios:
            terrain_types = self._terrain.terrain_types[env_ids]
            goals = torch.tensor(
                [
                    (*get_scenario(name, self.cfg.benchmark_seed).goal, 0.05)
                    for name in self.cfg.benchmark_scenarios
                ],
                dtype=torch.float32,
                device=self.device,
            )
            self._goal_pos_w[env_ids] = self._terrain.env_origins[env_ids] + goals[terrain_types]
        else:
            valid_goals = self._terrain.flat_patches["goal"]
            patch_ids = torch.randint(valid_goals.shape[2], (count,), device=self.device)
            self._goal_pos_w[env_ids] = valid_goals[
                self._terrain.terrain_levels[env_ids],
                self._terrain.terrain_types[env_ids],
                patch_ids,
            ]
            self._goal_pos_w[env_ids, 2] += 0.05
        self._forward_scanner.reset(env_ids)
        self._contact_sensor.reset(env_ids)
        self._safety.reset(env_ids)
        self._walking_action[env_ids] = 0.0
        self._desired_command[env_ids] = 0.0
        self._fill_walking_history(env_ids)
        self._success[env_ids] = False
        self._collision[env_ids] = False
        self._fall[env_ids] = False
        self._fallback_used[env_ids] = False
        self._invalid_action[env_ids] = False
        self._collision_count[env_ids] = 0
        self._collision_force[env_ids] = 0.0
        self._collision_body_id[env_ids] = -1
        self._termination_reason[env_ids] = 0
        self._dwa_fallback.reset(env_ids)
        _, distance = self._goal_body()
        self._previous_goal_distance[env_ids] = distance[env_ids]
        self._step_progress[env_ids] = 0.0

        frame = self._upper_frame()
        self._upper_history.reset(frame, env_ids)
        self.extras["log"] = {}
        for name, values in self._episode_sums.items():
            self.extras["log"][f"Episode_Reward/{name}"] = values[env_ids].mean()
            values[env_ids] = 0.0
