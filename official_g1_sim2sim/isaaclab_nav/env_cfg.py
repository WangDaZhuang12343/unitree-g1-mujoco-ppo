"""Isaac Lab configuration for the frozen-policy G1 navigation task."""

from __future__ import annotations

import os
from pathlib import Path

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils import configclass

from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG, UnitreeUrdfFileCfg

from isaaclab_nav.walking_compatibility import POLICY_TRAINING_PROFILE, apply_actuator_profile


def _default_walking_policy() -> str:
    import unitree_rl_lab

    source_root = Path(unitree_rl_lab.__file__).resolve().parents[3]
    return str(
        source_root
        / "deploy/robots/g1_29dof/config/policy/velocity/v0/exported/policy.onnx"
    )


def _official_g1_robot_cfg() -> ArticulationCfg:
    """Use Unitree's checked-out URDF; upstream's optional USD path is a placeholder."""

    import unitree_rl_lab

    unitree_rl_lab_root = Path(unitree_rl_lab.__file__).resolve().parents[3]
    unitree_ros = Path(
        os.environ.get("UNITREE_ROS_PATH", unitree_rl_lab_root.parent / "unitree_ros")
    )
    urdf = unitree_ros / "robots/g1_description/g1_29dof_rev_1_0.urdf"
    if not urdf.is_file():
        raise FileNotFoundError(f"Set UNITREE_ROS_PATH; missing official G1 URDF: {urdf}")
    cfg = UNITREE_G1_29DOF_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    cfg.spawn = UnitreeUrdfFileCfg(asset_path=str(urdf))
    # The official collision meshes overlap during the frozen walking gait.
    # Keep the external asset untouched, but prevent those internal contacts
    # from being misreported as navigation collisions with terrain obstacles.
    cfg.spawn.articulation_props.enabled_self_collisions = False
    return cfg


NAVIGATION_TERRAIN_CFG = TerrainGeneratorCfg(
    seed=17,
    curriculum=False,
    size=(8.0, 8.0),
    border_width=10.0,
    num_rows=4,
    num_cols=8,
    color_scheme="random",
    difficulty_range=(0.0, 1.0),
    sub_terrains={
        "boxes": terrain_gen.MeshRepeatedBoxesTerrainCfg(
            proportion=1.0,
            object_params_start=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=4, height=0.45, size=(0.35, 0.35)
            ),
            object_params_end=terrain_gen.MeshRepeatedBoxesTerrainCfg.ObjectCfg(
                num_objects=10, height=0.80, size=(0.70, 0.70)
            ),
            platform_width=1.6,
            platform_height=0.0,
            flat_patch_sampling={
                "goal": terrain_gen.FlatPatchSamplingCfg(
                    num_patches=32,
                    patch_radius=0.45,
                    x_range=(2.5, 3.5),
                    y_range=(-1.0, 1.0),
                    z_range=(-0.05, 0.05),
                    max_height_diff=0.02,
                )
            },
        )
    },
)


@configclass
class G1VisualNavigationEnvCfg(DirectRLEnvCfg):
    """Ten-Hz upper policy over the immutable 50-Hz Unitree walking policy."""

    episode_length_s = 20.0
    decimation = 20
    action_space = 3
    observation_space = 483
    state_space = 0

    sim: SimulationCfg = SimulationCfg(
        dt=0.005,
        render_interval=decimation,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    scene: InteractiveSceneCfg = InteractiveSceneCfg(
        num_envs=32, env_spacing=8.0, replicate_physics=True
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=NAVIGATION_TERRAIN_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim.physics_material,
        debug_vis=False,
    )

    robot: ArticulationCfg = _official_g1_robot_cfg()
    contact_sensor = ContactSensorCfg(
        prim_path="/World/envs/env_.*/Robot/.*",
        history_length=3,
        update_period=sim.dt,
        track_air_time=False,
    )
    forward_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/torso_link",
        update_period=decimation * sim.dt,
        offset=RayCasterCfg.OffsetCfg(
            pos=(0.0576235, 0.01753, 0.42987),
            # LidarPatternCfg already uses the robotics x-forward frame. The
            # D435 URDF rotation describes a camera link and would pitch these
            # rays another 47.6 degrees into the floor.
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            # Cover both the high-mounted camera's near-field blind spot and
            # forward obstacles. Twenty channels keep low boxes visible before
            # the hips reach them without making the GPU ray count excessive.
            channels=20,
            vertical_fov_range=(-75.0, 8.0),
            horizontal_fov_range=(-55.0, 55.0),
            horizontal_res=5.0,
        ),
        max_distance=5.0,
        mesh_prim_paths=["/World/ground"],
        debug_vis=False,
    )

    walking_policy_path: str = _default_walking_policy()
    # The released ONNX was trained at unitree_rl_lab e3c0fe4. Upstream later
    # replaced its actuator limits without updating the policy artifact.
    walking_actuator_profile: str = POLICY_TRAINING_PROFILE
    walking_action_scale: float = 0.25
    walking_decimation: int = 4
    goal_range: float = 4.0
    success_radius: float = 0.30
    collision_force_threshold: float = 20.0
    minimum_clearance: float = 0.30
    obstacle_min_height: float = 0.035
    obstacle_max_height: float = 1.50
    ground_height_tolerance: float = 0.05
    distance_field_max_clearance: float = 5.0
    costmap_rear_range: float = 0.5
    costmap_front_range: float = 4.0
    costmap_side_range: float = 2.0
    costmap_inflation_radius: float = 0.28
    enable_dwa_fallback: bool = False
    dwa_fallback_clearance_trigger: float = 0.22
    # Empty during training. Benchmark scripts populate this tuple and replace
    # only the terrain generator; the navigation/walking architecture is shared.
    benchmark_scenarios: tuple[str, ...] = ()
    benchmark_seed: int = 7

    # A collision must cost more than all progress available in the longest
    # 6.5 m frozen benchmark scene. Otherwise rushing toward the goal and
    # crashing is a positive-return shortcut (6.5 * 30 - 20 previously).
    progress_reward_scale: float = 15.0
    success_reward: float = 100.0
    collision_penalty: float = -150.0
    fall_penalty: float = -150.0
    clearance_penalty_scale: float = -5.0
    action_rate_penalty_scale: float = -0.05
    action_bound_penalty_scale: float = -0.25
    timeout_penalty: float = -10.0

    def __post_init__(self):
        apply_actuator_profile(self.robot, self.walking_actuator_profile)
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        if self.decimation % self.walking_decimation:
            raise ValueError("upper decimation must be a multiple of walking_decimation")
