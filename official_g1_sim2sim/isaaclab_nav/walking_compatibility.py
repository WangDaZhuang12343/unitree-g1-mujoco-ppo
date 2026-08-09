"""Versioned Isaac actuator profiles for the immutable released Walking ONNX."""

from __future__ import annotations

import copy


WALKING_ONNX_SHA256 = "610c27e463a8f666aa50a06346678c00b4df3859f10b54bcc1f817c28251406f"
POLICY_RELEASE_COMMIT = "e3c0fe49b1e33e8fb985a4b43aaf9f93f94e3a7a"
POLICY_TRAINING_EFFORT_LIMITS = {"legs": 300, "feet": 20, "arms": 300}


def policy_training_actuators() -> dict[str, object]:
    """Return the G1 actuator config at the commit where the ONNX first appeared."""
    from isaaclab.actuators import ImplicitActuatorCfg

    return {
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_roll_joint",
                ".*_hip_yaw_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
                "waist_.*_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw_joint": 100.0,
                ".*_hip_roll_joint": 100.0,
                ".*_hip_pitch_joint": 100.0,
                ".*_knee_joint": 150.0,
                "waist_.*_joint": 200.0,
            },
            damping={
                ".*_hip_yaw_joint": 2.0,
                ".*_hip_roll_joint": 2.0,
                ".*_hip_pitch_joint": 2.0,
                ".*_knee_joint": 4.0,
                "waist_.*_joint": 5.0,
            },
            armature={
                ".*_hip_.*": 0.01,
                ".*_knee_joint": 0.01,
                "waist_.*_joint": 0.01,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit_sim=20,
            stiffness=40.0,
            damping=2.0,
            armature=0.01,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim=300,
            velocity_limit_sim=100.0,
            stiffness=40.0,
            damping=10.0,
            armature={
                ".*_shoulder_.*": 0.01,
                ".*_elbow_.*": 0.01,
                ".*_wrist_.*": 0.01,
            },
        ),
    }


def apply_actuator_profile(robot_cfg, profile: str) -> None:
    """Apply a named profile without changing the official policy artifact."""
    if profile == "current":
        from unitree_rl_lab.assets.robots.unitree import UNITREE_G1_29DOF_CFG

        robot_cfg.actuators = copy.deepcopy(UNITREE_G1_29DOF_CFG.actuators)
        return
    if profile == "policy_training_2025_07":
        robot_cfg.actuators = policy_training_actuators()
        return
    raise ValueError(f"unknown actuator profile: {profile}")
