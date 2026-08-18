# SPDX-License-Identifier: BSD-3-Clause
"""SO-ARM101 reach task.

Built on Isaac Lab's shared :class:`ReachEnvCfg`, retargeted from the 0.85 m-class arms it
ships with (Franka, UR10) to the SO-101's ~0.30 m envelope.

Two properties of this arm drive every override below:

1. **Reach.** The stock command ranges (``pos_x`` 0.35-0.65 m) sit entirely outside the
   SO-101's workspace, so every episode would be unsolvable. Ranges here are measured
   against the joint limits reported by ``scripts/scene_demo.py``.
2. **5 DOF.** ``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll`` cannot
   realise an arbitrary 6-DOF pose. Orientation is therefore soft-weighted, not dropped:
   keeping a small negative weight biases the wrist toward the commanded approach without
   letting orientation error dominate a position task it can never zero out.
"""

import math

import isaaclab.envs.mdp as mdp
from isaaclab.utils.configclass import configclass

from isaaclab_tasks.core.reach.reach_env_cfg import ReachEnvCfg

from so101_scene.tuning import so101_cfg  # isort: skip

# Verified against the spawned articulation, not guessed from the USD:
#   joint_names : shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
#   body_names  : base, shoulder, upper_arm, lower_arm, wrist, gripper, moving_jaw_so101_v1
# The end-effector body is "gripper" (there is also a *joint* named "gripper" — different
# namespaces). Re-run scripts/scene_demo.py if the asset ever changes.
SO101_EE_BODY = "gripper"
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]


@configclass
class SO101ReachEnvCfg(ReachEnvCfg):
    """Reach a randomised end-effector pose with the SO-ARM101."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = so101_cfg("{ENV_REGEX_NS}/Robot")

        # Seed each episode near the mid-range crouch rather than the stock (0.5, 1.5) scale.
        # A fully extended 5-DOF arm starts on its boundary singularity, where the IK-free
        # joint-space policy gets almost no usable gradient.
        self.events.reset_robot_joints.params["position_range"] = (0.9, 1.1)

        # Position is the real objective; orientation is best-effort on 5 DOF.
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [SO101_EE_BODY]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [SO101_EE_BODY]
        self.rewards.end_effector_orientation_tracking.weight = -0.02  # stock -0.1
        self.rewards.joint_vel.params["asset_cfg"].joint_names = SO101_ARM_JOINTS

        # Drive the 5 arm joints; the jaw is not part of a reach task.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=SO101_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )

        # Command targets sized to the SO-101 envelope (base at origin, table top at z=0).
        self.commands.ee_pose.body_name = SO101_EE_BODY
        self.commands.ee_pose.ranges.pos_x = (0.10, 0.26)
        self.commands.ee_pose.ranges.pos_y = (-0.14, 0.14)
        self.commands.ee_pose.ranges.pos_z = (0.08, 0.26)
        self.commands.ee_pose.ranges.pitch = (math.pi / 2, math.pi / 2)
        # 3 cm on a 30 cm arm ~= the 5 cm default on an 85 cm arm.
        self.commands.ee_pose.position_success_threshold = 0.03


@configclass
class SO101ReachEnvCfg_PLAY(SO101ReachEnvCfg):
    """Small, noise-free variant for visual inspection and policy playback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.0
        self.observations.policy.enable_corruption = False
