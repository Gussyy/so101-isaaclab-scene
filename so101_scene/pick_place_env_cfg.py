# SPDX-License-Identifier: BSD-3-Clause
"""SO-ARM101 pick-and-place task.

Built on Isaac Lab's ``contrib.lift`` LiftEnvCfg, whose staged reward already encodes
pick-and-place: approach the object, lift it clear of the table, then carry it to a
commanded pose.

Retargeting from the Franka reference is not just a robot swap -- the SO-101 is a ~0.30 m,
5-DOF, single-jaw arm where the Franka is a ~0.85 m, 7-DOF, parallel-jaw one. Every
constant below is resized accordingly; see the notes on each block.
"""

from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg, NewtonCollisionPipelineCfg, NewtonShapeCfg
from isaaclab_physx.physics import PhysxCfg

from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.physics import PhysxAutoCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.utils.configclass import configclass

import isaaclab.sim as sim_utils
from isaaclab.sim import RigidBodyMaterialCfg

from isaaclab_tasks.utils import PresetCfg

from isaaclab_tasks.contrib.lift import mdp
from isaaclab_tasks.contrib.lift.lift_env_cfg import LiftEnvCfg

from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
from so101_scene.tuning import so101_cfg  # isort: skip

# Verified by spawning the articulation (scripts/scene_demo.py), not read off the USD.
SO101_EE_BODY = "gripper"
SO101_ARM_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

# Jaw travel, from the reported joint limits [-0.175, +1.745] rad. "Open" is deliberately
# short of the hard stop so the controller is not fighting the limit while approaching.
GRIPPER_OPEN = 1.20
GRIPPER_CLOSE = 0.0

# A 2.5 cm cube. The Franka reference lifts a 4 cm DexCube, which the SO-101's single jaw
# cannot span -- object size is the difference between a task that trains and one that
# silently never achieves a grasp.
CUBE_SIZE = 0.025

# Grasp-point offset from the ``gripper`` body origin to the midpoint between the jaws,
# expressed in the gripper's LOCAL frame (which is what OffsetCfg wants). Measured with
# scripts/measure_ee.py, not guessed -- the world-frame delta is (+0.0094, -0.0125,
# +0.0091), so using it directly would put the grasp frame on the wrong axis.
EE_GRASP_OFFSET = (0.0101, 0.0094, -0.0117)

# SO101_CFG's default joint pose is all zeros, which extends the arm straight up and along
# -y -- 0.36 m from a cube sitting 0.18 m away in +x. With reaching_object's std of 0.04 that
# scores identically 0.0, so PPO sees no gradient at all and collapses (measured: 261
# iterations, every task reward term exactly 0.0, policy std 1.00 -> 0.03).
#
# Two corrections, both taken from Isaac Lab's own SO-101 stack task:
#   * a mid-range crouch, so the gripper starts above the workspace pointing down rather
#     than fully extended on its boundary singularity;
#   * a 90-deg base yaw, so the arm faces +x where the cube and command targets live.
# NOTE rot is (x, y, z, w) in Isaac Lab 3.0 -- the quaternion convention changed from wxyz
# in 2.x, and silently feeding a wxyz value here yaws the base to the wrong place.
SO101_INIT_JOINT_POS = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}
SO101_BASE_ROT = (0.0, 0.0, 0.70710678, 0.70710678)



@configclass
class PickPlacePhysicsCfg(PresetCfg):
    """Physics backends for SO-101 pick-and-place.

    ``contrib.lift`` hardcodes a single PhysX config, so ``physics=newton_mjwarp`` is not
    selectable on it out of the box. This preset restores backend choice.

    The Newton solver numbers are taken from Isaac Lab's own SO-101 *stack* task rather
    than from the reach preset: grasping is contact-rich, so ``njmax``/``nconmax`` are 3x
    and 10x the reach values, and the friction cone is elliptic with a high ``impratio``
    because a single-jaw pinch grasp lives or dies on friction fidelity. Reach's
    ``njmax=100, nconmax=20, cone="pyramidal"`` will drop contacts here.

    ``default`` is PhysX, matching upstream's own choice for SO-101 manipulation -- Newton
    grasping on this arm is newer and less validated. Select it explicitly with
    ``physics=newton_mjwarp``.
    """

    isaacsim_physx = PhysxCfg(
        bounce_threshold_velocity=0.01,
        gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
        gpu_total_aggregate_pairs_capacity=2**21,
        friction_correlation_distance=0.00625,
    )
    newton_mjwarp = NewtonCfg(
        solver_cfg=MJWarpSolverCfg(
            solver="newton",
            integrator="implicitfast",
            njmax=300,
            nconmax=200,
            impratio=10.0,
            cone="elliptic",
            update_data_interval=2,
            iterations=100,
            ls_iterations=15,
            ls_parallel=False,
            use_mujoco_contacts=False,
            ccd_iterations=35,
        ),
        collision_cfg=NewtonCollisionPipelineCfg(),
        default_shape_cfg=NewtonShapeCfg(),
        num_substeps=2,
        debug_mode=False,
        # Replays the stepping kernels from a captured graph instead of relaunching each
        # one per step: fewer CPU-side launches per frame, which is where a 20-thread CPU
        # feeding thousands of envs actually bottlenecks.
        use_cuda_graph=True,
    )
    physx = PhysxAutoCfg(isaacsim_physx=isaacsim_physx)
    default = isaacsim_physx


@configclass
class SO101CubePickPlaceEnvCfg(LiftEnvCfg):
    """Pick a cube off the table and place it at a commanded pose."""

    def __post_init__(self) -> None:
        super().__post_init__()

        # Restore backend choice; contrib.lift pins PhysX unconditionally.
        self.sim.physics = PickPlacePhysicsCfg()

        self.scene.robot = so101_cfg("{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state = ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            rot=SO101_BASE_ROT,
            joint_pos=SO101_INIT_JOINT_POS,
        )

        # 5 arm joints drive the pose; the jaw is a separate binary open/close action.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=SO101_ARM_JOINTS,
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": GRIPPER_OPEN},
            close_command_expr={"gripper": GRIPPER_CLOSE},
        )
        self.commands.object_pose.body_name = SO101_EE_BODY

        # Place targets inside the SO-101 envelope. The Franka defaults (x 0.4-0.6,
        # z 0.25-0.5) are entirely unreachable for a 0.30 m arm.
        self.commands.object_pose.ranges.pos_x = (0.15, 0.25)
        self.commands.object_pose.ranges.pos_y = (-0.10, 0.10)
        self.commands.object_pose.ranges.pos_z = (0.06, 0.16)

        # High static friction: a single-jaw gripper pinches rather than encloses, so the
        # grasp is friction-limited. A low-friction cube simply squirts out of the jaw.
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.20, 0.0, CUBE_SIZE / 2.0], rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(CUBE_SIZE, CUBE_SIZE, CUBE_SIZE),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.25, 0.15)),
                physics_material=RigidBodyMaterialCfg(static_friction=1.2, dynamic_friction=1.0),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.015),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        # Spawn jitter shrunk to the reachable patch of table in front of the arm.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.03, 0.03),
            "y": (-0.06, 0.06),
            "z": (0.0, 0.0),
        }

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.04, 0.04, 0.04)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/{SO101_EE_BODY}",
                    name="end_effector",
                    offset=OffsetCfg(pos=EE_GRASP_OFFSET),
                ),
            ],
        )

        # Shaping width tracks the *starting* EE-object distance, not the arm length.
        # Measured start distance is 0.177 m (scripts/diagnose_scene.py), where
        # 1 - tanh(d/std) gives: std=0.04 -> 0.0003, 0.08 -> 0.024, 0.10 -> 0.056.
        # An earlier 0.04 -- picked by scaling Franka's 0.1 down for a 3x smaller arm --
        # left the term at 0.0003 for the whole run, i.e. no gradient, and PPO collapsed.
        self.rewards.reaching_object.params["std"] = 0.10
        self.rewards.lifting_object.params["minimal_height"] = 0.025
        self.rewards.object_goal_tracking.params["minimal_height"] = 0.025
        self.rewards.object_goal_tracking.params["std"] = 0.12
        self.rewards.object_goal_tracking.params["success_threshold"] = 0.03
        self.rewards.object_goal_tracking_fine_grained.params["minimal_height"] = 0.025
        self.rewards.object_goal_tracking_fine_grained.params["std"] = 0.03

        self.rewards.joint_vel.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=SO101_ARM_JOINTS)

        # Drop detection: table top is z=0 in this scene, so -0.05 means it fell off.
        self.terminations.object_dropping.params["minimum_height"] = -0.05


@configclass
class SO101CubePickPlaceEnvCfg_PLAY(SO101CubePickPlaceEnvCfg):
    """Small, noise-free variant for visual inspection and policy playback."""

    def __post_init__(self) -> None:
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.0
        self.observations.policy.enable_corruption = False
