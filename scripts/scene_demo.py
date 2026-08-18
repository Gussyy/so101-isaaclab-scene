# SPDX-License-Identifier: BSD-3-Clause
"""Standalone SO-ARM101 scene: ground, light, arm on a pedestal, and a graspable cube.

Prints the articulation's real joint/body names on startup (ground truth for writing
task configs), then drives a sine sweep across the arm while cycling the gripper.

Usage:
    python scripts/scene_demo.py --steps 600                 # headless, bounded
    python scripts/scene_demo.py --steps 0 --viz kit         # interactive, runs until closed
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-ARM101 scene demo.")
parser.add_argument("--num_envs", type=int, default=4, help="Number of environments to spawn.")
parser.add_argument("--steps", type=int, default=600, help="Physics steps to run; 0 = until window closed.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math  # noqa: E402

import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.configclass import configclass  # noqa: E402

from isaaclab_assets.robots.so101 import SO101_CFG  # isort: skip

# The SO-101 base column sits slightly below its root origin; lift the whole arm onto a
# pedestal so the workspace is above the ground plane rather than intersecting it.
# ponytail: pedestal is a plain cuboid, not a USD table — one less remote asset to fetch.
PEDESTAL_H = 0.10


@configclass
class SO101SceneCfg(InteractiveSceneCfg):
    """Ground + light + SO-ARM101 + a cube within reach."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    pedestal = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Pedestal",
        spawn=sim_utils.CuboidCfg(
            size=(0.16, 0.16, PEDESTAL_H),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.25, 0.25, 0.28)),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, PEDESTAL_H / 2.0)),
    )

    robot: ArticulationCfg = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, PEDESTAL_H),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": -0.6,
            "elbow_flex": 0.8,
            "wrist_flex": 0.6,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
    )

    cube = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.9, 0.3, 0.2)),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.22, 0.0, 0.015)),
    )


def describe(robot) -> None:
    """Dump the articulation's real names/limits — the ground truth for task configs."""
    print("\n" + "=" * 62)
    print("SO-ARM101 articulation")
    print("=" * 62)
    print(f"  num_instances : {robot.num_instances}")
    print(f"  DOF           : {robot.num_joints}")
    print(f"  joint_names   : {robot.joint_names}")
    print(f"  body_names    : {robot.body_names}")
    limits = robot.data.joint_pos_limits.torch[0]
    for name, (lo, hi) in zip(robot.joint_names, limits.tolist()):
        print(f"    {name:<16} [{lo:+.3f}, {hi:+.3f}] rad")
    print("=" * 62 + "\n")


def run(sim: SimulationContext, scene: InteractiveScene, max_steps: int) -> None:
    robot = scene["robot"]
    describe(robot)

    sim_dt = sim.get_physics_dt()
    default_pos = robot.data.default_joint_pos.torch.clone()
    # Amplitude per joint: sweep the arm, and give the gripper its own cycle.
    amp = torch.zeros_like(default_pos)
    names = robot.joint_names
    for j, n in enumerate(names):
        amp[:, j] = {"shoulder_pan": 0.5, "shoulder_lift": 0.3, "elbow_flex": 0.3, "wrist_flex": 0.3}.get(n, 0.0)
    grip = names.index("gripper") if "gripper" in names else None

    count = 0
    while simulation_app.is_running():
        if max_steps and count >= max_steps:
            print(f"[INFO] Completed {count} steps — exiting.")
            break

        if count % 400 == 0:
            root_pose = robot.data.default_root_pose.torch.clone()
            root_pose[:, :3] += scene.env_origins
            robot.write_root_pose_to_sim_index(root_pose=root_pose)
            robot.write_root_velocity_to_sim_index(root_velocity=robot.data.default_root_vel.torch.clone())
            robot.write_joint_position_to_sim_index(position=default_pos.clone())
            robot.write_joint_velocity_to_sim_index(velocity=robot.data.default_joint_vel.torch.clone())
            scene.reset()
            print(f"[INFO] step {count}: reset scene")

        phase = 2.0 * math.pi * count / 240.0
        target = default_pos + amp * math.sin(phase)
        if grip is not None:
            # Gripper jaw is [~0, 1.745] rad; cycle open->closed at half the arm's rate.
            target[:, grip] = 0.87 * (1.0 - math.cos(phase * 0.5))
        robot.set_joint_position_target_index(target=target)

        scene.write_data_to_sim()
        sim.step()
        count += 1
        scene.update(sim_dt)

        if count % 200 == 0:
            ee = robot.data.joint_pos.torch[0]
            print(f"[step {count:4d}] joint_pos[env0] = " + " ".join(f"{v:+.3f}" for v in ee.tolist()))


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0))
    sim.set_camera_view(eye=[0.9, 0.9, 0.6], target=[0.0, 0.0, 0.15])
    scene = InteractiveScene(SO101SceneCfg(num_envs=args_cli.num_envs, env_spacing=1.2))
    sim.reset()
    print("[INFO] Scene ready.")
    run(sim, scene, args_cli.steps)


if __name__ == "__main__":
    main()
    simulation_app.close()
