# SPDX-License-Identifier: BSD-3-Clause
"""Measure the SO-ARM101 grasp point.

``pick_place_env_cfg.EE_GRASP_OFFSET`` positions the end-effector frame between the jaws
rather than at the ``gripper`` body origin. That offset is a property of the asset, so it
should be measured, not guessed. This prints the ``gripper`` and ``moving_jaw_so101_v1``
body poses in the robot root frame at the default joint configuration, plus the midpoint
between them -- which is the offset to use.

Usage:
    python scripts/measure_ee.py
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure SO-ARM101 grasp point.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402

from so101_scene.tuning import so101_cfg  # noqa: E402


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0))
    cfg = so101_cfg("/World/Robot")
    robot = Articulation(cfg)
    sim.reset()

    for _ in range(10):
        sim.step()
        robot.update(sim.get_physics_dt())

    names = robot.body_names
    root = robot.data.root_pose.torch[0, :3]
    pos = robot.data.body_pose_w.torch[0, :, :3]

    print("\n" + "=" * 60)
    print("Body positions in robot root frame (default joint pose)")
    print("=" * 60)
    for i, n in enumerate(names):
        rel = (pos[i] - root).tolist()
        print(f"  {n:<24} ({rel[0]:+.4f}, {rel[1]:+.4f}, {rel[2]:+.4f})")

    if "gripper" in names and "moving_jaw_so101_v1" in names:
        g = pos[names.index("gripper")]
        j = pos[names.index("moving_jaw_so101_v1")]
        mid = ((g + j) / 2.0 - g).tolist()
        print("-" * 60)
        print(f"  jaw midpoint relative to 'gripper' body:")
        print(f"    EE_GRASP_OFFSET = ({mid[0]:+.4f}, {mid[1]:+.4f}, {mid[2]:+.4f})")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
