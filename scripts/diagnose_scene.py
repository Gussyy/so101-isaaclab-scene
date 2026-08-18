# SPDX-License-Identifier: BSD-3-Clause
"""Print real world-frame geometry of a task scene.

Written after a 261-iteration pick-and-place run scored exactly 0.0 on every task reward
term: the console shows a flat curve but not *why*. This prints where the robot base, the
end-effector, the object and the command target actually are, which turns "it will not
learn" into a number you can act on.

Usage:
    python scripts/diagnose_scene.py --task SO101-PickPlace-v0 --num_envs 2
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump task scene geometry.")
parser.add_argument("--task", default="SO101-PickPlace-v0")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import so101_scene  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def main() -> None:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    for _ in range(5):
        env.step(torch.zeros(env.action_space.shape, device=env.device))

    robot = env.scene["robot"]
    obj = env.scene["object"]
    origin = env.scene.env_origins[0]

    base = robot.data.root_pos_w.torch[0] - origin
    bodies = robot.body_names
    ee_i = bodies.index("gripper")
    ee = robot.data.body_pos_w.torch[0, ee_i] - origin
    op = obj.data.root_pos_w.torch[0] - origin

    print("\n" + "=" * 68)
    print(f"{args_cli.task}  (env-local frame)")
    print("=" * 68)
    print(f"  robot base       ({base[0]:+.4f}, {base[1]:+.4f}, {base[2]:+.4f})")
    print(f"  ee 'gripper'     ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
    print(f"  object           ({op[0]:+.4f}, {op[1]:+.4f}, {op[2]:+.4f})")
    print(f"  |ee - object|    {torch.norm(ee - op).item():.4f} m   <-- reaching_object std is 0.04")
    print(f"  |base - object|  {torch.norm(base - op).item():.4f} m   <-- SO-101 reach is ~0.30 m")

    if "ee_frame" in env.scene.keys():
        tgt = env.scene["ee_frame"].data.target_pos_w.torch[0, 0] - origin
        print(f"  ee_frame target  ({tgt[0]:+.4f}, {tgt[1]:+.4f}, {tgt[2]:+.4f})")
        print(f"  |ee_frame - obj| {torch.norm(tgt - op).item():.4f} m")

    try:
        cmd = env.command_manager.get_command("object_pose")[0]
        print(f"  command (root frame) ({cmd[0]:+.4f}, {cmd[1]:+.4f}, {cmd[2]:+.4f})")
    except Exception as exc:  # noqa: BLE001
        print(f"  command: unavailable ({exc})")

    for name, asset in env.scene.rigid_objects.items():
        p = asset.data.root_pos_w.torch[0] - origin
        print(f"  [rigid] {name:<12} ({p[0]:+.4f}, {p[1]:+.4f}, {p[2]:+.4f})")
    print("=" * 68 + "\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
