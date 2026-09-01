# SPDX-License-Identifier: BSD-3-Clause
"""Print real world-frame geometry of a task scene.

Written after a 261-iteration pick-and-place run scored exactly 0.0 on every task reward
term: the console shows a flat curve but not *why*. This prints where the robot base, the
end-effector, the object and the command target actually are, which turns "it will not
learn" into a number you can act on.

Takes a registered task id, or a YAML config -- a config-built scene is the one whose geometry
is easiest to get wrong, since nothing in the YAML says where anything ends up.

Usage:
    python scripts/diagnose_scene.py --task SO101-PickPlace-v0 --num_envs 2
    python scripts/diagnose_scene.py --config configs/variants/goal_centre.yaml
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump task scene geometry.")
parser.add_argument("--task", default="SO101-PickPlace-v0")
parser.add_argument("--config", default=None, help="YAML config; overrides --task")
parser.add_argument("--num_envs", type=int, default=2)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Cameras in the config mean the renderer must be on before the app launches.
if args_cli.config:
    from simbridge.builder import load_config as _load

    if (_load(args_cli.config).get("scene") or {}).get("cameras"):
        args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import so101_scene  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

from simbridge.builder import build_env_cfg, load_config, resolve_task  # noqa: E402
from simbridge.registry import register_task  # noqa: E402

register_task("pick_place", "SO101-PickPlace-v0")
register_task("pick_place_play", "SO101-PickPlace-Play-v0")
register_task("reach", "SO101-Reach-v0")


def main() -> None:
    if args_cli.config:
        raw = load_config(args_cli.config)
        task = resolve_task(raw)
        cfg = build_env_cfg(raw, device=args_cli.device, num_envs=args_cli.num_envs)
    else:
        task = args_cli.task
        cfg = parse_env_cfg(task, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(task, cfg=cfg).unwrapped
    env.reset()
    for _ in range(5):
        env.step(torch.zeros(env.action_space.shape, device=env.device))

    robot = env.scene["robot"]
    obj = env.scene["object"]
    origin = env.scene.env_origins[0]

    base = robot.data.root_pos_w.torch[0] - origin
    bodies = robot.body_names
    # Do not hardcode the end-effector name. so101 calls it "gripper"; so101_full calls it
    # "gripper_base" and has no body called "gripper" at all, so assuming one crashes on a robot
    # the config is perfectly entitled to name.
    ee_name = next((n for n in ("gripper", "gripper_base") if n in bodies), None)
    if ee_name is None:
        raise SystemExit(f"no end-effector body found; robot has: {bodies}")
    ee_i = bodies.index(ee_name)
    ee = robot.data.body_pos_w.torch[0, ee_i] - origin
    op = obj.data.root_pos_w.torch[0] - origin

    print("\n" + "=" * 68)
    print(f"{args_cli.config or args_cli.task}  (env-local frame)")
    print("=" * 68)
    print(f"  robot base       ({base[0]:+.4f}, {base[1]:+.4f}, {base[2]:+.4f})")
    print(f"  ee {ee_name!r:<14} ({ee[0]:+.4f}, {ee[1]:+.4f}, {ee[2]:+.4f})")
    print(f"  object           ({op[0]:+.4f}, {op[1]:+.4f}, {op[2]:+.4f})")
    print(f"  |ee - object|    {torch.norm(ee - op).item():.4f} m   <-- reaching_object std is 0.04")
    print(f"  |base - object|  {torch.norm(base - op).item():.4f} m   <-- SO-101 reach is ~0.30 m")

    if "ee_frame" in env.scene.keys():
        tgt = env.scene["ee_frame"].data.target_pos_w.torch[0, 0] - origin
        print(f"  ee_frame target  ({tgt[0]:+.4f}, {tgt[1]:+.4f}, {tgt[2]:+.4f})")
        print(f"  |ee_frame - obj| {torch.norm(tgt - op).item():.4f} m")

    q = robot.data.root_quat_w.torch[0]
    print(f"  base rot (xyzw)  ({q[0]:+.4f}, {q[1]:+.4f}, {q[2]:+.4f}, {q[3]:+.4f})")

    try:
        cmd = env.command_manager.get_command("object_pose")[0]
        print(f"  command (root frame) ({cmd[0]:+.4f}, {cmd[1]:+.4f}, {cmd[2]:+.4f})")
        # The same target in the frame the config's `spawn` and the camera's `look_at` use.
        # These are different frames whenever the base is rotated, which is the single easiest
        # thing to get wrong in a config: a goal written like a table coordinate lands elsewhere.
        from isaaclab.utils.math import quat_apply

        goal_w = quat_apply(q.unsqueeze(0), cmd[:3].unsqueeze(0))[0] + robot.data.root_pos_w.torch[0] - origin
        print(f"  command (env frame)  ({goal_w[0]:+.4f}, {goal_w[1]:+.4f}, {goal_w[2]:+.4f})")
        print(f"  |object - goal|  {torch.norm(op - goal_w).item():.4f} m")
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
