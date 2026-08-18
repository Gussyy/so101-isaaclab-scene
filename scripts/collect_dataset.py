# SPDX-License-Identifier: BSD-3-Clause
"""Record a LeRobot dataset by replaying a driver in the simulator.

The point of the trained PPO policy, beyond being a policy: it is an expert that never gets
tired, so demonstrations cost simulation time instead of human time. At the measured 27,000
rendered frames/s, 200 episodes is a couple of seconds of stepping.

Episodes are buffered and only written if they succeed, so a 92%-success expert yields a clean
dataset rather than one where 8% of the demonstrations teach the wrong thing.

    python scripts/collect_dataset.py --config configs/pick_place_lerobot.yaml \\
        --episodes 200 --out datasets/so101_pickplace

Then, in a venv with LeRobot:

    lerobot-train --dataset.root=datasets/so101_pickplace --policy.type=act
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Record a LeRobot dataset from the simulator.")
parser.add_argument("--config", default="configs/pick_place_lerobot.yaml")
parser.add_argument("--out", default="datasets/so101_pickplace")
parser.add_argument("--episodes", type=int, default=50, help="successful episodes to keep")
parser.add_argument("--max-steps", type=int, default=250, help="cap per episode")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--task", default="Pick up the cube and place it at the target")
parser.add_argument("--keep-failures", action="store_true", help="record unsuccessful episodes too")
parser.add_argument("--success-threshold", type=float, default=0.05,
                    help="object-to-goal distance counted as success, metres")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simbridge.builder import load_config  # noqa: E402

_cfg_probe = load_config(args_cli.config)
if (_cfg_probe.get("scene") or {}).get("cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import simbridge.sources  # noqa: F401,E402
import so101_scene  # noqa: F401,E402
from simbridge.builder import build_env_cfg, describe, resolve_task  # noqa: E402
from simbridge.lerobot_recorder import EpisodeBuffer, LeRobotRecorder  # noqa: E402
from simbridge.registry import SOURCES, lookup, register_task  # noqa: E402
from simbridge.schema import ObsPacket  # noqa: E402

register_task("pick_place", "SO101-PickPlace-v0")
register_task("pick_place_play", "SO101-PickPlace-Play-v0")
register_task("reach", "SO101-Reach-v0")


def to_packet(step: int, obs, num_envs: int) -> ObsPacket:
    state, images = {}, {}
    if isinstance(obs, dict):
        for k, v in obs.items():
            if not hasattr(v, "shape"):
                continue
            arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            (images if (arr.ndim == 4 and arr.shape[-1] in (1, 3, 4)) else state)[k] = arr
    return ObsPacket(step=step, num_envs=num_envs, state=state, images=images)


def episode_success(env, threshold: float = 0.05, min_height: float = 0.025) -> np.ndarray:
    """Per-environment success: object lifted and close to its commanded goal.

    Not `terminated`. In contrib.lift the only non-timeout termination is `object_dropping`, so
    treating `terminated` as success records exactly the failures -- which is what a first run
    here did, reporting 2% "success" for a policy measured at 92%.

    Mirrors what the reward's success_threshold checks: the goal command is expressed in the
    robot root frame, so the object position is converted before comparing.

    Must be evaluated BEFORE env.step(). ManagerBasedRLEnv calls _reset_idx() inside step(), so
    reading the scene afterwards returns the next episode's freshly-reset state and every episode
    looks like a failure.
    """
    obj = env.scene["object"]
    robot = env.scene["robot"]
    cmd = env.command_manager.get_command("object_pose")          # (N, 7) in root frame
    obj_w = obj.data.root_pos_w.torch[:, :3]
    root_w = robot.data.root_pos_w.torch[:, :3]
    root_q = robot.data.root_quat_w.torch

    from isaaclab.utils.math import quat_apply_inverse

    obj_b = quat_apply_inverse(root_q, obj_w - root_w)
    dist = torch.norm(obj_b - cmd[:, :3], dim=1)
    lifted = obj_w[:, 2] > min_height
    return ((dist < threshold) & lifted).detach().cpu().numpy()


def camera_frames(env, names: list[str], env_id: int) -> dict[str, np.ndarray]:
    """Read the rendered frames straight from the sensors.

    Not from the observation dict: whether cameras appear there depends on the task's observation
    groups, and a dataset silently missing its images is a costly thing to discover after
    collection.
    """
    out = {}
    for name in names:
        sensor = env.scene[name]
        rgb = sensor.data.output["rgb"]
        if hasattr(rgb, "torch"):
            rgb = rgb.torch
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.dtype != np.uint8:
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        out[name] = arr[env_id, ..., :3]
    return out


def main() -> None:
    cfg = load_config(args_cli.config)
    print(describe(cfg), "\n")

    cams = list((cfg.get("scene") or {}).get("cameras") or {})
    env_cfg = build_env_cfg(cfg, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(resolve_task(cfg), cfg=env_cfg).unwrapped

    action_dim = int(env.action_space.shape[-1])
    ctl = cfg.get("control") or {}
    factory = lookup(SOURCES, ctl.get("source", "zero"), "action source")
    kwargs = {k: v for k, v in ctl.items() if k not in {"source", "transport"}}
    kwargs.setdefault("action_dim", action_dim)
    kwargs.setdefault("device", args_cli.device)
    source = factory(**kwargs)

    res = None
    if cams:
        cam_cfg = (cfg["scene"]["cameras"])[cams[0]]
        r = cam_cfg.get("resolution", [128, 128])
        res = (int(r[1]), int(r[0]))

    rec = LeRobotRecorder(
        args_cli.out,
        fps=int(round(1.0 / (env_cfg.sim.dt * env_cfg.decimation))),
        task=args_cli.task,
        cameras=cams,
        state_dim=action_dim,
        image_size=res or (128, 128),
    )
    print(f"[collect] driver={source}  envs={env.num_envs}  cameras={cams or 'none'}")
    print(f"[collect] target: {args_cli.episodes} successful episodes -> {args_cli.out}\n")

    obs, _ = env.reset()
    source.reset()
    buffers = [EpisodeBuffer() for _ in range(env.num_envs)]
    # Success is latched per episode: if the object was ever lifted and within threshold of its
    # goal, the demonstration is worth keeping. Sampling only the final frame would discard
    # episodes that reached the goal and then drifted a centimetre in the last step.
    achieved = np.zeros(env.num_envs, dtype=bool)
    kept = attempted = 0
    step = 0

    while kept < args_cli.episodes and simulation_app.is_running():
        packet = to_packet(step, obs, env.num_envs)
        action = source.advance(packet)

        joint_pos = packet.state.get("joint_pos")
        for e in range(env.num_envs):
            state_vec = joint_pos[e] if joint_pos is not None else np.zeros(action_dim, dtype=np.float32)
            buffers[e].add(state_vec, action[e], camera_frames(env, cams, e) if cams else {})

        # Evaluated before the step, because the step resets whatever finished.
        achieved |= episode_success(env, threshold=args_cli.success_threshold)

        obs, _, terminated, truncated, _ = env.step(
            torch.as_tensor(action, device=env.device, dtype=torch.float32)
        )
        step += 1

        done = (terminated | truncated).detach().cpu().numpy()
        for e in np.nonzero(done)[0]:
            attempted += 1
            ok = bool(achieved[e]) or args_cli.keep_failures
            if ok and len(buffers[e]) > 1 and kept < args_cli.episodes:
                rec.add_episode(buffers[e])
                kept += 1
                if kept % 10 == 0 or kept == 1:
                    print(f"[collect] {kept}/{args_cli.episodes} kept "
                          f"({attempted} attempted, {100 * kept / max(attempted, 1):.0f}% success)", flush=True)
            buffers[e] = EpisodeBuffer()
            achieved[e] = False

        for e in range(env.num_envs):
            if len(buffers[e]) >= args_cli.max_steps:
                buffers[e] = EpisodeBuffer()
                achieved[e] = False

        if bool(np.any(done)):
            source.reset()

    root = rec.finalize()
    print(f"\n[collect] done: {rec.summary()}")
    print(f"[collect] attempted {attempted}, kept {kept} "
          f"({100 * kept / max(attempted, 1):.0f}% success rate)")
    print(f"\ntrain with LeRobot:\n    lerobot-train --dataset.root={root} --policy.type=act")
    source.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
