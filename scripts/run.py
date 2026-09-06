# SPDX-License-Identifier: BSD-3-Clause
"""Run any scene/task described by a YAML config, driven by any registered action source.

    python scripts/run.py --config configs/pick_place.yaml
    python scripts/run.py --config configs/pick_place.yaml --set control.source=zero
    python scripts/run.py --config configs/pick_place_teleop.yaml    # needs a policy server

The point of this script is that it contains no task-specific logic. Swapping the robot, the
props, the cameras, or what drives the arm is a config edit; this file does not change.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run a YAML-described Isaac Lab scene.")
parser.add_argument("--config", required=True, help="path to a scene/task YAML")
parser.add_argument("--set", action="append", default=[], metavar="a.b=c",
                    help="dotted override, repeatable (e.g. --set control.source=zero)")
parser.add_argument("--steps", type=int, default=500, help="env steps to run; 0 = until closed")
parser.add_argument("--num_envs", type=int, default=None, help="override scene.num_envs")
parser.add_argument("--describe", action="store_true", help="print the resolved config and exit")
parser.add_argument("--log", default="logs/vr",
                    help="directory for a per-session JSONL of every step (action, grasp pose, joints); "
                         "written whenever the driver is remote, since that is a session worth keeping")
parser.add_argument("--no-log", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simbridge.builder import describe, load_config, resolve_task  # noqa: E402
from simbridge.registry import register_task  # noqa: E402

# Friendly YAML names -> registered Gym ids.
register_task("pick_place", "SO101-PickPlace-v0")
register_task("pick_place_play", "SO101-PickPlace-Play-v0")
register_task("reach", "SO101-Reach-v0")
register_task("reach_play", "SO101-Reach-Play-v0")


def _apply_overrides(cfg: dict, pairs: list[str]) -> dict:
    """Apply dotted key=value overrides, coercing scalars via YAML so types stay natural."""
    import yaml as _yaml
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"--set expects key=value, got {pair!r}")
        key, raw = pair.split("=", 1)
        node = cfg
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = _yaml.safe_load(raw)
    return cfg


cfg = _apply_overrides(load_config(args_cli.config), args_cli.set)
if args_cli.describe:
    print(describe(cfg))
    raise SystemExit(0)

# Cameras in the config mean the renderer must be on before the app launches.
if (cfg.get("scene") or {}).get("cameras"):
    args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101_scene  # noqa: F401,E402  (registers the tasks)
import simbridge.sources  # noqa: F401,E402  (registers the sources)
from simbridge.builder import build_env_cfg  # noqa: E402
from simbridge.registry import SOURCES, lookup  # noqa: E402
from simbridge.schema import ObsPacket  # noqa: E402


def make_source(ctl: dict, action_dim: int, device: str):
    """Instantiate the driver named by control.source."""
    name = ctl.get("source", "zero")
    factory = lookup(SOURCES, name, "action source")
    kwargs = {k: v for k, v in ctl.items() if k not in {"source", "transport", "actions", "ik_orientation_weight"}}
    kwargs.setdefault("action_dim", action_dim)
    kwargs.setdefault("device", device)
    return factory(**kwargs)


def to_packet(step: int, obs, num_envs: int) -> ObsPacket:
    """Isaac Lab observation dict -> the wire schema.

    An entry is treated as an image when it is 4-D with a trailing channel count of 1/3/4;
    everything else is state. Keeping the split here means a state-only driver never pays to
    decode frames it will not look at.
    """
    state, images = {}, {}
    if isinstance(obs, dict):
        for k, v in obs.items():
            if not hasattr(v, "shape"):
                continue
            arr = v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else np.asarray(v)
            is_image = arr.ndim == 4 and arr.shape[-1] in (1, 3, 4)
            (images if is_image else state)[k] = arr
    return ObsPacket(step=step, num_envs=num_envs, state=state, images=images)


def main() -> None:
    print(describe(cfg), "\n")
    env_cfg = build_env_cfg(cfg, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(resolve_task(cfg), cfg=env_cfg).unwrapped

    shape = env.action_space.shape
    action_dim = int(shape[-1])
    source = make_source(cfg.get("control") or {}, action_dim, args_cli.device)
    print(f"[run] driver: {source}  action_dim={action_dim}  envs={env.num_envs}\n")

    obs, _ = env.reset()
    source.reset()
    step = 0
    # Declared cameras ride the packet too, every second step. The observation manager does not
    # know about scene cameras unless a term names them; reading the sensor directly is five
    # lines where an observation group would be a new config surface.
    cams = list(((cfg.get("scene") or {}).get("cameras") or {}).keys())
    # So does the grasp point's pose, in the root frame: a teleop bridge anchors the operator's
    # orientation to where the fingers actually point, which only the simulator knows.
    from isaaclab.utils.math import subtract_frame_transforms

    # One grasp frame per arm, each in its own robot's root frame: ee_pose for the first,
    # ee_pose2 for a second arm (scene.robot2).
    arms = [(env.scene[r], env.scene[f], key) for r, f, key in
            (("robot", "ee_frame", "ee_pose"), ("robot2", "ee_frame2", "ee_pose2"))
            if f in env.scene.keys()]

    log = None
    if not args_cli.no_log and (cfg.get("control") or {}).get("source") == "zmq":
        import json
        import time
        from pathlib import Path

        log_dir = Path(args_cli.log)
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-sim.jsonl"
        log = open(log_path, "w", encoding="utf-8")
        print(f"[run] session log: {log_path}")
    t0 = __import__("time").time()
    rate_t0 = t0

    try:
        while simulation_app.is_running():
            if args_cli.steps and step >= args_cli.steps:
                print(f"[run] completed {step} steps")
                break
            packet = to_packet(step, obs, env.num_envs)
            if cams and step % 2 == 0:
                for name in cams:
                    rgb = env.scene[name].data.output.get("rgb")
                    if rgb is not None:
                        packet.images[name] = rgb[:1, ..., :3].detach().cpu().numpy()
            for robot, ee, key in arms:
                p_b, q_b = subtract_frame_transforms(
                    robot.data.root_pos_w.torch, robot.data.root_quat_w.torch,
                    ee.data.target_pos_w.torch[:, 0], ee.data.target_quat_w.torch[:, 0],
                )
                packet.state[key] = torch.cat([p_b, q_b], dim=-1).detach().cpu().numpy()
            action = source.advance(packet)
            was_reset = getattr(source, "last_reset", None) is not None
            if was_reset:
                # The operator asked for a fresh scene (B on the VR controller). Whole-env reset:
                # the task's per-env reset needs ids the socket does not carry, and one env is
                # the teleop case anyway.
                source.last_reset = None
                obs, _ = env.reset()
                source.reset()
                print(f"[run] scene reset at step {step} (operator)", flush=True)
            else:
                obs, _, terminated, truncated, _ = env.step(
                    torch.as_tensor(action, device=env.device, dtype=torch.float32)
                )
            if log is not None:
                rec = {"t": round(__import__("time").time() - t0, 4), "step": step, "reset": was_reset,
                       "action": np.asarray(action)[0].round(5).tolist(),
                       "joint_pos": robot.data.joint_pos.torch[0].detach().cpu().numpy().round(5).tolist()}
                if "ee_pose" in packet.state:
                    rec["ee_pose"] = np.asarray(packet.state["ee_pose"])[0].round(5).tolist()
                log.write(json.dumps(rec) + chr(10))
            if was_reset:
                step += 1
                continue
            if bool(torch.any(terminated | truncated)):
                source.reset()
            step += 1
            if step % 100 == 0:
                now = __import__("time").time()
                rate = 100.0 / max(1e-9, now - rate_t0); rate_t0 = now
                print(f"[run] step {step}  {rate:5.1f} steps/s", flush=True)
    finally:
        if log is not None:
            log.close()
        source.close()
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
