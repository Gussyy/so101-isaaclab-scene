# SPDX-License-Identifier: BSD-3-Clause
"""Render one config to a video clip, for documentation.

The README claims that editing a YAML line changes the scene. This is what turns that claim into
something a reader can check: run the same script against two configs that differ by one line and
compare the two clips. If the claim ever stops being true, the clips stop matching the configs.

    python scripts/capture_clip.py --config configs/variants/goal_centre.yaml \\
        --out docs/variants/goal_centre.mp4 --steps 200

Renders environment 0 of the first declared camera. Give that camera a documentation-sized
resolution (640x480); 128px training cameras look like mud at video scale.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Render a config to a video clip.")
parser.add_argument("--config", required=True)
parser.add_argument("--out", required=True)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--settle", type=int, default=8, help="steps to drop before recording")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--camera", default=None, help="camera name; defaults to the first declared")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import simbridge.sources  # noqa: F401,E402
import so101_scene  # noqa: F401,E402
from simbridge.builder import build_env_cfg, load_config, resolve_task  # noqa: E402
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


def frame(env, name: str) -> np.ndarray:
    rgb = env.scene[name].data.output["rgb"]
    if hasattr(rgb, "torch"):          # Isaac Lab 3.0 hands back a ProxyArray
        rgb = rgb.torch
    arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
    return arr[0, ..., :3]


def main() -> None:
    cfg = load_config(args_cli.config)
    cams = list((cfg.get("scene") or {}).get("cameras") or {})
    if not cams:
        raise SystemExit(f"{args_cli.config} declares no cameras; nothing to render")
    cam = args_cli.camera or cams[0]

    env_cfg = build_env_cfg(cfg, device=args_cli.device, num_envs=1)
    env = gym.make(resolve_task(cfg), cfg=env_cfg).unwrapped

    ctl = cfg.get("control") or {}
    kwargs = {k: v for k, v in ctl.items() if k not in {"source", "transport"}}
    kwargs.setdefault("action_dim", int(env.action_space.shape[-1]))
    kwargs.setdefault("device", args_cli.device)
    source = lookup(SOURCES, ctl.get("source", "zero"), "action source")(**kwargs)

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # macro_block_size=1 keeps 640x480 at 640x480 instead of padding to a multiple of 16.
    writer = imageio.get_writer(out, fps=args_cli.fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_log_level="error")

    obs, _ = env.reset()
    source.reset()
    print(f"[capture] {args_cli.config} -> {out}  camera={cam}  steps={args_cli.steps}")
    try:
        for step in range(args_cli.steps + args_cli.settle):
            action = source.advance(to_packet(step, obs, env.num_envs))
            obs, _, terminated, truncated, _ = env.step(
                torch.as_tensor(action, device=env.device, dtype=torch.float32)
            )
            # Frame 0 is often black -- the renderer has not produced anything yet.
            if step >= args_cli.settle:
                writer.append_data(frame(env, cam))
            if bool((terminated | truncated).any()):
                source.reset()
    finally:
        writer.close()

    print(f"[capture] wrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    source.close()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
