# SPDX-License-Identifier: BSD-3-Clause
"""Dump one frame per camera so the views can be inspected before any policy is trained.

Kill check 0a. A visuomotor policy cannot recover from a camera that does not see the cube: if
it spans only a few pixels in the scene view, or the wrist camera is framed inside the jaw, no
amount of training fixes it -- and the failure looks like a bad policy rather than a bad camera.
Twenty seconds here versus a night of training against an uninformative image.

Also prints per-camera pixel statistics. A frame that is constant (std ~ 0) is a camera pointing
at nothing, which is easy to miss by eye in a small thumbnail.

Usage:
    python scripts/dump_camera_views.py --config configs/pick_place.yaml --num_envs 4
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Dump camera views for inspection.")
parser.add_argument("--config", default="configs/pick_place.yaml")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--settle", type=int, default=12, help="steps to run before capturing")
parser.add_argument("--out", default="docs/camera_views")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import so101_scene  # noqa: F401,E402
from simbridge.builder import build_env_cfg, load_config, resolve_task  # noqa: E402
from simbridge.registry import register_task  # noqa: E402

register_task("pick_place", "SO101-PickPlace-v0")
register_task("reach", "SO101-Reach-v0")


def save_png(path: Path, img: np.ndarray) -> None:
    """Write RGB uint8 without adding an image dependency beyond what is installed."""
    try:
        import imageio.v3 as iio

        iio.imwrite(path, img)
        return
    except Exception:
        pass
    from PIL import Image

    Image.fromarray(img).save(path)


def main() -> None:
    cfg = load_config(args_cli.config)
    cams = list((cfg.get("scene") or {}).get("cameras") or {})
    if not cams:
        raise SystemExit(f"{args_cli.config} declares no cameras; nothing to dump")

    env_cfg = build_env_cfg(cfg, device=args_cli.device, num_envs=args_cli.num_envs)
    env = gym.make(resolve_task(cfg), cfg=env_cfg).unwrapped
    env.reset()

    # Let the cube settle and the renderer produce a real frame; frame 0 is often black.
    action = torch.zeros(env.action_space.shape, device=env.device)
    for _ in range(args_cli.settle):
        env.step(action)

    out = Path(args_cli.out)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"Camera views  ({args_cli.config}, after {args_cli.settle} steps)")
    print("=" * 72)

    for name in cams:
        sensor = env.scene[name]
        rgb = sensor.data.output["rgb"]
        # Isaac Lab 3.0 hands back a ProxyArray, not a torch.Tensor, so an isinstance check
        # falls through to np.asarray() and dies on a CUDA tensor. Go via .torch when present.
        if hasattr(rgb, "torch"):
            rgb = rgb.torch
        arr = rgb.detach().cpu().numpy() if hasattr(rgb, "detach") else np.asarray(rgb)
        if arr.dtype != np.uint8:  # normalize=True configs hand back float
            arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)
        arr = arr[..., :3]

        for env_id in range(min(2, arr.shape[0])):
            frame = arr[env_id]
            path = out / f"{name}_env{env_id}.png"
            save_png(path, frame)
            print(
                f"  {name:<12} env{env_id}  shape={frame.shape}  "
                f"mean={frame.mean():6.1f}  std={frame.std():6.1f}  "
                f"min={frame.min():3d} max={frame.max():3d}  -> {path}"
            )
            if frame.std() < 1.0:
                print(f"    WARNING: {name} env{env_id} is nearly constant -- camera may see nothing")

    print("=" * 72)
    print("Check by eye: is the cube visible and more than a few pixels across?")
    print("The wrist view should look along the jaw at the workspace, not into the gripper body.\n")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
