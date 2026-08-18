# SPDX-License-Identifier: BSD-3-Clause
"""Measure rendered-observation throughput for the SO-ARM101 pick-and-place scene.

State-only simulation runs at ~120k steps/s with 8192 envs on this machine. A visuomotor
policy needs rendered frames, and rendering is the step that decides whether image-based
training is feasible here at all -- so it gets measured on the actual GPU rather than
estimated from someone else's benchmark.

Attaches a TiledCamera (scene view) and optionally a wrist camera to the existing
SO101-PickPlace env, then reports env-steps/s, frames/s and peak VRAM.

Usage:
    python scripts/bench_camera.py --counts 32,128,512 --res 128
    python scripts/bench_camera.py --counts 64,256 --res 224 --wrist
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Benchmark camera rendering throughput.")
parser.add_argument("--counts", default="32,128,512", help="comma-separated num_envs values")
parser.add_argument("--res", type=int, default=128, help="square camera resolution")
parser.add_argument("--steps", type=int, default=60, help="timed steps per point")
parser.add_argument("--warmup", type=int, default=15, help="untimed warmup steps")
parser.add_argument("--wrist", action="store_true", help="also attach a wrist camera")
parser.add_argument("--task", default="SO101-PickPlace-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Rendering requires the Kit renderer; force it on regardless of --viz.
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import subprocess  # noqa: E402
import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402

import so101_scene  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402


def vram_mib() -> int:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
        return int(out)
    except Exception:
        return 0


def scene_camera(res: int) -> TiledCameraCfg:
    """Third-person view of the workspace, framed on the cube and the arm."""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/SceneCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.55, 0.0, 0.35), rot=(0.0, 0.259, 0.0, 0.966), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, clipping_range=(0.01, 3.0)),
        width=res,
        height=res,
    )


def wrist_camera(res: int) -> TiledCameraCfg:
    """Eye-in-hand view. Diffusion Policy uses scene + wrist; the wrist view carries most
    of the fine alignment signal near grasp."""
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper/WristCam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, -0.04, 0.02), rot=(0.5, -0.5, 0.5, 0.5), convention="opengl"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=(0.01, 1.5)),
        width=res,
        height=res,
    )


def run_one(n: int, res: int) -> dict:
    cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=n)
    cfg.scene.scene_cam = scene_camera(res)
    ncam = 1
    if args_cli.wrist:
        cfg.scene.wrist_cam = wrist_camera(res)
        ncam = 2

    env = gym.make(args_cli.task, cfg=cfg).unwrapped
    env.reset()
    action = torch.zeros(env.action_space.shape, device=env.device)

    for _ in range(args_cli.warmup):
        env.step(action)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(args_cli.steps):
        env.step(action)
    torch.cuda.synchronize()
    dt = time.time() - t0

    mem = vram_mib()
    env.close()

    sps = n * args_cli.steps / dt
    return {
        "num_envs": n, "res": res, "cams": ncam,
        "env_steps_per_s": round(sps),
        "frames_per_s": round(sps * ncam),
        "ms_per_step": round(dt / args_cli.steps * 1000, 1),
        "vram_mib": mem,
    }


def main() -> None:
    rows = []
    for n in [int(x) for x in args_cli.counts.split(",")]:
        try:
            r = run_one(n, args_cli.res)
        except Exception as exc:  # noqa: BLE001
            r = {"num_envs": n, "res": args_cli.res, "error": str(exc)[:90]}
        rows.append(r)
        print(f"  -> {r}", flush=True)

    print("\n" + "=" * 82)
    print(f"Rendered throughput | {args_cli.task} | {args_cli.res}x{args_cli.res} | "
          f"{'scene+wrist' if args_cli.wrist else 'scene only'}")
    print("=" * 82)
    print(f"{'envs':>6} {'cams':>5} {'env_steps/s':>12} {'frames/s':>10} {'ms/step':>9} {'VRAM MiB':>9}")
    for r in rows:
        if "error" in r:
            print(f"{r['num_envs']:>6}  FAILED: {r['error']}")
        else:
            print(f"{r['num_envs']:>6} {r['cams']:>5} {r['env_steps_per_s']:>12} "
                  f"{r['frames_per_s']:>10} {r['ms_per_step']:>9} {r['vram_mib']:>9}")
    print("=" * 82)
    print("state-only reference on this machine: 8192 envs -> ~120,000 env-steps/s, 5.7 GB")


if __name__ == "__main__":
    main()
    simulation_app.close()
