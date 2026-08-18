# SPDX-License-Identifier: BSD-3-Clause
"""Drive each joint through its full range and record it, so the reach is visible rather than tabulated.

Joint positions are written straight to the simulation, bypassing the action space. That is the
point: the action space is `default + 0.5 * action`, which biases what a policy commands, and the
question here is what the *arm* can do. What you see is the asset's own limits.

    python scripts/sweep_limits.py --out docs/reach_sweep.mp4

Prints commanded against reached for each extreme, and says which way it missed:

* blocked short  -- something is in the way. `elbow_flex` stops ~0.3 rad early in this scene;
  turning off self-collision recovers only a quarter of that, so most of it is the table.
* past the limit -- the drive overshoots a limit PhysX is not enforcing hard. `shoulder_lift`
  settles around +1.8 to +1.96 against a +1.745 limit.

Both are properties of the scene and the drive, not of the asset's declared range. In free space
(scripts/measure_workspace.py, no table, gravity off) every joint reaches its full limit.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Sweep every joint through its range and record it.")
parser.add_argument("--config", default="configs/variants/goal_centre.yaml")
parser.add_argument("--out", default="docs/reach_sweep.mp4")
parser.add_argument("--hold", type=int, default=44, help="frames to hold at each extreme")
parser.add_argument("--travel", type=int, default=34, help="frames to travel between extremes")
parser.add_argument("--fps", type=int, default=30)
parser.add_argument("--reach", type=float, default=0.354,
                    help="reach outline radius in metres; 99th-pct measured value")
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
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

import so101_scene  # noqa: F401,E402
from simbridge.builder import build_env_cfg, load_config, resolve_task  # noqa: E402
from simbridge.registry import register_task  # noqa: E402

register_task("pick_place", "SO101-PickPlace-v0")
register_task("reach", "SO101-Reach-v0")

BG, FG, DIM, ACCENT, GOOD = (13, 17, 23), (201, 209, 217), (110, 120, 130), (86, 182, 255), (86, 211, 100)


def _font(size: int, bold: bool = False):
    p = Path("C:/Windows/Fonts") / ("consolab.ttf" if bold else "consola.ttf")
    if p.exists():
        return ImageFont.truetype(str(p), size)
    return ImageFont.load_default()


def sphere_outline(cam_pos, look_at, focal_mm: float, width: int, height: int,
                   centre=(0.0, 0.0, 0.0), radius: float = 0.354, aperture_mm: float = 20.955):
    """Where the reach limit's silhouette lands in the image, as (cx, cy, r) in pixels.

    Drawn as an outline rather than spawned as a translucent shell: ``PreviewSurfaceCfg`` has an
    ``opacity`` field, but the RTX path renders the sphere solid anyway, and a solid sphere hides
    the robot it is supposed to describe.
    """
    c = np.asarray(cam_pos, dtype=float)
    fwd = np.asarray(look_at, dtype=float) - c
    d = float(np.linalg.norm(fwd))
    if d < 1e-9:
        return None
    fwd /= d
    right = np.cross(fwd, (0.0, 0.0, 1.0))
    n = np.linalg.norm(right)
    if n < 1e-9:
        return None
    right /= n
    up = np.cross(right, fwd)

    v = np.asarray(centre, dtype=float) - c
    z = float(v @ fwd)
    if z <= radius:                      # camera inside or level with the shell
        return None
    f_px = focal_mm / aperture_mm * width
    cx = width / 2.0 + f_px * float(v @ right) / z
    cy = height / 2.0 - f_px * float(v @ up) / z
    r_px = f_px * radius / float(np.sqrt(max(z * z - radius * radius, 1e-9)))
    return cx, cy, r_px


def annotate(frame: np.ndarray, joint: str, q: float, lo: float, hi: float, ee_r: float, ee_z: float,
             outline=None, radius: float = 0.354):
    """Caption a frame with the joint being swept and where it currently is in its range."""
    img = Image.fromarray(frame).convert("RGB")
    if outline is not None:
        cx, cy, r = outline
        d0 = ImageDraw.Draw(img, "RGBA")
        d0.ellipse([cx - r, cy - r, cx + r, cy + r], outline=(86, 182, 255, 210), width=2)
        d0.ellipse([cx - r, cy - r * 0.30, cx + r, cy + r * 0.30],
                   outline=(86, 182, 255, 90), width=1)      # equator, for a sense of depth
        d0.text((cx - 52, cy - r - 18), f"reach {radius * 100:.0f} cm",
                font=_font(14), fill=(86, 182, 255, 230))
    w, h = img.size
    panel = Image.new("RGB", (w, h + 104), BG)
    panel.paste(img, (0, 0))
    d = ImageDraw.Draw(panel)

    d.text((16, h + 10), joint, font=_font(20, bold=True), fill=ACCENT)
    d.text((16, h + 38), f"{q:+.3f} rad   ({np.degrees(q):+6.1f} deg)", font=_font(16), fill=FG)
    d.text((16, h + 60), f"limits {lo:+.3f} .. {hi:+.3f}", font=_font(15), fill=DIM)
    d.text((16, h + 80), f"gripper at {ee_r:.3f} m out, {ee_z:.3f} m up", font=_font(15), fill=GOOD)

    # A bar showing where in its travel the joint currently is.
    x0, x1, y = 300, w - 24, h + 46
    d.rectangle([x0, y, x1, y + 12], outline=(48, 54, 61), width=1)
    # Clamped: a joint can read slightly outside its limits -- shoulder_lift settles at +1.96
    # against a +1.745 limit -- and an unclamped fraction draws a rectangle backwards.
    frac = (q - lo) / (hi - lo) if hi > lo else 0.5
    cx = x0 + min(max(frac, 0.0), 1.0) * (x1 - x0)
    d.rectangle([x0, y, cx, y + 12], fill=(30, 80, 110))
    d.line([(cx, y - 4), (cx, y + 16)], fill=ACCENT, width=3)
    d.text((x0, y + 18), f"{lo:+.2f}", font=_font(13), fill=DIM)
    d.text((x1 - 40, y + 18), f"{hi:+.2f}", font=_font(13), fill=DIM)
    return np.asarray(panel)


def main() -> None:
    cfg = load_config(args_cli.config)
    cams = list((cfg.get("scene") or {}).get("cameras") or {})
    if not cams:
        raise SystemExit(f"{args_cli.config} declares no cameras; nothing to record")
    cam = cams[0]

    env = gym.make(resolve_task(cfg), cfg=build_env_cfg(cfg, device=args_cli.device, num_envs=1)).unwrapped
    env.reset()

    robot = env.scene["robot"]
    names = list(robot.joint_names)
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()
    default = robot.data.default_joint_pos.torch[0].cpu().numpy()
    ee_i = robot.body_names.index("gripper")

    # The reach boundary, drawn from the camera the config already declares.
    cam_spec = cfg["scene"]["cameras"][cam]
    reach_r = float(args_cli.reach)
    res = cam_spec.get("resolution", [128, 128])
    outline = None
    if "look_at" in cam_spec:
        outline = sphere_outline(cam_spec.get("pos", (0.55, 0.0, 0.35)), cam_spec["look_at"],
                                 float(cam_spec.get("focal_length", 18.0)),
                                 int(res[0]), int(res[1]), radius=reach_r)
    else:
        print("[sweep] camera has no look_at, so the reach outline cannot be placed; skipping it")

    out = Path(args_cli.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio.get_writer(out, fps=args_cli.fps, codec="libx264", quality=8,
                                macro_block_size=1, ffmpeg_log_level="error")

    print(f"\n{'joint':<16}{'commanded':>11}{'reached':>10}{'gap':>8}   note")
    print("-" * 62)
    reached_report = []

    try:
        for j, name in enumerate(names):
            lo, hi = float(limits[j, 0]), float(limits[j, 1])
            # default -> lower -> upper -> default, so each joint returns the arm to the pose the
            # next joint starts from and one joint moves at a time.
            legs = [(default[j], lo), (lo, hi), (hi, default[j])]
            for a, b in legs:
                n = args_cli.travel + (args_cli.hold if b in (lo, hi) else 0)
                for k in range(n):
                    t = min(1.0, k / max(args_cli.travel - 1, 1))
                    q = default.copy()
                    q[j] = a + (b - a) * t
                    tq = torch.as_tensor(q, device=env.device, dtype=torch.float32).unsqueeze(0)
                    # Both the state and the target. Writing the state alone leaves the implicit
                    # PD actuator holding its previous target, which drags every joint about
                    # 0.1 rad back from wherever it was placed -- and reads as "stopped short"
                    # when nothing is stopping it.
                    robot.write_joint_position_to_sim(tq)
                    robot.write_joint_velocity_to_sim(torch.zeros_like(tq))
                    robot.set_joint_position_target(tq)
                    robot.write_data_to_sim()
                    env.sim.step()
                    env.scene.update(env.physics_dt)

                    rgb = env.scene[cam].data.output["rgb"]
                    if hasattr(rgb, "torch"):
                        rgb = rgb.torch
                    arr = rgb.detach().cpu().numpy()
                    if arr.dtype != np.uint8:
                        arr = (np.clip(arr, 0.0, 1.0) * 255).astype(np.uint8)

                    ee = (robot.data.body_pos_w.torch[0, ee_i] - robot.data.root_pos_w.torch[0]).cpu().numpy()
                    got = float(robot.data.joint_pos.torch[0, j].item())
                    writer.append_data(annotate(arr[0, ..., :3], name, got, lo, hi,
                                                float(np.hypot(ee[0], ee[1])), float(ee[2]),
                                                outline=outline, radius=reach_r))

                if b in (lo, hi):
                    got = float(robot.data.joint_pos.torch[0, j].item())
                    gap = got - b
                    # Direction matters. Falling short means something is in the way; going past
                    # means the drive overshoots a limit PhysX is not enforcing hard. Calling
                    # both "stopped short" is how this script first reported shoulder_lift
                    # settling at +1.96 against a +1.745 limit.
                    outward = 1.0 if b == hi else -1.0
                    if abs(gap) < 0.02:
                        note = ""
                    elif gap * outward > 0:
                        note = "settles PAST the limit"
                    else:
                        note = "blocked short of it"
                    print(f"{name:<16}{b:>11.3f}{got:>10.3f}{gap:>8.3f}   {note}")
                    reached_report.append((name, b, got, outward))
    finally:
        writer.close()

    off = [r for r in reached_report if abs(r[2] - r[1]) >= 0.02]
    blocked = [r for r in off if (r[2] - r[1]) * r[3] < 0]
    past = [r for r in off if (r[2] - r[1]) * r[3] > 0]
    print("-" * 62)
    print(f"{len(reached_report) - len(off)}/{len(reached_report)} extremes reached within 0.02 rad")
    if blocked:
        print("blocked short: " + ", ".join(f"{n} ({w:+.2f} -> {g:+.2f})" for n, w, g, _ in blocked))
    if past:
        print("past the limit: " + ", ".join(f"{n} ({w:+.2f} -> {g:+.2f})" for n, w, g, _ in past))
    print(f"\nwrote {out} ({out.stat().st_size / 1e6:.1f} MB)")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
