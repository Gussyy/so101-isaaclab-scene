# SPDX-License-Identifier: BSD-3-Clause
"""Measure the SO-ARM101's real reachable envelope, and check a task is physically possible.

``simbridge.objective`` rejects goals the arm cannot reach, which is only as good as the numbers
it compares against. Those started as estimates. This measures them: sample joint configurations
inside the limits, drive the arm there, read where the gripper actually ends up.

Reach is not the whole question. A cube can sit inside the envelope and still be impossible to
pick because the jaw cannot span it or the arm cannot hold it. So this also reports:

* the envelope for *any* gripper orientation, and separately for a downward-facing gripper,
  which is what a top-down grasp needs and is a much smaller region;
* the jaw opening in metres, against the object size in the config;
* the payload implied by the actuator effort limits, against the object mass.

Prints constants ready to paste into ``simbridge/objective.py``.

Usage:
    python scripts/measure_workspace.py --samples 4000
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Measure the SO-ARM101 workspace.")
parser.add_argument("--samples", type=int, default=3000, help="joint configurations to sample")
parser.add_argument("--batch", type=int, default=500, help="configurations evaluated per sim step")
parser.add_argument("--down-cos", type=float, default=0.7,
                    help="min alignment of the gripper axis with -Z to count as top-down")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import Articulation  # noqa: E402
from isaaclab.sim import SimulationContext  # noqa: E402
from isaaclab.utils.math import quat_apply  # noqa: E402

from so101_scene.tuning import so101_cfg  # noqa: E402


def percentile_envelope(radial: np.ndarray, z: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> dict:
    """Envelope at percentiles rather than min/max.

    The extremes of a sampled workspace are single fully-extended configurations the arm can
    technically reach and cannot usefully work at. Percentiles give a boundary a task can be
    written against.
    """
    return {
        "radial_min": float(np.percentile(radial, lo)),
        "radial_max": float(np.percentile(radial, hi)),
        "z_min": float(np.percentile(z, lo)),
        "z_max": float(np.percentile(z, hi)),
    }


def main() -> None:
    sim = SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / 120.0))
    cfg = so101_cfg("/World/Robot")
    cfg.spawn.rigid_props.disable_gravity = True   # kinematic probe; gravity would sag the arm
    robot = Articulation(cfg.replace(prim_path="/World/Robot"))
    sim.reset()

    names = robot.joint_names
    limits = robot.data.joint_pos_limits.torch[0].cpu().numpy()   # (dof, 2)
    arm = [i for i, n in enumerate(names) if n != "gripper"]
    grip_i = names.index("gripper") if "gripper" in names else None
    ee_i = robot.body_names.index("gripper")

    rng = np.random.default_rng(0)
    pos_all, down_mask = [], []

    remaining = args_cli.samples
    while remaining > 0:
        n = min(args_cli.batch, remaining)
        remaining -= n

        q = np.tile(robot.data.default_joint_pos.torch[0].cpu().numpy(), (1, 1))
        q = np.repeat(q, robot.num_instances, axis=0)
        # one configuration at a time across the single instance, batched by stepping
        for _ in range(n):
            sample = q[0].copy()
            for j in arm:
                sample[j] = rng.uniform(limits[j, 0], limits[j, 1])
            t = torch.as_tensor(sample, device=robot.device, dtype=torch.float32).unsqueeze(0)
            robot.write_joint_position_to_sim_index(position=t)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(t))
            sim.step()
            robot.update(sim.get_physics_dt())

            root = robot.data.root_pos_w.torch[0]
            ee = robot.data.body_pos_w.torch[0, ee_i] - root
            quat = robot.data.body_quat_w.torch[0, ee_i]
            # gripper approach axis in world frame; compare against -Z for a top-down grasp
            axis = quat_apply(quat.unsqueeze(0), torch.tensor([[0.0, 0.0, 1.0]], device=quat.device))[0]
            pos_all.append(ee.cpu().numpy())
            down_mask.append(float(-axis[2].item()))

    pos = np.asarray(pos_all)
    downs = np.asarray(down_mask)
    radial = np.hypot(pos[:, 0], pos[:, 1])
    z = pos[:, 2]

    full = percentile_envelope(radial, z)
    sel = downs >= args_cli.down_cos
    top = percentile_envelope(radial[sel], z[sel]) if sel.sum() > 50 else None

    print("\n" + "=" * 74)
    print(f"SO-ARM101 reachable envelope   ({len(pos)} sampled configurations)")
    print("=" * 74)
    print(f"  joints sampled : {[names[i] for i in arm]}")
    print(f"  absolute max reach : {radial.max():.3f} m   max height : {z.max():.3f} m")
    print()
    print("  any gripper orientation (1st-99th percentile):")
    print(f"    radial  {full['radial_min']:.3f} .. {full['radial_max']:.3f} m")
    print(f"    height  {full['z_min']:.3f} .. {full['z_max']:.3f} m")
    if top:
        print(f"\n  top-down grasp only (gripper axis within {args_cli.down_cos:.2f} of -Z, "
              f"{100 * sel.mean():.0f}% of samples):")
        print(f"    radial  {top['radial_min']:.3f} .. {top['radial_max']:.3f} m")
        print(f"    height  {top['z_min']:.3f} .. {top['z_max']:.3f} m")
        print("    a top-down pick is only possible inside this smaller region")

    print("\n  paste into simbridge/objective.py:")
    e = top or full
    print(f"    REACH_RADIAL = ({e['radial_min']:.2f}, {e['radial_max']:.2f})")
    print(f"    REACH_HEIGHT = ({max(0.0, e['z_min']):.2f}, {e['z_max']:.2f})")

    # ---- is a grasp physically possible at all? -------------------------
    print("\n" + "-" * 74)
    print("  graspability")
    print("-" * 74)
    if grip_i is not None:
        lo, hi = limits[grip_i]
        jaw_i = robot.body_names.index("moving_jaw_so101_v1")
        spans = []
        for val in (lo, hi):
            t = robot.data.default_joint_pos.torch[0].clone().unsqueeze(0)
            t[0, grip_i] = float(val)
            robot.write_joint_position_to_sim_index(position=t)
            robot.write_joint_velocity_to_sim_index(velocity=torch.zeros_like(t))
            # One step is not enough for the articulation to settle at the written pose; a single
            # step reported a 0.0 mm opening, which is obviously wrong and worth guarding against.
            for _ in range(20):
                sim.step()
                robot.update(sim.get_physics_dt())
            a = robot.data.body_pos_w.torch[0, ee_i]
            b = robot.data.body_pos_w.torch[0, jaw_i]
            spans.append(float(torch.norm(a - b).item()))
        opening = abs(spans[1] - spans[0])
        print(f"    jaw joint range   : {lo:+.3f} .. {hi:+.3f} rad")
        print(f"    jaw body separation at closed / open : {spans[0]*1000:.1f} / {spans[1]*1000:.1f} mm")
        if opening < 1e-4:
            print("    WARNING: no measurable jaw travel -- the joint did not move; treat as unmeasured")
        else:
            print(f"    jaw travel        : {opening * 1000:.1f} mm")
            print(f"    -> an object wider than roughly {opening * 1000:.0f} mm cannot be spanned")

    efforts = robot.data.joint_effort_limits.torch[0].cpu().numpy() if hasattr(
        robot.data, "joint_effort_limits") else None
    if efforts is not None and len(efforts):
        # crude static bound: torque available at the shoulder over the horizontal lever arm
        lever = float(np.percentile(radial, 90))
        payload = float(efforts[arm[1]] / max(lever, 1e-6) / 9.81)
        print(f"    shoulder effort   : {efforts[arm[1]]:.1f} N m at a {lever:.2f} m lever")
        print(f"    -> stall-torque payload bound {payload * 1000:.0f} g -- an UPPER bound, and a")
        print("       badly optimistic one: it ignores the arm's own weight, gearing losses and")
        print("       all dynamics. The real SO-101 is quoted nearer 200-300 g.")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
    simulation_app.close()
