# SPDX-License-Identifier: BSD-3-Clause
"""Plot training curves from the rsl_rl TensorBoard logs.

The console prints only ``Mean reward``, which is not enough to tell "still exploring"
apart from "structurally cannot learn". The per-term ``Episode_Reward/*`` scalars are:
a task reward pinned at exactly 0.0 means that term never fires at all, which is a broken
scene or mis-scaled shaping rather than a slow start.

Usage:
    python scripts/plot_rewards.py                       # newest run
    python scripts/plot_rewards.py --run logs/rsl_rl/pick_place_so101/2026-08-18_15-32-36
    python scripts/plot_rewards.py --compare             # overlay every run found
"""

import argparse
import glob
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # noqa: E402

# Panels: (title, [tag substrings], y-label)
PANELS = [
    ("Mean reward", ["Train/mean_reward"], "reward"),
    ("Task reward terms", [
        "Episode_Reward/reaching_object",
        "Episode_Reward/lifting_object",
        "Episode_Reward/object_goal_tracking",
    ], "reward"),
    ("Success rate", ["Metrics/success_rate"], "fraction"),
    ("Distance to goal", ["Metrics/object_pose/position_error"], "m"),
    ("Policy std", ["Policy/mean_std"], "std"),
    ("Episode length", ["Train/mean_episode_length"], "steps"),
]


def load(run: str) -> dict[str, tuple[list[int], list[float]]]:
    ea = EventAccumulator(run, size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ea.Tags()["scalars"]:
        pts = ea.Scalars(tag)
        out[tag] = ([p.step for p in pts], [p.value for p in pts])
    return out


def newest_run(root: str) -> str:
    runs = [d for d in glob.glob(os.path.join(root, "*", "*", "")) if glob.glob(os.path.join(d, "events.out.tfevents*"))]
    if not runs:
        raise SystemExit(f"no TensorBoard runs under {root}")
    return max(runs, key=os.path.getmtime)


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot rsl_rl training curves.")
    ap.add_argument("--root", default="logs/rsl_rl")
    ap.add_argument("--run", default=None, help="specific run dir; default = newest")
    ap.add_argument("--compare", action="store_true", help="overlay all runs found under --root")
    ap.add_argument("--out", default="docs/reward_curve.png")
    args = ap.parse_args()

    if args.compare:
        runs = sorted(d for d in glob.glob(os.path.join(args.root, "*", "*", ""))
                      if glob.glob(os.path.join(d, "events.out.tfevents*")))
    else:
        runs = [args.run or newest_run(args.root)]

    series = {r: load(r) for r in runs}

    fig, axes = plt.subplots(2, 3, figsize=(16, 8))
    for ax, (title, tags, ylabel) in zip(axes.flat, PANELS):
        plotted = False
        for run, data in series.items():
            label_run = os.path.basename(os.path.normpath(run))
            for tag in tags:
                if tag not in data:
                    continue
                steps, vals = data[tag]
                short = tag.split("/")[-1]
                label = f"{short}" if len(series) == 1 else f"{label_run[:16]}:{short}"
                ax.plot(steps, vals, linewidth=1.4, label=label)
                plotted = True
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("iteration")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25)
        if plotted:
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, "no data", ha="center", va="center", transform=ax.transAxes)

    run_names = ", ".join(os.path.basename(os.path.normpath(r)) for r in runs)
    fig.suptitle(f"SO-ARM101 pick-and-place — {run_names}", fontsize=12)
    fig.tight_layout()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"wrote {args.out}")

    # Console summary so the numbers are readable without opening the image.
    for run, data in series.items():
        print(f"\n{os.path.basename(os.path.normpath(run))}")
        for tag in ("Train/mean_reward", "Metrics/success_rate",
                    "Episode_Reward/reaching_object", "Episode_Reward/lifting_object",
                    "Episode_Reward/object_goal_tracking", "Metrics/object_pose/position_error"):
            if tag in data:
                _, v = data[tag]
                print(f"  {tag:<44} first={v[0]:+.5f}  last={v[-1]:+.5f}  max={max(v):+.5f}")


if __name__ == "__main__":
    main()
