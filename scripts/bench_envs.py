# SPDX-License-Identifier: BSD-3-Clause
"""Sweep ``--num_envs`` and report throughput, VRAM and GPU utilisation.

Answers the only question that matters for saturating a GPU on this task: how many
parallel environments before throughput stops scaling or VRAM runs out.

Each point is a separate process -- Isaac Sim cannot rebuild a scene at a new env count
in-process -- so the sweep is slow but honest. Startup cost is excluded from steps/s;
that number comes from the trainer's own steady-state report.

Usage:
    python scripts/bench_envs.py --counts 1024,2048,4096,8192
    python scripts/bench_envs.py --counts 2048,4096 --physics newton_mjwarp
"""

import argparse
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TRAIN = REPO / "scripts" / "train.py"

_SPS = re.compile(r"Steps per second:\s*([0-9.]+)")
_ITER_T = re.compile(r"Iteration time:\s*([0-9.]+)s")


def gpu_stats() -> tuple[int, int]:
    """(VRAM MiB used, GPU util %) — 0,0 if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        ).stdout.strip().splitlines()[0]
        mem, util = (int(x.strip()) for x in out.split(","))
        return mem, util
    except Exception:
        return 0, 0


def run_one(task: str, physics: str, n: int, iters: int) -> dict:
    cmd = [sys.executable, str(TRAIN), "--rl_library", "rsl_rl", "--task", task,
           "--num_envs", str(n), "--max_iterations", str(iters)]
    if physics:
        cmd.append(f"physics={physics}")

    t0 = time.time()
    proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)

    sps, iter_t, peak_mem, peak_util, first_iter_at = [], [], 0, 0, None
    assert proc.stdout is not None
    for line in proc.stdout:
        if "Learning iteration" in line:
            if first_iter_at is None:
                first_iter_at = time.time() - t0
            m, u = gpu_stats()
            peak_mem, peak_util = max(peak_mem, m), max(peak_util, u)
        if m := _SPS.search(line):
            sps.append(float(m.group(1)))
        if m := _ITER_T.search(line):
            iter_t.append(float(m.group(1)))
    proc.wait()

    # Drop the first few samples: cold caches and CUDA-graph capture are not steady state.
    steady = sps[len(sps) // 2:] or sps
    return {
        "num_envs": n,
        "ok": proc.returncode == 0,
        "startup_s": round(first_iter_at, 1) if first_iter_at else None,
        "steps_per_s": round(sum(steady) / len(steady)) if steady else None,
        "iter_s": round(sum(iter_t[len(iter_t) // 2:] or iter_t) / max(1, len(iter_t[len(iter_t) // 2:] or iter_t)), 3) if iter_t else None,
        "peak_vram_mib": peak_mem,
        "peak_gpu_pct": peak_util,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep num_envs for throughput/VRAM.")
    ap.add_argument("--task", default="SO101-PickPlace-v0")
    ap.add_argument("--physics", default="", help="e.g. newton_mjwarp; empty = task default (PhysX)")
    ap.add_argument("--counts", default="1024,2048,4096,8192")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()

    rows = []
    for n in [int(x) for x in args.counts.split(",")]:
        print(f"\n=== num_envs={n} ===", flush=True)
        r = run_one(args.task, args.physics, n, args.iters)
        rows.append(r)
        print(f"  -> {r}", flush=True)

    print("\n" + "=" * 78)
    print(f"{args.task}  physics={args.physics or 'default(physx)'}")
    print("=" * 78)
    print(f"{'envs':>6} {'ok':>4} {'startup_s':>10} {'steps/s':>10} {'iter_s':>8} {'VRAM MiB':>9} {'GPU %':>6}")
    for r in rows:
        print(f"{r['num_envs']:>6} {str(r['ok']):>4} {str(r['startup_s']):>10} "
              f"{str(r['steps_per_s']):>10} {str(r['iter_s']):>8} {r['peak_vram_mib']:>9} {r['peak_gpu_pct']:>6}")
    print("=" * 78)


if __name__ == "__main__":
    main()
