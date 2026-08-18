# SPDX-License-Identifier: BSD-3-Clause
"""Play back a trained SO-ARM101 policy.

Usage:
    python scripts/play.py --rl_library rsl_rl --task SO101-Reach-Play-v0 --viz newton
"""

import warp as wp

wp.config.enable_backward = False

import so101_scene  # noqa: F401,E402  -- registers SO101-Reach-Play-v0
from isaaclab_rl.entrypoints import run_play_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_play_cli())
