# SPDX-License-Identifier: BSD-3-Clause
"""Train an SO-ARM101 policy.

Mirrors Isaac Lab's own train entrypoint, but imports :mod:`so101_scene` first so this
repo's environments are in the Gym registry before the CLI resolves ``--task``.

Usage:
    python scripts/train.py --rl_library rsl_rl --task SO101-Reach-v0 physics=newton_mjwarp
"""

# Warp captures ``enable_backward`` when a module is created, which happens at import time,
# so it has to be set before importing anything that defines Warp kernels.
import warp as wp

wp.config.enable_backward = False

import so101_scene  # noqa: F401,E402  -- registers SO101-Reach-v0
from isaaclab_rl.entrypoints import run_train_cli  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(run_train_cli())
