# SPDX-License-Identifier: BSD-3-Clause
"""Reference policy server: the process a YAML control.source of zmq connects to.

Runs entirely outside Isaac Sim -- no Kit, no USD, no simulator import. That separation is the
point: a flow-matching visuomotor policy can live in its own process with its own CUDA context
and its own framework, and the simulator only holds a socket.

    python scripts/policy_server.py --policy zero --action-dim 6
    python scripts/policy_server.py --policy sine --action-dim 6
    python scripts/policy_server.py --policy checkpoint --checkpoint logs/.../model_1499.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simbridge.schema import ObsPacket  # noqa: E402
from simbridge.transport import ZmqPolicyServer  # noqa: E402


def build_policy(args):
    """Return an ObsPacket -> actions callable for the requested policy kind."""
    if args.policy == "checkpoint":
        if not args.checkpoint:
            raise SystemExit("--policy checkpoint requires --checkpoint")
        from simbridge.sources.rl_checkpoint import RslRlCheckpointSource

        src = RslRlCheckpointSource(args.checkpoint, device=args.device)
        return src._predict_chunk  # noqa: SLF001  -- the server owns this source

    if args.policy == "sine":
        counter = {"t": 0}

        def sine(obs: ObsPacket) -> np.ndarray:
            counter["t"] += 1
            a = np.zeros((obs.num_envs, args.action_dim), dtype=np.float32)
            a[:, 0] = 0.5 * np.sin(counter["t"] / 40.0)
            return a

        return sine

    def zero(obs: ObsPacket) -> np.ndarray:
        return np.zeros((obs.num_envs, args.action_dim), dtype=np.float32)

    return zero


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve actions over ZeroMQ.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--policy", default="zero", choices=["zero", "sine", "checkpoint"])
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    fn = build_policy(args)
    print(f"[policy-server] {args.policy} policy on {args.endpoint} (ctrl-c to stop)", flush=True)
    server = ZmqPolicyServer(fn, endpoint=args.endpoint)
    try:
        server.serve_forever(log_every=200)
    except KeyboardInterrupt:
        print("\n[policy-server] stopped")
    finally:
        server.close()


if __name__ == "__main__":
    main()
