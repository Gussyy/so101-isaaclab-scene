# SPDX-License-Identifier: BSD-3-Clause
"""Drive this environment from LeRobot: a leader arm, or a trained policy.

Run this in a venv with ``lerobot`` installed. It imports no Isaac Sim, so LeRobot keeps its own
torch and dependency tree and the simulator keeps ours. They meet only at the socket.

    # a physical SO-101 leader arm teleoperating the sim
    python scripts/lerobot_server.py teleop --port COM5

    # a policy trained with lerobot-train
    python scripts/lerobot_server.py policy --path outputs/train/act_so101/checkpoints/last/pretrained_model

    # check the wiring without any hardware or model
    python scripts/lerobot_server.py mock

Then, in the environment's venv:

    python scripts/run.py --config configs/pick_place_lerobot.yaml

Setup (separate venv):

    python -m venv .lerobot && .lerobot\\Scripts\\activate
    pip install lerobot pyzmq msgpack
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simbridge.lerobot import (  # noqa: E402
    ARM_JOINTS,
    JOINTS,
    lerobot_action_to_array,
    obs_to_lerobot,
)
from simbridge.schema import ObsPacket  # noqa: E402
from simbridge.transport import ZmqPolicyServer  # noqa: E402


def make_teleop(args):
    """A LeRobot leader arm, read each step and mapped to environment actions."""
    from lerobot.teleoperators.so_leader import SOLeader, SOLeaderConfig

    leader = SOLeader(SOLeaderConfig(port=args.port, id=args.id))
    leader.connect()
    print(f"[lerobot] leader arm connected on {args.port}")

    def fn(obs: ObsPacket) -> np.ndarray:
        action = leader.get_action()          # {"shoulder_pan.pos": ..., ...}
        vec = lerobot_action_to_array(action, binary_gripper=not args.continuous_gripper)
        return np.tile(vec, (obs.num_envs, 1))

    return fn, leader.disconnect


def make_policy(args):
    """A LeRobot policy, stepped on one environment of the batch."""
    from lerobot.policies.factory import make_policy_from_pretrained

    policy = make_policy_from_pretrained(args.path)
    policy.eval()
    print(f"[lerobot] policy loaded from {args.path}")

    import torch

    def fn(obs: ObsPacket) -> np.ndarray:
        lr_obs = obs_to_lerobot(obs.state, obs.images, env_id=0)
        batch = {}
        for k, v in lr_obs.items():
            if isinstance(v, np.ndarray):     # image: HWC uint8 -> 1CHW float
                t = torch.from_numpy(v).permute(2, 0, 1).float().div(255.0).unsqueeze(0)
                batch[f"observation.images.{k}"] = t
            else:
                batch.setdefault("observation.state", []).append(float(v))
        if "observation.state" in batch:
            batch["observation.state"] = torch.tensor([batch["observation.state"]], dtype=torch.float32)

        with torch.inference_mode():
            out = policy.select_action(batch)
        vec = out.squeeze(0).cpu().numpy()

        # LeRobot policies emit whatever their training data used. If that was normalised joint
        # targets, convert; if it is already in this environment's units, pass it through.
        if args.raw_actions:
            arr = np.asarray(vec, dtype=np.float32)
        else:
            arr = lerobot_action_to_array(
                {f"{j}.pos": float(vec[i]) for i, j in enumerate(JOINTS[: len(vec)])},
                binary_gripper=not args.continuous_gripper,
            )
        return np.tile(arr, (obs.num_envs, 1))

    return fn, lambda: None


def make_mock(args):
    """No hardware, no model: a slow sweep, to prove the wiring end to end."""
    print("[lerobot] mock driver -- sweeping the arm, no LeRobot import")
    t0 = time.time()

    def fn(obs: ObsPacket) -> np.ndarray:
        phase = (time.time() - t0) * 0.6
        vec = np.zeros(len(JOINTS), dtype=np.float32)
        for i in range(len(ARM_JOINTS)):
            vec[i] = 0.35 * np.sin(phase + i * 0.7)
        vec[5] = 1.0 if np.sin(phase * 0.5) > 0 else -1.0
        return np.tile(vec, (obs.num_envs, 1))

    return fn, lambda: None


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve actions from LeRobot over ZeroMQ.")
    ap.add_argument("mode", choices=["teleop", "policy", "mock"])
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--port", default="COM5", help="serial port of the leader arm (teleop)")
    ap.add_argument("--id", default="so101_leader", help="LeRobot device id (teleop)")
    ap.add_argument("--path", default=None, help="pretrained policy directory or hub id (policy)")
    ap.add_argument("--raw-actions", action="store_true",
                    help="policy already emits environment-space actions; skip unit conversion")
    ap.add_argument("--continuous-gripper", action="store_true",
                    help="task uses a continuous jaw action rather than BinaryJointPositionAction")
    args = ap.parse_args()

    if args.mode == "policy" and not args.path:
        raise SystemExit("policy mode needs --path")

    builder = {"teleop": make_teleop, "policy": make_policy, "mock": make_mock}[args.mode]
    try:
        fn, cleanup = builder(args)
    except ImportError as exc:
        raise SystemExit(
            f"could not import LeRobot ({exc}).\n"
            "  This script is meant to run in a venv with lerobot installed:\n"
            "      python -m venv .lerobot && .lerobot\\Scripts\\activate\n"
            "      pip install lerobot pyzmq msgpack\n"
            "  Use `mock` mode to test the wiring without it."
        ) from exc

    print(f"[lerobot] serving {args.mode} on {args.endpoint}")
    server = ZmqPolicyServer(fn, endpoint=args.endpoint)
    try:
        server.serve_forever(log_every=200)
    except KeyboardInterrupt:
        print("\n[lerobot] stopped")
    finally:
        cleanup()
        server.close()


if __name__ == "__main__":
    main()
