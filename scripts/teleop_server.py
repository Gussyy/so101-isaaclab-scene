# SPDX-License-Identifier: BSD-3-Clause
"""Keyboard teleoperation over ZeroMQ, with no simulator in the process.

The same interface a policy server uses, driven by a human instead of a model. Useful as a
worked example of the boundary: the environment does not know whether the actions arriving on
its socket came from a neural network or from someone holding a key down.

Reads keys from this terminal using the standard library -- ``msvcrt`` on Windows, ``termios``
elsewhere -- so it needs no Isaac Sim, no Kit window, and no extra dependency. Type in this
terminal; the simulator can be on another monitor, another machine, or headless.

    # terminal 1
    python scripts/teleop_server.py --action-dim 6

    # terminal 2
    python scripts/run.py --config configs/pick_place_teleop.yaml

For teleoperation with focus in the Isaac Sim window instead, use ``control.source: keyboard``
(see simbridge/sources/keyboard.py).
"""

from __future__ import annotations

import argparse
import sys
import threading
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from simbridge.objective import resolve_action_limits  # noqa: E402
from simbridge.schema import ObsPacket  # noqa: E402
from simbridge.transport import ZmqPolicyServer  # noqa: E402

BINDINGS = [("q", "a"), ("w", "s"), ("e", "d"), ("r", "f"), ("t", "g")]
LABELS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
# Truncating the names collides shoulder_pan/shoulder_lift and wrist_flex/wrist_roll, so the
# status line uses distinct short forms.
SHORT = ["pan", "lift", "elbow", "wflex", "wroll"]


def _read_keys(on_key, stop: threading.Event) -> None:
    """Feed single keypresses to ``on_key`` until ``stop`` is set.

    Raw/unbuffered mode matters: line-buffered input would only deliver keys on Enter, which is
    unusable for driving an arm.
    """
    if sys.platform == "win32":
        import msvcrt

        while not stop.is_set():
            if msvcrt.kbhit():
                try:
                    ch = msvcrt.getch().decode("utf-8", "ignore").lower()
                except Exception:
                    continue
                if ch:
                    on_key(ch)
            else:
                stop.wait(0.01)
        return

    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        while not stop.is_set():
            ch = sys.stdin.read(1)
            if ch:
                on_key(ch.lower())
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


class KeyboardDriver:
    """Holds a joint target that keys nudge, and serves it as an action."""

    def __init__(self, action_dim: int, step: float, gripper_open: float, gripper_close: float,
                 action_limit=None) -> None:
        self.action_dim = action_dim
        self.step = step
        self.gripper_open = gripper_open
        self.gripper_close = gripper_close
        # Same limits as the in-process source. Clipping to +/-1 here caps every joint at half a
        # radian from its default pose -- the arm stops because the driver stopped.
        self.limits = resolve_action_limits(action_limit, action_dim)
        self.target = np.zeros(action_dim, dtype=np.float32)
        self.closed = False
        self._lock = threading.Lock()

    def on_key(self, ch: str) -> None:
        with self._lock:
            for i, (inc, dec) in enumerate(BINDINGS):
                if i >= self.action_dim:
                    break
                lo, hi = self.limits[i]
                if ch == inc:
                    self.target[i] = float(np.clip(self.target[i] + self.step, lo, hi))
                elif ch == dec:
                    self.target[i] = float(np.clip(self.target[i] - self.step, lo, hi))
            if ch == " ":
                self.closed = not self.closed
            elif ch == "n":
                self.target[:] = 0.0
                self.closed = False
        self.print_state()

    def print_state(self) -> None:
        vals = "  ".join(f"{SHORT[i]}={self.target[i]:+.2f}" for i in range(min(5, self.action_dim)))
        grip = "closed" if self.closed else "open"
        print(f"\r  {vals}   gripper={grip}   ", end="", flush=True)

    def __call__(self, obs: ObsPacket) -> np.ndarray:
        with self._lock:
            cmd = self.target.copy()
            if self.action_dim >= 6:
                cmd[5] = self.gripper_close if self.closed else self.gripper_open
        return np.tile(cmd, (obs.num_envs, 1)).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description="Keyboard teleop over ZeroMQ.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555")
    ap.add_argument("--action-dim", type=int, default=6)
    ap.add_argument("--step", type=float, default=0.05, help="target change per keypress")
    ap.add_argument("--gripper-open", type=float, default=1.0)
    ap.add_argument("--gripper-close", type=float, default=-1.0)
    ap.add_argument("--action-limit", type=float, default=None,
                    help="symmetric cap per joint in action units; default spans the arm's real range")
    args = ap.parse_args()

    driver = KeyboardDriver(args.action_dim, args.step, args.gripper_open, args.gripper_close,
                            action_limit=args.action_limit)

    print(f"[teleop] serving on {args.endpoint}\n")
    for i, (inc, dec) in enumerate(BINDINGS):
        print(f"    {inc} / {dec}   {LABELS[i]}")
    print("    space   toggle gripper")
    print("    n       zero all targets")
    print("    ctrl-c  quit\n")
    driver.print_state()

    stop = threading.Event()
    reader = threading.Thread(target=_read_keys, args=(driver.on_key, stop), daemon=True)
    reader.start()

    server = ZmqPolicyServer(driver, endpoint=args.endpoint)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[teleop] stopped")
    finally:
        stop.set()
        server.close()


if __name__ == "__main__":
    main()
