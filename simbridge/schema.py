# SPDX-License-Identifier: BSD-3-Clause
"""The wire contract between an environment and whatever drives it.

Everything that produces actions -- a trained RL checkpoint, a teleop device, a flow-matching
visuomotor policy, a scripted tester -- speaks these two messages and nothing else. That is what
makes the driver swappable without the environment knowing which one is attached.

Design notes:

* **Arrays travel as raw bytes plus shape/dtype, not as nested lists.** A 256x128x128x3 uint8
  image batch is 12 MB; JSON-encoding it costs more time than the simulation step that produced
  it. msgpack framing keeps the payload zero-copy on both ends.
* **The envelope is versioned.** A policy server and a simulator are separate processes with
  separate lifetimes; they will drift. ``PROTOCOL_VERSION`` makes that a clear error instead of
  a silent shape mismatch.
* **Observations are grouped, not flattened.** ``state`` and ``images`` stay separate because a
  visuomotor policy consumes them through different encoders, and a state-only policy should be
  able to ignore images without paying to decode them.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import msgpack
import numpy as np

PROTOCOL_VERSION = 1


class ProtocolError(RuntimeError):
    """Raised when a peer sends a message this build cannot interpret."""


def _pack_array(a: np.ndarray) -> dict[str, Any]:
    """Encode an ndarray as {shape, dtype, raw-bytes}, contiguous so the copy is one memcpy."""
    a = np.ascontiguousarray(a)
    return {"shape": list(a.shape), "dtype": a.dtype.str, "data": a.tobytes()}


def _unpack_array(d: dict[str, Any]) -> np.ndarray:
    return np.frombuffer(d["data"], dtype=np.dtype(d["dtype"])).reshape(tuple(d["shape"]))


def _pack_group(g: dict[str, np.ndarray]) -> dict[str, Any]:
    return {k: _pack_array(v) for k, v in g.items()}


def _unpack_group(g: dict[str, Any]) -> dict[str, np.ndarray]:
    return {k: _unpack_array(v) for k, v in g.items()}


@dataclass
class ObsPacket:
    """Environment -> driver.

    Attributes:
        step: Environment step counter. Echoed back in the ActionPacket so a driver that falls
            behind can be detected rather than silently acted upon.
        num_envs: Batch dimension shared by every array here.
        state: Named proprioceptive/privileged arrays, each ``(num_envs, ...)``.
        images: Named image arrays, each ``(num_envs, H, W, C)`` uint8.
        done: ``(num_envs,)`` bool, episode boundaries.
        reward: ``(num_envs,)`` float32, present for RL drivers, ignorable otherwise.
        info: Small JSON-able extras. Not for bulk data.
    """

    step: int
    num_envs: int
    state: dict[str, np.ndarray] = field(default_factory=dict)
    images: dict[str, np.ndarray] = field(default_factory=dict)
    done: np.ndarray | None = None
    reward: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        payload = {
            "v": PROTOCOL_VERSION,
            "kind": "obs",
            "step": self.step,
            "num_envs": self.num_envs,
            "state": _pack_group(self.state),
            "images": _pack_group(self.images),
            "done": _pack_array(self.done) if self.done is not None else None,
            "reward": _pack_array(self.reward) if self.reward is not None else None,
            "info": self.info,
            "t": time.time(),
        }
        return msgpack.packb(payload, use_bin_type=True)

    @staticmethod
    def from_bytes(raw: bytes) -> ObsPacket:
        d = msgpack.unpackb(raw, raw=False)
        _check(d, "obs")
        return ObsPacket(
            step=d["step"],
            num_envs=d["num_envs"],
            state=_unpack_group(d.get("state") or {}),
            images=_unpack_group(d.get("images") or {}),
            done=_unpack_array(d["done"]) if d.get("done") else None,
            reward=_unpack_array(d["reward"]) if d.get("reward") else None,
            info=d.get("info") or {},
        )


@dataclass
class ActionPacket:
    """Driver -> environment.

    Attributes:
        step: The ``ObsPacket.step`` this responds to. The environment rejects a mismatch rather
            than applying a stale action.
        action: ``(num_envs, action_dim)`` float32.
        reset: Optional ``(num_envs,)`` bool requesting per-env reset -- used by teleop and by
            dataset collection to discard a bad episode.
        info: Small extras (e.g. a teleop 'gripper' toggle, or a policy's inference latency).
    """

    step: int
    action: np.ndarray
    reset: np.ndarray | None = None
    info: dict[str, Any] = field(default_factory=dict)

    def to_bytes(self) -> bytes:
        payload = {
            "v": PROTOCOL_VERSION,
            "kind": "action",
            "step": self.step,
            "action": _pack_array(np.asarray(self.action, dtype=np.float32)),
            "reset": _pack_array(self.reset) if self.reset is not None else None,
            "info": self.info,
            "t": time.time(),
        }
        return msgpack.packb(payload, use_bin_type=True)

    @staticmethod
    def from_bytes(raw: bytes) -> ActionPacket:
        d = msgpack.unpackb(raw, raw=False)
        _check(d, "action")
        return ActionPacket(
            step=d["step"],
            action=_unpack_array(d["action"]),
            reset=_unpack_array(d["reset"]) if d.get("reset") else None,
            info=d.get("info") or {},
        )


def _check(d: dict[str, Any], kind: str) -> None:
    if d.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"protocol version {d.get('v')} != {PROTOCOL_VERSION}; rebuild both ends")
    if d.get("kind") != kind:
        raise ProtocolError(f"expected a {kind!r} message, got {d.get('kind')!r}")


def demo() -> None:
    """Round-trip self-check: bytes in == arrays out, and version skew is caught."""
    rng = np.random.default_rng(0)
    obs = ObsPacket(
        step=7,
        num_envs=4,
        state={"joint_pos": rng.random((4, 6), dtype=np.float32)},
        images={"scene_cam": rng.integers(0, 255, (4, 8, 8, 3), dtype=np.uint8)},
        done=np.zeros(4, dtype=bool),
        reward=np.ones(4, dtype=np.float32),
    )
    back = ObsPacket.from_bytes(obs.to_bytes())
    assert back.step == 7 and back.num_envs == 4
    assert np.array_equal(back.state["joint_pos"], obs.state["joint_pos"])
    assert np.array_equal(back.images["scene_cam"], obs.images["scene_cam"])
    assert back.done is not None and back.done.shape == (4,)

    act = ActionPacket(step=7, action=rng.random((4, 6), dtype=np.float32))
    aback = ActionPacket.from_bytes(act.to_bytes())
    assert np.allclose(aback.action, act.action) and aback.step == 7

    # A packet of the wrong kind must be rejected, not silently misread.
    try:
        ActionPacket.from_bytes(obs.to_bytes())
    except ProtocolError:
        pass
    else:
        raise AssertionError("kind mismatch was not caught")

    print("schema demo OK: obs/action round-trip, kind mismatch rejected")


if __name__ == "__main__":
    demo()
