# SPDX-License-Identifier: BSD-3-Clause
"""Trivial sources. Not toys -- these are the controls you diff a real policy against.

A policy that beats ZeroSource by nothing is not learning, and that is a much faster thing to
check than staring at a reward curve.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simbridge.interfaces import ActionSource
from simbridge.registry import register_source
from simbridge.schema import ObsPacket


@register_source("zero")
class ZeroSource(ActionSource):
    """Holds every joint at its default target. The do-nothing baseline."""

    def __init__(self, action_dim: int, action_horizon: int = 1, **_: object) -> None:
        super().__init__(action_horizon=action_horizon)
        self.action_dim = int(action_dim)

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        return np.zeros((obs.num_envs, self.action_dim), dtype=np.float32)


@register_source("random")
class ScriptedSource(ActionSource):
    """Uniform random actions, or any user function of the observation.

    Args:
        action_dim: Width of the action vector.
        fn: Optional ``(ObsPacket) -> (num_envs, action_dim)``. Defaults to uniform noise.
        scale: Amplitude of the default noise.
        seed: Fixed so a "random" baseline is reproducible between runs.
    """

    def __init__(
        self,
        action_dim: int,
        fn: Callable[[ObsPacket], np.ndarray] | None = None,
        scale: float = 1.0,
        seed: int = 0,
        action_horizon: int = 1,
        **_: object,
    ) -> None:
        super().__init__(action_horizon=action_horizon)
        self.action_dim = int(action_dim)
        self._fn = fn
        self._scale = float(scale)
        self._rng = np.random.default_rng(seed)

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        if self._fn is not None:
            return self._fn(obs)
        return (self._rng.uniform(-1.0, 1.0, (obs.num_envs, self.action_dim)) * self._scale).astype(np.float32)

@register_source("gripper_cycle")
class GripperCycleSource(ActionSource):
    """Hold the arm still and cycle the gripper open/closed. For checking a gripper works.

    The arm has to be driven too, not just left alone: the action is an offset from the default
    pose, so emitting zeros for the arm joints holds the start pose, which is what we want while
    watching the fingers.

    Args:
        action_dim: Width of the action vector; the last element is taken to be the gripper.
        period: Steps per full open-close cycle.
        open_first: Whether the cycle starts open.
    """

    def __init__(
        self,
        action_dim: int = 6,
        period: int = 60,
        open_first: bool = True,
        action_horizon: int = 1,
        **_: object,
    ) -> None:
        super().__init__(action_horizon=action_horizon)
        self.action_dim = int(action_dim)
        self.period = max(int(period), 2)
        self.open_first = bool(open_first)
        self._step = 0

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        cmd = np.zeros(self.action_dim, dtype=np.float32)
        phase = (self._step // (self.period // 2)) % 2 == 0
        cmd[-1] = 1.0 if (phase == self.open_first) else -1.0
        self._step += 1
        return np.tile(cmd, (obs.num_envs, 1))

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        # Deliberately not resetting the phase: an episode reset mid-cycle should not restart
        # the cycle, or a short episode length would freeze the gripper in one state forever.

@register_source("keyframes")
class KeyframeSource(ActionSource):
    """Replay a fixed list of held actions. A scripted demo, not a policy.

    Each entry is ``{steps: N, action: [...]}``: hold that action for N steps, then move to the
    next. The list does not loop -- the last frame is held forever, because a grasp that loops
    lets go of what it is holding.

    An action here is an *offset from the robot's default pose*, since the task uses
    ``JointPositionActionCfg(scale=0.5, use_default_offset=True)``. So the joint target is
    ``default + 0.5 * action``, and an action of zero holds the start pose rather than driving
    the joints to zero. The last element is the binary gripper: +1 open, -1 closed.

    Short actions are padded with zeros, so ``action: [0, 0.4]`` means "shoulder_lift, everything
    else unchanged" without writing four trailing zeros. A too-long one is an error rather than a
    silent truncation -- the usual way to write this wrong is to forget the gripper slot.

    Unlike ``gripper_cycle``, a reset DOES restart the script: a scripted demo that resumes
    mid-grasp after the episode restarts is showing you a lift with nothing in the jaw.
    """

    def __init__(
        self,
        action_dim: int = 6,
        keyframes: list[dict] | None = None,
        action_horizon: int = 1,
        **_: object,
    ) -> None:
        super().__init__(action_horizon=action_horizon)
        self.action_dim = int(action_dim)
        if not keyframes:
            raise ValueError(
                "the keyframes source needs a 'keyframes' list, e.g.\n"
                "  control:\n    source: keyframes\n"
                "    keyframes:\n      - {steps: 40, action: [0, 0, 0, 0, 0, 1]}"
            )
        self._frames: list[tuple[int, np.ndarray]] = []
        for i, frame in enumerate(keyframes):
            steps = int(frame.get("steps", 1))
            if steps < 1:
                raise ValueError(f"keyframe {i}: steps must be >= 1, got {steps}")
            act = list(frame.get("action", []))
            if len(act) > self.action_dim:
                raise ValueError(
                    f"keyframe {i}: action has {len(act)} entries, the action space is "
                    f"{self.action_dim}"
                )
            padded = np.zeros(self.action_dim, dtype=np.float32)
            padded[: len(act)] = act
            self._frames.append((steps, padded))
        self._step = 0

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        t = self._step
        cmd = self._frames[-1][1]
        for steps, action in self._frames:
            if t < steps:
                cmd = action
                break
            t -= steps
        self._step += 1
        return np.tile(cmd, (obs.num_envs, 1))

    def reset(self, env_ids=None) -> None:
        super().reset(env_ids)
        self._step = 0


def demo() -> None:
    """Self-check: a keyframe script has to hold each frame for exactly its own step count."""
    from simbridge.schema import ObsPacket

    src = KeyframeSource(
        action_dim=6,
        keyframes=[
            {"steps": 2, "action": [0, 0, 0, 0, 0, 1]},
            {"steps": 3, "action": [0, 0.4]},
        ],
    )
    packet = ObsPacket(step=0, num_envs=1, state={}, images={})
    seen = [src._predict_chunk(packet)[0].copy() for _ in range(7)]

    assert [a[5] for a in seen[:2]] == [1.0, 1.0], seen[:2]
    assert [a[1] for a in seen[2:5]] == [0.4] * 3, seen[2:5]
    # Past the end it holds the last frame rather than looping back to the open gripper.
    assert seen[5][1] == 0.4 and seen[6][5] == 0.0, (seen[5], seen[6])
    # Short actions pad, they do not shift.
    assert seen[2][0] == 0.0 and seen[2][5] == 0.0, seen[2]

    src.reset()
    assert src._predict_chunk(packet)[0][5] == 1.0, "reset must restart the script"

    for bad, why in (
        ({"steps": 0, "action": [1]}, "steps"),
        ({"steps": 1, "action": [0] * 7}, "action space"),
    ):
        try:
            KeyframeSource(action_dim=6, keyframes=[bad])
        except ValueError as exc:
            assert why in str(exc), exc
        else:
            raise AssertionError(f"{bad} should have been refused")

    print("basic sources: keyframe self-check passed")


if __name__ == "__main__":
    demo()
