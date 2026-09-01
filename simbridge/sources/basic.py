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
