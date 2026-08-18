# SPDX-License-Identifier: BSD-3-Clause
"""The two interfaces everything else plugs into: :class:`ActionSource` and :class:`Transport`.

The shape here deliberately follows NVlabs/RoboLab's ``InferenceClient`` (four required hooks,
action chunking handled by the base class). RoboLab itself cannot be installed on this stack --
it pins Isaac Sim 5.0/5.1, Isaac Lab 2.2/2.3, Python 3.11 and Ubuntu, against our Isaac Sim
6.0.1 / Isaac Lab 3.0 / Python 3.12 / Windows -- but its interface is proven against real
flow-matching VLA servers (pi0, pi0.5 via OpenPI), so it is worth matching rather than inventing.

Why chunking belongs in the base class
--------------------------------------
Generative action models do not emit one action per observation; they emit a short trajectory.
Diffusion Policy predicts ``Tp`` steps and executes ``Ta`` of them before re-planning, and pi0
serves horizons of 10-15. Running the model every control step would be both wasteful and
worse -- open-loop chunk execution is what gives these policies their temporal consistency.
So :meth:`ActionSource.advance` caches a chunk and steps through it, calling the expensive
:meth:`_predict_chunk` only when the cache runs dry.

A source that genuinely is per-step (teleop, a scripted tester, an RL MLP) simply returns a
chunk of length 1 and pays nothing for the machinery.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from simbridge.schema import ActionPacket, ObsPacket


class Transport(ABC):
    """Moves :class:`ObsPacket` / :class:`ActionPacket` between an environment and a driver.

    Kept separate from :class:`ActionSource` so the same policy code runs in-process during
    development and over a socket in deployment, with no change to the policy.
    """

    @abstractmethod
    def request(self, obs: ObsPacket) -> ActionPacket:
        """Send an observation, block until the matching action returns."""

    def close(self) -> None:
        """Release sockets/handles. Safe to call twice."""


class ActionSource(ABC):
    """Anything that can drive an environment.

    Subclasses implement :meth:`_predict_chunk`; the base class handles chunk caching, step
    accounting and the horizon bookkeeping.

    Args:
        action_horizon: How many actions of each predicted chunk to execute before re-planning
            (``Ta`` in Diffusion Policy terms). 1 means re-plan every step.
    """

    def __init__(self, action_horizon: int = 1) -> None:
        if action_horizon < 1:
            raise ValueError("action_horizon must be >= 1")
        self.action_horizon = int(action_horizon)
        self._chunk: np.ndarray | None = None  # (horizon, num_envs, action_dim)
        self._cursor: int = 0

    # ---- required hook -------------------------------------------------

    @abstractmethod
    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        """Produce actions for the next ``>=1`` steps.

        Returns:
            Either ``(num_envs, action_dim)`` for a single step, or
            ``(horizon, num_envs, action_dim)`` for a chunk. The base class normalises both.
        """

    # ---- optional hooks ------------------------------------------------

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        """Drop any cached chunk. Called on episode boundaries.

        A stale chunk replayed across a reset is a real and quiet failure mode: the actions were
        planned for the previous episode's object pose, so the policy looks subtly broken rather
        than obviously broken. Always clear.
        """
        self._chunk = None
        self._cursor = 0

    def close(self) -> None:
        """Release any resources."""

    # ---- driver loop ---------------------------------------------------

    def advance(self, obs: ObsPacket) -> np.ndarray:
        """Return one ``(num_envs, action_dim)`` action, re-planning when the chunk is spent."""
        if self._chunk is None or self._cursor >= min(self.action_horizon, len(self._chunk)):
            chunk = np.asarray(self._predict_chunk(obs), dtype=np.float32)
            if chunk.ndim == 2:  # single step -> chunk of length 1
                chunk = chunk[None, ...]
            if chunk.ndim != 3:
                raise ValueError(
                    f"_predict_chunk must return (num_envs, act) or (horizon, num_envs, act); got {chunk.shape}"
                )
            self._chunk, self._cursor = chunk, 0

        action = self._chunk[self._cursor]
        self._cursor += 1
        return action

    def __repr__(self) -> str:
        return f"{type(self).__name__}(action_horizon={self.action_horizon})"


class RemoteActionSource(ActionSource):
    """An :class:`ActionSource` whose prediction happens in another process, over a transport.

    This is the RoboLab server-client split: the model runs wherever it likes (its own CUDA
    context, its own framework, even another machine) and the simulator holds only a socket.
    """

    def __init__(self, transport: Transport, action_horizon: int = 1) -> None:
        super().__init__(action_horizon=action_horizon)
        self.transport = transport

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        reply = self.transport.request(obs)
        if reply.step != obs.step:
            raise RuntimeError(
                f"driver replied to step {reply.step} while the env is at {obs.step}; "
                "the peer is out of sync and its actions are stale"
            )
        return reply.action

    def close(self) -> None:
        self.transport.close()


def demo() -> None:
    """Self-check: chunking executes Ta steps per prediction, and reset drops the cache."""

    class Counting(ActionSource):
        def __init__(self, horizon: int, chunk_len: int) -> None:
            super().__init__(action_horizon=horizon)
            self.calls = 0
            self.chunk_len = chunk_len

        def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
            self.calls += 1
            return np.full((self.chunk_len, obs.num_envs, 3), float(self.calls), dtype=np.float32)

    obs = ObsPacket(step=0, num_envs=2)

    # Ta=4 over chunks of 8: one prediction should cover 4 steps.
    src = Counting(horizon=4, chunk_len=8)
    vals = [src.advance(obs)[0, 0] for _ in range(8)]
    assert src.calls == 2, f"expected 2 predictions for 8 steps at Ta=4, got {src.calls}"
    assert vals[:4] == [1.0] * 4 and vals[4:] == [2.0] * 4, vals

    # reset must force a fresh prediction even mid-chunk.
    src.reset()
    src.advance(obs)
    assert src.calls == 3, "reset did not invalidate the cached chunk"

    # A per-step source (Ta=1, chunk of 1) predicts every step.
    per_step = Counting(horizon=1, chunk_len=1)
    for _ in range(5):
        per_step.advance(obs)
    assert per_step.calls == 5, per_step.calls

    print("interfaces demo OK: chunk reuse, reset invalidation, per-step degenerate case")


if __name__ == "__main__":
    demo()
