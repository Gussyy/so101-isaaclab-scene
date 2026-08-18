# SPDX-License-Identifier: BSD-3-Clause
"""Zero-copy, same-process transport.

Exists so that "run the policy in-process" and "run the policy behind a socket" are the same
code path with a different transport, rather than two code paths that drift. Skips
serialisation entirely -- there is no wire, so paying msgpack costs would be pure waste.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from simbridge.interfaces import Transport
from simbridge.schema import ActionPacket, ObsPacket


class InProcessTransport(Transport):
    """Calls a local function instead of crossing a process boundary.

    Args:
        fn: Maps an :class:`ObsPacket` to actions -- either a raw ``(num_envs, action_dim)``
            array or a full :class:`ActionPacket`.
    """

    def __init__(self, fn: Callable[[ObsPacket], np.ndarray | ActionPacket]) -> None:
        self._fn = fn

    def request(self, obs: ObsPacket) -> ActionPacket:
        out = self._fn(obs)
        if isinstance(out, ActionPacket):
            return out
        return ActionPacket(step=obs.step, action=np.asarray(out, dtype=np.float32))
