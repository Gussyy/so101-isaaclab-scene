# SPDX-License-Identifier: BSD-3-Clause
"""Sources whose actions arrive over a transport -- a policy server, or a teleop process."""

from __future__ import annotations

from simbridge.interfaces import RemoteActionSource
from simbridge.registry import register_source
from simbridge.transport import ZmqClientTransport


@register_source("zmq")
def make_remote_source(
    endpoint: str = "tcp://127.0.0.1:5555",
    timeout_ms: int = 30_000,
    retries: int = 2,
    action_horizon: int = 1,
    **_: object,
) -> RemoteActionSource:
    """Attach to a policy server over ZeroMQ.

    The server can be anything that speaks the schema: a flow-matching visuomotor policy in its
    own CUDA context, a human teleop bridge, or another simulator. The environment neither knows
    nor cares which.
    """
    return RemoteActionSource(
        ZmqClientTransport(endpoint=endpoint, timeout_ms=timeout_ms, retries=retries),
        action_horizon=action_horizon,
    )
