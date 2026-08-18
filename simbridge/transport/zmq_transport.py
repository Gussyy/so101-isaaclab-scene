# SPDX-License-Identifier: BSD-3-Clause
"""ZeroMQ REQ/REP transport, plus the server side a policy runs behind.

REQ/REP rather than PUB/SUB deliberately. Simulation is lock-step: the environment cannot take
a step until it has an action, so the request/reply pairing *is* the control flow, and REQ/REP
enforces it. PUB/SUB would silently drop or reorder under load, which for a control loop means
acting on a stale observation -- the kind of bug that looks like a bad policy rather than a bad
socket.

The timeout matters for the same reason. A blocking recv with no deadline turns a crashed
policy server into a hung simulator with no error, so the client fails loudly instead.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import zmq

from simbridge.interfaces import Transport
from simbridge.schema import ActionPacket, ObsPacket

DEFAULT_ENDPOINT = "tcp://127.0.0.1:5555"


class ZmqClientTransport(Transport):
    """Environment side: ships observations out, waits for actions back.

    Args:
        endpoint: ZeroMQ endpoint, e.g. ``tcp://127.0.0.1:5555``.
        timeout_ms: How long to wait for a reply before raising.
        retries: Reconnect attempts on timeout. ZeroMQ REQ sockets latch into a broken state
            after a timeout, so recovery requires closing and re-opening the socket, not just
            retrying the recv.
    """

    def __init__(self, endpoint: str = DEFAULT_ENDPOINT, timeout_ms: int = 30_000, retries: int = 2) -> None:
        self.endpoint = endpoint
        self.timeout_ms = int(timeout_ms)
        self.retries = int(retries)
        self._ctx = zmq.Context.instance()
        self._sock: zmq.Socket | None = None
        self._connect()

    def _connect(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
        self._sock = self._ctx.socket(zmq.REQ)
        self._sock.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.connect(self.endpoint)

    def request(self, obs: ObsPacket) -> ActionPacket:
        payload = obs.to_bytes()
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            try:
                assert self._sock is not None
                self._sock.send(payload)
                return ActionPacket.from_bytes(self._sock.recv())
            except zmq.Again as exc:
                last = exc
                if attempt < self.retries:
                    # REQ is stuck after a timeout; a fresh socket is the only way back.
                    self._connect()
        raise TimeoutError(
            f"no reply from policy server at {self.endpoint} within {self.timeout_ms} ms "
            f"after {self.retries + 1} attempts"
        ) from last

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close(linger=0)
            self._sock = None


class ZmqPolicyServer:
    """Policy side: receives observations, replies with actions.

    Runs in whatever process the model wants -- its own CUDA context, its own framework, its own
    machine. The simulator holds only a socket.

    Args:
        fn: Maps an :class:`ObsPacket` to ``(num_envs, action_dim)`` or
            ``(horizon, num_envs, action_dim)`` actions.
        endpoint: Endpoint to bind.
    """

    def __init__(self, fn: Callable[[ObsPacket], np.ndarray], endpoint: str = DEFAULT_ENDPOINT) -> None:
        self._fn = fn
        self.endpoint = endpoint
        self._ctx = zmq.Context.instance()
        self._sock = self._ctx.socket(zmq.REP)
        self._sock.setsockopt(zmq.LINGER, 0)
        self._sock.bind(endpoint)
        self._stop = False

    def serve_forever(self, log_every: int = 0) -> None:
        n, t0 = 0, time.time()
        while not self._stop:
            try:
                obs = ObsPacket.from_bytes(self._sock.recv())
            except zmq.ContextTerminated:
                break
            action = np.asarray(self._fn(obs), dtype=np.float32)
            self._sock.send(ActionPacket(step=obs.step, action=action).to_bytes())
            n += 1
            if log_every and n % log_every == 0:
                print(f"[policy-server] {n} requests, {n / max(1e-9, time.time() - t0):.1f} req/s", flush=True)

    def stop(self) -> None:
        self._stop = True

    def close(self) -> None:
        self._sock.close(linger=0)
