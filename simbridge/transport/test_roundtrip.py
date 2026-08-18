# SPDX-License-Identifier: BSD-3-Clause
"""Cross-process check for the ZeroMQ transport: real socket, real serialisation.

Covers the failure path too. A policy server that dies must surface as a loud TimeoutError,
not a simulator that hangs forever on recv.
"""

from __future__ import annotations

import sys
import threading
import time

import numpy as np

sys.path.insert(0, ".")

from simbridge.interfaces import RemoteActionSource  # noqa: E402
from simbridge.schema import ObsPacket  # noqa: E402
from simbridge.transport import InProcessTransport, ZmqClientTransport, ZmqPolicyServer  # noqa: E402

ENDPOINT = "tcp://127.0.0.1:5599"


def _policy(obs: ObsPacket) -> np.ndarray:
    """Toy policy: act on the mean of joint_pos so we can verify payload fidelity."""
    jp = obs.state["joint_pos"]
    return np.repeat(jp.mean(axis=1, keepdims=True), 6, axis=1).astype(np.float32)


def main() -> None:
    # --- in-process transport -----------------------------------------
    obs = ObsPacket(step=3, num_envs=4, state={"joint_pos": np.arange(24, dtype=np.float32).reshape(4, 6)})
    src = RemoteActionSource(InProcessTransport(_policy), action_horizon=1)
    a = src.advance(obs)
    assert a.shape == (4, 6), a.shape
    assert np.allclose(a[:, 0], obs.state["joint_pos"].mean(axis=1))
    print("  in-process transport OK", a.shape)

    # --- zmq across threads (real sockets, real msgpack) ---------------
    server = ZmqPolicyServer(_policy, endpoint=ENDPOINT)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    client = ZmqClientTransport(ENDPOINT, timeout_ms=5000)
    rsrc = RemoteActionSource(client, action_horizon=1)

    # payload fidelity, including a realistic image batch
    big = ObsPacket(
        step=11,
        num_envs=8,
        state={"joint_pos": np.random.rand(8, 6).astype(np.float32)},
        images={"scene_cam": np.random.randint(0, 255, (8, 128, 128, 3), dtype=np.uint8)},
    )
    out = rsrc.advance(big)
    assert out.shape == (8, 6), out.shape
    assert np.allclose(out[:, 0], big.state["joint_pos"].mean(axis=1), atol=1e-5)
    print("  zmq round-trip OK", out.shape, "(incl. 8x128x128x3 image batch)")

    # throughput on a realistic visuomotor payload
    n = 40
    t0 = time.time()
    for i in range(n):
        big.step = 100 + i
        rsrc.reset()
        rsrc.advance(big)
    dt = time.time() - t0
    mb = 8 * 128 * 128 * 3 / 1e6
    print(f"  zmq throughput: {n/dt:.0f} req/s on {mb:.1f} MB payloads ({mb*n/dt:.0f} MB/s)")

    client.close()
    server.stop()
    server.close()

    # --- failure path: dead server must raise, not hang ---------------
    dead = ZmqClientTransport("tcp://127.0.0.1:5601", timeout_ms=400, retries=1)
    t0 = time.time()
    try:
        dead.request(ObsPacket(step=0, num_envs=1, state={"joint_pos": np.zeros((1, 6), np.float32)}))
    except TimeoutError as exc:
        print(f"  dead server raised TimeoutError after {time.time()-t0:.1f}s (did not hang): OK")
        assert "5601" in str(exc)
    else:
        raise AssertionError("a dead policy server did not raise")
    finally:
        dead.close()

    print("transport round-trip demo OK")


if __name__ == "__main__":
    main()
