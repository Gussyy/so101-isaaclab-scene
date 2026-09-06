# SPDX-License-Identifier: BSD-3-Clause
"""Serve a VR controller as gripper poses, over the socket everything else already uses.

The Quest's own browser opens ``scripts/vr/index.html``, reads the right controller through
WebXR, and streams its pose over a WebSocket to this process. This process turns that into the
8-wide action -- ``[x, y, z, qx, qy, qz, qw, jaw]`` -- and answers the simulator's ZeroMQ
requests with it, exactly as ``scripts/hand_gripper_server.py`` does for a webcam hand. The
simulator does not know which one is on the other end.

    # this venv (websockets, cryptography, zmq, msgpack, scipy: all already installed)
    python scripts/vr_gripper_server.py                # prints the URL to open on the Quest
    python scripts/vr_gripper_server.py --fake         # no headset: a scripted controller
    python scripts/vr_gripper_server.py --no-tls       # http/ws on localhost, for a desktop browser

    # the simulator, in another terminal
    python scripts/run.py --config configs/vr_teleop.yaml --viz kit --steps 0

Controls, on the right Touch controller:

    GRIP (squeeze)   hold to move the arm -- a clutch, like lifting a mouse. Let go and the
                     arm stays put while you reposition your hand.
    TRIGGER          close the jaw while held.
    A / X            re-centre: the arm goes to `--home`, and motion resumes from there.

Frames
------
WebXR ``local-floor`` is +x right, +y up, -z forward. The scene is +x forward, +y left, +z up.
:func:`webxr_to_sim` is the one place that mapping lives, and :func:`demo` checks it with a
controller held out in front, to the left, and raised. Orientation is pinned to jaws-down by
default -- ``--track-rot`` uses the controller's orientation relative to where it was when the
clutch engaged -- for the same reason the webcam and Quest-hand drivers pin theirs: a wrong
rotation is unusable, and the controller's rest orientation has not been measured on hardware.

WebXR needs a secure context. The page is served over HTTPS with a self-signed certificate
(generated once into ``scripts/vr/``, gitignored); the Quest browser asks once, then remembers.
"""

from __future__ import annotations

import argparse
import ipaddress
import json
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from hand_tracker import Clutch  # noqa: E402  -- pure numpy; no cv2, no mediapipe

from simbridge.schema import ObsPacket  # noqa: E402
from simbridge.transport import ZmqPolicyServer  # noqa: E402

ACTION_DIM = 8
PAGE = Path(__file__).resolve().parent / "vr" / "index.html"
CERT = Path(__file__).resolve().parent / "vr" / "cert.pem"
KEY = Path(__file__).resolve().parent / "vr" / "key.pem"

# Jaws pointing straight down, (x, y, z, w). Same four numbers hand_gripper_server.py uses.
JAWS_DOWN = np.array([0.70710678, 0.0, 0.0, 0.70710678])

# sim = P @ webxr. Rows: sim_x = -webxr_z (forward), sim_y = -webxr_x (left), sim_z = webxr_y (up).
_P = np.array([[0.0, 0.0, -1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def webxr_to_sim(pos, quat_xyzw):
    """A WebXR pose -> the scene's frame. Both quaternions are (x, y, z, w)."""
    from scipy.spatial.transform import Rotation as R

    p = _P @ np.asarray(pos, dtype=np.float64)
    # A rotation expressed in one basis, re-expressed in another: P R P^T. Permuting the
    # quaternion's components instead is wrong for exactly the same reason it looks right.
    r = R.from_quat(np.asarray(quat_xyzw, dtype=np.float64))
    q = R.from_matrix(_P @ r.as_matrix() @ _P.T).as_quat()
    return p, q


class VrDriver:
    """Latest controller sample -> action. Holds the last action when nothing has arrived.

    Holding is the important behaviour, as with the webcam driver: the page reconnects, the
    controller loses tracking behind the operator's back, a frame is dropped -- answering any
    of those with zeros would fling the arm to the origin.
    """

    def __init__(self, home=(0.27, 0.0, 0.12), scale: float = 1.0, smooth: float = 0.5,
                 track_rot: bool = False) -> None:
        self.clutch = Clutch(np.asarray(home, dtype=np.float64), scale=scale)
        self.smooth = float(smooth)
        self.track_rot = bool(track_rot)
        self._lock = threading.Lock()
        self._sample = None                     # latest dict from the page, in the sim frame
        self._squeezing = False
        self._recentre_down = False
        self._q_anchor = None                   # controller orientation when the clutch engaged
        self._pos = None
        self.last = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last[:3] = home
        self.last[3:7] = JAWS_DOWN
        self.last[7] = 1.0
        self.received = 0
        self.served = 0

    # -- from the WebSocket thread ------------------------------------------------------------
    def push(self, msg: dict) -> None:
        pos, quat = webxr_to_sim(msg["pos"], msg.get("quat", (0.0, 0.0, 0.0, 1.0)))
        # The raw numbers, for the first contact with real hardware: the first few messages and
        # then one in every 300. This is the only place the "derived, not measured" facts about
        # the page -- the frame, the button indices -- can be checked against a real controller.
        if self.received < 5 or self.received % 300 == 0:
            p = msg["pos"]
            print(f"\n[vr] page msg {self.received}: webxr pos=({p[0]:+.3f},{p[1]:+.3f},{p[2]:+.3f})"
                  f" -> sim ({pos[0]:+.3f},{pos[1]:+.3f},{pos[2]:+.3f})  squeeze={msg.get('squeeze', 0):.2f}"
                  f" trigger={msg.get('trigger', 0):.2f} recentre={msg.get('recentre', False)}", flush=True)
        with self._lock:
            self._sample = {
                "pos": pos, "quat": quat,
                "trigger": float(msg.get("trigger", 0.0)),
                "squeeze": float(msg.get("squeeze", 0.0)),
                "recentre": bool(msg.get("recentre", False)),
            }
            self.received += 1

    # -- from the ZeroMQ thread ----------------------------------------------------------------
    def __call__(self, obs: ObsPacket) -> np.ndarray:
        with self._lock:
            s = self._sample
        if s is not None:
            raw = s["pos"]
            squeezing = s["squeeze"] > 0.5
            if squeezing != self._squeezing:            # clutch on the grip button's edges
                self.clutch.toggle(raw)
                self._squeezing = squeezing
                self._q_anchor = s["quat"] if squeezing else self._q_anchor
            recentre = s["recentre"]
            if recentre and not self._recentre_down:    # one action per press, not per frame
                self.clutch.recentre(raw)
                self._pos = None
            self._recentre_down = recentre

            out = self.clutch.update(raw)
            self._pos = out if self._pos is None or self.smooth >= 1.0 else (
                (1.0 - self.smooth) * self._pos + self.smooth * out)
            self.last[:3] = self._pos
            if self.track_rot and self._squeezing and self._q_anchor is not None:
                from scipy.spatial.transform import Rotation as R

                # Orientation relative to where the controller was when the grip closed, applied
                # on top of jaws-down. Absolute controller orientation would make "rest" mean
                # "pointing the jaws wherever the controller happens to point", which is never
                # down.
                rel = R.from_quat(s["quat"]) * R.from_quat(self._q_anchor).inv()
                self.last[3:7] = (rel * R.from_quat(JAWS_DOWN)).as_quat()
            self.last[7] = -1.0 if s["trigger"] > 0.5 else 1.0

        self.served += 1
        if self.served % 60 == 0:
            x, y, z = self.last[:3]
            state = "ENGAGED" if self.clutch.engaged else "held"
            print(f"\r  {state:<8} pos=({x:+.3f}, {y:+.3f}, {z:+.3f})  "
                  f"jaw={'closed' if self.last[7] < 0 else 'open  '}  "
                  f"page msgs={self.received}   ", end="", flush=True)
        return np.tile(self.last, (obs.num_envs, 1)).astype(np.float32)


class FakeController:
    """A scripted right hand, in WebXR's frame, so the whole chain runs with no headset.

    The motion is a person reaching forward and down to the block, gripping, lifting, carrying
    left, releasing -- with the clutch held the whole time. It feeds :meth:`VrDriver.push` with
    the same dict the page sends, so everything from the frame conversion onward is the
    production path. What it cannot prove is the page itself in a real Quest browser.
    """

    KEYS = [                                   # (t, dx_right, dy_up, dz_forward(-z), squeeze, trigger)
        (0.0, 0.00, 0.00, 0.00, 0.0, 0.0),
        (1.0, 0.00, 0.00, 0.00, 1.0, 0.0),     # grip: clutch engages here, this pose = home
        (3.0, 0.00, -0.10, 0.00, 1.0, 0.0),    # down onto the block (home is 0.12 up)
        (4.0, 0.00, -0.10, 0.00, 1.0, 1.0),    # trigger: close
        (6.0, 0.00, -0.02, 0.00, 1.0, 1.0),    # lift
        (8.0, -0.10, -0.02, 0.00, 1.0, 1.0),   # carry to the operator's left (-x in WebXR)
        (9.0, -0.10, -0.08, 0.00, 1.0, 1.0),   # lower
        (10.0, -0.10, -0.08, 0.00, 1.0, 0.0),  # release
        (12.0, -0.10, -0.02, 0.00, 1.0, 0.0),  # back up
    ]

    def __init__(self, driver: VrDriver, rate_hz: float = 72.0) -> None:
        self.driver, self.dt = driver, 1.0 / rate_hz
        self.t0 = None       # set when the simulator first asks -- see run_forever

    def sample(self, t: float) -> dict:
        k = self.KEYS
        if t >= k[-1][0]:
            _, dx, dy, dz, sq, tr = k[-1]
        else:
            for (t0, x1, y1, z1, s0, r0), (t1, x2, y2, z2, s1, r1) in zip(k, k[1:]):
                if t < t1:
                    a = (t - t0) / (t1 - t0)
                    dx, dy, dz = x1 + a * (x2 - x1), y1 + a * (y2 - y1), z1 + a * (z2 - z1)
                    sq, tr = (s0 if a < 0.5 else s1), (r0 if a < 0.5 else r1)
                    break
        # WebXR: the controller starts 0.4 m in front of (-z) and 1.0 m above the floor.
        return {"pos": [dx, 1.0 + dy, -0.4 + dz], "quat": [0.0, 0.0, 0.0, 1.0],
                "squeeze": sq, "trigger": tr, "recentre": False}

    def run_forever(self) -> None:
        # Hold the first pose until the simulator's first request. Isaac Sim takes ~20 s to boot,
        # and a clock started at bridge launch had the whole 12 s script play out to nobody: the
        # first thing the simulator ever saw was the final, static pose, grip already held, so
        # the clutch anchored there and the arm never received a moving command. Measured.
        while self.driver.served == 0:
            self.driver.push(self.sample(0.0))
            time.sleep(self.dt)
        self.t0 = time.time()
        while True:
            self.driver.push(self.sample(time.time() - self.t0))
            time.sleep(self.dt)


# ---------------------------------------------------------------- HTTPS + WebSocket

def lan_ip() -> str:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))            # no packet is sent; this just picks the interface
        return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"
    finally:
        s.close()


def ensure_cert(ip: str) -> None:
    """A self-signed certificate for the LAN address, made once. WebXR refuses plain http."""
    if CERT.exists() and KEY.exists():
        return
    import datetime as dt

    from cryptography import x509
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import rsa
    from cryptography.x509.oid import NameOID

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    name = x509.Name([x509.NameAttribute(NameOID.COMMON_NAME, "so101-vr-bridge")])
    san = x509.SubjectAlternativeName([
        x509.DNSName("localhost"),
        x509.IPAddress(ipaddress.ip_address("127.0.0.1")),
        x509.IPAddress(ipaddress.ip_address(ip)),
    ])
    now = dt.datetime.now(dt.timezone.utc)
    cert = (x509.CertificateBuilder().subject_name(name).issuer_name(name)
            .public_key(key.public_key()).serial_number(x509.random_serial_number())
            .not_valid_before(now - dt.timedelta(days=1)).not_valid_after(now + dt.timedelta(days=3650))
            .add_extension(san, critical=False).sign(key, hashes.SHA256()))
    CERT.parent.mkdir(parents=True, exist_ok=True)
    CERT.write_bytes(cert.public_bytes(serialization.Encoding.PEM))
    KEY.write_bytes(key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8,
                                      serialization.NoEncryption()))
    print(f"[vr] wrote a self-signed certificate for {ip} to {CERT.parent}")


def serve_page_and_socket(driver: VrDriver, host: str, port: int, tls: bool) -> None:
    """One port: GET / serves the page, an upgrade on /ws is the controller stream."""
    import ssl

    from websockets.datastructures import Headers
    from websockets.http11 import Response
    from websockets.sync.server import serve

    page = PAGE.read_bytes()

    def process_request(connection, request):
        if request.headers.get("Upgrade", "").lower() == "websocket":
            return None                                   # let the handshake proceed
        if request.path in ("/", "/index.html"):
            return Response(200, "OK", Headers([("Content-Type", "text/html; charset=utf-8"),
                                                ("Content-Length", str(len(page)))]), page)
        return Response(404, "Not Found", Headers(), b"")

    def handler(ws):
        print(f"\n[vr] page connected from {ws.remote_address[0]}", flush=True)
        try:
            for raw in ws:
                try:
                    driver.push(json.loads(raw))
                except (ValueError, KeyError, TypeError) as exc:
                    print(f"\n[vr] bad message ignored: {exc}", flush=True)
        finally:
            print("\n[vr] page disconnected; holding the last pose", flush=True)

    ctx = None
    if tls:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(CERT, KEY)
    server = serve(handler, host, port, ssl=ctx, process_request=process_request)
    server.serve_forever()


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve a VR controller as gripper actions.")
    ap.add_argument("--endpoint", default="tcp://127.0.0.1:5555", help="ZeroMQ, for the simulator")
    ap.add_argument("--port", type=int, default=8443, help="HTTPS/WSS port the Quest connects to")
    ap.add_argument("--no-tls", action="store_true",
                    help="plain http/ws. WebXR then works only on localhost -- for a desktop test")
    ap.add_argument("--fake", action="store_true", help="scripted controller, no page, no headset")
    ap.add_argument("--home", type=float, nargs=3, default=[0.27, 0.0, 0.12],
                    help="grasp point before the clutch is first engaged, robot root frame. "
                         "0.27 m out: the only place this arm reaches table height (docs/VR.md)")
    ap.add_argument("--scale", type=float, default=1.0, help="hand travel : gripper travel")
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA factor; 1.0 = off")
    ap.add_argument("--track-rot", action="store_true",
                    help="follow the controller's orientation (relative to the grip anchor)")
    args = ap.parse_args()

    driver = VrDriver(home=tuple(args.home), scale=args.scale, smooth=args.smooth,
                      track_rot=args.track_rot)

    if args.fake:
        threading.Thread(target=FakeController(driver).run_forever, daemon=True).start()
        print("[vr] FAKE controller: reach, grip, lift, carry left, release (12 s, then hold)")
    else:
        ip = lan_ip()
        host = "127.0.0.1" if args.no_tls else "0.0.0.0"
        if not args.no_tls:
            ensure_cert(ip)
        scheme = "http" if args.no_tls else "https"
        shown = "localhost" if args.no_tls else ip
        threading.Thread(target=serve_page_and_socket, args=(driver, host, args.port, not args.no_tls),
                         daemon=True).start()
        print(f"[vr] open on the Quest:  {scheme}://{shown}:{args.port}/")
        if not args.no_tls:
            print("[vr] the certificate is self-signed: accept it once (Advanced -> proceed)")
        print("[vr] GRIP = move,  TRIGGER = close jaw,  A/X = re-centre")

    print(f"[vr] serving actions on {args.endpoint}\n")
    server = ZmqPolicyServer(driver, endpoint=args.endpoint)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        print("\n[vr] stopped")
        server.close()


def demo() -> None:
    """Self-check: the frame mapping, the clutch on the grip button, the trigger on the jaw."""
    # Forward in WebXR is -z; it has to come out as +x. Left is -x; it has to be +y. Up stays up.
    p, q = webxr_to_sim([0.0, 0.0, -1.0], [0, 0, 0, 1])
    assert np.allclose(p, [1.0, 0.0, 0.0]) and np.allclose(q, [0, 0, 0, 1]), (p, q)
    p, _ = webxr_to_sim([-1.0, 0.0, 0.0], [0, 0, 0, 1])
    assert np.allclose(p, [0.0, 1.0, 0.0]), p
    p, _ = webxr_to_sim([0.0, 1.0, 0.0], [0, 0, 0, 1])
    assert np.allclose(p, [0.0, 0.0, 1.0]), p
    # A quarter turn about WebXR's up axis must be a quarter turn about the scene's up axis.
    from scipy.spatial.transform import Rotation as R

    _, q = webxr_to_sim([0, 0, 0], R.from_euler("y", 90, degrees=True).as_quat())
    assert np.allclose(R.from_quat(q).as_euler("xyz", degrees=True), [0, 0, 90], atol=1e-6), q

    d = VrDriver(home=(0.2, 0.0, 0.15), smooth=1.0)
    obs = ObsPacket(step=0, num_envs=1, state={}, images={})
    assert np.allclose(d(obs)[0, :3], [0.2, 0.0, 0.15]), "nothing received: hold home"
    d.push({"pos": [0, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 0, "trigger": 0})
    d(obs)
    d.push({"pos": [0, 1, -0.5], "quat": [0, 0, 0, 1], "squeeze": 0, "trigger": 0})
    assert np.allclose(d(obs)[0, :3], [0.2, 0.0, 0.15]), "grip not held: motion must be ignored"
    d.push({"pos": [0, 1, -0.5], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0})
    d(obs)                                               # engage: anchors here, no jump
    d.push({"pos": [0, 1, -0.6], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 1})
    a = d(obs)[0]
    assert np.allclose(a[:3], [0.3, 0.0, 0.15]), a      # 0.1 m further forward (-z) -> +x
    assert a[7] == -1.0, "trigger must close the jaw"
    d.push({"pos": [0, 1, -0.6], "quat": [0, 0, 0, 1], "squeeze": 0, "trigger": 0})
    d(obs)
    d.push({"pos": [5, 5, 5], "quat": [0, 0, 0, 1], "squeeze": 0, "trigger": 0})
    assert np.allclose(d(obs)[0, :3], [0.3, 0.0, 0.15]), "released: the arm must stay put"
    assert FakeController(d).sample(4.5)["trigger"] == 1.0
    print("vr_gripper_server demo OK: frames, clutch edges, jaw, hold")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
