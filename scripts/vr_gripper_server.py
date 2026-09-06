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
    A / X            home: the arm glides back to its start pose (position and the way the
                     fingers point) over a few seconds, and motion resumes from there.
    B / Y            reset the scene: objects back to their start, arm to its rest pose.

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
import math
import socket
import sys
import threading
import time
from pathlib import Path

import numpy as np


def yaw_lock(pos, quat):
    """Turn the commanded orientation about the world's up so the fingers head toward `pos`.

    Five joints: base yaw, three pitches in one vertical plane, wrist roll. The fingers' heading
    IS the base yaw, and the base yaw is wherever the arm is reaching, so a commanded heading
    other than atan2(y, x) of the target is unreachable and the IK trades position for it
    (37 mm off at the tray, measured). Projecting it out costs nothing the arm could have done;
    pitch and roll -- the parts five joints can follow -- are kept. Skipped when the fingers
    point within ~11 degrees of vertical, where "heading" is noise.

    The fingers may point either way along that plane: at the shipped start pose they point
    BACK toward the base, hooked under the wrist (heading 180 degrees from the target). So the
    turn is the smallest one that puts the heading in the plane, wrapped to +-90 degrees --
    the first version turned the target half a circle and the arm contorted trying to follow.
    """
    from scipy.spatial.transform import Rotation as R

    r = R.from_quat(quat)
    f = r.apply([0.0, -1.0, 0.0])           # gripper_base -y is the finger direction (body_offset)
    if math.hypot(f[0], f[1]) < 0.2:
        return np.asarray(quat, dtype=np.float64)
    d = math.atan2(pos[1], pos[0]) - math.atan2(f[1], f[0])
    d = (d + math.pi / 2) % math.pi - math.pi / 2
    return (R.from_euler("z", d) * r).as_quat()

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class Clutch:
    """Holds the output still while disengaged, and re-anchors on every engage.

    ``home`` is where the output sits before the controller has ever been engaged, and where A
    puts it back. ``scale`` trades reachable volume against precision. (The same class as the
    webcam hand tracker's; copied rather than imported so this script runs from a clean clone.)
    """

    def __init__(self, home: np.ndarray, scale: float = 1.0) -> None:
        self.home = np.asarray(home, dtype=np.float64)
        self.scale = float(scale)
        self.engaged = False
        self._out = self.home.copy()     # last emitted position, held across disengage
        self._anchor = np.zeros(3)       # raw controller position at the moment of engaging

    def toggle(self, raw: np.ndarray) -> None:
        self.engaged = not self.engaged
        if self.engaged:
            self._anchor = np.asarray(raw, dtype=np.float64).copy()

    def recentre(self, raw: np.ndarray) -> None:
        self._out = self.home.copy()
        self._anchor = np.asarray(raw, dtype=np.float64).copy()

    def update(self, raw: np.ndarray) -> np.ndarray:
        if self.engaged:
            self._out = self._out + self.scale * (np.asarray(raw, dtype=np.float64) - self._anchor)
            self._anchor = np.asarray(raw, dtype=np.float64).copy()
        return self._out.copy()


from simbridge.schema import ActionPacket, ObsPacket  # noqa: E402
from simbridge.transport import ZmqPolicyServer  # noqa: E402

ACTION_DIM = 8
LOG_DIR = Path(__file__).resolve().parent.parent / "logs" / "vr"
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

    def __init__(self, home=None, scale: float = 1.0, smooth: float = 0.5,
                 track_rot: bool = True, log_dir: Path | None = None) -> None:
        # Home is the start pose: where the arm is when the simulator first reports, unless
        # `--home` says otherwise. So the first grip never jumps and A has somewhere to go.
        self._home_from_sim = home is None
        self.clutch = Clutch(np.zeros(3) if home is None else np.asarray(home, dtype=np.float64), scale=scale)
        self.smooth = float(smooth)
        self.track_rot = bool(track_rot)
        # Where the fingers actually point, from the simulator (root frame, xyzw). The operator's
        # rotation is applied RELATIVE to this, anchored at the moment the grip closes -- so the
        # arm never has to jump to an orientation it may not be able to reach, and turning the
        # controller turns the fingers from wherever they are.
        self.ee_quat: np.ndarray | None = None
        self._q_ee_anchor: np.ndarray | None = None
        self.frames: dict = {}                  # camera name -> latest frame; a sender thread encodes
        self._frames_lock = threading.Lock()
        self.cam_names: list[str] = []
        self._log = None
        if log_dir is not None:
            log_dir.mkdir(parents=True, exist_ok=True)
            path = log_dir / f"{time.strftime('%Y%m%d-%H%M%S')}-bridge.jsonl"
            self._log = open(path, "w", encoding="utf-8")
            print(f"[vr] session log: {path}")
        self._t0 = time.time()
        self._lock = threading.Lock()
        self._sample = None                     # latest dict from the page, in the sim frame
        self._squeezing = False
        self._recentre_down = False
        self._reset_down = False
        self._reset_pending = False             # B was pressed; say so on the next reply, once
        self._orient_fresh = True               # take the orientation target from the next sim pose
        self._ee_prev: np.ndarray | None = None  # the pose before this one: a start pose must repeat
        self.home = self.clutch.home.copy()
        self._q_start: np.ndarray | None = None  # how the fingers pointed at the start pose
        self._glide = False                     # A: gliding back to the start pose
        self._t_prev: float | None = None
        self._clock = time.time                 # the demo swaps this for a fake clock
        # A takes the arm home at these rates, not in one jump: 0.10 m/s is a slow reach,
        # 1 rad/s a slow wrist turn -- about three seconds from the far side of the table.
        self.glide_v, self.glide_w = 0.10, 1.0
        self._q_anchor = None                   # controller orientation when the clutch engaged
        self._pos = None
        self.last = np.zeros(ACTION_DIM, dtype=np.float32)
        self.last[:3] = self.clutch.home
        self.last[3:7] = JAWS_DOWN
        self.last[7] = 1.0
        self.received = 0
        self.served = 0
        self.pages: set = set()                 # live page connections, for frames going back
        self._last_frame_t = 0.0

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
                "reset": bool(msg.get("reset", False)),
            }
            self.received += 1

    # -- frames to the page --------------------------------------------------------------------
    def take_frames(self, obs: ObsPacket) -> None:
        """Keep the latest frame per camera. Encoding and sending happen on their own thread,
        so a slow page or a big JPEG never delays the reply the simulator is waiting on."""
        if not obs.images:
            return
        names = sorted(obs.images)
        with self._frames_lock:
            self.frames = {n: np.asarray(obs.images[n])[0] if np.asarray(obs.images[n]).ndim == 4
                           else np.asarray(obs.images[n]) for n in names}
        if names != self.cam_names:
            self.cam_names = names
            self.announce_cams()

    def announce_cams(self, ws=None) -> None:
        msg = json.dumps({"type": "cams", "names": self.cam_names})
        for conn in ([ws] if ws is not None else list(self.pages)):
            try:
                conn.send(msg)
            except Exception:  # noqa: BLE001
                self.pages.discard(conn)

    def stream_forever(self, max_hz: float = 30.0, quality: int = 60) -> None:
        """Sender thread: every camera's latest frame as `index byte + JPEG`, rate-capped."""
        import io as _io

        from PIL import Image

        period = 1.0 / max_hz
        while True:
            t = time.time()
            with self._frames_lock:
                frames = dict(self.frames)
                self.frames = {}
            if frames and self.pages:
                for i, name in enumerate(self.cam_names):
                    arr = frames.get(name)
                    if arr is None:
                        continue
                    buf = _io.BytesIO()
                    Image.fromarray(np.ascontiguousarray(arr[..., :3]).astype(np.uint8)).save(
                        buf, "JPEG", quality=quality)
                    data = bytes([i]) + buf.getvalue()
                    for ws in list(self.pages):
                        try:
                            ws.send(data)
                        except Exception:  # noqa: BLE001  -- a page that went away
                            self.pages.discard(ws)
            time.sleep(max(0.0, period - (time.time() - t)))

    # -- from the ZeroMQ thread ----------------------------------------------------------------
    def __call__(self, obs: ObsPacket) -> np.ndarray:
        self.take_frames(obs)
        ee = obs.state.get("ee_pose") if obs.state else None
        if ee is not None:
            ee = np.asarray(ee, dtype=np.float64).reshape(-1)
            self.ee_quat = ee[3:7]
            if self._home_from_sim or self._orient_fresh:
                # The start pose is taken only once two consecutive reports agree: the very
                # first packet after a reset can carry the pose from BEFORE it (the frame
                # sensor lags a step), and a home 52 mm off pulled the arm into the table.
                from scipy.spatial.transform import Rotation as R

                prev, self._ee_prev = self._ee_prev, ee.copy()
                stable = prev is not None and np.linalg.norm(prev[:3] - ee[:3]) < 1e-3 and                     (R.from_quat(prev[3:7]) * R.from_quat(ee[3:7]).inv()).magnitude() < 0.01
                if stable and self._home_from_sim:
                    self._home_from_sim = False
                    self.home = ee[:3].copy()
                    self.clutch.home = self.home.copy()
                    self.clutch._out = self.home.copy()
                    self.last[:3] = self.home
                    print(f"[vr] home = the start pose the simulator reports, {np.round(self.home, 3).tolist()}", flush=True)
                if stable and self._orient_fresh:
                    # The first steady pose the simulator reports (and the first after a reset)
                    # is the orientation target until the controller turns it. It must be a
                    # FIXED target: feeding the arm's own orientation back as the target left
                    # the orientation unconstrained, and with three position constraints on
                    # five weakly-driven joints the arm fell through its null space under
                    # gravity -- the wrist went from -1.2 to +0.6 rad in 90 steps with the
                    # grasp point never moving. That is the "gripper turns by itself" of the
                    # first sessions. Taken here, before any controller message, so the arm
                    # holds its start pose while the page is still connecting.
                    self.last[3:7] = ee[3:7].copy()
                    self._q_start = ee[3:7].copy()
                    self._orient_fresh = False
        with self._lock:
            s = self._sample
        if self._home_from_sim and ee is not None:
            # Until the start pose has been reported twice, ask for exactly where the arm is:
            # the first reply used to carry zeros and the old jaws-down default.
            self.last[:3] = ee[:3]
            self.last[3:7] = ee[3:7]
            s = None
        if s is not None:
            raw = s["pos"]
            squeezing = s["squeeze"] > 0.5
            if squeezing != self._squeezing:            # clutch on the grip button's edges
                self.clutch.toggle(raw)
                self._squeezing = squeezing
                if squeezing:
                    self._q_anchor = s["quat"]
                    self._q_ee_anchor = self.ee_quat if self.ee_quat is not None else self.last[3:7].copy()
            recentre = s["recentre"]
            if recentre and not self._recentre_down:    # one action per press, not per frame
                # A: the arm goes back to its start pose slowly -- position and the way the
                # fingers point -- while the controller's current position is mapped to home,
                # so motion resumes from there without a jump when the glide ends.
                self.clutch.recentre(raw)
                if self._pos is None:
                    self._pos = self.home.copy()
                self._glide = True
                print("[vr] A: gliding back to the start pose", flush=True)
            self._recentre_down = recentre
            reset = s["reset"]
            if reset and not self._reset_down:          # B: the scene goes back to its start,
                self._reset_pending = True              # and the target goes home with it
                self.clutch.recentre(raw)
                self._pos = None
                self._q_anchor = self._q_ee_anchor = None   # re-anchor on the pose after the reset
                self._orient_fresh = True
                self._ee_prev = None                        # and only on a pose reported twice
                print("[vr] B: scene reset requested", flush=True)
            self._reset_down = reset

            now = self._clock()
            dt = min(now - self._t_prev, 0.1) if self._t_prev is not None else 0.0
            self._t_prev = now
            out = self.clutch.update(raw)
            if self._glide:
                self.clutch.recentre(raw)               # keep the controller mapped to home
                d = self.home - self._pos
                n = float(np.linalg.norm(d))
                step = self.glide_v * dt
                self._pos = self.home.copy() if n <= step else self._pos + d / n * step
            else:
                self._pos = out if self._pos is None or self.smooth >= 1.0 else (
                    (1.0 - self.smooth) * self._pos + self.smooth * out)
            self.last[:3] = self._pos
            if self.track_rot:
                from scipy.spatial.transform import Rotation as R

                if self._squeezing and self._q_anchor is None and not self._reset_pending:
                    # Gripping across a reset: the anchor was dropped, this is the first pose
                    # the simulator reports after it, so this is where the fingers point now.
                    self._q_anchor = s["quat"]
                    self._q_ee_anchor = self.ee_quat if self.ee_quat is not None else self.last[3:7].copy()
                if self._squeezing and self._q_anchor is not None:
                    # How far the controller has turned since the grip closed, applied to how
                    # the fingers were pointing when it closed. Absolute controller orientation
                    # would make "rest" mean "wherever the controller happens to point", and
                    # jaws-down as the base is unreachable below 0.22 m on this arm (docs/VR.md).
                    rel = R.from_quat(s["quat"]) * R.from_quat(self._q_anchor).inv()
                    self.last[3:7] = (rel * R.from_quat(self._q_ee_anchor)).as_quat()
                if self._glide and self._q_start is not None:
                    # Turn toward the start orientation at glide_w, and re-anchor the relative
                    # rotation on the way so the controller's wrist counts from here afterwards.
                    r0, r1 = R.from_quat(self.last[3:7]), R.from_quat(self._q_start)
                    to_go = r1 * r0.inv()
                    ang = float(to_go.magnitude())
                    f = 1.0 if ang < 1e-6 else min(1.0, self.glide_w * dt / ang)
                    self.last[3:7] = (R.from_rotvec(to_go.as_rotvec() * f) * r0).as_quat()
                    self._q_anchor, self._q_ee_anchor = s["quat"], self.last[3:7].copy()
                    if np.linalg.norm(self.home - self._pos) < 2e-3 and ang * (1.0 - f) < 0.02:
                        self._glide = False
                        print("[vr] A: at the start pose", flush=True)
                elif self._glide and np.linalg.norm(self.home - self._pos) < 2e-3:
                    self._glide = False
                self.last[3:7] = yaw_lock(self.last[:3], self.last[3:7])
            self.last[7] = -1.0 if s["trigger"] > 0.5 else 1.0

        self.served += 1
        if self._log is not None:
            rec = {"t": round(time.time() - self._t0, 4), "step": obs.step, "engaged": self.clutch.engaged,
                   "action": self.last.round(5).tolist()}
            if s is not None:
                rec.update({"ctrl_pos": np.round(s["pos"], 5).tolist(), "ctrl_quat": np.round(s["quat"], 5).tolist(),
                            "squeeze": s["squeeze"], "trigger": s["trigger"], "recentre": s["recentre"],
                            "reset": s["reset"]})
            if self.ee_quat is not None:
                rec["ee_quat"] = np.round(self.ee_quat, 5).tolist()
            self._log.write(json.dumps(rec) + chr(10))
        if self.served % 60 == 0:
            x, y, z = self.last[:3]
            state = "ENGAGED" if self.clutch.engaged else "held"
            print(f"\r  {state:<8} pos=({x:+.3f}, {y:+.3f}, {z:+.3f})  "
                  f"jaw={'closed' if self.last[7] < 0 else 'open  '}  "
                  f"page msgs={self.received}   ", end="", flush=True)
        action = np.tile(self.last, (obs.num_envs, 1)).astype(np.float32)
        if self._reset_pending:
            self._reset_pending = False
            return ActionPacket(step=obs.step, action=action, reset=np.ones(obs.num_envs, dtype=bool))
        return action


class FakeController:
    """A scripted right hand, in WebXR's frame, so the whole chain runs with no headset.

    The motion is a person reaching forward and down to the block, gripping, lifting, carrying
    left, releasing -- with the clutch held the whole time. It feeds :meth:`VrDriver.push` with
    the same dict the page sends, so everything from the frame conversion onward is the
    production path. What it cannot prove is the page itself in a real Quest browser.
    """

    # (t, dx_right, dy_up, dz (WebXR z: forward is negative), squeeze, trigger, wrist pitch deg)
    # From the folded rest pose -- grasp point about (0.11, 0, 0.09), fingers pointing down --
    # out and up to over the block at (0.27, 0, 0.13) with the wrist tilted back 80 degrees so
    # the finger tips point slightly up (the tilt that reaches the table, docs/VR.md), then the
    # same pick as before: down, close, lift, carry left, lower onto the tray, release.
    KEYS = [
        (0.0, 0.00, 0.00, 0.000, 0.0, 0.0, 0.0),
        (1.0, 0.00, 0.00, 0.000, 1.0, 0.0, 0.0),     # grip: clutch engages here, this pose = home
        (4.0, 0.00, +0.04, -0.155, 1.0, 0.0, 80.0),  # out over the block, wrist tilted back
        (5.5, 0.00, -0.02, -0.155, 1.0, 0.0, 80.0),  # down, pads 60 mm down the block's sides
        (6.5, 0.00, -0.02, -0.155, 1.0, 1.0, 80.0),  # trigger: close
        (8.5, 0.00, +0.06, -0.155, 1.0, 1.0, 80.0),  # lift 8 cm clear of the table
        (10.5, -0.10, +0.06, -0.155, 1.0, 1.0, 80.0),  # carry to the operator's left (-x in WebXR)
        (11.5, -0.10, -0.015, -0.155, 1.0, 1.0, 80.0), # lower onto the tray (its lip is 10 mm up)
        (12.5, -0.10, -0.015, -0.155, 1.0, 0.0, 80.0), # release
        (14.0, -0.10, +0.06, -0.155, 1.0, 0.0, 80.0),  # back up
    ]

    def __init__(self, driver: VrDriver, rate_hz: float = 72.0) -> None:
        self.driver, self.dt = driver, 1.0 / rate_hz
        self.t0 = None       # set when the simulator first asks -- see run_forever

    def sample(self, t: float) -> dict:
        k = self.KEYS
        if t >= k[-1][0]:
            _, dx, dy, dz, sq, tr, pitch = k[-1]
        else:
            for (t0, x1, y1, z1, s0, r0, p1), (t1, x2, y2, z2, s1, r1, p2) in zip(k, k[1:]):
                if t < t1:
                    a = (t - t0) / (t1 - t0)
                    dx, dy, dz = x1 + a * (x2 - x1), y1 + a * (y2 - y1), z1 + a * (z2 - z1)
                    pitch = p1 + a * (p2 - p1)
                    sq, tr = (s0 if a < 0.5 else s1), (r0 if a < 0.5 else r1)
                    break
        # WebXR: the controller starts 0.4 m in front of (-z) and 1.0 m above the floor. From
        # 6 s it also turns 40 degrees about the vertical, so orientation tracking is measured.
        yaw = 40.0 * min(max((t - 10.5) / 2.0, 0.0), 1.0)
        from scipy.spatial.transform import Rotation as R

        q = R.from_euler("y", yaw, degrees=True) * R.from_euler("x", pitch, degrees=True)
        return {"pos": [dx, 1.0 + dy, -0.4 + dz], "quat": q.as_quat().tolist(),
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
        driver.pages.add(ws)
        if driver.cam_names:
            driver.announce_cams(ws)
        try:
            for raw in ws:
                if isinstance(raw, bytes):
                    continue                        # the page sends text only; frames go the other way
                try:
                    driver.push(json.loads(raw))
                except (ValueError, KeyError, TypeError) as exc:
                    print(f"\n[vr] bad message ignored: {exc}", flush=True)
        finally:
            driver.pages.discard(ws)
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
    ap.add_argument("--home", type=float, nargs=3, default=None,
                    help="grasp point before the clutch is first engaged, robot root frame. "
                         "Where configs/vr_teleop.yaml's start pose puts it, so nothing jumps")
    ap.add_argument("--scale", type=float, default=1.0, help="hand travel : gripper travel")
    ap.add_argument("--smooth", type=float, default=0.5, help="EMA factor; 1.0 = off")
    ap.add_argument("--no-track-rot", action="store_true",
                    help="pin the fingers' orientation instead of following the controller's")
    ap.add_argument("--fps", type=float, default=30.0, help="camera stream cap, frames per second")
    ap.add_argument("--quality", type=int, default=60, help="JPEG quality of the stream")
    ap.add_argument("--log", default=str(LOG_DIR), help="directory for the per-session JSONL")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    driver = VrDriver(home=None if args.home is None else tuple(args.home), scale=args.scale, smooth=args.smooth,
                      track_rot=not args.no_track_rot,
                      log_dir=None if args.no_log else Path(args.log))
    threading.Thread(target=driver.stream_forever, args=(args.fps, args.quality), daemon=True).start()

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
        print("[vr] GRIP = move,  TRIGGER = close jaw,  A/X = glide home,  B/Y = reset scene")

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
    # B: exactly one reply carries reset=True, and the target goes home with it.
    d.push({"pos": [5, 5, 5], "quat": [0, 0, 0, 1], "squeeze": 0, "trigger": 0, "reset": True})
    r = d(obs)
    assert isinstance(r, ActionPacket) and bool(r.reset.all()) and np.allclose(r.action[0, :3], [0.2, 0.0, 0.15]), r
    assert not isinstance(d(obs), ActionPacket), "held B must not reset again"
    assert d._q_anchor is None and d._q_ee_anchor is None and d._orient_fresh, "B drops the orientation anchor"
    d.push({"pos": [5, 5, 5], "quat": [0, 0, 0, 1], "squeeze": 0, "trigger": 0, "reset": False})
    assert not isinstance(d(obs), ActionPacket)
    assert FakeController(d).sample(7.0)["trigger"] == 1.0 and FakeController(d).sample(4.5)["trigger"] == 0.0
    from scipy.spatial.transform import Rotation as _R
    assert abs(_R.from_quat(FakeController(d).sample(4.5)["quat"]).magnitude() - np.radians(80)) < 1e-6, "wrist tilted back by 4 s"

    # Frames are parked for the sender thread, and a new camera set is announced to the pages.
    class _Page:
        def __init__(self): self.got = []
        def send(self, data): self.got.append(data)
    page = _Page(); d.pages.add(page)
    frame = np.zeros((1, 12, 16, 3), dtype=np.uint8); frame[..., 0] = 200
    d(ObsPacket(step=1, num_envs=1, state={}, images={"front": frame, "top": frame}))
    assert d.cam_names == ["front", "top"] and set(d.frames) == {"front", "top"}
    assert page.got and json.loads(page.got[0]) == {"type": "cams", "names": ["front", "top"]}

    # Orientation follows the controller RELATIVE to how the fingers pointed at grip time.
    from scipy.spatial.transform import Rotation as R
    d2 = VrDriver(home=(0.2, 0.0, 0.15), smooth=1.0, track_rot=True)
    ee0 = R.from_euler("x", 90, degrees=True).as_quat()                    # the sim says: jaws down
    for i in range(2):
        d2(ObsPacket(step=i, num_envs=1, state={"ee_pose": np.array([[0.2, 0, 0.15, *ee0]])}, images={}))
    assert np.allclose(d2(obs)[0, 3:7], ee0), "before any grip: the target is the pose the sim reported, fixed"
    d2.push({"pos": [0, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0}); d2(obs)
    q_ctrl = R.from_euler("y", 30, degrees=True).as_quat()                 # turn 30 deg about WebXR up
    d2.push({"pos": [0, 1, -0.4], "quat": q_ctrl.tolist(), "squeeze": 1, "trigger": 0})
    got = R.from_quat(d2(obs)[0, 3:7])
    want = R.from_euler("z", 30, degrees=True) * R.from_quat(ee0)          # = 30 deg about the scene's up
    assert (got * want.inv()).magnitude() < 1e-6, (got.as_quat(), want.as_quat())
    # Yaw lock: fingers level and pointing +x, target moved to the left -> they head for it.
    d3 = VrDriver(home=(0.2, 0.0, 0.15), smooth=1.0, track_rot=True)
    ee1 = R.from_euler("z", 90, degrees=True).as_quat()                    # -y of the body -> +x
    for i in range(2):
        d3(ObsPacket(step=i, num_envs=1, state={"ee_pose": np.array([[0.2, 0, 0.15, *ee1]])}, images={}))
    d3.push({"pos": [0, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0}); d3(obs)
    d3.push({"pos": [-0.2, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0})
    a3 = d3(obs)[0]
    assert np.allclose(a3[:3], [0.2, 0.2, 0.15]), a3[:3]
    f3 = R.from_quat(a3[3:7]).apply([0, -1, 0])
    assert np.allclose(f3, [np.sqrt(0.5), np.sqrt(0.5), 0.0], atol=1e-6), f3
    # Hooked: fingers pointing back at the base (-x) stay hooked -- turned 45, not 225 degrees.
    q_hook = R.from_euler("z", -90, degrees=True).as_quat()                # body -y -> -x
    f4 = R.from_quat(yaw_lock([0.2, 0.2, 0.15], q_hook)).apply([0, -1, 0])
    assert np.allclose(f4, [-np.sqrt(0.5), -np.sqrt(0.5), 0.0], atol=1e-6), f4
    assert np.allclose(yaw_lock([0.3, 0.0, 0.1], q_hook), q_hook), "already in the plane: untouched"
    # No --home: the start pose the simulator first reports is home, and is held.
    d5 = VrDriver(smooth=1.0)
    d5.push({"pos": [0, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0})
    a0 = d5(ObsPacket(step=0, num_envs=1, state={"ee_pose": np.array([[0.3, 0.0, 0.2, *ee0]])}, images={}))  # stale: pre-reset
    assert np.allclose(a0[0, :3], [0.3, 0.0, 0.2]) and np.allclose(a0[0, 3:7], ee0), "before home is known: hold where the arm is"
    d5(ObsPacket(step=1, num_envs=1, state={"ee_pose": np.array([[0.11, 0.0, 0.09, *ee1]])}, images={}))
    assert d5._home_from_sim, "a pose reported once is not home yet"
    a5 = d5(ObsPacket(step=2, num_envs=1, state={"ee_pose": np.array([[0.11, 0.0, 0.09, *ee1]])}, images={}))
    assert np.allclose(d5.home, [0.11, 0.0, 0.09]) and np.allclose(a5[0, :3], [0.11, 0.0, 0.09]), (d5.home, a5)
    assert np.allclose(a5[0, 3:7], ee1), "the orientation target is the steady start pose, before any controller message"
    d5.push({"pos": [0, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0}); d5(obs)
    d5.push({"pos": [0, 1.05, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0})
    assert np.allclose(d5(obs)[0, :3], [0.11, 0.0, 0.14]), "motion counts from the reported start pose"

    # A: from the far side of the table, the target glides home at glide_v, never jumps, and the
    # fingers turn back to how they pointed at the start.
    d4 = VrDriver(home=(0.2, 0.0, 0.15), smooth=1.0, track_rot=True)
    clock = [0.0]; d4._clock = lambda: clock[0]
    for i in range(2):
        d4(ObsPacket(step=i, num_envs=1, state={"ee_pose": np.array([[0.2, 0, 0.15, *ee1]])}, images={}))
    d4.push({"pos": [0, 1, -0.4], "quat": [0, 0, 0, 1], "squeeze": 1, "trigger": 0}); d4(obs)
    q30 = R.from_euler("y", 30, degrees=True).as_quat()
    d4.push({"pos": [-0.2, 1.0, -0.5], "quat": q30.tolist(), "squeeze": 1, "trigger": 0}); a4 = d4(obs)[0]
    assert np.allclose(a4[:3], [0.3, 0.2, 0.15]) and (R.from_quat(a4[3:7]) * R.from_quat(ee1).inv()).magnitude() > 0.3
    d4.push({"pos": [-0.2, 1.0, -0.5], "quat": q30.tolist(), "squeeze": 1, "trigger": 0, "recentre": True}); d4(obs)
    d4.push({"pos": [-0.2, 1.0, -0.5], "quat": q30.tolist(), "squeeze": 1, "trigger": 0, "recentre": False})
    prev = np.linalg.norm(d4(obs)[0, :3] - d4.home); steps = 0
    while d4._glide and steps < 100:
        clock[0] += 0.05; a4 = d4(obs)[0]; steps += 1
        dist = np.linalg.norm(a4[:3] - d4.home)
        assert dist <= prev + 1e-6 and prev - dist <= d4.glide_v * 0.05 + 1e-6, (prev, dist)
        prev = dist
    assert not d4._glide and dist < 2e-3 and 20 <= steps <= 60, (steps, dist)
    assert (R.from_quat(a4[3:7]) * R.from_quat(ee1).inv()).magnitude() < 0.03, "fingers back to the start"
    d4.push({"pos": [-0.3, 1.0, -0.5], "quat": q30.tolist(), "squeeze": 1, "trigger": 0})
    clock[0] += 0.05; a5 = d4(obs)[0]
    assert np.allclose(a5[:3], [0.2, 0.1, 0.15]), a5[:3]   # motion resumes from home, no jump
    print("vr_gripper_server demo OK: frames, clutch edges, jaw, hold, cameras, orientation, yaw lock, reset, glide home, home from sim")


if __name__ == "__main__":
    if "--demo" in sys.argv:
        demo()
    else:
        main()
