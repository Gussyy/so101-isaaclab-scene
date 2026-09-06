# VR teleoperation: a Quest controller drives the SO-101

```bash
# terminal 1 -- the bridge. Serves the page the headset opens and answers the simulator.
python scripts/vr_gripper_server.py

# terminal 2 -- the simulator, on the monitor
python scripts/run.py --config configs/vr_teleop.yaml --viz kit --steps 0

# on the Quest: open the URL the bridge printed (https://<this PC's LAN IP>:8443/),
# accept the certificate once, press Enter VR.
```

Or set the scene up in a window first and press Start — `python scripts/scene_gui.py` — which
writes the YAML and launches both of the above.

| right Touch controller | |
|---|---|
| **GRIP** (hold) | move the arm. A clutch, like lifting a mouse: let go, reposition your hand, grip again — the arm stays where it was |
| **TRIGGER** (hold) | close the jaw |
| **A** | re-centre: the grasp point returns to `--home` |
| **B** | reset the scene: objects back to their start, arm to its rest pose, target home |

**With a Link cable, use the PC's browser instead:** start the bridge with `--no-tls`, open
`http://localhost:8443/` in Chrome or Edge, Enter VR — the session goes to the headset through
the Meta runtime, and `localhost` is a secure context so there is no certificate step at all.
That is the route verified on hardware.

**What you see in the headset is a floating screen** showing the simulator's camera — 1.2 m
wide, 1.5 m in front of where you stood when the session began, at chest height. The first
version showed nothing at all: passthrough is not available through Chrome over Link, so the
`immersive-vr` fallback was a black void and the operator had to steer from the monitor. The
frame gets there without a new socket: `run.py` attaches each declared camera's image to the
observation packet every third step, the bridge JPEG-encodes it at up to 15 Hz and pushes it
down the same WebSocket the controller comes up, and the page draws it on one textured quad.
It is also drawn on the page itself, so the stream can be checked from any browser.

## Why not Isaac Lab's own XR teleop

Isaac Lab ships one (`isaaclab_teleop`, CloudXR, `--xr`), and its guide is explicit:

> XR teleoperation is supported on **Linux x86_64 only**. The `teleop` extra gates `isaacteleop`
> and `dex-retargeting` behind platform markers, so on Windows or aarch64 the extra resolves but
> installs nothing usable.

`scripts/floating_gripper.py` already carries the other Windows-capable route — Kit's own
`omni.kit.xr.system.openxr` over Quest Link, with hand tracking, driving an arm-less gripper. It
is the *immersive* fork of this problem and is unverified on hardware for the same reason this
page is. The two share `hand_tracker.Clutch` and the 8-wide action, so whichever gets a headset
first informs the other.

## How it is put together

```
Quest browser  --WebXR-->  scripts/vr/index.html  --WSS JSON @60Hz-->  vr_gripper_server.py
                                                                            |  clutch, frame
                                                                            |  conversion, jaw
                                                             ZeroMQ REQ/REP |  8-wide action
                                                                            v
                                             run.py + control.actions: ik  -->  SO-101 arm
```

One message shape, `{pos, quat, trigger, squeeze, recentre}` in WebXR's `local-floor` frame.
One action shape, `[x, y, z, qx, qy, qz, qw, jaw]` in the robot's root frame — the same eight
numbers the webcam hand tracker sends to the floating gripper, so the simulator side neither
knows nor cares which device is on the socket.

**`control.actions: ik`** is the new piece on the simulator side. It swaps the task's five
joint-offset actions for Isaac Lab's `DifferentialInverseKinematicsAction`: the arm is
commanded by a grasp-point pose and damped-least-squares IK finds the joints. Isaac Lab 3.0
takes the quaternion as `(x, y, z, w)` — its identity fallback is `[0, 0, 0, 1]` — and the
command is in the articulation root frame, which for `so101_full` at identity rotation is the
env frame.

**Frames.** WebXR is +x right, +y up, −z forward; the scene is +x forward, +y left, +z up. The
mapping is one permutation matrix in `webxr_to_sim`, applied to positions directly and to
orientations as `P R Pᵀ` — permuting a quaternion's components instead is wrong for exactly the
reason it looks right. The bridge's `--demo` checks a controller held forward, left and up.

**Orientation.** The bridge sends jaws-down by default, and `--track-rot` follows the
controller's orientation *relative to where it was when the grip closed*, on top of that — the
same choice the webcam and Quest-hand drivers make, for the same reason: the controller's rest
orientation has not been measured on hardware. On the simulator side the shipped config gives
orientation **zero weight**, so neither reaches the joints; the measurements below say why, and
what it would take for them to.

**HTTPS.** WebXR refuses to start from a plain `http://` page on anything but `localhost`. The
bridge generates a self-signed certificate for this PC's LAN address on first run
(`scripts/vr/cert.pem`, gitignored) and serves page and WebSocket on one port. The Quest
browser will warn once — Advanced → proceed — and then remember it.

## What is verified, and what is not

Everything below the page is exercised without a headset, in two ways:

- **`python scripts/vr_gripper_server.py --fake`** replaces the page with a scripted controller
  — reach down, grip, close, lift, carry left, release — fed into the same `push()` the page
  hits. The simulator then runs the production path end to end.
- **The page's "Mouse test" button** drives the same message from an ordinary browser. Run
  with `--no-tls`, open `http://localhost:8443/`, drag on the panel. Verified here: the page
  connected, reconnected on its own when the bridge restarted, and the bridge counted 2,299
  messages from it while answering the simulator's requests.

| | status |
|---|---|
| bridge: frame mapping, clutch edges, jaw, hold-on-silence | **verified**, `--demo` |
| page → WebSocket → bridge → ZeroMQ → action | **verified**, mouse mode in a browser |
| page served over HTTPS with the generated certificate | **verified**, `curl -k` |
| `control.actions: ik` builds an 8-wide action on the grasp point | **verified**, tests |
| fake controller → IK → arm follows the pose, grasps and lifts a block | **verified** — 4–20 mm tracking, block carried; measurements below |
| the page with a real controller: session, `gripSpace`, frame mapping, button indices | **verified on a Quest 3 over Link, through desktop Chrome** — raw numbers below |

The button mapping follows the WebXR `xr-standard` profile — 0 trigger, 1 squeeze, 4 A/X —
and the frame conversion follows the spec; both were "derived, not measured" until the first
session with hardware, which the bridge logged (it prints the first messages and every 300th):

```
[vr] page connected from 192.168.1.103              <- Chrome on the PC, over Link
[vr] page msg 300: webxr pos=(+0.269,+0.181,-0.394) -> sim (+0.394,-0.269,+0.181)  squeeze=0.00 trigger=0.00
[vr] page msg 600: webxr pos=(+0.262,+0.497,-0.345) -> sim (+0.345,-0.262,+0.497)  squeeze=1.00 trigger=0.00
  ENGAGED  pos=(+0.215, +0.001, +0.227)  jaw=open
  held     pos=(+0.243, +0.146, +0.211)  jaw=open      <- released: the arm stays put
[vr] page msg 900: webxr pos=(+0.316,+0.479,-0.404) -> sim (+0.404,-0.316,+0.479)  squeeze=1.00 trigger=1.00
  ENGAGED  pos=(+0.414, +0.062, +0.186)  jaw=closed
```

Forward (−z) came out as +x, right (+x) as −y, up as +z; GRIP is button 1, TRIGGER is
button 0, and B (button 5) reaches the simulator as `ActionPacket.reset` — a field the wire
schema already had and nothing used: the run loop resets the environment when it sees it
(`[run] scene reset at step 1789 (operator)`, measured). The simulator stepped on those actions throughout (4,600 steps, no drop — including
a 20-second bridge restart mid-session, which the page and the ZeroMQ client both rode out).

**The route that worked was Link, not the Quest browser.** With the headset on a Link cable,
`https://localhost:8443/` in Chrome or Edge on the PC hands the WebXR session to the Meta
runtime, which renders it to the headset and reads the controllers. No LAN address, no Wi-Fi.
The Quest-browser route over the LAN address remains the wireless option and is untested.

**Scale.** The operator's hand travelled ~40 cm in that session; the arm reaches 30. Targets
went to 0.41 m (beyond reach — the arm stalls at its extent) and to 0.05 m (inside the base).
`--scale 0.6` on the bridge makes hand travel cover the workspace rather than overshoot it.

## Measured: does the arm follow?

The fake controller reaches down, grips, lifts, carries 160 mm left and releases, over 12 s.
Error is the distance between the commanded grasp point and the one the `ee_frame` sensor
reads back, over the 1,100 steps after the first request:

| `ik_orientation_weight` | mean error | settled (static target) | outcome |
|---|---|---|---|
| default (1.0, full pose) | 224 mm | 226 mm | parks at a joint limit and stays |
| 0.2 | 161 mm | 157 mm | parks |
| `[1, 1, 0]` (tilt held, yaw free) | 159 mm | 157 mm | parks |
| **0.0** (position only) | **19 mm** | **12.5 mm** | **tracks the whole path** |

Five joints cannot meet an arbitrary six-degree orientation, and damped least squares trades
position away trying: from a rest pose 90° from "jaws down", even a 0.2 weight makes the
orientation term three times the position term, and the solver drives into a limit before it
ever tracks. The per-axis weights did not rescue it because they act on the axis-angle *error
vector*, which only means "tilt versus yaw" once the tool is already near the goal.

So the shipped config sets the weight to zero: the arm follows the controller's position, and
the jaws' orientation follows the arm. That is what a five-joint arm can do, and it is what the
e-Yantra video's operators are working with too.

**Why "just start jaws-down" does not work either — measured.** The obvious fix is to start the
arm with the fingers already pointing down, so the orientation error is small from step one.
A 27-point grid over shoulder, elbow and wrist angles found where the fingers can point down at
all, reading the tool axis back from the simulator (`tool_down`: −1 is straight down):

| posture | tool_down | grasp point height |
|---|---|---|
| default rest | **+0.76** (pointing up) | 0.116 m |
| best jaws-down: lift 0, elbow 0.3, wrist −1.65 | −0.96 | **0.287 m** |
| lowest reach: lift 0.6, elbow 1.0, wrist −1.2 | +0.26 | **0.036 m** |

Jaws-down (≤ −0.83) exists only with the grasp point **above 0.22 m**; every posture that
reaches the table has the fingers near-horizontal. The wrist limit (±1.658 rad) is what binds.
`docs/workspace.txt` measured the same fact on the single-jaw arm long ago — *"top-down grasp
only: height 0.142 .. 0.457 m"*. Starting jaws-down and asking the IK to hold it while
descending gave 162–169 mm error, as this table says it must.

So this arm picks things off a table **from the front**, fingers roughly level, like a person
reaching across a desk. Position-only IK is not a compromise here; it is the right description
of what the arm can do. The bridge still sends a jaws-down quaternion — the IK ignores it at
weight 0, and it is there for anyone who raises the weight for a task above 0.22 m.

**Where the table is reachable.** Low reach exists at x ≈ 0.26–0.31 m from the base, not at
0.20 (the weight-0 run above asked for x = 0.20 at z = 0.03 and got z = 0.095 — a workspace
edge, not a solver failure). The shipped config puts the object at x = 0.27 and the bridge's
default `--home` there too. Re-run there, position-only: **4–20 mm** error through the whole
reach, the descent stopping at z ≈ 63 mm — the arm's floor at that distance.

**And the object has to suit the gripper.** The parallel fingers close to **56 mm**, so the
task's 30 mm cube can never be gripped by them, and was not. The shipped config uses a
70 × 70 × 90 mm, 50 g block: wide enough to be closed on, tall enough to be met at 60 mm.

**The final run, fake controller, block:**

```
step=120  jaw=closed   object_z= 63 mm      <- closed on it
step=210              object_z= 59 mm      <- lifted (rests at 45)
step=240              object_z= 63 mm      <- carried left, y 0.00 -> 0.03
step=270              object_z= 50 mm      <- tipping during the carry
step=300              object_z= 32 mm      <- on its side
PROBE object peak height 76.1 mm  (rests at 45 mm; CARRIED)
```

Grasped, lifted 31 mm, carried part of the way, dropped on its side. That is a front-approach
grasp on a block being swung sideways by a scripted hand that does not know it is slipping — a
person watching the monitor would slow down, or would not. It is the whole chain working; it is
not a claim that every pick succeeds.

**Also found by this test, not by reading:** the fake controller's clock used to start when the
*bridge* started, ~25 s before the simulator's first request, so the whole 12 s script had
played out to nobody and the clutch anchored on the final static pose. The command never moved
and the arm sat still for 900 steps. It now starts on the first request.

## The scene GUI

`python scripts/scene_gui.py [config.yaml]` — a tkinter window: robot, physics, object rows
(type, catalogue name, position, static), control source, IK on/off, demo camera, and whether
to launch the VR bridge alongside. Every control is one key of the YAML the builder already
takes; the window writes `configs/gui_scene.yaml` and runs `scripts/run.py --config` on it in
its own console. Load and Save round-trip any config in `configs/`.

Two things it fills in rather than asks: the robot's base rotation (+90° for `so101`, none for
`so101_full` — a sign that was got wrong twice by reasoning about it), and the default joint
pose. It refuses a duplicate object key, and runs the builder's own validation before a
two-minute Kit boot rather than after.
