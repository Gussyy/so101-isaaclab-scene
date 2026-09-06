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
| **GRIP** (hold) | move the arm — and turn it: the fingers follow the controller's rotation, relative to how they pointed when you gripped. A clutch, like lifting a mouse: let go, reposition your hand, grip again — the arm stays where it was |
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

**Orientation follows the controller, relative to the arm.** Each step the simulator puts the
grasp point's pose in the root frame on the packet (`state["ee_pose"]`). When the grip closes,
the bridge anchors *both* the controller's orientation and the fingers' actual orientation; from
then on the target is "however far the controller has turned since, applied to how the fingers
pointed then". Two things fall out of that. The arm never has to jump to an orientation it may
not be able to reach — the target starts at zero error and moves only as fast as your wrist.
And it is what fixed the gripper "rotating by itself": with orientation unweighted, five joints
against three position constraints left `wrist_roll` free to spin without moving the grasp
point, and damped least squares let it. The first session showed exactly that. Orientation now
carries weight 0.3 in the IK, measured below. `--no-track-rot` pins it instead.

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

## Where the arm starts

The task's rest pose puts `gripper_base` at z = 0.06 m, 0.28 m out — which is *inside* a 90 mm
block placed where the arm reaches the table. The first live session started in collision.

Five start poses were tried before one picked the block up. Each failure was measured rather
than guessed at, and each measurement changed something that was not the start pose:

| start pose (shoulder, elbow, wrist) | fingers | what happened |
|---|---|---|
| 0.3, 0.6, −1.5 | 30° down | descent stalled at 164 mm: this arm cannot hold 30° down below ~0.16 m |
| 0.3, 0.6, −1.2 | 17° down | stalled at 92 mm, level with the block's top |
| 0.0, 1.0, −1.2 | 7° down | the arm *rose* to 190 mm before the first grip (the target must be fixed, below) |
| −0.3, 1.0, −0.6 | 9° up | stalled at 115 mm with the block at x 0.27 or 0.30; the arm gains, below |
| −0.3, 0.4, +0.6 | 41° up | stalled at 122 mm: the block was under the housing, not between the pads |
| **−0.3, 1.0, −0.6, block at x = 0.35** | 9° up | **the pads go 60 mm down the block's sides** |

The gripper's collision meshes were then measured in the `gripper_base` frame (the finger
line is −y, the grasp point is at y = −74.8 mm), and again in the world from the simulator's
body poses with the fingers open and closed:

| part | along the fingers (y) | across (world y, open → closed) | up/down (z) |
|---|---|---|---|
| finger `arm_r` | −78 … +20 mm | −8 … +67 → −52 … +24 mm | ±30 mm |
| finger `arm_l` | −78 … +20 mm | −68 … +7 → −25 … +52 mm | ±30 mm |
| gear between them | −83 … −70 mm | ±15 mm | ±15 mm |
| motor housing | −125 … −63 mm | ±60 mm | −26 … +35 mm |

Three facts fall out. The grasp point is the **palm face**: the pads run 95 mm *forward* of it
and reach **41 mm below** the finger line; the gear sits 16 mm below the line just ahead of the
palm; the housing runs 62 mm *behind* the palm and 28 mm below the line. So an object under
the grasp point is under the housing, and the housing lands on it — 115 mm for a 90 mm block,
as measured, with the block at x 0.27, 0.30 and even 0.33, where its near top edge still caught
the housing's far corner. The fingers are open at joint 0; the asset's authored stop was
−0.044, pads 134 mm apart open and 49 mm closed, level at `wrist_roll` 0 (±0.35 rad tilts
them 20 mm apart in height; the body *origins* sit at different heights, which misled one
attempt). The stop is now −0.068 — see "The jaw closes" under the shirt scene.

The scene follows from the numbers: the block centred at **x = 0.35**, under the pads (x
0.27–0.37 when the grasp point is at 0.273) and 27 mm forward of the housing. With the finger
tips 9° up the descent is the arm's own floor — the same tilt is reachable down to
(0.269, 0.036) in the grid and was measured to 68 mm with the scene empty — which puts the
pads 60 mm down the block's sides.

**The start pose is the real arm's rest pose.** Upper arm leaning back, forearm folded down
over it, the gripper resting in front of the base with the fingers pointing down, 20° forward
of vertical — the pose the arm is parked in on the desk. From a 57-posture grid of the folded
region: shoulder −1.65, elbow 1.65, wrist 1.3 (just inside the ±1.745 / 1.69 limits), grasp
point (0.112, 0, 0.091), finger tips at (0.144, 0.002), the housing on the table. The bridge
takes **home** from the first pose the simulator reports, so `--home` is no longer needed and
the first grip does not jump. B resets the scene, which puts the arm back there. **A glides
the arm back there** — position at 0.10 m/s, the way the fingers point at 1 rad/s, about
three seconds from the far side of the table — while the controller's current position is
mapped to home, so motion resumes from the start pose without a jump. To pick from the table
the operator tilts the wrist back so the finger tips point slightly up: that is the tilt that
reaches 68 mm at x = 0.27, and the fake controller does the same (80° about the WebXR x axis).

**The final mock run, fake controller, from the rest pose:** reach out and up from the folded pose while tilting the wrist back 80°, descend, close at 29 mm, lift the block to 183 mm, carry it 100 mm left and lower it onto the tray. Tracking error over the 940 steps after the first request **15 mm mean, 57 mm max** (the max while the arm unfolds and where the 80° tilt is not quite reachable at the table, so the solver trades 4 cm of height for it); orientation error **6° mean, 29° max**; object peak height 183 mm from a 45 mm rest — **CARRIED**.

**The previous mock run, from a start pose over the block:** reach down to 68 mm (1.7 mm off), close at 69 mm, lift the block to 121 mm, carry it 100 mm left at 121 mm and lower it onto the tray. Tracking error over the 840 steps after the first request **2.2 mm mean, 3.1 mm max**; orientation error **1.2° mean, 2.9° max**; object peak height 121 mm from a 45 mm rest — **CARRIED**.

**The finger colliders were convex hulls, and that is what every block was hitting.** With the
housing and gear cleared, the block at x = 0.35 was still met at 107 mm — and the finger
joints then closed to −0.040, nearly their full travel, straight through where the block's
sides were. Each finger is an L: a pad plus a rack bar running across the gripper to the
central gear, and the asset approximated each as a single convex hull, a wedge whose
underside slopes from the pad's bottom up to the rack and whose inner face is not the pad.
The block's top edge met that slope; the pads had nothing to close with. Moving the block to
x = 0.40, beyond the pads, let the arm descend freely to 68 mm, which located the contact on
the fingers themselves. The two finger colliders in
`robot_description/IsaacAssets/SO-ARM101-FULL/payloads/instances.usda` are now
`convexDecomposition`; the arm's floor beside the block went from 107 mm to its own 68 mm,
and the pads close on the block's sides.

**The orientation target must be fixed, not followed.** An early version asked, before the
first grip, for whatever orientation the arm currently had. That leaves the orientation
unconstrained, and three position constraints on five weakly-driven joints leave a null space
that gravity walks the arm through: the wrist went from −1.2 to +0.6 rad in 90 steps with the
grasp point never moving, and the arm settled 6 cm higher in a posture it then anchored on.
That is the "gripper turns by itself" of the first sessions, reproduced in the mock. The bridge
now takes the orientation target once, from the first pose the simulator reports (and again
after a B reset), and only the controller's own rotation changes it.

**The arm servos were too soft to close the last 45 mm.** The asset's arm gains are 17.8 N·m/rad
and 0.6 N·m·s/rad. Holding a target 70 mm above the table at reach, the shoulder sat 0.10–0.12
rad short of the IK's joint target on 1.8–2.2 N·m — the torque it takes to hold the arm's own
weight there — and since the task-space IK steps from the *current* pose each time, that gap
never closes: a commanded 70 mm stopped at 118 mm, on CPU and GPU physics alike. The config's
`scene.robot.arm` block sets **200 / 5**; the remaining gap is 0.01 rad, about 3 mm, and the
STS3215 in the real arm is a stiff position servo, so this is nearer the hardware, not further.

**The object spawns where the YAML says.** The task re-places it at every reset, ±3 cm in x and
±6 cm in y — right for training, wrong for an operator, and it made the mock's failures
unrepeatable (the block under the housing in one run, clear of it in the next).
`scene.spawn_jitter: false` pins it; the GUI has the checkbox.

## Every session is logged

`logs/vr/<timestamp>-bridge.jsonl` (the bridge) and `logs/vr/<timestamp>-sim.jsonl` (`run.py`,
whenever the driver is remote) — one JSON object per request / per step:

```
bridge:  t, step, engaged, squeeze, trigger, recentre, reset, ctrl_pos, ctrl_quat, ee_quat, action
sim:     t, step, reset, action, ee_pose (root frame, xyz + xyzw), joint_pos
```

Join them on `step`. The first session's complaint — "the gripper rotates by itself" — could
not be answered from what was logged then (every 300th message, position only); this is what
answers the next one. `--no-log` on either side turns it off; `--log DIR` moves it.

## More than one screen, streamed off the action thread

Three cameras in the config — `a_front` large, `b_top` and `c_side` smaller, turned 28° to face
you — each a floating panel; the page lays them out by the camera list the bridge announces, so
adding a fourth is a config line. Names sort alphabetically, hence the prefixes.

Speed came from three changes, in order of effect: the frames are encoded and sent on their
**own thread** with a "latest frame" slot, so the reply the simulator is waiting on is never
delayed by a JPEG or a slow page; the simulator attaches frames every **second** step instead of
every third; and the cap went from 15 to **30 fps** at JPEG quality 60. Binary frames carry a one-
byte camera index ahead of the JPEG.

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

**With orientation following the controller** — relative to the arm, the fake controller
turning its wrist 40° during the carry, error between the commanded and actual grasp frame:

| `ik_orientation_weight` | orientation error (mean / p95) | position error (mean / p95) |
|---|---|---|
| 0.0 | not tracked; `wrist_roll` drifts | 4–20 mm |
| **0.1** | **17° / 17°** | **22 mm / 49 mm** |
| 0.2 | 24° / 25° | 60 mm / 61 mm |
| 0.3 | 16° / 20° | 59 mm / 77 mm |

0.1 is shipped: the fingers follow the wrist to within about 17° for a 22 mm cost in position,
and the self-rotation is gone. Above that the solver gives up twice the position for no more
orientation. The residual 17° is the arm, not the solver — five joints meet a commanded
orientation only approximately, and the anchor keeps that approximation from ever having to
be large.

**The heading is the arm's, so the bridge stops asking for it.** Five joints are base yaw,
three pitches in one vertical plane, and wrist roll. The fingers' heading *is* the base yaw,
and the base yaw is wherever the arm is reaching — so a commanded heading other than
`atan2(y, x)` of the target is unreachable, and the solver was paying position for it: at the
tray (y = 0.10) the grasp point sat 37 mm off with the fingers 6° from a heading the arm could
never take. `yaw_lock()` in the bridge now turns the commanded orientation about the world's
up until the finger axis lies in the arm's plane through the commanded point — pointing either
way along it, by the smaller turn, since at the start pose the fingers point back at the base
(the first version turned them a half circle and the arm contorted trying to follow). Pitch and
roll — the parts five joints can follow — are kept, and it is skipped within ~11° of straight
down, where a heading is noise.
It costs nothing the arm could have done, and it is what a person's wrist does anyway: you
turn toward what you are reaching for.

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

**Found on the headset, not by reading: the right arm would not move.** The page's send
function rate-limited per *message*; with two controllers the first one in a frame set the
timestamp and the second, a microsecond later, was dropped. The Quest lists the left
controller first, so one session's logs held 1,133 engaged samples on the left driver and 91
on the right. The gate is per frame now and both hands go out together.

## The buttons, and recording in the LeRobot layout

| controller | button | does |
|---|---|---|
| right | **B** | start recording an episode |
| right | **A** | end the episode |
| left | **X** | reset the scene: objects to their start, both arms to their rest pose |
| left | **Y** | both arms glide back to their start pose |
| either | GRIP | move and turn that arm (a clutch) |
| either | TRIGGER | close that arm's jaw |

The page sends buttons 4 and 5 of each controller as it always did; the bridge gives them
their jobs per hand, and the drivers underneath still only know "home" and "reset". While an
episode is recording the main panel in the headset carries a red **● REC** badge (the bridge
tells the page on every change and on connect; the page composites it onto the frames). A
recording request travels on the reply's `info` (`{"record": "start" | "stop"}`), once, and
`run.py` does the recording — it is the process that has the joint state, the joint targets
and the camera frames, and it writes with the same `LeRobotRecorder` that
`scripts/collect_dataset.py` uses, so the result trains with `lerobot-train` unconverted:

```
datasets/vr_<time>/            (or run.py --dataset DIR; --task sets the language string)
  meta/info.json               fps 50, features, counts       meta/episodes.jsonl   meta/tasks.jsonl
  data/chunk-000/episode_000000.parquet
  videos/chunk-000/a_wrist_right/episode_000000.mp4   videos/chunk-000/b_wrist_left/...
```

`observation.state` and `action` are twelve wide — `right_shoulder_pan` … `right_gripper`,
then the left arm — the state being the joints as measured and the action the joint *targets*
the IK set that step, which is what a policy trained on the data will be asked to produce.
The images are the wrist cameras (every camera with `attach:`), at their own sizes, 50
frames a second with the camera's 20 fps frames repeated. An episode ends on A, on a scene
reset, or when the simulator closes; one-frame episodes are dropped. The dataset directory
is created at the first B and its metadata rewritten after every episode, so it is valid at
any moment.

**Verified with the fake controller** (B at 1.2 s, A at 15 s, the mug pick in between):
one episode of 449 frames at 50 fps, `observation.state` and `action` (449, 12), two videos of 449 frames each at 480×360 and 320×240, `meta/info.json` with the features above, `episodes.jsonl` and `tasks.jsonl` — the right arm's joints move through the pick in the data, the left arm's hold still, and the wrist video shows the fingers over the table.

## A camera on each gripper

The view the LeRobot wrist camera gives — both finger tips in the bottom corners of a wide
frame, the table ahead — is the one the operator picks by, so each gripper now carries one.
`attach: gripper_base` on a camera mounts it on that body and it rides the arm; `pos` is then
the offset in the gripper's own frame (fingers along −y toward the palm, +z the top of the
housing), `pitch` the tilt down from looking along the fingers, `robot: robot2` puts it on the
second arm, and `fisheye: 170` swaps the pinhole for the 170° Kannala-Brandt lens
[liorbenhorin/lerobot_so101_teleop](https://github.com/liorbenhorin/lerobot_so101_teleop)
uses for its ego camera (their polynomial, scaled to the resolution). The mount was placed by
rendering, five mounts compared against the photo: 10 cm behind the palm, 6 cm above the
finger line, 10° down. Closer or steeper (7.5 cm, 35°) filled the frame with fingers and lost
the object; 20° down brought the housing into the bottom third. The right wrist camera is the headset's big front panel; the left wrist camera,
the front view and the top view float around it. The side camera went: each camera render
is 26 ms and four at 20 fps is where 50 steps/s still holds.

## Two arms, a crate, a mug and a soup can

`scene.robot2` adds a second SO-101 beside the first: its own articulation, grasp frame
(`ee_frame2`, reported as `ee_pose2`), IK and gripper actions, so the action grows from 8 to
16 — the first arm's eight, then the second's. The bridge needs no flag: the first packet that
carries `ee_pose2` wakes the left driver, and from then on the **right controller drives the
right arm and the left controller the left**, each with its own home, clutch, glide and
orientation anchor; B on either resets the scene. Only the parallel gripper with
`control.actions: ik`; the task's reward, observations and resets still watch one robot and
one object, so the second arm is scenery to the score and a robot to the operator.

The crate scene (`configs/vr_teleop_crate.yaml`) puts the arms 30 cm apart (y = ∓0.15, right and left as the operator sees
them) with a crate between them — a 16 cm floor and four 6 cm walls, static boxes — and real
things from Isaac Sim's YCB set 0.35 m in front of each arm, under the finger pads at the
start pose: a **mug** (81 mm across, 120 g, handle turned away) for the right arm and a
**tomato soup can** (68 mm, 200 g) for the left. Both are a pinch across the body for fingers
that close to 49 mm and open to 134. The props stream from the asset server on first use and
are cached after. The arms' reach at table height is 0.26–0.31 m from their own base, which
is why each object sits in front of its own arm and the crate sits between them: lifting over
the wall takes 7 cm, and the crate's near half is within both arms' reach.

The goal-pose and grasp-frame markers (`/Visuals/Command/*` and the IK target frame) are off:
`scene.debug_markers: false`. They are for a policy's author and float in the operator's view.

## Clothes folding: the shirt, on PhysX

The shipped scene (`configs/vr_teleop.yaml`) is the two arms and a T-shirt flat on the table
between them, `shirt: {type: physx_cloth, usd_path: assets/garment/shirt.usd, scale: 0.18}`.
LeHome's engine was the first thing tried, since that is where the shirt asset came from:
LeHome runs its cloth as **PhysX particle cloth**, and that schema is gone from the PhysX in
Isaac Sim 5+ (`isaaclab_physx` has no particle system at all; only the OmniPhysics deformable
bodies). The Newton VBD path (`configs/lehome_bedroom_shirt.yaml`) stays a separate config:
every camera render there does a full particle writeback and a device stall, 1.9–4.2 steps/s
(docs/PHYSICS.md). So the shirt is a **PhysX surface deformable** — the same engine as the arms,
the same contacts, and a camera costs what it costs on a rigid scene. It is GPU-only, so this
config is the one VR scene on `sim.device: cuda:0`. The material is the set Isaac Lab's own PhysX
cloth-lifting task ships (Young's modulus 1e6, 1 mm thick, 5 mm rest offset) except the
friction, 1.0 instead of their 10 (the mock below says why); solver stiffnesses, not an
identified fabric. The task's terms still need a rigid `object`, so a 2 cm cube sits
parked behind the left arm, out of every camera.

**The jaw closes.** The asset was authored with the finger stop at −0.044, which leaves the
pads 49 mm apart when closed: fine for a mug, and no use on cloth — a layer of cloth is two
rest offsets thick to a rigid pad (10 mm here), so with a 49 mm gap a single layer can never
be pinched whatever the material. The prismatic stop in `physics.usda` is now −0.068 (each
pad moves 0.97 mm per mm of travel), and `tuning.py`'s travel and close command follow.
Measured after the change: the close command drives the fingers to −0.062, where the pads meet and stop (the −0.068 stop itself is never reached); open is unchanged at 134 mm. ASSUMED, not measured on the real arm, that its
jaw closes fully; if it stops short, put the real gap's travel back in both places.

**Measured, headless, four cameras at 20 fps:** **15.6 steps/s** with the shirt and four cameras at 20 fps (400 steps); 19.2 with the cameras off, 19.1 with the shirt removed instead, 15.3 with the 9 mm shirt mesh, 15.8 with the cameras at 10 fps. The floor is GPU PhysX itself — a substep is 5.3 ms on the GPU against 0.6 on the CPU, and two arms' IK terms run per substep — not the cloth (4 steps/s) or the cameras (4 steps/s). That is 0.3× real time: the arms follow the operator at a third of their speed, and the recording's 50 fps is simulator time. The rigid crate scene stays on CPU physics at 55–60.

**The mock on the shirt, fake controller on the right arm:** the fake controller's 12-second script (reach out from the rest pose, tilt, descend, close, lift, carry 150 mm left, release, rise) with the shirt placed under its grasp (world y −0.15, `pcloth.yaml`): the jaw closes at 37 mm on the shirt's collar bulk, the shirt's highest node goes from 51 mm at rest to **175 mm** in the carry, and drops back to 55 mm when the jaw opens — **CARRIED**. With the reference friction of 10 the shirt stayed hung over the pads after release (top at 185 mm); friction 1.0 holds and lets go, and is what ships. Tracking over the 940 steps after the first request **8.6 mm mean**; two transient dips of ~40 mm (15°) during the fast reach and the descent. The same script shows them with no shirt in the scene at all, on GPU and on CPU physics alike: with nothing under the pads the descent runs `wrist_flex` to its −1.66 stop and the pads onto the table, and the solver trades 15° of pitch for it. A transient of the script, not the cloth or the device; the crate mock (6.6 mm mean) had the mug under the pads.

**The two-arm mock, fake controller on the right arm, the left arm holding:** the right arm reaches out from its rest pose, tilts, closes on the mug at 71 mm, lifts it to 123 mm, carries it 150 mm to the crate and releases it inside (the mug settles at 83 mm, leaning on a wall). Tracking error over the 940 steps after the first request **6.6 mm mean**, orientation **4.4° mean**; the left arm holds its rest pose to the millimetre throughout — **CARRIED**.

## Making it fast: where a step goes

The target is 50 steps/s — the environment steps at 50 Hz (`dt` 0.01, decimation 2), so that
is real time. The first live sessions ran at 16–18. RTX settings and lights were tried first
(a "cheap" render block: 1 sample per pixel, no bounces, reflections, GI, AO, translucency or
denoiser; a distant light instead of the dome) and changed nothing: 17–19 either way. So one
step was profiled phase by phase instead, one boot, 100–150 repetitions each
(RTX 4070 Ti, three tiled cameras, headless unless said):

| phase | GPU physics (`cuda:0`) | CPU physics |
|---|---|---|
| one PhysX substep | 5.3 ms | **0.6 ms** |
| IK action term, `apply_action` (runs per substep) | 3.9 ms | 1.4 ms |
| `write_data_to_sim` | 1.7 ms | – |
| rewards + observations + terminations | 2.0 ms | – |
| **`env.step`, no camera read** | **31 ms (32/s)** | **8.1 ms (123/s)** |
| one camera render + readback (all three cameras) | 26 ms | 26 ms |
| `env.step` + cameras at 30 fps (a render every 2nd step) | 43 ms (23/s) | **20 ms (50/s)** |
| `env.step` + cameras at 15 fps | 38 ms (26/s) | **14 ms (70/s)** |
| packet build + msgpack, 1 MB of frames | 0.15 ms | – |
| ZeroMQ round trip, 1 MB | 0.9 ms | – |
| the Kit viewer window, per step on top of everything | +25 ms | +25 ms |

Three findings, in order of size:

1. **One arm on GPU PhysX is all launch overhead.** A substep is 5.3 ms on the GPU and 0.6 ms
   on the CPU, and the IK term — which fetches the Jacobian every substep — halves too. The
   config now sets `sim.device: cpu`; the step went from 31 ms to 8. Rendering stays on the GPU
   regardless; Newton backends need `cuda`, this is PhysX.
2. **A camera render is 26 ms whatever you turn off.** The RTX toggles above, `--kit_args`
   with DLSS off, AA off, sampled lighting off and `waitIdle` false: all within 1 ms of each
   other. It is the render *pipeline* per frame, not the shading, so the only lever is how
   often it runs: `update_period` on each camera. 30 fps costs 13 ms a step averaged, 15 fps
   costs 6.5. The transport was never the problem — a megabyte of frames is a millisecond.
   Fewer cameras help some (one instead of three: a render of ~15 ms instead of ~24) and
   halving every resolution helps little (~3 ms), so the three screens stay and the rate is
   the dial.
3. **The viewer window is a second render.** 25 ms per step, every step. The GUI now starts
   the simulator without it (a checkbox brings it back); the VR screens do not need it.

**Live, bridge running and the page connected** (the bridge JPEG-encodes three streams on the
same CPU the physics now runs on): cameras at 30 fps gave **43–45 steps/s**; at 20 fps
(`update_period: 0.05`, the shipped value) **55–60 steps/s**. The environment steps at 50 Hz,
so that is real time with a margin; the headset sees each camera at 20 fps.

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
