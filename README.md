# SO-ARM101 simulation environment for Isaac Lab 3.0

A config-driven simulation environment for the
[TheRobotStudio SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) 5-DOF arm, with a
swappable boundary between the environment and whatever drives it.

Scenes are declared in YAML. What controls the arm — a trained policy, a teleoperation device,
a policy server in another process — is one line of that YAML, and the environment never learns
which one is attached.

Built and measured on Isaac Sim 6.0.1 / Isaac Lab 3.0, Windows 11, RTX 4070 Ti 12 GB.

> This page is the environment: how to set it up, how to declare a scene, and how to talk to it.
> **[`experiment/`](experiment/)** is a worked *example* of that interface — driving this
> environment with a reinforcement-learning policy, both in-process and over a socket.

---

## Install

Requires an existing Isaac Lab 3.0 environment (Isaac Sim 6.0.1, Python 3.12).

```bash
pip install -e .
```

Pulls `pyzmq`, `msgpack` and `pyyaml`. Isaac Sim and Isaac Lab are expected to be present already.

---

## Quick start

```bash
# Standalone scene: arm on a pedestal, cube, sine sweep.
# Prints the articulation's real joint and body names.
python scripts/scene_demo.py --steps 400 --num_envs 4

# Run a YAML-declared scene, driven by whatever the config names
python scripts/run.py --config configs/pick_place.yaml

# Same scene, different driver -- nothing else changes
python scripts/run.py --config configs/pick_place.yaml --set control.source=zero

# Check a config without booting Isaac Sim (fast)
python scripts/run.py --config configs/pick_place.yaml --describe
```

---

## Declaring a scene

A config names a task and declares the robot, props, cameras and driver:

```yaml
task: pick_place

scene:
  num_envs: 256
  robot:
    type: so101
    rot: [0.0, 0.0, 0.70710678, 0.70710678]   # (x,y,z,w) in Isaac Lab 3.0
    joint_pos: {shoulder_lift: -0.6, elbow_flex: 0.8, wrist_flex: 0.6}
  cameras:
    scene_cam:
      type: tiled
      prim_path: "{ENV_REGEX_NS}/SceneCam"
      resolution: [128, 128]

sim:
  episode_length_s: 5.0

control:
  source: rl_checkpoint            # zero | random | rl_checkpoint | zmq
  checkpoint: path/to/model.pt
  action_horizon: 1                # raise for chunked generative policies
```

Reward and termination logic stays in Python (`so101_scene/*_env_cfg.py`). Those are code, and a
config language that tries to express them becomes one nobody can debug.

Validation is strict: an unknown key is an error listing the valid ones, not a silent no-op. A
typo'd `resolutoin` that quietly leaves the camera at its default is exactly the bug worth
catching at parse time.

### Vocabulary

| kind | registered names |
|---|---|
| robots | `so101` |
| objects | `cuboid`, `static_cuboid`, `usd` |
| cameras | `tiled` |
| action sources | `zero`, `random`, `rl_checkpoint`, `zmq` |

Extend without editing the package:

```python
from simbridge.registry import register_object

@register_object("my_widget")
def _widget(spec):
    return RigidObjectCfg(...)
```

---

## Driving the environment from another process

`control.source: zmq` points the environment at a policy server. The server imports no
simulator — no Kit, no USD — so a model can run in its own process, its own CUDA context, even
on another machine.

```bash
# terminal 1
python scripts/policy_server.py --policy sine --action-dim 6

# terminal 2
python scripts/run.py --config configs/pick_place_teleop.yaml
```

Writing your own server means supplying one function:

```python
from simbridge.schema import ObsPacket
from simbridge.transport import ZmqPolicyServer

def policy(obs: ObsPacket):
    obs.state["policy"]           # (num_envs, obs_dim) float32
    obs.images["scene_cam"]       # (num_envs, H, W, 3) uint8
    return my_model(obs)          # (num_envs, action_dim)

ZmqPolicyServer(policy, endpoint="tcp://127.0.0.1:5555").serve_forever()
```

### The wire contract

Two messages, versioned, msgpack-framed. Arrays travel as raw bytes plus shape and dtype — a
256×128×128×3 batch is 12 MB, and JSON-encoding it costs more than the simulation step that
produced it.

| | contents |
|---|---|
| `ObsPacket` | `step`, `num_envs`, `state{}`, `images{}`, `done`, `reward`, `info` |
| `ActionPacket` | `step`, `action (num_envs, action_dim)`, `reset`, `info` |

`step` is echoed back and checked, so a driver that falls behind is an error rather than a stale
action silently applied.

Transport is REQ/REP because simulation is lock-step: the environment cannot advance without an
action, so the request/reply pairing *is* the control flow. PUB/SUB would drop or reorder under
load, and the environment would act on stale observations — a failure that looks like a bad
policy rather than a bad socket. Client-side timeouts turn a dead server into a loud error
instead of a hung simulator.

Measured: 432 MB/s, and a dead server raises in 0.8 s.

### Action chunking

Generative policies emit trajectories, not single actions — Diffusion Policy executes `Ta` of
`Tp` predicted steps, π0 serves horizons of 10–15. `ActionSource` caches a chunk and re-plans
only when it is spent; set `control.action_horizon`. Per-step drivers return a chunk of length 1
and pay nothing for the machinery.

The interface shape follows [NVlabs/RoboLab](https://github.com/NVlabs/RoboLab)'s
`InferenceClient` (four hooks, chunking in the base class), which is proven against π0/π0.5
flow-matching servers. RoboLab itself cannot be installed on this stack — it pins Isaac Sim
5.0/5.1, Isaac Lab 2.2/2.3, Python 3.11 and Ubuntu.

---

## The robot

Verified by spawning it, not read off the USD:

```
DOF         : 6
joint_names : shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
body_names  : base, shoulder, upper_arm, lower_arm, wrist, gripper, moving_jaw_so101_v1
```

| joint | limits (rad) |
|---|---|
| `shoulder_pan` | −1.920 … +1.920 |
| `shoulder_lift` | −1.745 … +1.745 |
| `elbow_flex` | −1.690 … +1.690 |
| `wrist_flex` | −1.658 … +1.658 |
| `wrist_roll` | −2.744 … +2.841 |
| `gripper` | −0.175 … +1.745 |

The end-effector body is **`gripper`** — note there is also a *joint* named `gripper`.
Re-run `scripts/scene_demo.py` if the upstream asset changes; everything keys off these names.

Two things about this arm that break code ported from larger robots:

- **5 DOF.** It cannot reach an arbitrary 6-DOF pose. Task-space controllers should command the
  full pose but soft-weight orientation so position tracks exactly and orientation is best-effort.
- **Default joint pose is unusable.** `SO101_CFG`'s all-zero pose extends the arm up and along
  −y, away from the workspace. Configs here seat it in a mid-range crouch with a 90° base yaw.

---

## Registered tasks

| id | what it is |
|---|---|
| `SO101-Reach-v0` | reach a randomised end-effector pose |
| `SO101-PickPlace-v0` | pick a 2.5 cm cube, place it at a commanded pose |
| `SO101-PickPlace-Play-v0` | 16 envs, no observation noise, for playback |
| `SO101-PickPlace-Wide-v0` | wider cube spawn range |
| `SO101-PickPlace-Blind-v0` | cube position removed from the observation (a diagnostic, not a task) |

---

## Utilities

| script | what it answers |
|---|---|
| `scripts/scene_demo.py` | does the robot load, and what are its real joint/body names? |
| `scripts/diagnose_scene.py` | where actually are the base, end-effector, object and goal? |
| `scripts/measure_ee.py` | where is the grasp point between the jaws? |
| `scripts/dump_camera_views.py` | what do the cameras actually see? |
| `scripts/run.py` | run any YAML scene with any driver |
| `scripts/policy_server.py` | serve actions over ZeroMQ |

These exist because guessing geometry was the source of every real bug in this repo — a grasp
offset off by a cube width, an arm facing away from its workspace. Measuring takes a minute.

---

## Layout

```
so101_scene/     task and scene definitions (rewards, terminations, robot tuning)
simbridge/       the policy-environment boundary: schema, interfaces, transports, YAML builder
configs/         YAML scene declarations
scripts/         environment utilities and the runner
experiment/      worked example: driving the env with an RL policy -> experiment/README.md
```

Self-checks (each module has one, runnable):

```bash
python -m simbridge.schema
python -m simbridge.interfaces
python -m simbridge.transport.test_roundtrip
```
