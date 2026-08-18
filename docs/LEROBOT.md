# Using LeRobot with this environment

Three things [LeRobot](https://github.com/huggingface/lerobot) can do here:

| | |
|---|---|
| **Teleoperate** | drive the simulated arm from a physical SO-101 leader arm |
| **Collect data** | record simulated episodes as a LeRobot dataset, ready for `lerobot-train` |
| **Run a policy** | drive the simulated arm with an ACT / Diffusion / π0 / SmolVLA policy |

## Install LeRobot separately

LeRobot goes in **its own venv**, not beside Isaac Sim.

```bash
python -m venv .lerobot
.lerobot\Scripts\activate          # Linux/macOS: source .lerobot/bin/activate
pip install lerobot pyzmq msgpack
```

Isaac Sim 6.0.1 pins torch 2.11.0+cu128 and Python 3.12; LeRobot pins its own torch and a large
dependency tree. They do not need to agree, because they only exchange messages over a socket.
Installing both into one environment is a fight with no upside.

Nothing in `simbridge` imports LeRobot, so the environment side works whether or not it is
installed.

## The joint mapping

LeRobot's `SOFollower` declares its motors in the same order as the Isaac Lab articulation:

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Two things differ and are converted in `simbridge/lerobot.py`:

- **Units.** LeRobot normalises arm joints to −100..100 and the gripper to 0..100
  (`MotorNormMode.RANGE_0_100` — note it is not symmetric). This environment uses −1..1 offsets
  from a default pose.
- **Gripper action.** The pick-and-place task drives the jaw with `BinaryJointPositionActionCfg`,
  so only the sign matters. Pass `--continuous-gripper` if your task uses a continuous jaw.

**Calibration caveat.** The conversion assumes a physical arm's −100..100 spans the same range as
the URDF joint limits. That is close enough to teleoperate with and *not* close enough for
sim-to-real. Replace `simbridge.lerobot.JOINT_LIMITS` with a measured calibration if you need
transfer.

---

## 1. Teleoperate from a leader arm

In the LeRobot venv:

```bash
python scripts/lerobot_server.py teleop --port COM5 --id so101_leader
```

In the environment venv:

```bash
python scripts/run.py --config configs/pick_place_lerobot.yaml --set control.source=zmq
```

Check the wiring first without any hardware:

```bash
python scripts/lerobot_server.py mock      # sweeps the arm, imports no LeRobot
```

## 2. Collect a dataset

The trained PPO policy is an expert that never tires, so demonstrations cost simulation time
rather than human time.

```bash
python scripts/collect_dataset.py \
    --config configs/pick_place_lerobot.yaml \
    --episodes 200 \
    --out datasets/so101_pickplace
```

Measured on one RTX 4070 Ti, 32 envs, two 128 px cameras:

```
20 episodes, 5000 frames, 2 cameras, 60 MB
attempted 32, kept 20 (62% of started episodes ran to completion; 100% of completed ones succeeded)
```

Only successful episodes are written. Success is measured directly — object lifted and within
5 cm of its commanded goal — not by `terminated`, which in `contrib.lift` means *dropped*.

Output is the LeRobot v2.1 layout:

```
meta/info.json          fps, feature schema, counts
meta/episodes.jsonl     one line per episode
meta/tasks.jsonl        task index -> language string
data/chunk-000/episode_000000.parquet
videos/chunk-000/scene_cam/episode_000000.mp4
videos/chunk-000/wrist_cam/episode_000000.mp4
```

Train on it directly:

```bash
lerobot-train --dataset.root=datasets/so101_pickplace --policy.type=act
```

### Sizing

At 128 px with two cameras, roughly 3 MB per 250-step episode. 200 episodes is about 600 MB and
a few seconds of simulation. Generating far more than needed is cheap; the constraint is disk,
not time.

Keep the objective matching the distribution the expert trained on. Narrowing it produces an
expert evaluated off-distribution — a first run here scored 2% for a policy measured at 92%.

## 3. Run a LeRobot policy

```bash
# LeRobot venv
python scripts/lerobot_server.py policy \
    --path outputs/train/act_so101/checkpoints/last/pretrained_model

# environment venv
python scripts/run.py --config configs/pick_place_lerobot.yaml --set control.source=zmq
```

Add `--raw-actions` if the policy already emits actions in this environment's units rather than
LeRobot's normalised ones. Which applies depends on what the training data contained, so check
before trusting the numbers.

For a chunked policy (ACT, Diffusion, π0 all predict action sequences), set the execution horizon
in the config and the chunk is cached and stepped for you:

```yaml
control:
  source: zmq
  action_horizon: 8
```

---

## What runs where

```
LeRobot venv                          environment venv
------------                          ----------------
lerobot_server.py    --ZeroMQ-->      run.py
  leader arm                            Isaac Sim 6.0.1
  or a policy                           the task and scene
  imports no Isaac Sim                  imports no LeRobot
```

Dataset collection runs entirely on the environment side; LeRobot only reads the result.

## Testing without hardware or LeRobot

```bash
python -m simbridge.lerobot              # unit conversion, joint order, round-trip
python -m simbridge.lerobot_recorder     # writes a small dataset and reads it back
python scripts/lerobot_server.py mock    # serves actions with no LeRobot import
python scripts/run_all_tests.py          # everything above, plus the rest
```
