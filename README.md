# SO-ARM101 scene & reach task — Isaac Lab 3.0

A minimal, self-contained Isaac Lab 3.0 project for the
[TheRobotStudio SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) 5-DOF arm:
a standalone physics scene, plus a Gym-registered PPO reach task that runs on either the
**Newton** or **PhysX** backend.

Built and verified on Isaac Sim 6.0.1 + Isaac Lab 3.0 (`develop`), Windows 11, RTX 4070 Ti.

## Contents

| Path | What it is |
|---|---|
| `scripts/scene_demo.py` | Standalone scene — arm on a pedestal, cube, sine sweep. Dumps real joint/body names. |
| `scripts/train.py` | Training entrypoint (registers this repo's envs, then defers to Isaac Lab). |
| `scripts/play.py` | Policy playback. |
| `so101_scene/reach_env_cfg.py` | The reach task, retargeted from Isaac Lab's `ReachEnvCfg`. |
| `so101_scene/agents/rsl_rl_ppo_cfg.py` | PPO hyperparameters. |

Registered env ids: `SO101-Reach-v0`, `SO101-Reach-Play-v0`.

## Setup

Requires an existing Isaac Lab 3.0 environment (Isaac Sim 6.0.1, Python 3.12).

```bash
pip install -e .
```

## Run

```bash
# Scene only — no RL. Prints the articulation's joint/body names and limits.
python scripts/scene_demo.py --steps 400 --num_envs 4

# Same scene, interactive viewport
python scripts/scene_demo.py --steps 0 --num_envs 4 --viz kit

# Train (Newton backend)
python scripts/train.py --rl_library rsl_rl --task SO101-Reach-v0 physics=newton_mjwarp --num_envs 256

# Train (PhysX backend)
python scripts/train.py --rl_library rsl_rl --task SO101-Reach-v0 physics=isaacsim_physx --num_envs 256

# Play a trained policy
python scripts/play.py --rl_library rsl_rl --task SO101-Reach-Play-v0 --viz newton
```

Omitting `--viz` runs headless — Isaac Lab 3.0 has no `--headless` flag.

## The robot

Verified by spawning it, not read off the USD:

```
DOF         : 6
joint_names : shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
body_names  : base, shoulder, upper_arm, lower_arm, wrist, gripper, moving_jaw_so101_v1
```

| Joint | Limits (rad) |
|---|---|
| `shoulder_pan` | -1.920 … +1.920 |
| `shoulder_lift` | -1.745 … +1.745 |
| `elbow_flex` | -1.690 … +1.690 |
| `wrist_flex` | -1.658 … +1.658 |
| `wrist_roll` | -2.744 … +2.841 |
| `gripper` | -0.175 … +1.745 |

The end-effector body is **`gripper`** — note there is also a *joint* named `gripper`.
Re-run `scripts/scene_demo.py` if the upstream asset changes; everything else keys off these names.

## Why the task overrides what it does

Isaac Lab's stock `ReachEnvCfg` targets 0.85 m-class arms (Franka, UR10). Two things had to change:

1. **Workspace.** Stock command ranges (`pos_x` 0.35–0.65 m) fall entirely outside the SO-101's
   ~0.30 m envelope — every episode would be unsolvable. Ranges were resized against the joint
   limits above, and the success threshold tightened 5 cm → 3 cm to stay proportional.
2. **5 DOF, not 6.** The arm cannot reach an arbitrary 6-DOF pose. Orientation tracking is
   soft-weighted (-0.1 → -0.02) rather than removed: it still biases the wrist toward the
   commanded approach, without an unzeroable orientation error swamping the position objective.

Reset scale is also narrowed to `(0.9, 1.1)` — the stock `(0.5, 1.5)` can start the arm fully
extended on its boundary singularity, where a joint-space policy gets little usable gradient.

## Notes

- Assets stream from NVIDIA's S3 asset root on first run and are cached locally; the first
  launch is slow, later ones are not.
- `gripper` is excluded from the action space — a reach task does not actuate the jaw. Add it to
  `SO101_ARM_JOINTS` if you extend this toward grasping.
