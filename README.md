# SO-ARM101 scene, reach & pick-and-place — Isaac Lab 3.0

A self-contained Isaac Lab 3.0 project for the
[TheRobotStudio SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) 5-DOF arm: a
standalone physics scene plus two Gym-registered PPO tasks — **reach** and
**pick-and-place** — running on either the PhysX or Newton backend.

Built and measured on Isaac Sim 6.0.1 + Isaac Lab 3.0 (`develop`), Windows 11,
RTX 4070 Ti 12 GB, i5-13600KF.

## Contents

| Path | What it is |
|---|---|
| `scripts/scene_demo.py` | Standalone scene — arm on a pedestal, cube, sine sweep. Dumps real joint/body names. |
| `scripts/measure_ee.py` | Measures the grasp point between the jaws (source of `EE_GRASP_OFFSET`). |
| `scripts/bench_envs.py` | Sweeps `--num_envs`, reporting throughput, VRAM and GPU utilisation. |
| `scripts/train.py` / `play.py` | Training / playback entrypoints. |
| `so101_scene/reach_env_cfg.py` | Reach task, retargeted from `ReachEnvCfg`. |
| `so101_scene/pick_place_env_cfg.py` | Pick-and-place, retargeted from `contrib.lift` + a physics preset. |
| `so101_scene/tuning.py` | Shared robot config and the measured backend notes. |

Env ids: `SO101-Reach-v0`, `SO101-Reach-Play-v0`, `SO101-PickPlace-v0`, `SO101-PickPlace-Play-v0`.

## Setup

Requires an existing Isaac Lab 3.0 environment (Isaac Sim 6.0.1, Python 3.12).

```bash
pip install -e .
```

## Run

```bash
# Scene only. Prints the articulation's joint/body names and limits.
python scripts/scene_demo.py --steps 400 --num_envs 4

# Pick-and-place, tuned for a 12 GB card (see GPU scaling below)
python scripts/train.py --rl_library rsl_rl --task SO101-PickPlace-v0 --num_envs 8192

# Newton backend instead of PhysX — read the startup-cost note first
python scripts/train.py --rl_library rsl_rl --task SO101-PickPlace-v0 physics=newton_mjwarp --num_envs 8192

# Watch a trained policy
python scripts/play.py --rl_library rsl_rl --task SO101-PickPlace-Play-v0 --viz newton
```

Omitting `--viz` runs headless — Isaac Lab 3.0 has no `--headless` flag.

## GPU scaling

Measured with `scripts/bench_envs.py` on `SO101-PickPlace-v0`, PhysX, 12 GB card.
`steps/s` is steady-state (first half of samples discarded — cold caches are not steady state).

| envs | startup | steps/s | iter time | VRAM | GPU % | gain |
|---|---|---|---|---|---|---|
| 2048 | 27 s | 55,425 | 1.18 s | 3.5 GB | 40 % | — |
| 4096 | 38 s | 87,974 | 1.49 s | 4.2 GB | 59 % | +59 % |
| **8192** | **64 s** | **119,754** | **2.19 s** | **5.7 GB** | **79 %** | **+36 %** |
| 12288 | 93 s | 134,959 | 2.92 s | 7.3 GB | 55 % | +13 % |
| 16384 | 126 s | 142,034 | 3.69 s | 8.3 GB | 88 % | +5 % |

**8192 is the default worth using on 12 GB**: 84 % of peak throughput at two-thirds the VRAM
and half the startup. Past it the curve flattens hard — 16384 costs 2x the memory and 2x the
startup to buy 19 % more throughput, and leaves little headroom if you later add cameras or a
Kit viewport.

Below 4096 the GPU is simply idle waiting for work; 2048 (a common tutorial default) leaves
more than half the card unused on this task.

## Backend choice: PhysX vs Newton

**PhysX is the default here, deliberately** — the same choice Isaac Lab's own SO-101 stack
task makes.

`physics=newton_mjwarp` pays a large one-time CPU cost before training starts: Newton runs
CoACD convex decomposition over the SO-101 collision meshes at scene-build time. A 256-env
run **burned 900 CPU-seconds inside CoACD and never reached iteration 0**. The same scene on
PhysX does zero CoACD and reaches iteration 0 in 64 s at 8192 envs.

It is a *startup* cost, not per-step, so it amortises over a long run — but it makes Newton
painful for short iteration cycles. Both backends are supported; pick knowingly.

`contrib.lift` hardcodes a single PhysX config, so `physics=` is not selectable on it
upstream. `PickPlacePhysicsCfg` restores the choice, with Newton solver values taken from
Isaac Lab's SO-101 *stack* task (`njmax=300, nconmax=200`, elliptic cone, `impratio=10`)
rather than the reach preset — grasping is contact-rich and reach's `njmax=100, nconmax=20`
drops contacts.

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

## Why the tasks override what they do

Isaac Lab's stock reach and lift tasks target 0.85 m-class arms (Franka, UR10). The SO-101 is
a ~0.30 m, 5-DOF, single-jaw arm, so a robot swap alone produces a task that runs and never
learns:

1. **Workspace.** Stock command ranges (`pos_x` 0.35–0.65 m) fall entirely outside the
   SO-101's envelope — every episode is unsolvable. Ranges are resized against the measured
   joint limits, and success thresholds tightened proportionally (5 cm → 3 cm).
2. **5 DOF, not 6.** The arm cannot reach an arbitrary 6-DOF pose. Reach soft-weights
   orientation tracking (-0.1 → -0.02) rather than removing it: it still biases the wrist
   toward the commanded approach without an unzeroable error swamping the position objective.
3. **Object size.** The Franka reference lifts a 4 cm DexCube; the SO-101's single jaw cannot
   span it. The cube here is 2.5 cm with high static friction — a pinch grasp is
   friction-limited where a parallel jaw is not.
4. **Reward geometry.** Shaping `std` values are ~3x tighter, matching the ~3x smaller arm.
   At Franka scale, "close enough" spans the whole workspace and the gradient vanishes.
5. **Reset scale.** Narrowed to `(0.9, 1.1)` — stock `(0.5, 1.5)` can start the arm fully
   extended on its boundary singularity.

## Notes

- Assets stream from NVIDIA's S3 asset root on first run and cache locally; the first launch
  is slow, later ones are not.
- `gripper` is excluded from the reach action space — a reach task does not actuate the jaw.
  Pick-and-place drives it as a separate binary open/close action.
- Overriding the robot's collision approximation to `convexHull` to dodge CoACD does **not**
  work: `modify_collision_properties` only touches prims that already carry the collision
  schema, and the SO-101 asset authors none, so the override silently no-ops. See
  `so101_scene/tuning.py`.
