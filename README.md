# SO-ARM101 scene, reach & pick-and-place — Isaac Lab 3.0

A self-contained Isaac Lab 3.0 project for the
[TheRobotStudio SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100) 5-DOF arm: a
standalone physics scene plus two Gym-registered PPO tasks — **reach** and
**pick-and-place** — running on either the PhysX or Newton backend.

Built and measured on Isaac Sim 6.0.1 + Isaac Lab 3.0 (`develop`), Windows 11,
RTX 4070 Ti 12 GB, i5-13600KF.


## Results

Trained policy: **92 % success rate** on pick-and-place, 1500 iterations / 8192 parallel envs,
78 minutes on one RTX 4070 Ti.

![reward curve](docs/reward_curve.png)

[Policy video](docs/pick_place_policy.mp4) � grasp, lift, carry to the commanded pose.

| metric | start | final | best |
|---|---|---|---|
| `Metrics/success_rate` | 0.000 | **0.921** | 0.936 |
| `Train/mean_reward` | 0.45 | **158.5** | 162.2 |
| `Episode_Reward/lifting_object` | 0.04 | 14.02 | 14.09 |
| `Episode_Reward/object_goal_tracking` | 0.005 | 13.91 | 13.98 |
| `Metrics/object_pose/position_error` | 0.272 m | **0.073 m** | � |

The balance between `lifting_object` (14.02) and `object_goal_tracking` (13.91) is the thing to
look at. In the run that failed they were 4.53 and 0.05: the policy was paid to lift and not to
carry, so it lifted and held. Comparable magnitudes mean both stages are worth doing.

Success is defined as the object reaching within 5 cm of the commanded pose. On a 2.5 cm cube
that is a real but not strict tolerance; the training env also carries observation noise that
the play env does not.

Reproduce:

```bash
python scripts/train.py --rl_library rsl_rl --task SO101-PickPlace-v0 --num_envs 8192
python scripts/plot_rewards.py
python scripts/play.py --rl_library rsl_rl --task SO101-PickPlace-Play-v0 --num_envs 9 --video
```

Video recording needs `pip install "moviepy<2" imageio-ffmpeg` � the isaacsim bundle ships
neither, and moviepy 2.x fails because Isaac Lab's recorder uses the v1 API.

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


## Reward shaping: the failure mode that cost two runs

Both failed runs had the same shape of bug, and it is worth stating as a rule because it is
easy to reintroduce:

> A dense shaping term only helps if it has gradient **at the distances the task actually
> starts from**. Its width must span the distance the thing has to travel � it is not a
> function of the robot's size.

`1 - tanh(d/std)` saturates fast. Scaling `std` down "because the arm is 3x smaller" produced
a reward that was numerically dense but functionally constant:

| term | `std` used | distance at start | reward | outcome |
|---|---|---|---|---|
| `reaching_object` | 0.04 | 0.177 m | 0.0003 | never approached the cube; policy std collapsed 1.00 -> 0.03 |
| `object_goal_tracking` | 0.12 | 0.335 m | 0.007 | learned to lift and hold; success 0.13 % |
| `object_goal_tracking` | **0.30** | 0.335 m | **0.194** | **success 89 %** |

That middle row is the interesting one. A dense reward flat at 0.007 across the whole working
range is *functionally sparse* � but without any of the machinery that makes sparse rewards
learnable. It is strictly worse than either honest option.

### Why that matters, from the literature

[Hindsight Experience Replay](https://arxiv.org/abs/1707.01495) (Andrychowicz et al., 2017)
reports two results that frame this directly:

- **Their dense shaping failed outright.** On the shaped reward `|g - s_object|^2`, Figure 5
  reads "Both algorithms fail on all tasks"; they also tried linear/quadratic terms pulling
  the gripper toward the object and none "led to successful training". Sparse binary rewards
  plus HER is what worked.
- **Pick-and-place needed an exploration prior.** They recorded one state with the box grasped
  and "start half of the training episodes from this state", because random exploration
  essentially never discovers a grasp. Isaac Lab ships the same idea as the
  `start_grasped_then_assembled` reset strategy in its NIST/factory tasks.

Those look like they contradict the result here, and the reconciliation is the point: HER is an
*off-policy* method (DDPG/SAC), unavailable to rsl_rl's on-policy PPO. What substitutes for it
is 8192 parallel environments � enough exploration volume that a genuinely dense reward finds
the grasp without hindsight relabelling. The failed runs did not have a dense reward; they had
a flat one.

If a harder variant of this task does stall, the demonstration-state trick is the documented
next lever, not more shaping tweaks.

### This task is known-hard upstream

Isaac Lab's own Franka lift carries open reports of the identical "reaches but will not lift"
symptom: [#204](https://github.com/isaac-sim/IsaacLab/issues/204),
[#1697](https://github.com/isaac-sim/IsaacLab/discussions/1697), one at 45 M steps with zero
successes. Treat any inherited constant that carries a magnitude as suspect on a 0.30 m arm.

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
