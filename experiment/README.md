# Example: driving the environment with a reinforcement-learning policy

A worked example of the interface described in the [main README](../README.md) — the same
environment, driven three different ways, with nothing changing on the environment side.

The RL result here is incidental. It exists to give the example a real policy to talk to.

---

## The three ways to attach a driver

### 1. In-process, via the YAML

The environment loads the checkpoint itself. Simplest, lowest latency, no second process.

```yaml
control:
  source: rl_checkpoint
  checkpoint: logs/rsl_rl/pick_place_so101/<run>/model_1499.pt
  action_horizon: 1
```

```bash
python scripts/run.py --config configs/pick_place.yaml
```

### 2. Over a socket, policy in its own process

The environment holds only a ZeroMQ client. The server imports no simulator, so the policy can
use a different framework, a different CUDA context, or a different machine.

```bash
# terminal 1 -- no Isaac Sim import here
python scripts/policy_server.py --policy checkpoint \
    --checkpoint logs/rsl_rl/pick_place_so101/<run>/model_1499.pt

# terminal 2
python scripts/run.py --config configs/pick_place_teleop.yaml
```

The only difference from (1) is the `control` block:

```yaml
control:
  source: zmq
  endpoint: tcp://127.0.0.1:5555
  timeout_ms: 30000
```

### 3. Baselines, for comparison

```bash
python scripts/run.py --config configs/pick_place.yaml --set control.source=zero
python scripts/run.py --config configs/pick_place.yaml --set control.source=random
```

Worth running before trusting any policy: a policy that does not beat `zero` by a clear margin
is not doing anything, and that is far quicker to check than reading a reward curve.

---

## Writing your own driver

Any driver is one method. The base class handles chunk caching, step accounting and horizons.

```python
from simbridge.interfaces import ActionSource
from simbridge.registry import register_source

@register_source("my_policy")
class MyPolicy(ActionSource):
    def _predict_chunk(self, obs):
        obs.state["policy"]        # (num_envs, obs_dim) float32
        obs.images["scene_cam"]    # (num_envs, H, W, 3) uint8
        return actions             # (num_envs, act) or (horizon, num_envs, act)
```

Return a `(horizon, num_envs, act)` chunk and set `control.action_horizon` to execute it
open-loop before re-planning — this is what a generative policy needs, and it is why chunking
lives in the base class rather than in each driver.

---

## Training, for reproduction

```bash
python experiment/scripts/train.py --rl_library rsl_rl --task SO101-PickPlace-v0 --num_envs 8192
python experiment/scripts/plot_rewards.py
python experiment/scripts/play.py --rl_library rsl_rl --task SO101-PickPlace-Play-v0 --num_envs 9 --video
```

Omitting `--viz` runs headless — Isaac Lab 3.0 has no `--headless` flag. Video recording needs
`pip install "moviepy<2" imageio-ffmpeg`; moviepy 2.x fails because Isaac Lab's recorder uses
the v1 API.

### Result

92 % success, 1500 iterations over 8192 parallel envs, 78 minutes on one RTX 4070 Ti.

![reward curve](docs/reward_curve.png)

[Policy video](docs/pick_place_policy.mp4)

| metric | start | final |
|---|---|---|
| `success_rate` | 0.000 | **0.921** |
| `lifting_object` | 0.04 | 14.02 |
| `object_goal_tracking` | 0.005 | 13.91 |
| object-to-goal distance | 0.272 m | 0.073 m |

The balance between `lifting_object` and `object_goal_tracking` is the thing to read. An earlier
run finished at 4.53 and 0.05 — the policy was paid to lift and not to carry, so it lifted the
cube and held it. Comparable magnitudes mean both halves of the task are worth doing.

---

## Measured throughput

Both from this machine, via `experiment/scripts/bench_envs.py` and `bench_camera.py`.

**State-only** (`SO101-PickPlace-v0`, PhysX):

| envs | steps/s | VRAM | GPU % |
|---|---|---|---|
| 2048 | 55,425 | 3.5 GB | 40 % |
| **8192** | **119,754** | **5.7 GB** | **79 %** |
| 16384 | 142,034 | 8.3 GB | 88 % |

8192 is the value worth using on a 12 GB card: 84 % of peak throughput at two-thirds the VRAM.
Past it the curve flattens — 16384 costs twice the memory and twice the startup for 19 % more.

**With cameras** (128 px, `TiledCamera`):

| envs | cams | env-steps/s | frames/s | VRAM |
|---|---|---|---|---|
| 512 | 1 | 27,378 | 27,378 | 10.1 GB |
| 256 | 2 | 13,904 | 27,809 | 9.1 GB |

Rendering costs ~4.4× versus state-only, not the order of magnitude usually assumed. `ms/step`
is flat regardless of env count, so tiled rendering amortises a fixed cost and throughput scales
linearly. The ceiling is ~27,000 frames/s, spendable on either envs or cameras; VRAM binds
before speed does.

Consequence: generating demonstrations is not a bottleneck. 200 episodes is ~2 s of simulation
and 2.5 GB on disk.

---

## Notes worth keeping

Things that cost real time here, recorded so they cost nobody else any:

- **Isaac Lab 3.0 changed the quaternion convention** to `(x, y, z, w)`. A `wxyz` value copied
  from 2.x code yaws the base somewhere silently wrong.
- **Shaping width tracks the distance the object travels, not the robot's size.** Scaling
  `std` down "because the arm is 3× smaller" produced rewards of 0.0003 and 0.007 — numerically
  dense, functionally constant. Two runs died this way before the pattern was obvious.
- **Inherited constants are suspect on a 0.30 m arm.** `contrib.lift`'s curriculum ramps a
  smoothness penalty to −1e-1 at 10k env-steps; on this arm that produced −4.07 against a task
  reward of +3.19, so the policy was paid more to hold still than to lift.
- **Newton (`physics=newton_mjwarp`) pays ~900 CPU-seconds** of CoACD convex decomposition at
  scene build. PhysX does none. Everything here is PhysX.
- **VRAM overflow does not raise on Windows.** WDDM silently spills to host RAM and runs 11–151×
  slower, so "it fits" has to be judged by ms/iteration, never by absence of a crash.

---

## Contents

```
scripts/train.py          RL training entrypoint
scripts/play.py           policy playback, optional video
scripts/plot_rewards.py   TensorBoard curves incl. per-term reward breakdown
scripts/bench_envs.py     num_envs sweep: throughput, VRAM, GPU utilisation
scripts/bench_camera.py   rendered-observation throughput
docs/                     curves, video, raw benchmark output, and the
                          visuomotor flow-matching design study
```
