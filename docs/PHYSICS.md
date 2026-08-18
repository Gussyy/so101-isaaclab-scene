# Physics backends: PhysX, Newton, and deformables

Pick a backend in the config:

```yaml
sim:
  physics: newton_vbd     # physx | isaacsim_physx | newton_mjwarp | newton_vbd
```

An unknown name is refused at build time with the list the task actually offers.

| backend | solver | use it for |
|---|---|---|
| `physx` / `isaacsim_physx` | PhysX | the default. Rigid only, fastest to start |
| `newton_mjwarp` | MuJoCo-Warp | rigid on Newton |
| `newton_vbd` | MuJoCo-Warp + VBD, coupled | **required** for `cloth` and `soft_body` |

## This did not work before

`sim.physics` was an accepted key that nothing applied. Two mechanisms hid it:
`parse_env_cfg` calls `resolve_presets(cfg)` with no selection, collapsing every `PresetCfg` to
its `.default`; and `SimulationContext` does the same to any preset that still reaches it —

```python
if not hasattr(physics_cfg, "class_type") and hasattr(physics_cfg, "default"):
    physics_cfg = physics_cfg.default
```

— silently, both times. Every run in this repo was PhysX regardless of the config, including the
ones whose purpose was comparing backends. Selecting a backend means resolving the preset by
name *before* `parse_env_cfg` discards the alternatives.

## Startup cost

Newton runs CoACD convex decomposition over the robot's 17 collision meshes at scene build. The
asset explicitly asks for decomposition and the meshes are large — 161,982 points on the wrist
alone, and the servo housing at 57,240 appears four times.

Measured here, 4 envs / 30 steps, process start to exit:

| | wall | CoACD log lines |
|---|---|---|
| `physx` | **9 s** | 0 |
| `newton_mjwarp`, asset as authored | **285 s** | 1779 |
| `newton_mjwarp` + `collision_approximation: convexHull` | **24 s** | 234 |

```yaml
scene:
  robot:
    collision_approximation: convexHull
```

This keeps decomposition on `moving_jaw` and `wrist_roll_follower` — a convex hull across those
fills in the notch the jaw grips with. The other 15 are structural links and servo housings.

It is a *startup* cost, not a per-step one, so a long training run can ignore it.

An earlier note in `tuning.py` blamed this on the asset authoring no collision schema. That was
wrong: all 17 meshes already carry `MeshCollisionAPI` with `approximation=convexDecomposition`.
The override no-opped because those prims are **instance proxies**, which cannot be written to.
`make_uninstanceable=True` makes them writable.

## Deformables

```yaml
scene:
  objects:
    sheet:
      type: cloth
      size: [0.12, 0.12]
      resolution: [16, 16]
    blob:
      type: soft_body
      size: [0.05, 0.05, 0.05]
      k_mu: 3.0e3
sim:
  physics: newton_vbd
```

Declaring a deformable without `newton_vbd` is refused at parse time. Under any other backend
the asset spawns, renders, and never moves — the deformable builder hook is only installed by
the VBD manager, and "my cloth is broken" is a bad way to learn that.

`soft_body` is tetrahedralised at spawn, which needs a package a default Isaac Lab install omits:

```bash
pip install "pytetwild[all]>=0.3.0,<0.4"
```

### Measured state

300 steps at 1 env, deformation measured as per-node displacement after removing rigid
translation, then `env.reset()`:

| | peak deformation | |
|---|---|---|
| `cloth` draped over a static post | **43.5 mm** | settles by step 40 and holds to 200 |
| `soft_body` landing on the table | **5.0 mm** | ~2% strain, stable |
| `cloth` landing flat on a flat table | 0.0 mm | correct — uniform contact, no relative deformation |

Both survive `env.reset()`, and the task's rigid cube stays on the table throughout.

### Why the cloth used to fall straight through everything

This took a long time to find, so it is worth writing down properly.

**Static shapes belonged to no solver entry.** `CouplerEntryCfg.bodies` claims bodies and, with
`include_body_shapes=True`, the shapes attached to them. The table and ground plane are *static*
shapes — Newton body index `-1` — so naming `/Robot` and `/Object` never claimed them. And per
`CouplerCfg`:

> Bodies, particles, joints, and shapes may be assigned to at most one entry. **Unassigned model
> elements remain outside the nested solvers.**

So the table was in neither solver. The cloth had nothing to land on, and the cube had nothing to
rest on — it fell through to `z = -0.05`, tripped `object_dropping`, and reset. Forever.

Copying `include_static_shapes=True` onto the *soft* entry, as the shipped Franka task does, is
equally wrong here: it takes the table away from the rigid solver instead of leaving it nowhere.
Both arrangements produce **byte-identical** behaviour, which is what made this expensive — I
changed `num_substeps` from 2 to 4 and the trajectory did not move by a single digit. That should
have been the tell immediately: if doubling the substep count changes nothing, the thing you are
configuring is not in the loop.

The fix is `include_static_shapes=True` on the **rigid** entry. The cloth still feels the table,
through the proxy's shared outer contacts (`collision_pipeline=None`, which requires
`collide_interval=None`).

The Franka task gets away with the other arrangement because its arm is fixed-base and its cloth
rests on rigid supports — it has no free-floating rigid body that needs a floor.

### Do not proxy light free-floating bodies

Adding the 2.5 cm cube to the proxy mapping does let the cloth collide with it. It also launches
it: measured `z = +10.7 m` at step 40, `+25.7 m` by step 120. The lagged impulse exchange is not
stable for a 15 g free body.

Everything the shipped Franka task proxies is effectively immovable — the hand and fingers of a
fixed-base arm, and static supports. Follow that. To give cloth something to drape over, use a
`static_cuboid`, as `configs/cloth.yaml` does; static shapes belong to the rigid entry and reach
the soft solver through shared outer contacts, and cannot be destabilised.

### Stability limits, measured

The soft body is stable at `k_mu = 3e3` dropped from 0.10 m. Two nearby settings are not, both
dying with `CUDA error: an illegal memory access was encountered`:

| | |
|---|---|
| `k_mu = 8e2`, drop 0.20 m | crashes |
| `k_mu = 1.5e3`, drop 0.20 m | crashes |
| `k_mu = 3e3`, drop 0.10 m | **stable** |

Impact energy matters as much as stiffness — the same stiffness that survives a 0.10 m drop can
fail at 0.20 m. If you soften the material, lower the drop with it.

### A diagnostic that misled me

A flat sheet landing on a flat table reports **0.00 mm** deformation, and that is correct: every
particle contacts at once, so there is no relative motion to measure. It is indistinguishable
from a cloth that is not simulating at all. Give the cloth an obstacle before concluding
anything from a deformation number.

### Full-surface contact is off, deliberately

The shipped Franka cloth task sets `enable_rigid_soft_full_surface_contact=True`. We cannot: it
samples each rigid shape's signed-distance field, the SO-101's collision shapes are meshes with
no SDF, and Isaac Lab's `NewtonShapeCfg` exposes no way to request one — its fields are `margin`,
`gap`, `ke`, `kd`, `mu`. Enabling it raises at finalize, naming all 500-odd shapes.

Contacts are therefore per-particle, and the cloth is only as grippable as its particle spacing
is fine. At 16×16 over a 120 mm sheet that is ~8 mm against a ~36 mm jaw.

## Performance

Cost scales with particles × substeps × solver iterations. A 16×16 cloth is 289 nodes per
environment; the 8192-env rigid training does not survive this. **Tens of environments, not
thousands.**

## Realism — read this before trusting any of it

Rigid sim-to-real on this arm is reasonable. Deformable is much weaker, and it is worth being
blunt:

- **`k_mu`, `k_lambda`, `tri_ke`, `edge_ke` are solver stiffnesses, not material properties.**
  Nothing here has been identified against a real fabric or foam. The cloth numbers are Isaac
  Lab's own shipped values, chosen because they simulate stably.
- **The soft-body values were chosen to be visible, not accurate.** The first attempt used
  `k_mu = 2e4` at density 100 — a 5 cm cube weighing 12.5 g, which is a strain well under a
  percent. It simulated correctly and looked completely rigid. Softening it makes the
  deformation legible; it does not make it truer.
- **Contact friction dominates cloth grasping** and is unmeasured here.
- **A single 36 mm jaw is a poor cloth gripper** regardless of solver quality.

Treat the behaviour as plausible, not predictive. Matching a real material means identifying
these parameters against real footage of that material — see
`docs/source/overview/core-concepts/physical-backends/newton/using-vbd-solver.rst` in the Isaac
Lab tree for a symptom→parameter table to tune against.

## Trying it

```bash
python scripts/run.py --config configs/pick_place_newton.yaml    # rigid, on Newton
python scripts/run.py --config configs/cloth.yaml --steps 0 --viz kit
python scripts/run.py --config configs/soft_body.yaml --steps 0 --viz kit
```
