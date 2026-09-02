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

### A real garment, not a rectangle

`configs/lehome_bedroom_shirt.yaml` simulates LeHome's shirt mesh instead of a flat grid:

```yaml
scene:
  objects:
    shirt:
      type: cloth
      usd_path: "assets/garment/shirt.usd"
      scale: 0.25
      particle_radius: 0.0015
```

No new spawner was needed. `spawn_from_usd` already routes `deformable_props` to
`define_deformable_body_properties`, picks `"surface"` from the material being a
`SurfaceDeformableBodyMaterialBaseCfg`, and finds the single Mesh under the prim itself. The
`cloth` factory just hands it a `UsdFileCfg` instead of a `MeshRectangleCfg`.

Getting a *usable* garment took three measured findings.

**The source mesh diverges at step 7.** LeHome's shirt is 14,746 vertices / 28,726 triangles.
At the scale this arm can reach, its triangles are 2.4 mm across against a 2 mm particle radius,
so the particles start out overlapping and the solver explodes immediately. For reference, Isaac
Lab's own cloth task uses an 8x8 grid — 81 particles (`resolution` counts elements, so a
grid of n has n+1 nodes per side).

**How you reduce it matters more than how far.** Two ways to get to ~3000 vertices:

| | edge lengths at scale | spread |
|---|---|---|
| quadric decimation (`fast_simplification`) | 0.21 .. 35.5 mm | **170x** |
| voxel clustering (`open3d`) | 1.04 .. 11.07 mm | **11x** |

Quadric decimation optimises for silhouette — right for rendering, wrong here. A self-contact
radius has to sit *below the smallest edge* or every particle collides with its own neighbour, and
at 0.21 mm there is no usable radius. The quadric mesh simulates fine with self-collision off and
cannot have it on at all. That is why `scripts/make_garment.py` uses clustering and why its knob
is **particle spacing**, not vertex count.

**Self-collision needed one non-obvious field.** Even on the uniform mesh, turning
`particle_enable_self_contact` on tore the shirt apart by step 60 — z drifting to −0.96 m, peak
deformation 613 mm. The fix is `particle_rest_shape_contact_exclusion_radius`, which Isaac Lab
defaults to **0.0**: particles that were 6 mm apart in the rest shape were colliding with each
other. Set to 8 mm — just above the mesh spacing — and it is stable.

Measured, 1 environment, 150 steps, parallel gripper, self-collision on:

| spacing | particles | steps/s | stable |
|---|---|---|---|
| source mesh | 14,746 | — | **diverges at step 7** |
| 4 mm | 5,314 | 20.8 | yes |
| **6 mm** | **2,572** | **24.8** | yes — the shipped `shirt.usd` |
| 9 mm | 1,207 | 26.6 | yes |

Cost is sublinear in particle count here — 4x the particles for 22% of the rate — because the
rigid solve and the coupling dominate at these sizes. Paying for the finer mesh is cheap until
it isn't: 6000 particles drops to 23.8 steps/s.

### The gripper cannot pick the shirt up

Worth stating with numbers, because "cloth is grippable" is the easy thing to assume from a
render. A scripted grasp (`source: keyframes`) pans the gripper over the settled shirt, descends,
closes, and lifts 100 mm:

```
step=80   cloth_top= 12.6mm  ee=(+0.251,-0.116,+0.060)  near_jaw=564
step=120  cloth_top= 12.5mm  ee=(+0.241,-0.111,+0.026)  near_jaw=519   <- closed
step=220  cloth_top= 12.4mm  ee=(+0.257,-0.120,+0.125)  near_jaw=566   <- lifted 100 mm
step=319  cloth_top= 12.4mm
```

564 particles between the jaws, and the cloth does not move by a tenth of a millimetre. The
reason is already in this page: contacts are per-particle because
`enable_rigid_soft_full_surface_contact` cannot be enabled without SDFs, so there is no *surface*
between the fingers to pinch — only particles 6 mm apart, and a finger passes between them.

Two other things this turned up, both stable-vs-NaN rather than subtle:

- **A garment spawned on top of the robot NaNs the articulation on step 1.** Particles starting
  inside a rigid body is not something the proxy recovers from. The shipped config spawns the
  shirt beside the arm for that reason, and it then runs 320 steps clean.
- **Pressing the gripper hard into the cloth also diverges**, at step 117 of a descent driving
  the end-effector to z = 0.019. Stopping at z = 0.025 is stable. So there is a working envelope
  for touching cloth with this arm; it just does not include holding it.

### The coupler follows the robot now

`newton_vbd`'s proxy named `Robot/gripper` and `Robot/moving_jaw_so101_v1` — the single-jaw
robot's bodies, flat under `/Robot` because that asset is flat. Selecting `so101_full` refused to
build at all:

```
ValueError: CouplerCfg proxy 'rigid'->'soft': body-label regex
'/World/envs/env_[^/]+/Robot/gripper' matched no Newton bodies.
```

There is no body called `gripper` on that robot, and it nests its bodies as the kinematic chain,
so a one-level path would not have matched even if there were. `apply_cloth_proxy_bodies` now
rewrites the list from `SO101_FULL_CLOTH_BODIES` whenever the config names that robot — the same
pattern as the ee_frame and gripper action.

### Rendering a deformable scene costs 13x, and no render knob fixes it

`{type: light, kind: dome|distant|sphere}` replaces the task's `DomeLight`. **The config entry
has to be named `light`** -- under any other key the builder gives it `{ENV_REGEX_NS}/<key>`, so
it would add a per-environment light and leave the task's dome untouched. That is refused rather
than allowed to happen quietly. `intensity` is not
comparable across kinds — a dome's is a radiance over the whole sphere, a distant light's an
irradiance from one direction — so changing `kind` without retuning gives a black or blown-out
image, which reads as a broken light rather than a unit mismatch.

It will not make a deformable scene faster. Measured on the shirt config, 1 env, 200 steps:

| | steps/s |
|---|---|
| shirt, **headless** | **24.8** |
| shirt + 640x480 camera, dome light | 1.9 |
| shirt + camera, distant light | 1.7 |
| shirt + camera, sphere light | 1.9 |
| shirt + camera, `samples_per_pixel: 1`, no GI/reflections/AO | 1.9 |
| shirt + camera at **160x120** (16x fewer pixels) | 1.9 |
| shirt + camera, 9 mm mesh (1207 particles, not 2572) | 1.9 |

Everything lands on 1.9. Light type, RTX quality, resolution and particle count are all not the
lever. Two more runs say what is:

| | steps/s |
|---|---|
| rigid scene + the same camera, **PhysX** | **45.6** |
| Newton VBD + the same camera, **no deformable at all** | **4.2** |
| shirt + camera, Newton's own **Warp rasteriser** instead of RTX | 1.9 |

So the cost is the **backend**, not the cloth and not the renderer's workload: Newton with a
camera is ~11x slower than PhysX with the same camera before a single particle exists, and the
cloth then takes 4.2 down to 1.9.

Swapping the renderer does not help either, and that is the useful part. Setting
`TiledCameraCfg.renderer_cfg = NewtonWarpRendererCfg()` really does take -- verified by reading
back `type(camera._renderer)` at runtime, which is `NewtonWarpRenderer` rather than
`IsaacRtxRenderer` -- and the rate is unchanged at 1.9. So the cost is not in the renderer at
all; it is upstream of whoever consumes the frame.

That points at `SimulationContext.render()`, which calls `physics_manager.pre_render()` on every
render regardless of renderer. In `isaaclab_newton`'s manager that is a USD/Fabric writeback of
transforms and particles ending in a `wp.synchronize_device()` -- a full device stall, explicitly
excluded from the CUDA graph. Enabling cameras is also what arms that path in the first place:
the manager keeps a physics-only clone when no camera is requested, which is exactly the
headless case that runs at 24.8. That is a reading of the Isaac Lab source rather than something
measured directly, but it is consistent with every number above: O(1) in resolution, in particle
count, and in renderer.

Nothing exposed through this repo's config touches it.

Practical consequence: **collect and train headless, and render clips as a separate pass** —
which is what `scripts/capture_clip.py` already does. A camera left on during a deformable
rollout costs an order of magnitude and buys nothing you cannot capture afterwards.

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
python scripts/run.py --config configs/lehome_bedroom_shirt.yaml --steps 0 --viz kit
```

Enabling self-collision changed the flat sheet not at all, which is the check that it was turned
on for the garment and not at the rectangle's expense: `configs/cloth.yaml` still peaks at
**43.6 mm** of deformation draped over its post, against the 43.5 mm measured before.
