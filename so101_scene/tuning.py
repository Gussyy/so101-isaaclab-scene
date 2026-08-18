# SPDX-License-Identifier: BSD-3-Clause
"""Shared SO-ARM101 robot config, and the measured notes behind the run settings.

Measured on this machine (RTX 4070 Ti 12 GB, i5-13600KF 20 threads, Isaac Sim 6.0.1,
Isaac Lab 3.0 ``develop``). Numbers are from ``scripts/bench_envs.py``.

Backend startup cost
--------------------
``physics=newton_mjwarp`` pays a large one-time CPU cost before training starts: Newton
runs CoACD convex decomposition over the SO-101 collision meshes at scene-build time. A
256-env run burned **900 CPU-seconds in CoACD and never reached iteration 0**. The default
PhysX backend does zero CoACD on the same scene and reaches iteration 0 in well under a
minute.

That is why ``PickPlacePhysicsCfg.default`` is PhysX -- the same choice Isaac Lab's own
SO-101 stack task makes. Newton still works and is selectable; budget the startup.

Note this is a *startup* cost, not a per-step one, so it amortises over a long run and is
mostly a problem for short iteration cycles.

Why the decomposition happens, and what to do about it
------------------------------------------------------
An earlier note here blamed the asset for authoring no collision schema. That was wrong.
Inspecting the composed stage shows **17 collision meshes, every one of them already
carrying** ``UsdPhysics.MeshCollisionAPI`` with ``approximation = convexDecomposition``.
CoACD runs because the asset asks it to. And they are not small::

    wrist/collisions/wrist_roll_pitch_so101_v2      161,982 points
    lower_arm/collisions/under_arm_so101_v1         118,548
    base/collisions/base_motor_holder_so101_v1      112,620
    gripper/collisions/wrist_roll_follower_so101_v1  86,388
    upper_arm/collisions/upper_arm_so101_v1          78,204
    ... plus sts3215_03a_v1 (57,240) four times over

The real reason the ``CollisionPropertiesCfg`` override no-opped is that those prims are
**instance proxies**. They live inside an instanced prototype, and a proxy cannot be
written to -- so ``modify_collision_properties`` had nothing it was allowed to change.
Spawning with ``make_uninstanceable=True`` makes them writable, which is what
:func:`so101_cfg` does when asked for a cheaper approximation.

Measured on this machine, 4 envs, 30 steps, from process start to exit:

    physx                                 9 s
    newton_mjwarp, asset as authored    285 s     (1779 lines of CoACD log)
    newton_mjwarp, convexHull            24 s     (234 lines -- just the two pinch bodies)

So the override recovers most of the gap: 12x faster to start, and Newton lands within 3x of
PhysX rather than 32x.

``collision_approximation="convexHull"`` trades grasp fidelity for startup time, so it is
opt-in and it deliberately **keeps decomposition on the two bodies that form the pinch**
(``moving_jaw`` and ``wrist_roll_follower``). A convex hull fills in the notch the jaw
grips with, which is precisely the geometry this task depends on.

Not measured: whether hulling the other 15 bodies changes what the arm can do. It should not
-- they are structural links and servo housings, and the arm is not supposed to be touching
anything with them -- but "should not" is not "does not", and a self-collision that used to
clear now may not. Re-check success rate before trusting a policy trained this way.
"""

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.so101 import SO101_CFG


# The two bodies whose concavity the grasp depends on. A convex hull across these fills in
# the notch the jaw closes into, which would quietly make grasping worse.
GRASP_BODIES = ("moving_jaw", "wrist_roll_follower")


def _override_approximation(approximation: str, keep: tuple[str, ...]):
    """Spawner that rewrites collision-mesh approximations after the asset is on the stage.

    Wraps ``spawn_from_usd`` rather than replacing it. The meshes are instance proxies until
    ``make_uninstanceable=True`` has been applied, so this must run after the spawn, not via a
    ``CollisionPropertiesCfg`` on the way in.
    """

    def spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        from pxr import Usd, UsdPhysics

        prim = spawn_from_usd(prim_path, cfg, translation, orientation)
        changed = kept = 0
        for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
            api = UsdPhysics.MeshCollisionAPI(child)
            if not api:
                continue
            if any(k in str(child.GetPath()) for k in keep):
                kept += 1
                continue
            attr = api.GetApproximationAttr()
            if attr:
                attr.Set(approximation)
                changed += 1
        print(f"[so101] collision approximation -> {approximation} on {changed} meshes "
              f"({kept} left as authored: {', '.join(keep)})")
        return prim

    return spawn


def so101_cfg(
    prim_path: str = "{ENV_REGEX_NS}/Robot",
    collision_approximation: str | None = None,
    keep_decomposition: tuple[str, ...] = GRASP_BODIES,
) -> ArticulationCfg:
    """SO-101 articulation config at ``prim_path``.

    Single definition shared by every task here, so robot-level changes land in one place.

    Args:
        prim_path: Where to spawn it.
        collision_approximation: Override the asset's ``convexDecomposition`` -- pass
            ``"convexHull"`` to skip most of Newton's CoACD startup cost. ``None`` leaves the
            asset as authored, which is the right default for PhysX (it does no decomposition).
        keep_decomposition: Substrings of prim paths to leave alone. Defaults to the bodies that
            form the pinch, because a hull across those changes what the gripper can hold.
    """
    cfg = SO101_CFG.replace(prim_path=prim_path)
    if collision_approximation:
        cfg.spawn = cfg.spawn.replace(
            # Without this the collision meshes are instance proxies and cannot be written to.
            make_uninstanceable=True,
            func=_override_approximation(collision_approximation, tuple(keep_decomposition)),
        )
    return cfg
