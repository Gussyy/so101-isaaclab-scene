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

What does not work
------------------
Overriding the robot's collision approximation to ``convexHull`` via
``CollisionPropertiesCfg(mesh_collision_property=ConvexHullPropertiesCfg())`` on the
``UsdFileCfg`` spawn does **not** avoid the decomposition. ``modify_collision_properties``
only touches prims that already have the collision schema applied, and the SO-101 asset
authors none -- the override silently no-ops. Forcing it would mean authoring
``UsdPhysics.MeshCollisionAPI`` on each collision mesh prim directly. Left undone rather
than half-done; the backend choice above is the effective lever.
"""

from isaaclab.assets import ArticulationCfg

from isaaclab_assets.robots.so101 import SO101_CFG


def so101_cfg(prim_path: str = "{ENV_REGEX_NS}/Robot") -> ArticulationCfg:
    """SO-101 articulation config at ``prim_path``.

    Single definition shared by every task here, so robot-level changes land in one place.
    """
    return SO101_CFG.replace(prim_path=prim_path)
