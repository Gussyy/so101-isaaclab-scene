# SPDX-License-Identifier: BSD-3-Clause
"""The default vocabulary a config file can name.

Each factory takes the YAML sub-dict for that entry and returns an Isaac Lab cfg object. Keeping
the mapping explicit (rather than reflecting over Isaac Lab) means a config error is reported as
"unknown object 'cubeoid'" at build time, not as an obscure USD failure ten seconds into a
simulation.
"""

from __future__ import annotations

from typing import Any

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.sim import RigidBodyMaterialCfg
from isaaclab.sim.schemas.schemas_cfg import MassPropertiesCfg, RigidBodyPropertiesCfg

from simbridge.objective import PROP_CATALOGUE
from simbridge.registry import register_camera, register_object, register_robot


def _pos(spec: dict[str, Any], key: str = "pos", default=(0.0, 0.0, 0.0)) -> tuple:
    return tuple(spec.get(key, default))


# ---------------------------------------------------------------- robots

@register_robot("so101")
def _so101(spec: dict[str, Any]) -> ArticulationCfg:
    """TheRobotStudio SO-ARM101, 5-DOF + single-jaw gripper."""
    from so101_scene.tuning import so101_cfg

    # collision_approximation: "convexHull" skips most of Newton's CoACD startup cost, at the
    # price of grasp fidelity everywhere except the pinch bodies. See so101_scene.tuning.
    cfg = so101_cfg(
        spec.get("prim_path", "{ENV_REGEX_NS}/Robot"),
        collision_approximation=spec.get("collision_approximation"),
    )
    init = ArticulationCfg.InitialStateCfg(
        pos=_pos(spec),
        # Isaac Lab 3.0 quaternions are (x, y, z, w); 2.x was (w, x, y, z).
        rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        joint_pos=dict(spec.get("joint_pos", {})),
    )
    cfg.init_state = init
    return cfg


# --------------------------------------------------------------- objects

@register_object("cuboid")
def _cuboid(spec: dict[str, Any]) -> RigidObjectCfg:
    size = tuple(spec.get("size", (0.025, 0.025, 0.025)))
    return RigidObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Object"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_pos(spec, default=(0.2, 0.0, size[2] / 2)),
            rot=tuple(spec.get("rot", (1.0, 0.0, 0.0, 0.0))),
        ),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.get("color", (0.9, 0.25, 0.15)))),
            physics_material=RigidBodyMaterialCfg(
                static_friction=float(spec.get("static_friction", 1.2)),
                dynamic_friction=float(spec.get("dynamic_friction", 1.0)),
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=float(spec.get("mass", 0.015))),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
        ),
    )


@register_object("static_cuboid")
def _static_cuboid(spec: dict[str, Any]) -> AssetBaseCfg:
    """Non-physical prop (pedestal, wall). Collides but never moves."""
    size = tuple(spec.get("size", (0.16, 0.16, 0.1)))
    return AssetBaseCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Prop"),
        spawn=sim_utils.CuboidCfg(
            size=size,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec.get("color", (0.25, 0.25, 0.28)))),
            collision_props=sim_utils.CollisionPropertiesCfg(),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=_pos(spec, default=(0.0, 0.0, size[2] / 2))),
    )


def ycb_usd_path(name: str) -> str:
    """Asset URL for a named prop. Raises with the full list on a typo."""
    from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

    if name not in PROP_CATALOGUE:
        raise KeyError(
            f"unknown ycb prop {name!r}. Available: {', '.join(sorted(PROP_CATALOGUE))}. "
            "Isaac Sim's YCB set has no apple and no orange; docs/OBJECTS.md lists which of "
            "these this arm can actually close its jaw on."
        )
    return f"{ISAAC_NUCLEUS_DIR}/Props/YCB/Axis_Aligned/{PROP_CATALOGUE[name][0]}.usd"


def _spawn_prop(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
    """Spawn a YCB prop and give it the physics it does not ship with.

    The Axis_Aligned YCB files are visual meshes: a single ``Mesh`` with a material binding, no
    ``RigidBodyAPI``, no collider, no mass. Isaac Lab's ``UsdFileCfg`` only *modifies* physics
    that already exists, so pointing a ``RigidObjectCfg`` straight at one of these fails with
    "Could not perform 'modify_rigid_body_properties'" and then "Expected 1 prims, found 0".

    So: spawn the visuals, then define the schemas -- rigid body and mass on the root, a convex
    hull collider on each mesh underneath.
    """
    # pxr is imported here, not at module scope: importing it before the Kit app exists loads a
    # second USD runtime and every usd_*.dll in the extension cache then fails to load.
    from isaaclab.sim import schemas
    from isaaclab.sim.schemas.schemas_cfg import (
        CollisionPropertiesCfg,
        MassPropertiesCfg,
        MeshCollisionPropertiesCfg,
    )
    from isaaclab.sim.spawners.from_files import spawn_from_usd
    from pxr import Usd, UsdGeom

    visual = cfg.replace(rigid_props=None, collision_props=None, mass_props=None)
    prim = spawn_from_usd(prim_path, visual, translation, orientation)

    schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
    schemas.define_mass_properties(prim_path, cfg.mass_props or MassPropertiesCfg(mass=0.05))

    # Convex hull, not convex decomposition: decomposition is what made a Newton scene build cost
    # ~900 CPU-seconds here, and these props are single convex-ish objects anyway.
    for child in Usd.PrimRange(prim):
        if child.IsA(UsdGeom.Mesh):
            schemas.define_collision_properties(str(child.GetPath()), CollisionPropertiesCfg())
            schemas.define_mesh_collision_properties(
                str(child.GetPath()), MeshCollisionPropertiesCfg(mesh_approximation_name="convexHull")
            )
    return prim


@register_object("ycb")
def _ycb(spec: dict[str, Any]) -> RigidObjectCfg:
    """A named prop from Isaac Sim's YCB set: ``{type: ycb, name: banana}``.

    Streamed from the Isaac asset server on first use, so the first spawn of each is slow and
    needs a network connection. Not every prop is pickable by this arm -- see docs/OBJECTS.md,
    which is generated by measuring them rather than by reading the YCB spec sheet.
    """
    name = spec.get("name")
    if not name:
        raise KeyError("a ycb object needs a 'name' (e.g. name: banana)")
    size = float(spec.get("scale", 1.0))
    return RigidObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Object"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_pos(spec, default=(0.2, 0.0, 0.05)),
            rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        ),
        spawn=sim_utils.UsdFileCfg(
            func=_spawn_prop,
            usd_path=ycb_usd_path(name),
            scale=(size, size, size),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # These assets author no mass, so PhysX would derive one from density -- about 200 g
            # for the gelatin box, on an arm whose reference cube is 15 g. Pick one instead, and
            # let a config say otherwise.
            mass_props=MassPropertiesCfg(mass=float(spec.get("mass", 0.03))),
            # The friction the shipped materials carry is tuned for a parallel jaw. A single-jaw
            # pinch needs more, exactly as the task's own cube does.
            physics_material=RigidBodyMaterialCfg(
                static_friction=float(spec.get("static_friction", 1.2)),
                dynamic_friction=float(spec.get("dynamic_friction", 1.0)),
            ),
        ),
    )


@register_object("usd")
def _usd(spec: dict[str, Any]) -> RigidObjectCfg:
    """Any USD asset by path or URL."""
    return RigidObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Object"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_pos(spec), rot=tuple(spec.get("rot", (1.0, 0.0, 0.0, 0.0)))),
        spawn=sim_utils.UsdFileCfg(usd_path=spec["usd_path"], scale=tuple(spec.get("scale", (1.0, 1.0, 1.0)))),
    )


# --------------------------------------------------------------- cameras

def look_at_quat(pos, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """Orientation that points a camera at ``target``, as (x, y, z, w) in the *world* convention.

    Aiming a camera by writing a quaternion by hand is not a thing anyone can do, and getting it
    wrong is not obvious from the config -- it is obvious from a rendered frame of the floor,
    which is two minutes of Isaac Sim later. ``look_at`` states the intent instead.

    World convention: the camera looks along its +X axis with +Z up, which is the one of Isaac
    Lab's three that reads like a position in the scene rather than a graphics convention.
    """
    import math

    f = [t - p for t, p in zip(target, pos)]
    n = math.sqrt(sum(c * c for c in f))
    if n < 1e-9:
        raise ValueError(f"camera look_at target {tuple(target)} coincides with its position")
    x_axis = [c / n for c in f]

    # Re-orthogonalise `up` against the view direction. A camera looking straight down has `up`
    # parallel to it, which leaves no valid frame -- fall back to +X so the result is still sane.
    d = sum(u * a for u, a in zip(up, x_axis))
    z_axis = [u - d * a for u, a in zip(up, x_axis)]
    n = math.sqrt(sum(c * c for c in z_axis))
    if n < 1e-6:
        z_axis = [1.0 - x_axis[0] * x_axis[0], -x_axis[0] * x_axis[1], -x_axis[0] * x_axis[2]]
        n = math.sqrt(sum(c * c for c in z_axis))
    z_axis = [c / n for c in z_axis]
    y_axis = [
        z_axis[1] * x_axis[2] - z_axis[2] * x_axis[1],
        z_axis[2] * x_axis[0] - z_axis[0] * x_axis[2],
        z_axis[0] * x_axis[1] - z_axis[1] * x_axis[0],
    ]

    # Columns are the camera axes expressed in world coordinates.
    m = [[x_axis[i], y_axis[i], z_axis[i]] for i in range(3)]
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    return (x, y, z, w)          # Isaac Lab 3.0 order; 2.x was (w, x, y, z)


@register_camera("tiled")
def _tiled(spec: dict[str, Any]) -> TiledCameraCfg:
    """TiledCamera: the batched renderer. Measured ~27k frames/s at 128px on a 4070 Ti,
    with cost flat per step regardless of env count -- see docs/bench_camera.txt.

    Aim it with ``look_at: [x, y, z]`` (env-relative), or with an explicit ``rot`` quaternion.
    """
    res = spec.get("resolution", [128, 128])
    pos = _pos(spec, default=(0.55, 0.0, 0.35))
    if "look_at" in spec:
        if "rot" in spec:
            raise ValueError("camera takes look_at or rot, not both")
        rot = look_at_quat(pos, tuple(spec["look_at"]), tuple(spec.get("up", (0.0, 0.0, 1.0))))
        convention = "world"
    else:
        rot = tuple(spec.get("rot", (0.0, 0.259, 0.0, 0.966)))
        convention = spec.get("convention", "opengl")
    return TiledCameraCfg(
        prim_path=spec["prim_path"],
        offset=TiledCameraCfg.OffsetCfg(pos=pos, rot=rot, convention=convention),
        data_types=list(spec.get("data_types", ["rgb"])),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(spec.get("focal_length", 18.0)),
            clipping_range=tuple(spec.get("clipping_range", (0.01, 3.0))),
        ),
        width=int(res[0]),
        height=int(res[1]),
    )


def demo() -> None:
    """Self-check for look_at_quat: rotating the world axes must reproduce the view direction."""
    import math

    def rotate(q, v):
        x, y, z, w = q
        t = [2 * (y * v[2] - z * v[1]), 2 * (z * v[0] - x * v[2]), 2 * (x * v[1] - y * v[0])]
        return [v[i] + w * t[i] + (y * t[2] - z * t[1], z * t[0] - x * t[2], x * t[1] - y * t[0])[i]
                for i in range(3)]

    for pos, target in (((0.5, 0.0, 0.3), (0.0, 0.0, 0.0)),
                        ((0.6, -0.5, 0.4), (0.18, 0.0, 0.08)),
                        ((0.0, 0.0, 1.0), (0.0, 0.0, 0.0)),      # straight down: up is degenerate
                        ((-0.3, 0.4, 0.2), (0.2, -0.1, 0.05))):
        q = look_at_quat(pos, target)
        assert abs(math.sqrt(sum(c * c for c in q)) - 1.0) < 1e-6, f"not unit: {q}"

        # The camera's +X axis, rotated into the world, must point from pos to target.
        fwd = rotate(q, (1.0, 0.0, 0.0))
        want = [t - p for t, p in zip(target, pos)]
        n = math.sqrt(sum(c * c for c in want))
        want = [c / n for c in want]
        assert all(abs(a - b) < 1e-6 for a, b in zip(fwd, want)), f"{pos}->{target}: {fwd} != {want}"

        # +Z must stay as close to world up as the view allows: never below the horizon.
        assert rotate(q, (0.0, 0.0, 1.0))[2] > -1e-6, f"{pos}->{target}: camera is upside down"

    try:
        look_at_quat((0.1, 0.1, 0.1), (0.1, 0.1, 0.1))
    except ValueError:
        pass
    else:
        raise AssertionError("a target equal to the position should be rejected")

    print("scene builtins OK: look_at aims the camera, stays upright, rejects a degenerate target")


if __name__ == "__main__":
    # Run as a path, not `-m`: importing the package registers these factories once already,
    # and `-m` would execute this file a second time into a registry that rejects duplicates.
    #     python simbridge/scene/builtins.py
    demo()
