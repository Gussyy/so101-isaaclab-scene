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


@register_robot("so101_full")
def _so101_full(spec: dict[str, Any]) -> ArticulationCfg:
    """SO-ARM101-FULL: the same 5-DOF arm with a parallel gripper instead of the single jaw.

    Selecting this changes what the task's end-effector *is*, so the builder also repoints the
    ee_frame and the gripper action -- see :func:`simbridge.builder.apply_robot_wiring`. Naming a
    robot in a config cannot silently leave the task addressing a body that no longer exists.
    """
    from so101_scene.tuning import so101_full_cfg

    cfg = so101_full_cfg(
        spec.get("prim_path", "{ENV_REGEX_NS}/Robot"),
        gripper=spec.get("gripper"),
    )
    cfg.init_state = ArticulationCfg.InitialStateCfg(
        pos=_pos(spec),
        rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        joint_pos=dict(spec.get("joint_pos", {})),
    )
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


def _apply_physics_material(prim_path: str, cfg) -> None:
    """Create and bind the friction material, after the colliders exist.

    ``spawn_from_usd`` does this itself -- but *during* the spawn, which is before either of the
    spawners below has defined a single collider. The bind then finds nothing with a physics
    attribute to attach to and logs::

        [Warning] Could not perform 'bind_physics_material' on any prims under:
        '/World/envs/env_0/Object'.

    A warning, not an error. So the asset spawns and simulates with whatever friction the USD
    happened to ship, silently ignoring the config -- the failure mode this repo keeps meeting:
    a setting that looks applied and is not. Both spawners therefore hold the material back out
    of the spawn (``physics_material=None`` on the visual pass) and do it here instead.
    """
    if getattr(cfg, "physics_material", None) is None:
        return
    from isaaclab.sim import bind_physics_material

    rel = cfg.physics_material_path
    material_path = rel if rel.startswith("/") else f"{prim_path}/{rel}"
    cfg.physics_material.func(material_path, cfg.physics_material)
    bind_physics_material(prim_path, material_path)


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

    # physics_material is held back too -- see _apply_physics_material.
    visual = cfg.replace(rigid_props=None, collision_props=None, mass_props=None, physics_material=None)
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
    _apply_physics_material(prim_path, cfg)
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


# ------------------------------------------------------- LeHome household props
#
# https://github.com/lehome-official/lehome publishes a household asset library -- burger parts,
# tableware, towels, appliances -- as a public HuggingFace dataset. The USD loads here. LeHome's
# task code does not: it targets Isaac Lab 2.3 with PhysX particle cloth and fluid, bimanual, in a
# 4.8 GB apartment. docs/LEHOME.md says exactly what did and did not come across.
#
# Two things about these files decide how they have to be spawned, and both were measured rather
# than assumed (`python scripts/measure_lehome.py`):
#
#   1. Only about half of them author physics. The rest are visual meshes, like the YCB props.
#   2. Those that do put the rigid body on a CHILD of the default prim -- /root/Bowl016, not
#      /root. Isaac Lab's `modify_*` helpers address `prim_path` itself, so a `rigid_props` on the
#      UsdFileCfg lands on the wrong prim and raises for having no API there.
#
# One spawner handles both: find the body if there is one, define one if there is not.


def lehome_usd_path(name: str) -> str:
    """Absolute path to a named LeHome asset.

    Raises with the full list on a typo, and with the download command if the library is simply
    not there yet -- it is 1.5 GB and deliberately not committed.
    """
    from pathlib import Path

    from simbridge.objective import LEHOME_CATALOGUE

    if name not in LEHOME_CATALOGUE:
        raise KeyError(f"unknown lehome prop {name!r}. Available: {', '.join(sorted(LEHOME_CATALOGUE))}")
    path = Path(__file__).resolve().parent.parent.parent / "assets" / "lehome" / LEHOME_CATALOGUE[name].path
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. The LeHome assets are not part of this repository -- fetch them:"
            "\n    python scripts/fetch_lehome.py"
        )
    return path.as_posix()


def _lehome_spawner(approximation: str | None):
    """Spawn a LeHome asset and make sure it ends up with exactly one rigid body."""

    def _spawn(prim_path: str, cfg, translation=None, orientation=None, **kwargs):
        # pxr is imported here, not at module scope: importing it before the Kit app exists loads
        # a second USD runtime and every usd_*.dll in the extension cache then fails to load.
        from isaaclab.sim import schemas
        from isaaclab.sim.schemas.schemas_cfg import CollisionPropertiesCfg, MeshCollisionPropertiesCfg
        from isaaclab.sim.spawners.from_files import spawn_from_usd
        from pxr import Usd, UsdGeom, UsdPhysics

        # physics_material is held back too -- see _apply_physics_material.
        visual = cfg.replace(rigid_props=None, collision_props=None, mass_props=None, physics_material=None)
        prim = spawn_from_usd(prim_path, visual, translation, orientation)

        body = next((p for p in Usd.PrimRange(prim) if p.HasAPI(UsdPhysics.RigidBodyAPI)), None)
        if body is None:
            # Visuals only, exactly like the YCB props. Define the schemas after the spawn.
            schemas.define_rigid_body_properties(prim_path, cfg.rigid_props)
            body_path = prim_path
            for child in Usd.PrimRange(prim):
                if child.IsA(UsdGeom.Mesh):
                    schemas.define_collision_properties(str(child.GetPath()), CollisionPropertiesCfg())
                    schemas.define_mesh_collision_properties(
                        str(child.GetPath()),
                        MeshCollisionPropertiesCfg(mesh_approximation_name=approximation or "convexHull"),
                    )
        else:
            body_path = str(body.GetPath())
            schemas.modify_rigid_body_properties(body_path, cfg.rigid_props)
            if approximation:
                # Several of these ask for convexDecomposition, which is what makes a Newton
                # scene build cost minutes rather than seconds. Same dial the robot has.
                for child in Usd.PrimRange(prim):
                    if child.HasAPI(UsdPhysics.MeshCollisionAPI):
                        schemas.modify_mesh_collision_properties(
                            str(child.GetPath()),
                            MeshCollisionPropertiesCfg(mesh_approximation_name=approximation),
                        )

        if cfg.mass_props is not None:
            # define_, not modify_: the body may or may not already carry MassAPI.
            schemas.define_mass_properties(body_path, cfg.mass_props)
        _apply_physics_material(prim_path, cfg)
        return prim

    return _spawn


@register_object("lehome")
def _lehome(spec: dict[str, Any]) -> Any:
    """A named LeHome household prop: ``{type: lehome, name: burger_patty}``.

    ``static: true`` returns scenery rather than a rigid body, which is the only sane way to place
    the appliances -- a refrigerator is 824 mm across and this arm reaches 300 mm.
    """
    from simbridge.objective import LEHOME_CATALOGUE, graspable

    name = spec.get("name")
    if not name:
        raise KeyError("a lehome object needs a 'name' (e.g. name: burger_patty)")
    prop = LEHOME_CATALOGUE[name] if name in LEHOME_CATALOGUE else None
    usd_path = lehome_usd_path(name)          # raises with the list / the fetch command
    size = float(spec.get("scale", 1.0))

    if spec.get("static", False):
        return AssetBaseCfg(
            prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/" + name.title().replace("_", "")),
            init_state=AssetBaseCfg.InitialStateCfg(
                pos=_pos(spec), rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0)))
            ),
            spawn=sim_utils.UsdFileCfg(usd_path=usd_path, scale=(size, size, size)),
        )

    # A warning, not an error: a task may legitimately want something it pushes rather than lifts.
    # But a policy will not learn to lift what the gripper cannot close on, and that failure looks
    # exactly like a broken policy.
    ok, why = graspable(prop.width * size, float(spec.get("jaw", 0.1286)))
    if not ok:
        print(f"  [lehome] {name}: {why} -- the gripper cannot close on this. docs/LEHOME.md")

    mass = spec.get("mass")
    return RigidObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Object"),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=_pos(spec, default=(0.20, 0.0, 0.05)),
            rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        ),
        spawn=sim_utils.UsdFileCfg(
            func=_lehome_spawner(spec.get("collision_approximation")),
            usd_path=usd_path,
            scale=(size, size, size),
            # Collision meshes inside an instanced prototype are instance proxies and cannot be
            # written to, so an approximation override would silently no-op without this.
            make_uninstanceable=bool(spec.get("collision_approximation")),
            rigid_props=RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
                max_depenetration_velocity=5.0,
                disable_gravity=False,
            ),
            # These author household masses -- 0.5 kg for a bowl, 0.2 kg for a bun -- on an arm
            # whose reference cube is 15 g. Left alone by default rather than silently rescaled;
            # `mass:` in the config overrides it.
            mass_props=MassPropertiesCfg(mass=float(mass)) if mass is not None else None,
            physics_material=RigidBodyMaterialCfg(
                static_friction=float(spec.get("static_friction", 1.2)),
                dynamic_friction=float(spec.get("dynamic_friction", 1.0)),
            ),
        ),
    )


# --------------------------------------------------------------- deformables
#
# Cloth and soft bodies run on Newton's VBD solver, so a config declaring one must also select
# a backend that can simulate it: `sim.physics: newton_vbd`. Under PhysX or plain MJWarp the
# asset spawns and then sits inert, which is a confusing way to fail -- so the builder checks.
#
# The numbers below are the ones Isaac Lab's own shipped cloth task uses
# (isaaclab_tasks/core/lift/config/franka_soft/franka_cloth_env_cfg.py), not the class defaults,
# which are an order of magnitude stiffer and were not chosen for a scene like this. They are
# still solver stiffnesses rather than material properties: nothing here has been identified
# against a real fabric. See docs/PHYSICS.md.


@register_object("cloth")
def _cloth(spec: dict[str, Any]) -> Any:
    """A rectangular sheet of cloth. Needs ``sim.physics: newton_vbd``.

    ``resolution`` is the performance dial and the fidelity dial at once: it is the particle grid,
    and it counts ELEMENTS, so a grid of n has n+1 nodes a side: 8x8 is 81 particles per
    environment and 32x32 is 1089. Isaac Lab's shipped cloth task uses
    8x8.
    """
    from isaaclab.assets.deformable_object import DeformableObjectCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

    material = NewtonSurfaceDeformableBodyMaterialCfg(
        density=float(spec.get("density", 1.0)),
        particle_radius=float(spec.get("particle_radius", 0.002)),
        tri_ke=float(spec.get("tri_ke", 5e2)),      # in-plane stretch
        tri_ka=float(spec.get("tri_ka", 5e2)),      # in-plane shear
        tri_kd=float(spec.get("tri_kd", 1e-3)),
        edge_ke=float(spec.get("edge_ke", 0.5)),    # bending; low = drapes, high = card
        edge_kd=float(spec.get("edge_kd", 1e-3)),
    )
    visual = sim_utils.PreviewSurfaceCfg(diffuse_color=tuple(spec["color"])) if "color" in spec else None

    if "usd_path" in spec:
        # `color` is refused rather than ignored. The garment USD carries no material to
        # override, so the PreviewSurface is created and bound to a prim that has nothing to
        # bind it to -- the mesh renders in the default white and the config quietly lies.
        if "color" in spec:
            raise ValueError(
                "a cloth with a usd_path takes its appearance from the asset; `color` would be "
                "created and never bound. Author the material in the USD instead."
            )
        # An arbitrary garment mesh instead of a flat grid. No new spawner is needed:
        # spawn_from_usd already routes `deformable_props` to define_deformable_body_properties,
        # picks "surface" from the material being a SurfaceDeformableBodyMaterialBaseCfg, and
        # finds the single Mesh under the prim itself.
        # A scalar or a triple, so `scale` means the same thing here as on `usd` and `lehome`.
        raw = spec.get("scale", 1.0)
        scale = (float(raw),) * 3 if isinstance(raw, (int, float)) else tuple(float(v) for v in raw)
        spawn = sim_utils.UsdFileCfg(
            usd_path=spec["usd_path"],
            scale=scale,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=visual,
            physics_material=material,
        )
    else:
        spawn = sim_utils.MeshRectangleCfg(
            size=tuple(spec.get("size", (0.12, 0.12))),
            resolution=tuple(int(v) for v in spec.get("resolution", (8, 8))),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=visual or sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.85, 0.10)),
            physics_material=material,
        )

    return DeformableObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Cloth"),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=_pos(spec, default=(0.20, 0.0, 0.02)),
            rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        ),
        spawn=spawn,
    )


@register_object("soft_body")
def _soft_body(spec: dict[str, Any]) -> Any:
    """A solid deformable block. Needs ``sim.physics: newton_vbd``.

    Volume deformables are tetrahedralised at spawn by ``pytetwild``, which is the
    ``tetrahedralization`` extra and is not part of a default Isaac Lab install::

        pip install "pytetwild[all]>=0.3.0,<0.4"

    ``k_mu`` and ``k_lambda`` are the Lame parameters: ``k_mu`` resists shear, ``k_lambda``
    resists volume change. Raising both together makes it stiffer; raising only ``k_lambda``
    makes it more like a water balloon.
    """
    from isaaclab.assets.deformable_object import DeformableObjectCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonDeformableBodyMaterialCfg

    size = tuple(spec.get("size", (0.05, 0.05, 0.05)))
    return DeformableObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/SoftBody"),
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=_pos(spec, default=(0.20, 0.0, size[2] / 2)),
            rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=size,
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=tuple(spec.get("color", (0.25, 0.75, 0.35)))
            ),
            physics_material=NewtonDeformableBodyMaterialCfg(
                density=float(spec.get("density", 100.0)),
                particle_radius=float(spec.get("particle_radius", 0.005)),
                k_mu=float(spec.get("k_mu", 2e4)),
                k_lambda=float(spec.get("k_lambda", 2e4)),
                k_damp=float(spec.get("k_damp", 0.1)),
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


# --------------------------------------------------------------- lights
#
# The task ships one DomeLight at /World/light. Naming `light` in a config replaces it, because
# the builder reuses an existing scene entry's prim path -- no new registry kind needed.


@register_object("light")
def _light(spec: dict[str, Any]) -> AssetBaseCfg:
    """Replace the scene light: ``{type: light, kind: distant}``.

    A dome light is an environment light -- every camera ray that escapes the geometry samples
    it. A distant light is a single direction and costs almost nothing. Which one is faster
    depends on the renderer and the scene, so measure rather than assume; docs/PHYSICS.md has
    the numbers for this scene.

    ``intensity`` is NOT comparable across kinds: a dome's is a radiance over the whole sphere,
    a distant light's is an irradiance from one direction. Changing ``kind`` without re-tuning
    ``intensity`` gives a black or blown-out image, which reads as a broken light rather than a
    unit mismatch.
    """
    # The builder gives an entry the prim path of the scene entry it is overriding, and the task
    # calls its light `light`. Under any other key it resolves to {ENV_REGEX_NS}/<key>, which
    # spawns an extra per-environment light and leaves the task's dome exactly where it was --
    # brighter scene, no error, and nothing in the config saying so.
    prim_path = spec.get("prim_path", "/World/light")
    if "ENV_REGEX_NS" in prim_path:
        raise ValueError(
            f"a light entry has to be named 'light' to replace the task's own; as {prim_path!r} "
            "it would be an additional light rather than a replacement"
        )
    kind = spec.get("kind", "dome")
    color = tuple(spec.get("color", (0.75, 0.75, 0.75)))
    if kind == "dome":
        spawn = sim_utils.DomeLightCfg(color=color, intensity=float(spec.get("intensity", 3000.0)))
    elif kind == "distant":
        spawn = sim_utils.DistantLightCfg(
            color=color,
            intensity=float(spec.get("intensity", 2000.0)),
            angle=float(spec.get("angle", 0.53)),   # degrees; the sun's is 0.53
        )
    elif kind == "sphere":
        spawn = sim_utils.SphereLightCfg(
            color=color,
            intensity=float(spec.get("intensity", 30000.0)),
            radius=float(spec.get("radius", 0.05)),
        )
    else:
        raise KeyError(f"unknown light kind {kind!r}; expected one of dome, distant, sphere")

    return AssetBaseCfg(
        prim_path=prim_path,
        init_state=AssetBaseCfg.InitialStateCfg(
            # A distant light shines down -Z at identity, which is what a config usually wants.
            pos=_pos(spec, default=(0.0, 0.0, 2.0)),
            rot=tuple(spec.get("rot", (0.0, 0.0, 0.0, 1.0))),
        ),
        spawn=spawn,
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
    """TiledCamera: the batched renderer. Cheap on a rigid PhysX scene -- 45.6 steps/s at
    640x480, 1 env. Not cheap on a Newton deformable scene, where the same camera costs 13x
    and nothing exposed here changes that: light type, RTX quality, resolution, particle count
    and even swapping to Newton's own Warp rasteriser were each measured and each changed the
    rate by nothing. docs/PHYSICS.md has the table and the mechanism.

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
    cfg = TiledCameraCfg(
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
    return cfg


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
