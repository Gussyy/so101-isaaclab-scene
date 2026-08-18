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
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg

from simbridge.registry import register_camera, register_object, register_robot


def _pos(spec: dict[str, Any], key: str = "pos", default=(0.0, 0.0, 0.0)) -> tuple:
    return tuple(spec.get(key, default))


# ---------------------------------------------------------------- robots

@register_robot("so101")
def _so101(spec: dict[str, Any]) -> ArticulationCfg:
    """TheRobotStudio SO-ARM101, 5-DOF + single-jaw gripper."""
    from so101_scene.tuning import so101_cfg

    cfg = so101_cfg(spec.get("prim_path", "{ENV_REGEX_NS}/Robot"))
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


@register_object("usd")
def _usd(spec: dict[str, Any]) -> RigidObjectCfg:
    """Any USD asset by path or URL."""
    return RigidObjectCfg(
        prim_path=spec.get("prim_path", "{ENV_REGEX_NS}/Object"),
        init_state=RigidObjectCfg.InitialStateCfg(pos=_pos(spec), rot=tuple(spec.get("rot", (1.0, 0.0, 0.0, 0.0)))),
        spawn=sim_utils.UsdFileCfg(usd_path=spec["usd_path"], scale=tuple(spec.get("scale", (1.0, 1.0, 1.0)))),
    )


# --------------------------------------------------------------- cameras

@register_camera("tiled")
def _tiled(spec: dict[str, Any]) -> TiledCameraCfg:
    """TiledCamera: the batched renderer. Measured ~27k frames/s at 128px on a 4070 Ti,
    with cost flat per step regardless of env count -- see docs/bench_camera.txt."""
    res = spec.get("resolution", [128, 128])
    return TiledCameraCfg(
        prim_path=spec["prim_path"],
        offset=TiledCameraCfg.OffsetCfg(
            pos=_pos(spec, default=(0.55, 0.0, 0.35)),
            rot=tuple(spec.get("rot", (0.0, 0.259, 0.0, 0.966))),
            convention=spec.get("convention", "opengl"),
        ),
        data_types=list(spec.get("data_types", ["rgb"])),
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=float(spec.get("focal_length", 18.0)),
            clipping_range=tuple(spec.get("clipping_range", (0.01, 3.0))),
        ),
        width=int(res[0]),
        height=int(res[1]),
    )
