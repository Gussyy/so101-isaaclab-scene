# SPDX-License-Identifier: BSD-3-Clause
"""YAML -> a runnable Isaac Lab environment.

A config names a registered task as its base, then declares what to add or change: extra props,
cameras, robot start pose, episode length, and which driver supplies actions. Reward and
termination logic stays in Python, because those are code and pretending otherwise produces a
config language nobody can debug.

Validation is strict and up front. An unknown key is an error rather than a silent no-op --
a typo'd ``resolutoin`` that leaves the camera at its default is exactly the class of bug that
costs an afternoon.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from simbridge import scene  # noqa: F401  (registers builtins)
from simbridge.registry import CAMERAS, OBJECTS, ROBOTS, TASKS, lookup

_TOP_LEVEL = {"task", "scene", "sim", "control", "render", "objective", "meta"}
_SCENE_KEYS = {"num_envs", "env_spacing", "robot", "objects", "cameras"}
_SIM_KEYS = {"episode_length_s", "dt", "decimation", "physics", "device"}

# Named RTX settings, plus `carb_settings` for anything not surfaced here.
# DLSS upscales from a lower internal resolution; DLSS-G (`frame_generation`) interpolates
# extra frames. Frame generation is for display smoothness -- interpolated frames are not
# rendered from scene state, so they must not be used as policy observations.
_RENDER_KEYS = {
    "antialiasing", "dlss_mode", "frame_generation", "dl_denoiser",
    "reflections", "global_illumination", "translucency", "direct_lighting",
    "shadows", "ambient_occlusion", "samples_per_pixel", "max_bounces",
    "carb_settings",
}
_OBJECTIVE_KEYS = {"pickable", "sequence", "spawn"}
_AA_MODES = {"Off", "FXAA", "DLSS", "TAA", "DLAA"}
_DLSS_MODES = {0: "performance", 1: "balanced", 2: "quality", 3: "auto"}


def _reject_unknown(d: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"unknown key(s) {sorted(extra)} in '{where}'; allowed: {sorted(allowed)}")


def check_pickable(cfg: dict[str, Any], names: list[str]) -> list[str]:
    """Warn about objects the SO-101's single jaw cannot close on.

    Same reasoning as the reach check: a config that names ``pick[mug]`` is not malformed, it is
    impossible, and the way that failure shows up is a flat reward curve after an hour of
    training. The width is known without starting Isaac Sim -- a ``cuboid`` states its size, and
    a ``ycb`` prop has a measured one in the catalogue -- so it costs nothing to say so here.

    Warns rather than raises. The jaw figure is soft (see :data:`JAW_WIDTH`), and a task may
    legitimately want an object it pushes rather than lifts.
    """
    from simbridge.objective import PROP_CATALOGUE, graspable

    objects = (cfg.get("scene") or {}).get("objects") or {}
    out = []
    for name in names:
        spec = objects.get(name)
        if not isinstance(spec, dict):
            continue          # not declared here; the task's own object, whose size we know works
        kind = spec.get("type")
        if kind == "ycb":
            entry = PROP_CATALOGUE.get(spec.get("name", ""))
            width = entry[1] * float(spec.get("scale", 1.0)) if entry else None
        elif kind == "cuboid":
            width = min(float(c) for c in spec.get("size", (0.025, 0.025, 0.025)))
        else:
            width = None      # a bare `usd` asset: nothing here knows how big it is
        if width is None:
            continue
        ok, why = graspable(width)
        if not ok:
            out.append(f"pick[{name}] is {why}; the arm cannot close on it (see docs/OBJECTS.md)")
        elif "tight" in why:
            out.append(f"pick[{name}] is {why}")
    return out


def load_config(path: str | Path) -> dict[str, Any]:
    """Read and structurally validate a scene/task YAML."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}
    if not isinstance(cfg, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    _reject_unknown(cfg, _TOP_LEVEL, str(path))
    if "task" not in cfg:
        raise ValueError(f"{path}: 'task' is required (registered: {sorted(TASKS)})")
    _reject_unknown(cfg.get("scene") or {}, _SCENE_KEYS, "scene")
    _reject_unknown(cfg.get("sim") or {}, _SIM_KEYS, "sim")
    _reject_unknown(cfg.get("render") or {}, _RENDER_KEYS, "render")
    _reject_unknown(cfg.get("objective") or {}, _OBJECTIVE_KEYS, "objective")

    # Parse the objective here so a bad or unreachable goal fails in milliseconds,
    # rather than after Isaac Sim has spent two minutes starting up.
    obj_spec = cfg.get("objective") or {}
    if obj_spec.get("sequence"):
        from simbridge.objective import parse_objective, parse_region

        parsed = parse_objective(obj_spec["sequence"], obj_spec.get("pickable") or [])
        for w in parsed.warnings:
            print(f"[config] warning: {w}")
        if obj_spec.get("spawn"):
            spawn = parse_region(obj_spec["spawn"], label="objective.spawn")
            for w in spawn[1]:
                print(f"[config] warning: {w}")

        for w in check_pickable(cfg, parsed.pick_names):
            print(f"[config] warning: {w}")

    render = cfg.get("render") or {}
    # YAML 1.1 parses bare Off/On/Yes/No as booleans, so `antialiasing: Off` arrives as False.
    # Normalising is friendlier than making everyone remember to quote it.
    if isinstance(render.get("antialiasing"), bool):
        render["antialiasing"] = "Off" if render["antialiasing"] is False else "DLAA"
    aa = render.get("antialiasing")
    if isinstance(aa, str):
        canon = {m.lower(): m for m in _AA_MODES}
        aa = canon.get(aa.lower(), aa)
        render["antialiasing"] = aa
    if aa is not None and aa not in _AA_MODES:
        raise ValueError(f"render.antialiasing must be one of {sorted(_AA_MODES)}, got {aa!r}")
    dm = render.get("dlss_mode")
    if dm is not None and dm not in _DLSS_MODES:
        raise ValueError(
            f"render.dlss_mode must be one of {sorted(_DLSS_MODES)} "
            f"({', '.join(f'{k}={v}' for k, v in _DLSS_MODES.items())}), got {dm!r}"
        )
    if dm is not None and aa not in (None, "DLSS"):
        raise ValueError(f"render.dlss_mode is set but antialiasing is {aa!r}; set antialiasing: DLSS")
    return cfg


def apply_render_cfg(env_cfg, render: dict[str, Any], camera_names: list[str]):
    """Attach RTX render settings to every camera declared in the config.

    These settings live on each camera's ``renderer_cfg``, not on ``SimulationCfg`` -- there is
    no ``sim.renderer`` field, so assigning one creates a stray attribute that nothing reads and
    the settings silently do nothing. (Which is exactly what the first version of this function
    did, and the DLSS warning in the log is what caught it.)

    Applying to no cameras is a no-op, and reported as such by the caller rather than passing
    quietly: a ``render`` block in a config with no cameras means the author expected an effect
    that cannot happen.

    Two caveats worth knowing before enabling anything here:

    * DLSS renders below the requested resolution and upscales, but Isaac Sim will not go under
      300 px internally. At a 128 px training camera it upsamples the input instead, costing
      time and changing pixels for no benefit. Leave it off for training cameras.
    * Frame generation (DLSS-G) interpolates frames that were never rendered from scene state.
      Fine for a viewport, wrong as a policy observation, and it does not increase the rate at
      which the simulator produces distinct observations.
    """
    if not render:
        return env_cfg

    from isaaclab_physx.renderers.isaac_rtx_renderer_cfg import (
        IsaacRtxRendererCfg,
        IsaacRtxRendererGlobalSettingsCfg,
    )

    field_map = {
        "antialiasing": "antialiasing_mode",
        "dlss_mode": "dlss_mode",
        "frame_generation": "enable_dlssg",
        "dl_denoiser": "enable_dl_denoiser",
        "reflections": "enable_reflections",
        "global_illumination": "enable_global_illumination",
        "translucency": "enable_translucency",
        "direct_lighting": "enable_direct_lighting",
        "shadows": "enable_shadows",
        "ambient_occlusion": "enable_ambient_occlusion",
        "samples_per_pixel": "samples_per_pixel",
        "max_bounces": "max_bounces",
        "carb_settings": "carb_settings",
    }
    kwargs = {field_map[k]: v for k, v in render.items() if k in field_map and v is not None}
    if not kwargs:
        return env_cfg

    globals_cfg = IsaacRtxRendererGlobalSettingsCfg(**kwargs)
    for name in camera_names:
        cam = getattr(env_cfg.scene, name, None)
        if cam is None:
            continue
        existing = getattr(cam, "renderer_cfg", None)
        if isinstance(existing, IsaacRtxRendererCfg):
            existing.global_settings = globals_cfg
        else:
            cam.renderer_cfg = IsaacRtxRendererCfg(global_settings=globals_cfg)
    return env_cfg


def resolve_task(cfg: dict[str, Any]) -> str:
    """Map the YAML task name to a Gym env id."""
    name = cfg["task"] if isinstance(cfg["task"], str) else cfg["task"]["name"]
    return lookup(TASKS, name, "task")


def _default_prim_path(scene, name: str) -> str:
    """Prim path for a scene entry the config did not give one for.

    Overriding an entry the task already defines has to keep that entry's prim path. Isaac Lab
    terms address bodies by regex on the prim leaf -- ``contrib.lift`` resets the cube with
    ``SceneEntityCfg("object", body_names="Object")`` -- so deriving the path from the YAML key
    renames the prim to ``object`` and the reset term dies with "Not all regular expressions are
    matched", naming a term the config never mentions.
    """
    existing = getattr(scene, name, None)
    path = getattr(existing, "prim_path", None)
    return path if isinstance(path, str) else f"{{ENV_REGEX_NS}}/{name}"


def build_env_cfg(cfg: dict[str, Any], device: str = "cuda:0", num_envs: int | None = None):
    """Construct the Isaac Lab env cfg described by ``cfg``.

    The Isaac Lab import is deferred so this module stays importable without booting Isaac Sim;
    a config can then be validated in a fraction of a second instead of after a two-minute Kit
    startup.
    """
    from isaaclab_tasks.utils import parse_env_cfg

    gym_id = resolve_task(cfg)
    scene_spec = cfg.get("scene") or {}
    sim_spec = cfg.get("sim") or {}

    n = num_envs if num_envs is not None else int(scene_spec.get("num_envs", 64))
    env_cfg = parse_env_cfg(gym_id, device=sim_spec.get("device", device), num_envs=n)

    if "env_spacing" in scene_spec:
        env_cfg.scene.env_spacing = float(scene_spec["env_spacing"])
    if "episode_length_s" in sim_spec:
        env_cfg.episode_length_s = float(sim_spec["episode_length_s"])
    if "decimation" in sim_spec:
        env_cfg.decimation = int(sim_spec["decimation"])
    if "dt" in sim_spec:
        env_cfg.sim.dt = float(sim_spec["dt"])

    if (robot := scene_spec.get("robot")) is not None:
        env_cfg.scene.robot = lookup(ROBOTS, robot["type"], "robot")(robot)

    for name, spec in (scene_spec.get("objects") or {}).items():
        spec = {**spec, "prim_path": spec.get("prim_path", _default_prim_path(env_cfg.scene, name))}
        setattr(env_cfg.scene, name, lookup(OBJECTS, spec["type"], "object")(spec))

    for name, spec in (scene_spec.get("cameras") or {}).items():
        spec = {**spec, "prim_path": spec.get("prim_path", _default_prim_path(env_cfg.scene, name))}
        setattr(env_cfg.scene, name, lookup(CAMERAS, spec["type"], "camera")(spec))

    cam_names = list(scene_spec.get("cameras") or {})
    render_spec = cfg.get("render") or {}
    if render_spec and not cam_names:
        raise ValueError(
            "config has a 'render' block but declares no cameras; RTX settings attach to "
            "cameras, so nothing would apply"
        )
    apply_render_cfg(env_cfg, render_spec, cam_names)

    obj_spec = cfg.get("objective") or {}
    if obj_spec.get("sequence"):
        from simbridge.objective import apply_objective, parse_objective, parse_region

        parsed = parse_objective(obj_spec["sequence"], obj_spec.get("pickable") or [])
        spawn = parse_region(obj_spec["spawn"], label="objective.spawn")[0] if obj_spec.get("spawn") else None
        apply_objective(env_cfg, parsed, spawn=spawn)

    return env_cfg


def describe(cfg: dict[str, Any]) -> str:
    """One-screen summary, for confirming a config says what you meant before a long run."""
    s, sim, ctl = cfg.get("scene") or {}, cfg.get("sim") or {}, cfg.get("control") or {}
    rnd = cfg.get("render") or {}
    obj = cfg.get("objective") or {}
    lines = [
        f"task      : {cfg['task']}  -> {TASKS.get(cfg['task'] if isinstance(cfg['task'], str) else '', '?')}",
        f"envs      : {s.get('num_envs', 64)}  spacing={s.get('env_spacing', '-')}",
        f"robot     : {(s.get('robot') or {}).get('type', '-')}",
        f"objects   : {', '.join((s.get('objects') or {}) ) or '-'}",
        f"cameras   : {', '.join((s.get('cameras') or {})) or '-'}",
        f"sim       : dt={sim.get('dt','-')} decimation={sim.get('decimation','-')} "
        f"episode_s={sim.get('episode_length_s','-')}",
        f"control   : source={ctl.get('source','-')} transport={ctl.get('transport','inprocess')} "
        f"horizon={ctl.get('action_horizon',1)}",
        f"objective : {obj.get('sequence','-')}",
        f"spawn     : {obj.get('spawn','-')}",
        f"render    : aa={rnd.get('antialiasing','default')} "
        f"dlss_mode={rnd.get('dlss_mode','-')} framegen={rnd.get('frame_generation','-')}",
    ]
    return "\n".join(lines)
