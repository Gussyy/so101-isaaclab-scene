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

_TOP_LEVEL = {"task", "scene", "sim", "control", "meta"}
_SCENE_KEYS = {"num_envs", "env_spacing", "robot", "objects", "cameras"}
_SIM_KEYS = {"episode_length_s", "dt", "decimation", "physics", "device"}


def _reject_unknown(d: dict[str, Any], allowed: set[str], where: str) -> None:
    extra = set(d) - allowed
    if extra:
        raise ValueError(f"unknown key(s) {sorted(extra)} in '{where}'; allowed: {sorted(allowed)}")


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
    return cfg


def resolve_task(cfg: dict[str, Any]) -> str:
    """Map the YAML task name to a Gym env id."""
    name = cfg["task"] if isinstance(cfg["task"], str) else cfg["task"]["name"]
    return lookup(TASKS, name, "task")


def build_env_cfg(cfg: dict[str, Any], device: str = "cuda:0", num_envs: int | None = None):
    """Construct the Isaac Lab env cfg described by ``cfg``.

    Import is deferred: this module is importable (and testable) without booting Isaac Sim, so a
    config can be validated in a fraction of a second instead of after a two-minute Kit startup.
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
        spec = {**spec, "prim_path": spec.get("prim_path", f"{{ENV_REGEX_NS}}/{name}")}
        setattr(env_cfg.scene, name, lookup(OBJECTS, spec["type"], "object")(spec))

    for name, spec in (scene_spec.get("cameras") or {}).items():
        spec = {**spec, "prim_path": spec.get("prim_path", f"{{ENV_REGEX_NS}}/{name}")}
        setattr(env_cfg.scene, name, lookup(CAMERAS, spec["type"], "camera")(spec))

    return env_cfg


def describe(cfg: dict[str, Any]) -> str:
    """One-screen summary, for confirming a config says what you meant before a long run."""
    s, sim, ctl = cfg.get("scene") or {}, cfg.get("sim") or {}, cfg.get("control") or {}
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
    ]
    return "\n".join(lines)
