# SPDX-License-Identifier: BSD-3-Clause
"""Name -> factory lookups, so a YAML string can name a robot, object, camera or driver.

This is the piece that lets a new scene be a config file instead of a Python module. Everything
the YAML can refer to is registered here under a short name; the builder never imports by
dotted path from user config, which keeps a config file from being able to import arbitrary code.

Registering is a decorator, so third-party code can extend the vocabulary without editing this
file::

    @register_object("my_widget")
    def _widget(spec): ...
"""

from __future__ import annotations

from typing import Any, Callable

ROBOTS: dict[str, Callable[..., Any]] = {}
OBJECTS: dict[str, Callable[..., Any]] = {}
CAMERAS: dict[str, Callable[..., Any]] = {}
TASKS: dict[str, str] = {}          # task name -> gym env id
SOURCES: dict[str, Callable[..., Any]] = {}


def _adder(table: dict[str, Any], kind: str) -> Callable:
    def register(name: str) -> Callable:
        def deco(fn):
            if name in table:
                raise KeyError(f"{kind} {name!r} is already registered")
            table[name] = fn
            return fn
        return deco
    return register


register_robot = _adder(ROBOTS, "robot")
register_object = _adder(OBJECTS, "object")
register_camera = _adder(CAMERAS, "camera")
register_source = _adder(SOURCES, "action source")


def register_task(name: str, gym_id: str) -> None:
    """Map a friendly YAML task name to a registered Gym env id."""
    TASKS[name] = gym_id


def lookup(table: dict[str, Any], name: str, kind: str):
    """Fetch by name, failing with the list of valid options rather than a bare KeyError."""
    try:
        return table[name]
    except KeyError:
        raise KeyError(f"unknown {kind} {name!r}; registered: {sorted(table)}") from None
