# SPDX-License-Identifier: BSD-3-Clause
"""Action sources: everything that can drive an environment, behind one interface."""

from simbridge.sources.basic import ScriptedSource, ZeroSource
from simbridge.sources.remote import make_remote_source
from simbridge.sources.rl_checkpoint import RslRlCheckpointSource

# The in-process keyboard source imports carb, which only exists inside a running Kit app.
# Importing it unconditionally would break `python -m simbridge.schema` and every tool that
# touches simbridge without booting Isaac Sim, so it registers only when Kit is present.
try:  # pragma: no cover - depends on the runtime having Kit
    from simbridge.sources.keyboard import KeyboardSource
except Exception:  # noqa: BLE001
    KeyboardSource = None  # type: ignore[assignment]

__all__ = [
    "ZeroSource",
    "ScriptedSource",
    "RslRlCheckpointSource",
    "make_remote_source",
    "KeyboardSource",
]
