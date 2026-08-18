# SPDX-License-Identifier: BSD-3-Clause
"""simbridge -- a config-driven bridge between Isaac Lab environments and whatever drives them.

Three ideas, each earning its place:

1. **One wire contract** (:mod:`simbridge.schema`). An environment emits an ``ObsPacket`` and
   consumes an ``ActionPacket``. Everything that produces actions -- a trained RL checkpoint, a
   teleop device, a flow-matching visuomotor policy, a scripted baseline -- speaks only these,
   so the environment never learns which one is attached.

2. **Transport is pluggable** (:mod:`simbridge.transport`). The same policy code runs in-process
   during development and behind a ZeroMQ socket in deployment. That split is what lets a model
   live in its own process, its own CUDA context, even its own machine.

3. **Scenes are configuration** (:mod:`simbridge.builder`). A YAML file names a task and declares
   the robot, props, cameras and driver. Reward and termination logic stays in Python, because
   those are code, and a config language that tries to express them becomes one nobody can debug.

The interface shape follows NVlabs/RoboLab's ``InferenceClient`` -- four hooks, with action
chunking owned by the base class. RoboLab itself cannot be installed on this stack (it pins
Isaac Sim 5.0/5.1, Isaac Lab 2.2/2.3, Python 3.11 and Ubuntu, against our Isaac Sim 6.0.1 /
Isaac Lab 3.0 / Python 3.12 / Windows), but its interface is proven against real flow-matching
VLA servers, so matching it beats inventing something new.
"""

from simbridge.interfaces import ActionSource, RemoteActionSource, Transport
from simbridge.schema import ActionPacket, ObsPacket, ProtocolError, PROTOCOL_VERSION

__all__ = [
    "ActionSource",
    "RemoteActionSource",
    "Transport",
    "ObsPacket",
    "ActionPacket",
    "ProtocolError",
    "PROTOCOL_VERSION",
]
