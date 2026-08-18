# SPDX-License-Identifier: BSD-3-Clause
"""Keyboard teleoperation, driving joints directly.

Isaac Lab ships :class:`~isaaclab.devices.keyboard.Se3Keyboard`, which emits SE(3) end-effector
deltas. Those need an IK solver, and on a 5-DOF arm IK is already awkward: the SO-101 cannot
reach an arbitrary 6-DOF pose, so a solver has to soft-weight orientation to stay well behaved.

For teleoperation, driving the six joints directly avoids all of that and matches the action
space the tasks already use (``JointPositionActionCfg``, absolute targets). One key pair per
joint, which is also easier to learn than a 6-DOF SE(3) mapping on a small arm.

Keys act on a held target that persists between frames, rather than only while a key is down.
A per-frame delta would mean the arm falls back to its default pose the moment you stop typing,
which makes positioning something impossible to do slowly.

Requires the Isaac Sim window to have focus (input comes from carb). For a keyboard driver that
runs outside the simulator, see ``scripts/teleop_server.py``.
"""

from __future__ import annotations

import weakref

import numpy as np

from simbridge.interfaces import ActionSource
from simbridge.registry import register_source
from simbridge.schema import ObsPacket

# joint index -> (increase key, decrease key). Order matches the SO-101 articulation:
# shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
DEFAULT_BINDINGS: list[tuple[str, str]] = [
    ("Q", "A"),  # shoulder_pan
    ("W", "S"),  # shoulder_lift
    ("E", "D"),  # elbow_flex
    ("R", "F"),  # wrist_flex
    ("T", "G"),  # wrist_roll
]
GRIPPER_KEY = "SPACE"
RESET_KEY = "N"

JOINT_LABELS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


@register_source("keyboard")
class KeyboardSource(ActionSource):
    """Teleoperate the arm from the Isaac Sim window.

    Args:
        action_dim: Action width. 6 for the SO-101 (5 arm joints + gripper).
        step: How far each key press moves a joint target, in normalized action units.
        gripper_open: Gripper action value when open.
        gripper_close: Gripper action value when closed.
        broadcast: Apply the same command to every environment. Teleoperating one arm out of 256
            is rarely what anyone means, so this defaults to on.
    """

    def __init__(
        self,
        action_dim: int = 6,
        step: float = 0.05,
        gripper_open: float = 1.0,
        gripper_close: float = -1.0,
        broadcast: bool = True,
        action_horizon: int = 1,
        **_: object,
    ) -> None:
        super().__init__(action_horizon=action_horizon)
        self.action_dim = int(action_dim)
        self.step = float(step)
        self.gripper_open = float(gripper_open)
        self.gripper_close = float(gripper_close)
        self.broadcast = bool(broadcast)

        self._target = np.zeros(self.action_dim, dtype=np.float32)
        self._closed = False

        import carb  # noqa: PLC0415  -- only importable inside a running Kit app
        import omni.appwindow  # noqa: PLC0415

        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._carb = carb
        # weakref so this object stays collectable despite the subscription holding a callback
        self._sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_key(event, *args),
        )
        print(self.help_text(), flush=True)

    def __del__(self) -> None:
        try:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._sub)
        except Exception:
            pass

    @staticmethod
    def help_text() -> str:
        rows = "\n".join(
            f"    {inc:<6} / {dec:<6}  {JOINT_LABELS[i]}" for i, (inc, dec) in enumerate(DEFAULT_BINDINGS)
        )
        return (
            "\n  keyboard teleop (focus the Isaac Sim window)\n"
            f"{rows}\n"
            f"    {GRIPPER_KEY:<6} {'':<8} toggle gripper\n"
            f"    {RESET_KEY:<6} {'':<8} zero all targets\n"
        )

    def _on_key(self, event, *args) -> bool:
        if event.type != self._carb.input.KeyboardEventType.KEY_PRESS:
            return True
        name = event.input.name

        for i, (inc, dec) in enumerate(DEFAULT_BINDINGS):
            if name == inc:
                self._target[i] = float(np.clip(self._target[i] + self.step, -1.0, 1.0))
            elif name == dec:
                self._target[i] = float(np.clip(self._target[i] - self.step, -1.0, 1.0))

        if name == GRIPPER_KEY:
            self._closed = not self._closed
        elif name == RESET_KEY:
            self._target[:] = 0.0
            self._closed = False
        return True

    def _predict_chunk(self, obs: ObsPacket) -> np.ndarray:
        cmd = self._target.copy()
        if self.action_dim >= 6:
            cmd[5] = self.gripper_close if self._closed else self.gripper_open
        n = obs.num_envs if self.broadcast else 1
        return np.tile(cmd, (n, 1)).astype(np.float32)

    def reset(self, env_ids: np.ndarray | None = None) -> None:
        # Deliberately does NOT clear the held target. Episode resets happen constantly during
        # teleoperation, and zeroing the arm on each one would make it unusable. Press RESET_KEY.
        super().reset(env_ids)
