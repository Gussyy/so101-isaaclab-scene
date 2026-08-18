# SPDX-License-Identifier: BSD-3-Clause
"""Conversions between LeRobot and this environment.

LeRobot runs in its own process, talking over the ZeroMQ transport. That is not a workaround --
it is the reason the transport exists. LeRobot pins its own torch and a large dependency tree;
installing it alongside Isaac Sim 6.0.1 (which pins torch 2.11.0+cu128 and Python 3.12) is a
fight with no upside, because the only thing the two need to agree on is the wire format.

So: LeRobot in its own venv, this environment in ours, ``ObsPacket``/``ActionPacket`` between.

What actually has to line up
----------------------------
The motor order matches exactly -- LeRobot's ``SOFollower`` declares
``shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`` and so does the
Isaac Lab articulation. That is lucky and worth not breaking.

Two things do not line up and are handled here:

* **Units.** LeRobot normalises arm joints to -100..100 and the gripper to 0..100
  (``MotorNormMode.RANGE_0_100``). Isaac Lab actions here are -1..1 offsets from a default pose.
* **Calibration.** A physical SO-101's -100..100 maps to whatever its calibration says, not to
  the URDF joint limits. The mapping below assumes they coincide, which is close enough to
  teleoperate with and *not* close enough for sim-to-real transfer. :data:`JOINT_LIMITS` is
  exposed so it can be replaced with a measured calibration.

This module imports no LeRobot code, so it can be tested on either side of the boundary.
"""

from __future__ import annotations

import numpy as np

# LeRobot SOFollower motor order, which is also the Isaac Lab articulation order.
JOINTS: list[str] = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]
ARM_JOINTS: list[str] = JOINTS[:5]

# Radian limits from the spawned articulation (scripts/scene_demo.py), not from the USD.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.920, 1.920),
    "shoulder_lift": (-1.745, 1.745),
    "elbow_flex": (-1.690, 1.690),
    "wrist_flex": (-1.658, 1.658),
    "wrist_roll": (-2.744, 2.841),
    "gripper": (-0.175, 1.745),
}

# Matches so101_scene.pick_place_env_cfg: JointPositionActionCfg(scale=0.5, use_default_offset=True)
ACTION_SCALE = 0.5
DEFAULT_JOINT_POS: dict[str, float] = {
    "shoulder_pan": 0.0,
    "shoulder_lift": -0.6,
    "elbow_flex": 0.8,
    "wrist_flex": 0.6,
    "wrist_roll": 0.0,
    "gripper": 0.0,
}


def _norm_to_rad(joint: str, value: float) -> float:
    """LeRobot normalised units -> radians, using this joint's limits."""
    lo, hi = JOINT_LIMITS[joint]
    if joint == "gripper":                    # RANGE_0_100
        t = np.clip(value, 0.0, 100.0) / 100.0
    else:                                     # RANGE_M100_100
        t = (np.clip(value, -100.0, 100.0) + 100.0) / 200.0
    return lo + t * (hi - lo)


def _rad_to_norm(joint: str, rad: float) -> float:
    lo, hi = JOINT_LIMITS[joint]
    t = (np.clip(rad, lo, hi) - lo) / max(hi - lo, 1e-9)
    return t * 100.0 if joint == "gripper" else t * 200.0 - 100.0


def lerobot_action_to_array(action: dict[str, float], binary_gripper: bool = True) -> np.ndarray:
    """LeRobot action dict -> the ``(6,)`` action this environment expects.

    Args:
        action: ``{"shoulder_pan.pos": ..., ...}`` in LeRobot normalised units.
        binary_gripper: The pick-and-place task drives the jaw with
            ``BinaryJointPositionActionCfg``, so only the sign of the gripper action matters.
            Mapping a continuous jaw command through it and expecting proportional closure is a
            quiet way to lose a grasp.
    """
    out = np.zeros(len(JOINTS), dtype=np.float32)
    for i, joint in enumerate(JOINTS):
        key = f"{joint}.pos"
        if key not in action:
            continue
        rad = _norm_to_rad(joint, float(action[key]))
        if joint == "gripper":
            if binary_gripper:
                mid = sum(JOINT_LIMITS["gripper"]) / 2.0
                out[i] = 1.0 if rad > mid else -1.0
            else:
                out[i] = np.clip((rad - DEFAULT_JOINT_POS[joint]) / ACTION_SCALE, -1.0, 1.0)
        else:
            out[i] = np.clip((rad - DEFAULT_JOINT_POS[joint]) / ACTION_SCALE, -1.0, 1.0)
    return out


def array_to_lerobot_action(arr: np.ndarray) -> dict[str, float]:
    """The inverse: a ``(6,)`` environment action -> a LeRobot action dict."""
    out: dict[str, float] = {}
    for i, joint in enumerate(JOINTS[: len(arr)]):
        rad = float(arr[i]) * ACTION_SCALE + DEFAULT_JOINT_POS[joint]
        out[f"{joint}.pos"] = float(_rad_to_norm(joint, rad))
    return out


def obs_to_lerobot(
    state: dict[str, np.ndarray],
    images: dict[str, np.ndarray],
    env_id: int = 0,
    joint_pos_key: str = "joint_pos",
) -> dict:
    """Build the observation dict a LeRobot policy expects, for one environment.

    LeRobot policies act on a single robot, so a batched simulation has to pick an index. Feeding
    a policy the whole batch silently produces nonsense rather than an error.
    """
    obs: dict = {}
    q = state.get(joint_pos_key)
    if q is not None:
        row = np.asarray(q)[env_id]
        for i, joint in enumerate(JOINTS[: len(row)]):
            obs[f"{joint}.pos"] = float(_rad_to_norm(joint, float(row[i])))
    for name, img in images.items():
        obs[name] = np.asarray(img)[env_id]
    return obs


def demo() -> None:
    """Self-check: units, round-trip, gripper handling, missing keys."""
    # A LeRobot action at each joint's default pose should map to roughly zero action.
    at_default = {f"{j}.pos": _rad_to_norm(j, DEFAULT_JOINT_POS[j]) for j in JOINTS}
    a = lerobot_action_to_array(at_default, binary_gripper=False)
    assert np.allclose(a[:5], 0.0, atol=1e-4), a[:5]

    # Round-trip through the normalised representation.
    arr = np.array([0.3, -0.4, 0.5, -0.2, 0.1, 0.0], dtype=np.float32)
    back = lerobot_action_to_array(array_to_lerobot_action(arr), binary_gripper=False)
    assert np.allclose(arr, back, atol=1e-3), (arr, back)

    # Gripper normalisation is 0..100, not -100..100: 0 is fully closed, 100 fully open.
    closed = _norm_to_rad("gripper", 0.0)
    opened = _norm_to_rad("gripper", 100.0)
    assert abs(closed - JOINT_LIMITS["gripper"][0]) < 1e-6
    assert abs(opened - JOINT_LIMITS["gripper"][1]) < 1e-6
    assert lerobot_action_to_array({"gripper.pos": 100.0})[5] == 1.0
    assert lerobot_action_to_array({"gripper.pos": 0.0})[5] == -1.0

    # Out-of-range input clamps rather than flying off.
    assert lerobot_action_to_array({"shoulder_pan.pos": 1e6}, binary_gripper=False)[0] <= 1.0

    # Missing joints stay at zero instead of raising -- a partial dict is a normal thing to get.
    assert lerobot_action_to_array({"shoulder_pan.pos": 0.0})[3] == 0.0

    # Observation packing selects one environment out of the batch.
    state = {"joint_pos": np.tile(np.array([[0.0, -0.6, 0.8, 0.6, 0.0, 0.0]], dtype=np.float32), (4, 1))}
    images = {"scene_cam": np.zeros((4, 8, 8, 3), dtype=np.uint8)}
    obs = obs_to_lerobot(state, images, env_id=2)
    assert obs["scene_cam"].shape == (8, 8, 3)
    assert abs(obs["shoulder_lift.pos"] - _rad_to_norm("shoulder_lift", -0.6)) < 1e-4

    print("lerobot bridge OK: units, round-trip, 0..100 gripper, clamping, partial dicts, batching")
    print(f"  joint order matches LeRobot SOFollower: {', '.join(JOINTS)}")


if __name__ == "__main__":
    demo()
