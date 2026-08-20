# Robot Descriptions for SO-ARM101

Robot descriptions, URDF configurations, USD assets (for Isaac Sim / Isaac Lab 3.0), and 3D meshes for the **SO-ARM101** robotic arm variants (5-DOF arm with single-jaw and dual-jaw gripper options).

---

## Directory Overview

| Directory / File | Type | Description |
|---|---|---|
| [`SO-ARM101-FULL/`](SO-ARM101-FULL/) | URDF | Full robot assembly combining the calibrated 5-DOF manipulator (`so101_new_calib.urdf`) with the 2-jaw gear gripper (`Gripper.urdf`). |
| [`SO101/`](SO101/) | URDF / MJCF | Original SO-101 single-jaw arm exported via `onshape-to-robot`. Includes calibrated URDF and MuJoCo XML files. |
| [`SO-ARM101-OMNI-KIN/`](SO-ARM101-OMNI-KIN/) | URDF | Omni-Kin configuration containing `SO-ARM101-OMNI-KIN.urdf`, `SO-ARM101-ORIGINAL.urdf`, and standalone `Gripper.urdf`. |
| [`IsaacAssets/`](IsaacAssets/) | USD | OpenUSD asset packages for Isaac Sim 6.0.1 / Isaac Lab 3.0, including USD stages and payload files for simulation. |
| `SO-ARM101-OMNI-KIN.zip` | Archive | Compressed distribution package containing Omni-Kin URDFs and mesh files. |

---

## Kinematic Configurations & Variants

### 1. SO-ARM101-FULL (`robot_description/SO-ARM101-FULL/`)

- **URDF File**: [`SO-ARM101-FULL.urdf`](SO-ARM101-FULL/SO-ARM101-FULL.urdf)
- **Meshes Directory**: [`meshes/`](SO-ARM101-FULL/meshes/)
- **Description**: Merges the calibrated 5-DOF manipulator arm (`so101_new_calib.urdf`) with the 2-jaw parallel gear gripper (`Gripper.urdf`). All mesh assets are consolidated inside the local `meshes/` folder.

**Kinematic Tree Structure**:

```
base_link
 └── shoulder_link (joint: shoulder_pan)
      └── upper_arm_link (joint: shoulder_lift)
           └── lower_arm_link (joint: elbow_flex)
                └── wrist_link (joint: wrist_flex)
                     └── gripper_base (joint: wrist_roll)
                          ├── gripper_gear (joint: base_jaw_joint [revolute])
                          ├── arm_r (joint: base_gripper_right_joint [prismatic, mimic])
                          ├── arm_l (joint: base_gripper_left_joint [prismatic, mimic])
                          └── gripper_frame_link (joint: gripper_frame_joint [fixed])
```

---

### 2. SO101 Original (`robot_description/SO101/`)

- **Source**: [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
- **URDF File**: [`so101_new_calib.urdf`](SO101/so101_new_calib.urdf)
- **Assets Directory**: [`assets/`](SO101/assets/)
- **Description**: Original 5-DOF arm model with a single moving jaw (`moving_jaw_so101_v1`).

**Calibration Variants**:
- **`so101_new_calib.urdf` / `.xml` (Default)**: Virtual zero centered at the midpoint of joint ranges.
- **`so101_old_calib.urdf` / `.xml`**: Virtual zero defined with the arm fully extended horizontally.

---

### 3. SO-ARM101-OMNI-KIN (`robot_description/SO-ARM101-OMNI-KIN/`)

- **Files**:
  - [`SO-ARM101-OMNI-KIN.urdf`](SO-ARM101-OMNI-KIN/SO-ARM101-OMNI-KIN.urdf): Complete Omni-Kin arm assembly.
  - [`Gripper.urdf`](SO-ARM101-OMNI-KIN/Gripper.urdf): Standalone 2-jaw parallel gripper description.
  - [`SO-ARM101-ORIGINAL.urdf`](SO-ARM101-OMNI-KIN/SO-ARM101-ORIGINAL.urdf): Unmodified CAD export URDF.

---

### 4. Isaac Sim OpenUSD Assets (`robot_description/IsaacAssets/`)

- **Stages & Payloads**:
  - [`SO-ARM101-FULL/`](IsaacAssets/SO-ARM101-FULL/): `SO-ARM101-FULL.urdf` converted into `.usda` format.
  - [`Gripper/`](IsaacAssets/Gripper/): `Gripper.urdf` converted into `.usda` format.
  - [`Gripper_with_ROS2.usd`](IsaacAssets/Gripper_with_ROS2.usd): Test scene importing the gripper to evaluate and test the parallel joint.
- **Description**: OpenUSD stages and payload files configured with physics schemas, visual materials, and rigid body dynamics for Isaac Sim 6.0.1 / Isaac Lab 3.0.

---

## Joint Limits & Actuator Parameters

| Joint Name | Type | Range (rad / m) | Actuator / Note |
|---|---|---|---|
| `shoulder_pan` | Revolute | -1.920 … +1.920 | Feetech STS3215 (`motor1`) |
| `shoulder_lift` | Revolute | -1.745 … +1.745 | Feetech STS3215 (`motor2`) |
| `elbow_flex` | Revolute | -1.690 … +1.690 | Feetech STS3215 (`motor3`) |
| `wrist_flex` | Revolute | -1.658 … +1.658 | Feetech STS3215 (`motor4`) |
| `wrist_roll` | Revolute | -2.744 … +2.841 | Feetech STS3215 (`motor5`) |
| `base_jaw_joint` | Revolute | 0.000 … +3.140 | Feetech STS3215 (`motor6`) |
| `base_gripper_right_joint` | Prismatic | -0.044 … 0.000 | Mimics `base_jaw_joint` (-0.013721x) |
| `base_gripper_left_joint` | Prismatic | -0.044 … 0.000 | Mimics `base_gripper_right_joint` (1.0x) |

---