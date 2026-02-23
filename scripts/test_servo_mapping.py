import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ----------------------------
# Paths (repo-relative)
# ----------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "assets" / "ainex" / "ainex_physics.xml"


# ----------------------------
# Your current mappings (copy-pasted from replay_actiongroup.py)
# ----------------------------
SERVO_CENTER = 500.0
SERVO_MIN = 0.0
SERVO_MAX = 1000.0

# Per-servo maximum angle (radians) at the ends (0 and 1000).
# Order: Servo1..Servo22
SERVO_RANGE = np.array(
    [
        # ---- right leg (Servo1..Servo6) ----
        0.70,  # r_hip_yaw
        0.55,  # r_hip_roll
        0.80,  # r_hip_pitch
        1.00,  # r_knee
        0.55,  # r_ank_pitch
        0.45,  # r_ank_roll

        # ---- left leg (Servo7..Servo12) ----
        0.70,  # l_hip_yaw
        0.55,  # l_hip_roll
        0.80,  # l_hip_pitch
        1.00,  # l_knee
        0.55,  # l_ank_pitch
        0.45,  # l_ank_roll

        # ---- right arm (Servo13..Servo17) ----
        0.70,  # r_sho_pitch
        0.70,  # r_sho_roll
        0.80,  # r_el_pitch
        0.70,  # r_el_yaw
        0.50,  # r_gripper

        # ---- left arm (Servo18..Servo22) ----
        0.70,  # l_sho_pitch
        0.70,  # l_sho_roll
        0.80,  # l_el_pitch
        0.70,  # l_el_yaw
        0.50,  # l_gripper
    ],
    dtype=float,
)

# Servo1..Servo22 -> actuator indices in your model
SERVO_TO_ACT = [
    0, 1, 2, 3, 4, 5,          # Servo1-6   -> right leg
    6, 7, 8, 9, 10, 11,        # Servo7-12  -> left leg
    14, 15, 16, 17, 18,        # Servo13-17 -> right arm
    19, 20, 21, 22, 23         # Servo18-22 -> left arm
]

# +1 = normal, -1 = inverted, in CSV servo order
SERVO_SIGN = np.array(
    [
        +1,  # Servo1  -> 00_r_hip_yaw
        -1,  # Servo2  -> 01_r_hip_roll
        +1,  # Servo3  -> 02_r_hip_pitch
        +1,  # Servo4  -> 03_r_knee
        -1,  # Servo5  -> 04_r_ank_pitch
        -1,  # Servo6  -> 05_r_ank_roll

        +1,  # Servo7  -> 06_l_hip_yaw
        +1,  # Servo8  -> 07_l_hip_roll
        +1,  # Servo9  -> 08_l_hip_pitch
        +1,  # Servo10 -> 09_l_knee
        +1,  # Servo11 -> 10_l_ank_pitch
        +1,  # Servo12 -> 11_l_ank_roll

        +1,  # Servo13 -> 14_r_sho_pitch
        -1,  # Servo14 -> 15_r_sho_roll
        -1,  # Servo15 -> 16_r_el_pitch
        -1,  # Servo16 -> 17_r_el_yaw
        +1,  # Servo17 -> 18_r_gripper

        +1,  # Servo18 -> 19_l_sho_pitch
        +1,  # Servo19 -> 20_l_sho_roll
        +1,  # Servo20 -> 21_l_el_pitch
        +1,  # Servo21 -> 22_l_el_yaw
        +1,  # Servo22 -> 23_l_gripper
    ],
    dtype=float,
)

NUM_CSV_SERVOS = 22


# ----------------------------
# Test config
# ----------------------------
# Change these between runs:
TEST_SERVO = 12         # 1..22 (Servo number you want to test)
AMPLITUDE = 80.0       # servo units away from center (e.g., 50, 80, 120)
FREQ_HZ = 0.5          # oscillation frequency (0.25 to 1.0 is nice)
RUN_SECONDS = 10.0     # how long to run the test


def servo_to_radians(v: float, rad_range: float) -> float:
    """
    500  -> 0 rad
    0    -> -rad_range
    1000 -> +rad_range
    """
    x = (v - SERVO_CENTER) / (SERVO_MAX - SERVO_CENTER)
    x = max(-1.0, min(1.0, x))
    return x * float(rad_range)


def actuator_joint_info(model: mujoco.MjModel, act_idx: int):
    """
    For <position joint="..."> actuators:
      actuator_trnid[act_idx, 0] is the joint id.
    """
    j_id = int(model.actuator_trnid[act_idx, 0])
    if j_id < 0:
        return None

    qadr = int(model.jnt_qposadr[j_id])
    rmin = float(model.jnt_range[j_id, 0])
    rmax = float(model.jnt_range[j_id, 1])
    return (j_id, qadr, (rmin, rmax))


def clamp_to_joint_limits(rad: float, joint_range: tuple[float, float]) -> float:
    rmin, rmax = joint_range
    return float(np.clip(rad, rmin, rmax))


def set_all_neutral(model: mujoco.MjModel, data: mujoco.MjData):
    """
    Put robot in a consistent "neutral" pose:
      - all joint qpos = 0 (except freejoint)
      - all ctrl = 0
      - velocities = 0
    """
    data.ctrl[:] = 0.0

    # If you have a freejoint, qpos[0:7] are root pos+quat. Leave them as-is.
    # For hinge joints, just set them to 0.
    # (Safer than trying to reconstruct root.)
    if model.nq > 7:
        data.qpos[7:] = 0.0

    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print(f"Loaded model: {MODEL_PATH}")
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")

    # Print actuators for sanity
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(i, name)

    if not (1 <= TEST_SERVO <= NUM_CSV_SERVOS):
        raise ValueError("TEST_SERVO must be in 1..22")

    s_idx = TEST_SERVO - 1
    act_idx = SERVO_TO_ACT[s_idx]
    act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_idx)

    info = actuator_joint_info(model, act_idx)
    if info is None:
        raise RuntimeError(f"Actuator {act_idx} does not map to a joint as expected.")
    j_id, qadr, (rmin, rmax) = info
    j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)

    print()
    print("=== TEST CONFIG ===")
    print(f"TEST_SERVO: Servo{TEST_SERVO}")
    print(f"Mapped actuator: {act_idx} ({act_name})")
    print(f"Mapped joint: {j_id} ({j_name}) qpos_adr={qadr} range=[{rmin:.3f}, {rmax:.3f}]")
    print(f"SERVO_SIGN: {SERVO_SIGN[s_idx]:+g}")
    print(f"SERVO_RANGE: {SERVO_RANGE[s_idx]:.3f} rad")
    print("===================")
    print()

    # Start from consistent neutral
    set_all_neutral(model, data)

    # Keep head neutral (if present)
    if model.nu >= 14:
        data.ctrl[12] = 0.0
        data.ctrl[13] = 0.0

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        t0 = time.time()
        last_print = -1.0

        while viewer.is_running():
            now = time.time()
            elapsed = now - t0
            if elapsed >= RUN_SECONDS:
                print("Done.")
                break

            # Oscillate around 500
            # value = 500 + AMP * sin(2*pi*f*t)
            raw_servo = SERVO_CENTER + AMPLITUDE * np.sin(2.0 * np.pi * FREQ_HZ * elapsed)
            raw_servo = float(np.clip(raw_servo, SERVO_MIN, SERVO_MAX))

            # Convert -> rad with per-servo range and sign correction, then clamp to joint range
            rad = servo_to_radians(raw_servo, SERVO_RANGE[s_idx])
            rad = SERVO_SIGN[s_idx] * rad
            rad = clamp_to_joint_limits(rad, (rmin, rmax))

            # Drive ONLY this actuator; keep others neutral
            data.ctrl[:] = 0.0
            data.ctrl[act_idx] = rad
            if model.nu >= 14:
                data.ctrl[12] = 0.0
                data.ctrl[13] = 0.0

            mujoco.mj_step(model, data)
            viewer.sync()

            # Print at ~2 Hz
            if elapsed - last_print >= 0.5:
                last_print = elapsed
                q = float(data.qpos[qadr])
                print(
                    f"t={elapsed:5.2f}s  Servo{TEST_SERVO}={raw_servo:7.1f}  "
                    f"target={rad:+.3f} rad  qpos={q:+.3f} rad"
                )

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()