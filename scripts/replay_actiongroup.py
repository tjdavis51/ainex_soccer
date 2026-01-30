import time
import sqlite3
import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = "ainex/ainex_stable.xml"
D6A_PATH   = "action_groups_export/ActionGroups/forward_one_step.d6a"

# ---- Mapping: Servo1..Servo22 -> ctrl indices in your actuator order ----
# Your actuator order (from your printout):
# 0-5  right leg
# 6-11 left leg
# 12-13 head (skip)
# 14-18 right arm (5)
# 19-23 left arm (5)
SERVO_TO_CTRL = [
    0, 1, 2, 3, 4, 5,          # Servo1..6   -> right leg
    6, 7, 8, 9, 10, 11,        # Servo7..12  -> left leg
    14, 15, 16, 17, 18,        # Servo13..17 -> right arm
    19, 20, 21, 22, 23         # Servo18..22 -> left arm
]

# ---- Conversion: ticks (0..1000) -> radians ----
# Map 0..1000 to roughly [-2.09, +2.09] around center 500.
RAD_PER_TICK = 2.09 / 500.0

# Optional per-joint sign flips if any joint moves opposite direction.
# Start with all +1. If something is backwards, set that ctrl index to -1.
CTRL_SIGN = np.ones(24, dtype=float)

# Optional per-joint offsets (radians). Usually 0, but you can tweak.
CTRL_OFFSET = np.zeros(24, dtype=float)

def load_actiongroup_frames(d6a_path: str):
    con = sqlite3.connect(d6a_path)
    cur = con.cursor()
    rows = cur.execute(
        """
        SELECT Time,
               Servo1,Servo2,Servo3,Servo4,Servo5,Servo6,
               Servo7,Servo8,Servo9,Servo10,Servo11,Servo12,
               Servo13,Servo14,Servo15,Servo16,Servo17,
               Servo18,Servo19,Servo20,Servo21,Servo22
        FROM ActionGroup
        ORDER BY [Index] ASC;
        """
    ).fetchall()
    con.close()

    frames = []
    for row in rows:
        ms = int(row[0])
        servos = np.array(row[1:], dtype=float)  # length 22
        frames.append((ms, servos))
    return frames

def ticks_to_ctrl_targets(servos_22: np.ndarray, model: mujoco.MjModel):
    # Start from current ctrl (so untouched controls keep their value)
    target = np.array(model.key_qpos[0:0])  # dummy to avoid confusion
    ctrl = np.zeros(model.nu, dtype=float)

    # If you want: start from current data.ctrl instead, set later in loop.
    # We'll fill only mapped ctrl indices; others stay 0 unless you set them.
    for i, tick in enumerate(servos_22):
        ctrl_idx = SERVO_TO_CTRL[i]
        rad = (tick - 500.0) * RAD_PER_TICK
        rad = CTRL_SIGN[ctrl_idx] * rad + CTRL_OFFSET[ctrl_idx]

        # Clip to the joint range if possible (safe)
        # We can’t directly index joint range from actuator index cleanly here,
        # so just clip to [-2.2, 2.2] as a safe starting bound.
        rad = float(np.clip(rad, -2.2, 2.2))
        ctrl[ctrl_idx] = rad
    return ctrl

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data  = mujoco.MjData(model)

    frames = load_actiongroup_frames(D6A_PATH)
    print(f"Loaded model: {MODEL_PATH}")
    print(f"Loaded action: {D6A_PATH} ({len(frames)} frames, {sum(ms for ms,_ in frames)} ms total)")
    print(f"nu={model.nu}, timestep={model.opt.timestep}")

    # Start from a stable pose if you want:
    # If you have a saved qpos string, paste it here and uncomment:
    # data.qpos[:] = np.array([...], dtype=float)
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch(model, data) as viewer:
        # Let viewer fully initialize
        time.sleep(0.2)

        # Replay in a loop
        while viewer.is_running():
            for ms, servos in frames:
                if not viewer.is_running():
                    break

                # Compute ctrl targets for this keyframe
                ctrl_targets = ticks_to_ctrl_targets(servos, model)
                data.ctrl[:] = ctrl_targets

                # Hold this keyframe for its duration (ms)
                duration = ms / 1000.0
                steps = max(1, int(duration / model.opt.timestep))

                for _ in range(steps):
                    if not viewer.is_running():
                        break
                    mujoco.mj_step(model, data)
                    viewer.sync()

            # small pause between loops
            time.sleep(0.2)

if __name__ == "__main__":
    main()