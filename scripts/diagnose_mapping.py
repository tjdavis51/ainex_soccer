# scripts/diagnose_mapping.py
from __future__ import annotations

from pathlib import Path
import time

import mujoco
import mujoco.viewer
import numpy as np

# If your engine is in scripts/, this import works when running:
#   mjpython scripts/diagnose_mapping.py
from actiongroup_engine import ActionGroupEngine

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO_ROOT / "assets" / "ainex" / "ainex_physics.xml"


def print_actuator_map(model: mujoco.MjModel):
    print("\n=== MuJoCo actuator order (index -> actuator name -> joint) ===")
    for i in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        j_id = int(model.actuator_trnid[i, 0])
        j_name = None
        if j_id >= 0:
            j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id)
        print(f"{i:2d}  {act_name:18s}  joint={j_name}")
    print("=============================================================\n")


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    print_actuator_map(model)

    # Use your engine (so we're testing in the same way your system runs)
    engine = ActionGroupEngine(
        model,
        data,
        base_z=0.25,
        settle_seconds=0.8,
        transition_ms=400,
        abort_min_z=0.08,
        abort_min_up_z=0.15,
        sleep=True,
    )

    # Neutral CSV servo frame (you can change this if needed)
    # Many action-group formats treat ~500 as "neutral"
    neutral = np.full(engine.NUM_CSV_SERVOS, 500.0, dtype=float)

    # Start robot in a consistent pose using neutral
    engine.set_pose_from_frame(neutral)
    engine.settle()

    # Test parameters
    delta = 250.0  # how far from neutral to push one servo
    servo_idx = 0  # 0-based index (Servo1 is index 0)

    print("Controls:")
    print("  [ and ]  = previous / next servo")
    print("  - and =  = decrease / increase delta")
    print("  n      = reset to neutral (all servos at 500)")
    print("  p      = pulse current servo once (neutral -> +delta -> neutral)")
    print("  q      = quit")
    print()

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        while viewer.is_running():
            # Show which servo we are testing
            print(
                f"\rTesting Servo{servo_idx+1:02d}  delta={delta:.0f}  (press p to pulse)   ",
                end="",
                flush=True,
            )

            # Poll keyboard via viewer's key callbacks: we don't have direct callbacks,
            # so we'll use a simple approach: check viewer.opt flags? Not exposed.
            # Instead, we "pulse" using a time-based approach and let you edit quickly:
            # We'll simulate key input by reading from stdin if available.
            #
            # Practical approach: use simple stdin polling every loop.
            # If stdin blocks in your environment, run in terminal and press Enter after keys.

            # Step the sim a bit to keep it alive
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

            # Non-blocking-ish input: only read when a full line is entered.
            # You type a key then Enter.
            try:
                if not hasattr(main, "_last_read"):
                    main._last_read = ""
                # If there's buffered input in some terminals, this works fine.
                cmd = input("\nCommand ([ ] - = n p q): ").strip()
            except (EOFError, KeyboardInterrupt):
                break

            if cmd == "q":
                break
            elif cmd == "[":
                servo_idx = (servo_idx - 1) % engine.NUM_CSV_SERVOS
            elif cmd == "]":
                servo_idx = (servo_idx + 1) % engine.NUM_CSV_SERVOS
            elif cmd == "-":
                delta = max(10.0, delta - 25.0)
            elif cmd == "=":
                delta = min(450.0, delta + 25.0)
            elif cmd == "n":
                engine.set_pose_from_frame(neutral)
                engine.settle(viewer)
            elif cmd == "p":
                # Pulse: neutral -> target -> neutral, using ramps so it doesn't explode
                target = neutral.copy()
                target[servo_idx] = float(np.clip(500.0 + delta, 0.0, 1000.0))

                ctrl_neutral = engine.servo_frame_to_ctrl(neutral)
                ctrl_target = engine.servo_frame_to_ctrl(target)

                # print what actuator(s) this servo is mapped to right now
                act_idx = engine.SERVO_TO_ACT[servo_idx]
                act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, act_idx)
                j_id = int(model.actuator_trnid[act_idx, 0])
                j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) if j_id >= 0 else None
                print(f"\nServo{servo_idx+1} -> actuator[{act_idx}]={act_name}, joint={j_name}")

                ok = engine._ramp_ctrl(ctrl_neutral, ctrl_target, 500, viewer=viewer)
                if not ok:
                    print("ABORTED while ramping to target.")
                ok = engine._ramp_ctrl(ctrl_target, ctrl_neutral, 500, viewer=viewer)
                if not ok:
                    print("ABORTED while returning to neutral.")
            else:
                print("Unknown command. Use [, ], -, =, n, p, q.")


if __name__ == "__main__":
    main()