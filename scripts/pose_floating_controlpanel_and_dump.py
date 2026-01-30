import mujoco
import mujoco.viewer
import numpy as np
import sys

# run with: mjpython scripts/pose_floating_controlpanel_and_dump.py

MODEL_PATH = "assets/ainex/ainex_stable.xml"

def build_actuator_to_qpos_map(model: mujoco.MjModel):
    """
    Returns list of (actuator_name, joint_name, qpos_adr) for actuators that target joints.
    Works well for <position joint="..."> actuators.
    """
    mapping = []
    for a in range(model.nu):
        act_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) or f"act{a}"

        # actuator_trnid[a, 0] is typically the joint id for joint actuators
        j_id = int(model.actuator_trnid[a, 0])
        if j_id < 0:
            continue

        j_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j_id) or f"joint{j_id}"
        qadr = int(model.jnt_qposadr[j_id])

        # Skip free joint / multi-dof joints just in case (not expected here)
        j_type = int(model.jnt_type[j_id])
        # mjJNT_HINGE=3, mjJNT_SLIDE=2, mjJNT_BALL=1, mjJNT_FREE=0
        if j_type in (0, 1):  # FREE or BALL
            continue

        mapping.append((act_name, j_name, qadr))
    return mapping

def main():
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    data = mujoco.MjData(model)

    print(f"Loaded: {MODEL_PATH}")
    print(f"nq: {model.nq} nv: {model.nv} nu: {model.nu}")
    print("Floating pose mode: NO PHYSICS STEPPING (robot will not fall).")
    print("Use the viewer right panel: 'Control' sliders (these set data.ctrl).")
    print("Press Enter in this terminal anytime to dump qpos/qvel.\n")

    # Map actuator sliders -> joint qpos addresses
    act_map = build_actuator_to_qpos_map(model)
    if not act_map:
        print("No joint actuators found to drive pose.")
        print("Make sure your XML has <actuator><position joint='...'>...</position></actuator> entries.")
        sys.exit(1)

    # Initialize ctrl to current joint positions so sliders start “where the robot is”
    for a, (_, _, qadr) in enumerate(act_map):
        if a < model.nu:
            data.ctrl[a] = data.qpos[qadr]

    # Helpful printout
    print("Actuator -> Joint mapping (using Control sliders):")
    for i, (act_name, j_name, qadr) in enumerate(act_map):
        print(f"  ctrl[{i:02d}] {act_name} -> {j_name} (qpos[{qadr}])")
    print()

    # Launch passive viewer (macOS requires running under mjpython)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        # We want to keep it “frozen” unless you change sliders:
        # loop: copy ctrl -> qpos, forward kinematics, sync visuals + pull GUI inputs
        np.set_printoptions(suppress=True, precision=6, linewidth=200)

        while viewer.is_running():
            # Pull UI changes into data (includes ctrl sliders)
            viewer.sync()

            # Apply the control slider values as desired joint angles (qpos)
            # and keep velocities zero so it stays “still”
            for a, (_, _, qadr) in enumerate(act_map):
                if a < model.nu:
                    data.qpos[qadr] = float(data.ctrl[a])
                    data.qvel[qadr] = 0.0  # hinge joints: qvel index aligns with qpos adr for this model

            mujoco.mj_forward(model, data)

            # Terminal "press Enter to dump" without blocking the viewer:
            # Simple approach: check stdin non-blocking by using select (mac/linux).
            import select
            if select.select([sys.stdin], [], [], 0.0)[0]:
                _ = sys.stdin.readline()
                print("\nqpos length:", model.nq)
                print("qpos:\n", " ".join([f"{x:.6f}" for x in data.qpos]))
                print("\nqvel length:", model.nv)
                print("qvel:\n", " ".join([f"{x:.6f}" for x in data.qvel]))
                print()

if __name__ == "__main__":
    main()