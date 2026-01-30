"""
Minimal stable viewer for the AINex MuJoCo model.

What it does:
- Loads assets/ainex/ainex_stable.xml
- Opens MuJoCo's viewer with the right-side Control panel
- Uses the Control sliders (data.ctrl) to directly pose hinge joints
- Does NOT run physics stepping (robot will not fall)

How to run (from repo root):
mjpython scripts/view_ainex_stable.py
"""

from pathlib import Path

import mujoco
import mujoco.viewer


def build_actuator_to_qpos_map(model: mujoco.MjModel):
    """
    Build a mapping from each actuator index -> the joint qpos address it controls.

    This expects joint actuators like:
      <actuator>
        <position joint="r_hip_yaw" kp="50"/>
        ...
      </actuator>

    For each actuator, we find:
    - which joint it targets
    - that joint's qpos address in data.qpos
    """
    mapping = []
    for a in range(model.nu):
        # actuator_trnid[a, 0] is typically the joint id (for joint actuators)
        j_id = int(model.actuator_trnid[a, 0])
        if j_id < 0:
            continue

        # qpos address for this joint
        qadr = int(model.jnt_qposadr[j_id])

        # Only support 1-DOF joints (hinge/slide). Skip FREE/BALL.
        j_type = int(model.jnt_type[j_id])  # FREE=0, BALL=1, SLIDE=2, HINGE=3
        if j_type in (0, 1):
            continue

        mapping.append((a, qadr))
    return mapping


def main():
    # Resolve project root regardless of where the script is run from.
    root = Path(__file__).resolve().parents[1]
    model_path = root / "assets" / "ainex" / "ainex_stable.xml"

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    # Map each actuator to the qpos slot we want to set from its slider.
    act_map = build_actuator_to_qpos_map(model)
    if not act_map:
        raise RuntimeError(
            "No joint actuators found. Ensure your XML has <actuator><position joint='...'> entries."
        )

    # Initialize sliders to match current qpos so the model doesn't jump at start.
    for a, qadr in act_map:
        data.ctrl[a] = data.qpos[qadr]

    # Launch viewer. We do not step physics; we only forward the kinematics.
    with mujoco.viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        while viewer.is_running():
            # Pull UI changes (including control sliders) into `data`.
            viewer.sync()

            # Apply each control slider value directly to the corresponding joint angle.
            # This "poses" the robot and keeps it stable (no falling).
            for a, qadr in act_map:
                data.qpos[qadr] = float(data.ctrl[a])

            # Recompute forward kinematics with the updated joint positions.
            mujoco.mj_forward(model, data)


if __name__ == "__main__":
    main()