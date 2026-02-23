# scripts/run_sequence.py
from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import mujoco.viewer

# Make repo root importable even when running as a script with mjpython
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.actiongroup_engine import ActionGroupEngine  # noqa: E402


MODEL_PATH = REPO_ROOT / "assets" / "ainex" / "ainex_physics.xml"
AG_DIR = REPO_ROOT / "assets" / "action_groups" / "csv"


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    engine = ActionGroupEngine(
        model,
        data,
        base_z=0.34,
        settle_seconds=1.0,
        transition_ms=750,
        abort_min_z=0.12,
        abort_min_up_z=0.30,
        sleep=True,
    )

    # Load each CSV once (easy to edit / reuse)
    actions = {
        "ready": engine.load_csv(AG_DIR / "walk_ready.csv"),
        "stand": engine.load_csv(AG_DIR / "stand.csv"),
        "step": engine.load_csv(AG_DIR / "forward_step.csv"),
        "turn_l": engine.load_csv(AG_DIR / "turn_left.csv"),
        "turn_r": engine.load_csv(AG_DIR / "turn_right.csv"),
        "wave": engine.load_csv(AG_DIR / "wave.csv"),
        "twist": engine.load_csv(AG_DIR / "twist.csv"),
        "place_block": engine.load_csv(AG_DIR / "place_block.csv"),
        "hurdles": engine.load_csv(AG_DIR / "hurdles.csv"),
    }

    # Edit choreography here by reordering or repeating strings
    sequence = [
        "stand",
        "ready",
        "step",
        "step",
        "step",
        "step",
        "step",
        "step",
        "step",
        "turn_l",
        "turn_l",
        "turn_l",
        "turn_l",
        "step",
        "turn_r",
        "turn_r",
        "turn_r",
        "turn_r",
        "ready",
    ]

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        # Canonical start: READY pose, not whatever pose the model spawns into
        engine.reset_and_settle_to_action(actions["ready"], viewer)

        for name in sequence:
            action = actions[name]
            ok = engine.play_action(action, viewer, loop=False)
            if not ok:
                print(f"ABORTED during: {name}")
                break

            # Optional: return to ready between actions for stability (off by default)
            # if name != "ready":
            #     ok = engine.play_action(actions["ready"], viewer, loop=False)
            #     if not ok:
            #         print(f"ABORTED returning to ready after: {name}")
            #         break

        # Keep viewer alive
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    main()