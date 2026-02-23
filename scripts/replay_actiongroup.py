# scripts/replay_actiongroup.py
from __future__ import annotations

from pathlib import Path
import sys

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.actiongroup_engine import ActionGroupEngine

MODEL_PATH = REPO_ROOT / "assets" / "ainex" / "ainex_physics.xml"
DEFAULT_CSV = REPO_ROOT / "assets" / "action_groups" / "csv" / "forward_one_step.csv"


def replay_actiongroup(csv_path: Path):
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)

    engine = ActionGroupEngine(
        model,
        data,
        base_z=0.05,
        settle_seconds=1.0,
        transition_ms=1000,
        abort_min_z=0.12,
        abort_min_up_z=0.30,
        sleep=True,
        safety_enabled=False,
    )

    action = engine.load_csv(csv_path)

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        ok = engine.reset_and_settle_to_action(action, viewer)
        if ok:
            ok = engine.play_action(action, viewer, loop=False)

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()


if __name__ == "__main__":
    replay_actiongroup(DEFAULT_CSV)
