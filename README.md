# AINex Soccer: Sim-to-Real Policy over Action Groups

This project implements a sim-to-real control pipeline for the AINex humanoid robot.

Features Include:

- MuJoCo simulation tuned for walking/turning/kicking
- policy training with PPO over discrete **action groups** (which are CSV servo sequences)
- policy playback in simulation
- real-robot runtime using onboard camera snapshots and ROS action-group triggering

The target behavior is looks like: approach ball, align, and kick into goal.

## Current Project Status (End-of-Semester Snapshot)

What is working now:

- Action-group playback in MuJoCo through `ActionGroupEngine`
- Gymnasium-compatible RL environment (`AinexGymnasiumEnv`)
- PPO training and policy playback scripts
- Rule-based baseline controller for pipeline validation
- Real runtime stack for:
  - camera-based stand-in perception (`ball_visible/distance/bearing`, `goal_visible/distance/bearing`)
  - ROS action publishing bridge
  - real policy execution loop on AINex host OS
- Gesture demo module with probabilistic response variation (separate from soccer stack)

Known limitations:

- Real performance is sensitive to lighting and HSV tuning for ball/goal colors
- Ball/goal detection is still color-threshold based (not learned detection)
- Existing stock action groups can be inconsistent (especially stepping/turning precision)
- Real proprioceptive inputs are currently simplified in the real policy runner

My recommended continuation plan after this semester:

1. Strengthen recognition robustness (ball/goal) across lighting/background changes.
2. Make improved custom action groups for more consistent locomotion and turning.
3. Retrain PPO end-to-end using the improved action groups.

## Repository Layout

- `assets/ainex/`
  - MuJoCo XML and related robot assets
- `assets/action_groups/csv/`
  - discrete action-group CSVs used by sim and real execution
- `scripts/actiongroup_engine.py`
  - CSV pulse-to-joint conversion and action playback engine
- `scripts/ainex_env.py`
  - core RL environment + Gymnasium wrapper
- `scripts/run_baseline.py`
  - rule-based baseline behavior
- `scripts/train_ppo.py`
  - PPO training entrypoint
- `scripts/play_policy.py`
  - simulation policy replay
- `scripts/real/`
  - real runtime modules:
    - `perception_camera.py`
    - `calibrate_ball_distance.py`
    - `ros_action_bridge.py`
    - `run_policy_real.py`
- `gesture_recognition/`
  - standalone gesture-recognition projects and demo scripts
- `controller/`
  - keyboard action-group teleop controller

## Setup

### 1) Simulation / Training Environment

Use the project `.venv` (or equivalent) with MuJoCo and RL dependencies.
Tip: When making the virtual environment use either python 3.11 or 2.12 because they are more stable.

Minimum:

```bash
pip install mujoco numpy gymnasium stable-baselines3
```

For MuJoCo viewer on macOS, run viewer scripts with `mjpython`.

### 2) Real Runtime (AINex Host OS, outside Docker)

- Run soccer runtime on AINex host Python (outside container).
- Publish actions into ROS using Docker exec bridge.
- If internet is unavailable on robot, use offline wheel install workflow (`wheelhouse/`).

See:

- `scripts/real/README_REAL.md`
- `docs/final_run_commands/real_runtime_workflow.txt`

## Final Validation Command Sets

Saved command files are in:

- `docs/final_run_commands/`

Includes:

- `sim_baseline_and_policy.txt`
- `real_runtime_workflow.txt`
- `final_validation_checklist.txt`

## Reproducible High-Level Workflow

1. Validate simulation behavior with baseline.
2. Train PPO in simulation.
3. Replay policy in simulation.
4. On AINex host:
   - verify camera snapshot feed
   - tune HSV for ball/goal
   - calibrate ball distance scale
   - run policy dry-run
   - run policy live through ROS bridge

## Troubleshooting Notes

- `mujoco.viewer.launch_passive` on macOS requires `mjpython`.
- SB3 model loading errors usually indicate environment mismatch (venv / package versions).
- Real runtime failures are commonly due to:
  - incorrect HSV ranges
  - wrong camera URL/topic
  - missing or wrong ROS container id
  - overly strict kick gating thresholds
