# AINex Real Robot Policy Runtime (MVP)

This folder bridges your trained SB3 PPO policy to the real AINex robot.

## Files

- `perception_camera.py`
  - Reads snapshot frames from `web_video_server`.
  - Estimates camera-like fields:
    - `ball_visible`, `ball_distance`, `ball_bearing`
    - `goal_visible`, `goal_distance`, `goal_bearing`
- `ros_action_bridge.py`
  - Publishes action-group names to ROS (`/app/set_action`) either locally or through docker.
- `run_policy_real.py`
  - Loads PPO model and executes closed-loop control on robot.
  - Uses same observation key order from `env_interface.json` if present.
- `calibrate_ball_distance.py`
  - Quick scale calibration for ball distance from known test distance.

## Dependencies

Install in the same Python environment where you run these scripts:

```bash
pip install -r scripts/real/requirements_real.txt
```

These scripts are written to be compatible with Python 3.8+.

For offline wheel installs on the robot, put wheel files in a folder such as `/home/pi/wheels` and run:

```bash
python3 -m pip install --no-index --find-links /home/pi/wheels -r /home/pi/ainex_soccer/scripts/real/requirements_real.txt
```

## Quick checks

### 1) Perception-only check

```bash
python3 scripts/real/perception_camera.py \
  --url "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw" \
  --show --steps 200
```

### 2) Distance calibration

Put the ball at a measured distance (example 0.40m from robot camera baseline):

```bash
python3 scripts/real/calibrate_ball_distance.py \
  --url "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw" \
  --true-distance 0.40
```

Then pass the suggested scale to the runtime:

```bash
python3 scripts/real/run_policy_real.py \
  --model policies/<run_name>/final_model.zip \
  --url "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw" \
  --ball-distance-scale <recommended_distance_scale>
```

### 3) Dry-run policy (no ROS publish)

```bash
python3 scripts/real/run_policy_real.py \
  --model policies/<run_name>/final_model.zip \
  --url "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw" \
  --episodes 1 --max-steps 30 \
  --dry-run-actions
```

### 4) Live run (docker ROS bridge)

```bash
python3 scripts/real/run_policy_real.py \
  --model policies/<run_name>/final_model.zip \
  --url "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw" \
  --container-id <ROS_CONTAINER_ID> \
  --episodes 3 --max-steps 80
```

## Notes

- If your robot action-group names differ, pass a JSON mapping via `--action-map-json`.
- If timing is off, pass per-action wait durations with `--action-duration-json`.
- The current runner feeds zeros for proprioception fields (`base_yaw`, velocities, fallen flag).
  - This keeps observation shape compatible with your trained policy.
  - Next iteration should subscribe to IMU/odometry ROS topics and fill those fields with real values.
