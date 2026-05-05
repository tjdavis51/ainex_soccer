from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.real.perception_camera import CameraIntrinsics, RealCameraPerception, RealPerceptionConfig
from scripts.real.ros_action_bridge import ROSActionBridge, RosBridgeConfig


DEFAULT_OBS_KEYS = [
    "base_yaw",
    "base_linvel_x",
    "base_linvel_y",
    "base_angvel_z",
    "last_action_id",
    "ball_visible",
    "ball_distance",
    "ball_bearing",
    "is_fallen",
    "goal_visible",
    "goal_distance",
    "goal_bearing",
    "ball_to_goal_distance",
]

DEFAULT_ACTION_NAMES = [
    "stand",
    "ready",
    "step_forward",
    "turn_left",
    "turn_right",
    "kick_left",
    "kick_right",
    "kick",
]

DEFAULT_ACTION_MAP = {
    "stand": "stand",
    "ready": "walk_ready",
    "step_forward": "go_forward_low",
    "turn_left": "go_turn_left",
    "turn_right": "go_turn_right",
    "kick_left": "left_shot",
    "kick_right": "right_shot",
    "kick": "left_shot",
}

DEFAULT_ACTION_DURATION_S = {
    "stand": 1.0,
    "ready": 1.0,
    "step_forward": 1.6,
    "turn_left": 1.4,
    "turn_right": 1.4,
    "kick_left": 1.8,
    "kick_right": 1.8,
    "kick": 1.8,
}


def _load_json(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    return json.loads(path.read_text())


def _load_env_interface(model_path: Path) -> Tuple[List[str], List[str]]:
    meta_path = model_path.parent / "env_interface.json"
    if not meta_path.exists():
        return list(DEFAULT_OBS_KEYS), list(DEFAULT_ACTION_NAMES)
    payload = json.loads(meta_path.read_text())
    obs_keys = payload.get("obs_keys", DEFAULT_OBS_KEYS)
    action_names = payload.get("action_names", DEFAULT_ACTION_NAMES)
    return list(obs_keys), list(action_names)


def _estimate_ball_to_goal_distance(obs: Dict[str, Union[float, bool]], prev_value: float) -> float:
    ball_visible = bool(obs.get("ball_visible", False))
    goal_visible = bool(obs.get("goal_visible", False))
    if not (ball_visible and goal_visible):
        return float(prev_value)

    db = float(obs["ball_distance"])
    dg = float(obs["goal_distance"])
    ab = float(obs["ball_bearing"])
    ag = float(obs["goal_bearing"])

    delta = ab - ag
    value_sq = db * db + dg * dg - 2.0 * db * dg * math.cos(delta)
    return float(math.sqrt(max(0.0, value_sq)))


def _build_policy_obs(
    *,
    obs_keys: List[str],
    perception_obs: Dict[str, Union[float, bool]],
    last_action_id: int,
    n_actions: int,
    ball_to_goal_distance: float,
) -> np.ndarray:
    payload = {
        "base_yaw": 0.0,
        "base_linvel_x": 0.0,
        "base_linvel_y": 0.0,
        "base_angvel_z": 0.0,
        "last_action_id": -1.0,
        "ball_visible": 1.0 if bool(perception_obs.get("ball_visible", False)) else 0.0,
        "ball_distance": float(perception_obs.get("ball_distance", 0.0)),
        "ball_bearing": float(perception_obs.get("ball_bearing", 0.0)),
        "is_fallen": 0.0,
        "goal_visible": 1.0 if bool(perception_obs.get("goal_visible", False)) else 0.0,
        "goal_distance": float(perception_obs.get("goal_distance", 0.0)),
        "goal_bearing": float(perception_obs.get("goal_bearing", 0.0)),
        "ball_to_goal_distance": float(ball_to_goal_distance),
    }

    if last_action_id >= 0 and n_actions > 1:
        payload["last_action_id"] = float(last_action_id) / float(n_actions - 1)

    vec = np.array([float(payload.get(k, 0.0)) for k in obs_keys], dtype=np.float32)
    return vec


def _should_gate_kick(
    action_name: str,
    obs: Dict[str, Union[float, bool]],
    dist_thresh: float,
    bearing_thresh_rad: float,
) -> bool:
    if "kick" not in action_name:
        return False
    if not bool(obs.get("ball_visible", False)):
        return True
    if float(obs.get("ball_distance", 10.0)) > dist_thresh:
        return True
    if abs(float(obs.get("ball_bearing", 10.0))) > bearing_thresh_rad:
        return True
    return False


def _fallback_action(obs: Dict[str, Union[float, bool]]) -> str:
    if not bool(obs.get("ball_visible", False)):
        return "turn_left"
    bearing = float(obs.get("ball_bearing", 0.0))
    dist = float(obs.get("ball_distance", 10.0))
    if bearing > 0.15:
        return "turn_left"
    if bearing < -0.15:
        return "turn_right"
    if dist > 0.16:
        return "step_forward"
    return "kick"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run trained PPO policy on real AINex using snapshot camera + ROS action topic."
    )
    parser.add_argument("--model", type=Path, required=True, help="Path to trained SB3 .zip model")
    parser.add_argument("--url", type=str, required=True, help="Snapshot URL from web_video_server")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--stochastic", action="store_true")

    parser.add_argument("--container-id", type=str, default="", help="ROS docker container id (empty = local ROS shell)")
    parser.add_argument("--ros-topic", type=str, default="/app/set_action")
    parser.add_argument("--ros-setup-script", type=str, default="/opt/ros/noetic/setup.bash")
    parser.add_argument("--dry-run-actions", action="store_true", help="Do not publish actions, print only")

    parser.add_argument("--action-map-json", type=Path, default=None, help="JSON map from policy action names to robot action-group names")
    parser.add_argument("--action-duration-json", type=Path, default=None, help="JSON map from policy action names to wait time seconds")

    parser.add_argument("--kick-gate-distance", type=float, default=0.17)
    parser.add_argument("--kick-gate-bearing-deg", type=float, default=12.0)

    parser.add_argument("--fx", type=float, default=520.0)
    parser.add_argument("--fy", type=float, default=520.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--ball-diameter", type=float, default=0.08)
    parser.add_argument("--goal-inner-width", type=float, default=0.60)
    parser.add_argument("--ball-distance-scale", type=float, default=1.0, help="Multiply estimated ball distance by this factor")
    parser.add_argument("--goal-distance-scale", type=float, default=1.0, help="Multiply estimated goal distance by this factor")

    parser.add_argument("--ball-hsv-low", type=str, default="5,90,60")
    parser.add_argument("--ball-hsv-high", type=str, default="25,255,255")
    parser.add_argument("--ball-hsv-alt-low", type=str, default="", help="Optional second ball HSV range low h,s,v")
    parser.add_argument("--ball-hsv-alt-high", type=str, default="", help="Optional second ball HSV range high h,s,v")
    parser.add_argument("--goal-hsv-low", type=str, default="95,60,60")
    parser.add_argument("--goal-hsv-high", type=str, default="130,255,255")
    parser.add_argument("--ball-min-confidence", type=float, default=0.40)
    parser.add_argument("--ball-min-circularity", type=float, default=0.35)
    parser.add_argument("--ball-max-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--no-ball-tracking", action="store_true")
    parser.add_argument("--ball-max-center-jump", type=float, default=220.0)

    parser.add_argument("--csv-log", type=Path, default=None, help="Optional rollout log path")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Missing stable-baselines3. Install in this environment first.") from exc

    if not args.model.exists():
        raise SystemExit(f"Model not found: {args.model}")

    model = PPO.load(str(args.model))
    obs_keys, action_names = _load_env_interface(args.model)

    action_map = dict(DEFAULT_ACTION_MAP)
    action_map.update(_load_json(args.action_map_json))

    action_duration = dict(DEFAULT_ACTION_DURATION_S)
    action_duration.update(_load_json(args.action_duration_json))

    def parse_hsv(s: str) -> Tuple[int, int, int]:
        vals = [int(v.strip()) for v in s.split(",")]
        if len(vals) != 3:
            raise ValueError("HSV must be h,s,v")
        return (vals[0], vals[1], vals[2])

    if args.ball_hsv_alt_low and args.ball_hsv_alt_high:
        ball_alt_low = parse_hsv(args.ball_hsv_alt_low)
        ball_alt_high = parse_hsv(args.ball_hsv_alt_high)
    else:
        ball_alt_low = (-1, -1, -1)
        ball_alt_high = (-1, -1, -1)

    perception_cfg = RealPerceptionConfig(
        snapshot_url=args.url,
        ball_diameter_m=float(args.ball_diameter),
        goal_inner_width_m=float(args.goal_inner_width),
        ball_hsv_low=parse_hsv(args.ball_hsv_low),
        ball_hsv_high=parse_hsv(args.ball_hsv_high),
        ball_hsv_alt_low=ball_alt_low,
        ball_hsv_alt_high=ball_alt_high,
        goal_hsv_low=parse_hsv(args.goal_hsv_low),
        goal_hsv_high=parse_hsv(args.goal_hsv_high),
        ball_min_confidence=float(args.ball_min_confidence),
        ball_min_circularity=float(args.ball_min_circularity),
        ball_max_aspect_ratio=float(args.ball_max_aspect_ratio),
        ball_use_tracking=bool(not args.no_ball_tracking),
        ball_max_center_jump_px=float(args.ball_max_center_jump),
    )
    intr = CameraIntrinsics(fx_px=args.fx, fy_px=args.fy, cx_px=args.cx, cy_px=args.cy)
    perception = RealCameraPerception(perception_cfg, intrinsics=intr)

    bridge = ROSActionBridge(
        RosBridgeConfig(
            container_id=args.container_id,
            topic=args.ros_topic,
            ros_setup_script=args.ros_setup_script,
            dry_run=bool(args.dry_run_actions),
        )
    )

    log_writer = None
    log_file = None
    if args.csv_log is not None:
        args.csv_log.parent.mkdir(parents=True, exist_ok=True)
        log_file = args.csv_log.open("w", newline="")
        log_writer = csv.DictWriter(
            log_file,
            fieldnames=[
                "episode",
                "step",
                "ball_visible",
                "ball_distance",
                "ball_bearing",
                "goal_visible",
                "goal_distance",
                "goal_bearing",
                "ball_to_goal_distance",
                "policy_action",
                "executed_action",
                "robot_action_name",
                "gated",
                "publish_ok",
            ],
        )
        log_writer.writeheader()

    kick_gate_bearing_rad = math.radians(float(args.kick_gate_bearing_deg))

    print(f"[real_run] action_names={action_names}")
    print(f"[real_run] obs_keys={obs_keys}")

    try:
        for ep in range(int(args.episodes)):
            print(f"[ep {ep}] start")
            last_action_id = -1
            ball_to_goal_distance = 0.0

            for t in range(int(args.max_steps)):
                obs_cam = perception.observe()
                obs_cam["ball_distance"] = float(obs_cam["ball_distance"]) * float(args.ball_distance_scale)
                obs_cam["goal_distance"] = float(obs_cam["goal_distance"]) * float(args.goal_distance_scale)
                ball_to_goal_distance = _estimate_ball_to_goal_distance(obs_cam, ball_to_goal_distance)

                obs_vec = _build_policy_obs(
                    obs_keys=obs_keys,
                    perception_obs=obs_cam,
                    last_action_id=last_action_id,
                    n_actions=max(1, len(action_names)),
                    ball_to_goal_distance=ball_to_goal_distance,
                )

                action_id, _ = model.predict(obs_vec, deterministic=not bool(args.stochastic))
                action_id = int(action_id)
                if action_id < 0 or action_id >= len(action_names):
                    print(f"[ep {ep} step {t}] invalid action_id={action_id}, clamping")
                    action_id = max(0, min(action_id, len(action_names) - 1))

                policy_action = action_names[action_id]
                executed_action = policy_action
                gated = False

                if _should_gate_kick(policy_action, obs_cam, args.kick_gate_distance, kick_gate_bearing_rad):
                    executed_action = _fallback_action(obs_cam)
                    gated = True

                robot_action_name = action_map.get(executed_action, executed_action)
                publish_ok = bridge.send_action(robot_action_name)

                print(
                    f"[ep {ep} step {t:02d}] "
                    f"ball_d={float(obs_cam['ball_distance']):.3f} "
                    f"ball_b={float(obs_cam['ball_bearing']):+.3f} "
                    f"goal_d={float(obs_cam['goal_distance']):.3f} "
                    f"policy={policy_action:<11s} exec={executed_action:<11s} "
                    f"robot={robot_action_name:<16s} gated={int(gated)} ok={int(publish_ok)}"
                )

                if log_writer is not None:
                    log_writer.writerow(
                        {
                            "episode": ep,
                            "step": t,
                            "ball_visible": int(bool(obs_cam["ball_visible"])),
                            "ball_distance": float(obs_cam["ball_distance"]),
                            "ball_bearing": float(obs_cam["ball_bearing"]),
                            "goal_visible": int(bool(obs_cam["goal_visible"])),
                            "goal_distance": float(obs_cam["goal_distance"]),
                            "goal_bearing": float(obs_cam["goal_bearing"]),
                            "ball_to_goal_distance": float(ball_to_goal_distance),
                            "policy_action": policy_action,
                            "executed_action": executed_action,
                            "robot_action_name": robot_action_name,
                            "gated": int(gated),
                            "publish_ok": int(bool(publish_ok)),
                        }
                    )

                last_action_id = action_names.index(executed_action) if executed_action in action_names else action_id
                time.sleep(float(action_duration.get(executed_action, 1.2)))

    finally:
        if log_file is not None:
            log_file.close()


if __name__ == "__main__":
    main()
