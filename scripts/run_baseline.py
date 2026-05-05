from __future__ import annotations

"""
Rule-based baseline for AINex action-group soccer MVP.

How to run:
  mjpython scripts/run_baseline.py --episodes 1 --max-steps 20
  mjpython scripts/run_baseline.py --viewer --episodes 1 --max-steps 10

Notes:
  - Uses the same observation vector returned by `AinexActionEnv` (plus `obs_dict` for readability).
  - By default the env loads `ainex_soccer_task.xml` (if present), which includes a visible MuJoCo ball and goal pane.
  - If the model has no `soccer_ball` body, the env automatically falls back to a virtual ball state.
"""

import argparse
import csv
import math
import sys
from pathlib import Path
from typing import Optional

import numpy as np

# Make repo root importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ainex_env import AinexActionEnv  # noqa: E402


def choose_baseline_action(env: AinexActionEnv, obs: np.ndarray) -> int:
    d = env.obs_to_dict(obs)
    visible = d["ball_visible"] > 0.5
    bearing = d["ball_bearing"]
    distance = d["ball_distance"]
    base_angvel_z = d.get("base_angvel_z", 0.0)

    # Thresholds tuned for "action-group" granularity rather than fine control.
    turn_thresh = math.radians(12.0)
    approach_thresh = 0.20
    kick_bearing_thresh = math.radians(10.0)
    kick_distance_thresh = 0.145
    kick_settle_angvel_thresh = 0.45  # rad/s
    kick_force_after_ready_angvel_thresh = 1.0  # prevent "ready" deadlock
    near_ball_turn_thresh = math.radians(6.0)

    if not visible:
        return env.action_id("turn_left")

    if abs(bearing) > turn_thresh:
        if bearing > 0.0:
            return env.action_id("turn_left")
        return env.action_id("turn_right")

    if distance > approach_thresh:
        return env.action_id("step_forward")

    # Near-ball region: prioritize progress (turn/step) over indefinite READY loops.
    if abs(bearing) > near_ball_turn_thresh:
        if bearing > 0.0:
            return env.action_id("turn_left")
        return env.action_id("turn_right")

    if distance > kick_distance_thresh:
        return env.action_id("step_forward")

    # In kick zone: optional one-step READY for settling, then kick.
    if "ready" in env.action_index:
        ready_id = env.action_id("ready")
        if abs(bearing) <= kick_bearing_thresh and env.last_action_id != ready_id and abs(base_angvel_z) > kick_settle_angvel_thresh:
            return ready_id
        # If we've already used READY once and are still rotating a bit, allow the kick
        # unless angular velocity is extreme. This avoids freezing in repeated READY.
        if env.last_action_id == ready_id and abs(base_angvel_z) > kick_force_after_ready_angvel_thresh:
            return ready_id

    if bearing >= 0.0 and "kick_left" in env.action_index:
        return env.action_id("kick_left")
    if bearing < 0.0 and "kick_right" in env.action_index:
        return env.action_id("kick_right")
    if "kick" in env.action_index:
        return env.action_id("kick")

    # Fallback if no kick action is available.
    return env.action_id("step_forward")


def _open_csv_logger(path: Optional[Path]):
    if path is None:
        return None, None
    path.parent.mkdir(parents=True, exist_ok=True)
    f = path.open("w", newline="")
    writer = csv.DictWriter(
        f,
        fieldnames=[
            "episode",
            "step",
            "action_id",
            "action_name",
            "reward",
            "terminated",
            "truncated",
            "ball_visible",
            "ball_distance",
            "ball_bearing",
            "base_yaw",
            "base_x",
            "base_y",
            "base_z",
            "is_fallen",
            "kick_success",
            "ball_x",
            "ball_y",
            "ball_vx",
            "ball_vy",
            "goal_x",
            "goal_y",
            "goal_visible",
            "goal_distance",
            "goal_bearing",
            "ball_to_goal_distance",
            "ep_return_running",
            "ep_min_ball_distance",
            "ep_kick_attempts",
            "ep_kick_successes",
        ],
    )
    writer.writeheader()
    return f, writer


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a rule-based baseline on AINex action-group env.")
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--episodes", type=int, default=1, help="Number of episodes")
    parser.add_argument("--max-steps", type=int, default=25, help="Max env steps per episode")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument(
        "--log-csv",
        type=Path,
        default=REPO_ROOT / "logs" / "baseline_run.csv",
        help="CSV log output path",
    )
    args = parser.parse_args()

    log_fh, log_writer = _open_csv_logger(args.log_csv)

    env = AinexActionEnv(
        render=args.viewer,
        max_episode_steps=args.max_steps,
        seed=args.seed,
    )

    try:
        with env:
            print(f"[baseline] actions={env.available_actions()}")
            for ep in range(args.episodes):
                obs, info = env.reset(seed=args.seed + ep)
                ep_return = 0.0
                print(
                    f"[ep {ep}] reset "
                    f"ball_xy=({info['ball_xy'][0]:+.3f},{info['ball_xy'][1]:+.3f}) "
                    f"goal_xy=({info['goal_xy'][0]:+.3f},{info['goal_xy'][1]:+.3f}) "
                    f"base_pos=({info['base_pos_xyz'][0]:+.3f},{info['base_pos_xyz'][1]:+.3f},{info['base_pos_xyz'][2]:+.3f})"
                )

                for t in range(args.max_steps):
                    action_id = choose_baseline_action(env, obs)
                    action_name = env.action_names[action_id]
                    obs, reward, terminated, truncated, info = env.step(action_id)
                    ep_return += reward
                    d = info["obs_dict"]
                    base_pos = info["base_pos_xyz"]
                    ball_xy = info["ball_xy"]
                    ball_vel_xy = info["ball_vel_xy"]
                    goal_xy = info["goal_xy"]
                    epm = info.get("episode_metrics", {})

                    print(
                        f"[ep {ep} step {t:02d}] action={action_name:<11s} "
                        f"reward={reward:+.3f} "
                        f"dist={d['ball_distance']:.3f} "
                        f"bearing={d['ball_bearing']:+.3f} "
                        f"visible={int(d['ball_visible'] > 0.5)} "
                        f"goal_d={d.get('goal_distance', float('nan')):.3f} "
                        f"goal_b={d.get('goal_bearing', float('nan')):+.3f} "
                        f"yaw={d['base_yaw']:+.3f} "
                        f"base=({base_pos[0]:+.3f},{base_pos[1]:+.3f},{base_pos[2]:.3f}) "
                        f"fell={int(info['is_fallen'])} "
                        f"kick={int(info.get('kick_success', False))}"
                    )

                    if log_writer is not None:
                        log_writer.writerow(
                            {
                                "episode": ep,
                                "step": t,
                                "action_id": int(action_id),
                                "action_name": action_name,
                                "reward": float(reward),
                                "terminated": int(terminated),
                                "truncated": int(truncated),
                                "ball_visible": int(d["ball_visible"] > 0.5),
                                "ball_distance": float(d["ball_distance"]),
                                "ball_bearing": float(d["ball_bearing"]),
                                "base_yaw": float(d["base_yaw"]),
                                "base_x": float(base_pos[0]),
                                "base_y": float(base_pos[1]),
                                "base_z": float(base_pos[2]),
                                "is_fallen": int(info["is_fallen"]),
                                "kick_success": int(info.get("kick_success", False)),
                                "ball_x": float(ball_xy[0]),
                                "ball_y": float(ball_xy[1]),
                                "ball_vx": float(ball_vel_xy[0]),
                                "ball_vy": float(ball_vel_xy[1]),
                                "goal_x": float(goal_xy[0]),
                                "goal_y": float(goal_xy[1]),
                                "goal_visible": int(d.get("goal_visible", 0.0) > 0.5),
                                "goal_distance": float(d.get("goal_distance", float("nan"))),
                                "goal_bearing": float(d.get("goal_bearing", float("nan"))),
                                "ball_to_goal_distance": float(info.get("ball_to_goal_distance", float("nan"))),
                                "ep_return_running": float(epm.get("episode_return", ep_return)),
                                "ep_min_ball_distance": float(epm.get("min_ball_distance", float("nan"))),
                                "ep_kick_attempts": int(epm.get("kick_attempts", 0)),
                                "ep_kick_successes": int(epm.get("kick_successes", 0)),
                            }
                        )
                        log_fh.flush()

                    if terminated or truncated:
                        status = "terminated" if terminated else "truncated"
                        print(
                            f"[ep {ep}] {status} at step={t} return={ep_return:+.3f} "
                            f"min_ball_dist={epm.get('min_ball_distance', float('nan')):.3f} "
                            f"ball_to_goal_progress={epm.get('ball_to_goal_progress', float('nan')):+.3f} "
                            f"kicks={int(epm.get('kick_successes', 0))}/{int(epm.get('kick_attempts', 0))}"
                        )
                        break
                else:
                    final_epm = info.get("episode_metrics", {}) if "info" in locals() else {}
                    print(
                        f"[ep {ep}] completed max steps return={ep_return:+.3f} "
                        f"min_ball_dist={final_epm.get('min_ball_distance', float('nan')):.3f} "
                        f"ball_to_goal_progress={final_epm.get('ball_to_goal_progress', float('nan')):+.3f}"
                    )
    finally:
        if log_fh is not None:
            log_fh.close()


if __name__ == "__main__":
    main()
