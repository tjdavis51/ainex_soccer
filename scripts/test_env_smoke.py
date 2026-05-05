from __future__ import annotations

"""
Smoke tests for the AINex action-group environment.

Examples:
  mjpython scripts/test_env_smoke.py
  mjpython scripts/test_env_smoke.py --episodes 3 --steps 8
  mjpython scripts/test_env_smoke.py --gym
"""

import argparse
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ainex_env import AinexActionEnv, AinexGymnasiumEnv  # noqa: E402


def _assert_finite(name: str, arr) -> None:
    a = np.asarray(arr, dtype=float)
    if not np.all(np.isfinite(a)):
        raise AssertionError(f"{name} contains non-finite values: {a}")


def run_core_smoke(episodes: int, steps: int, seed: int) -> None:
    env = AinexActionEnv(render=False, max_episode_steps=steps, seed=seed, include_goal_obs=True)
    with env:
        print(f"[core] actions={env.available_actions()} obs_keys={env.obs_keys}")
        for ep in range(episodes):
            obs, info = env.reset(seed=seed + ep)
            _assert_finite("reset obs", obs)
            d = info["obs_dict"]
            print(
                f"[core ep {ep}] reset ball=({info['ball_xy'][0]:+.3f},{info['ball_xy'][1]:+.3f}) "
                f"goal=({info['goal_xy'][0]:+.3f},{info['goal_xy'][1]:+.3f}) "
                f"ball_d={d['ball_distance']:.3f} goal_d={d.get('goal_distance', math.nan):.3f}"
            )

            for t in range(steps):
                action_id = t % len(env.action_names)
                obs, reward, terminated, truncated, info = env.step(action_id)
                _assert_finite("step obs", obs)
                _assert_finite("reward", [reward])
                if info["obs_dict"]["ball_distance"] < 0.0:
                    raise AssertionError("ball_distance should never be negative")
                if "goal_distance" in info["obs_dict"] and info["obs_dict"]["goal_distance"] < 0.0:
                    raise AssertionError("goal_distance should never be negative")
                if abs(info["obs_dict"]["ball_bearing"]) > math.pi + 1e-6:
                    raise AssertionError("ball_bearing out of range [-pi, pi]")

                print(
                    f"[core ep {ep} step {t}] a={env.action_names[action_id]} r={reward:+.3f} "
                    f"ball_d={info['obs_dict']['ball_distance']:.3f} "
                    f"goal_d={info['obs_dict'].get('goal_distance', math.nan):.3f} "
                    f"fallen={int(info['is_fallen'])}"
                )
                if terminated or truncated:
                    print(
                        f"[core ep {ep}] end terminated={terminated} truncated={truncated} "
                        f"metrics={info.get('episode_metrics', {})}"
                    )
                    break


def run_gym_smoke(seed: int) -> None:
    try:
        env = AinexGymnasiumEnv(render=False, max_episode_steps=4, seed=seed, include_goal_obs=True)
    except ImportError as exc:
        print(f"[gym] skipped: {exc}")
        return

    obs, info = env.reset(seed=seed)
    _assert_finite("gym reset obs", obs)
    print(
        f"[gym] obs_shape={obs.shape} obs_space={env.observation_space.shape} "
        f"action_n={env.action_space.n}"
    )
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    _assert_finite("gym step obs", obs)
    _assert_finite("gym reward", [reward])
    print(
        f"[gym] one-step action={action} reward={reward:+.3f} "
        f"terminated={terminated} truncated={truncated}"
    )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke test the AINex env and Gymnasium adapter.")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--steps", type=int, default=6)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gym", action="store_true", help="Also test Gymnasium adapter")
    args = parser.parse_args()

    run_core_smoke(episodes=args.episodes, steps=args.steps, seed=args.seed)
    if args.gym:
        run_gym_smoke(seed=args.seed)


if __name__ == "__main__":
    main()
