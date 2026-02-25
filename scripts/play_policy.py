from __future__ import annotations

"""
Load and run a trained SB3 PPO policy in the AINex MuJoCo scene.

Examples:
  python3 scripts/play_policy.py --model policies/<run>/final_model.zip
  python3 scripts/play_policy.py --model policies/<run>/final_model.zip --episodes 3 --viewer
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ainex_env import AinexGymnasiumEnv  # noqa: E402


def _find_latest_model(policy_dir: Path) -> Optional[Path]:
    zips = sorted(policy_dir.glob("**/*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    return zips[0] if zips else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Play a trained PPO policy on the AINex env.")
    parser.add_argument("--model", type=Path, default=None, help="Path to SB3 .zip model")
    parser.add_argument("--policy-dir", type=Path, default=REPO_ROOT / "policies", help="Search base dir if --model not provided")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer")
    parser.add_argument("--stochastic", action="store_true", help="Use stochastic action sampling")
    parser.add_argument("--model-path", type=str, default="", help="Optional MuJoCo XML path override")
    parser.add_argument("--no-goal-obs", action="store_true", help="Disable goal obs features")
    args = parser.parse_args()

    model_path = args.model
    if model_path is None:
        model_path = _find_latest_model(args.policy_dir)
        if model_path is None:
            raise SystemExit(f"No .zip model found under {args.policy_dir}")
        print(f"[play_policy] using latest model: {model_path}")
    if not model_path.exists():
        raise SystemExit(f"Model not found: {model_path}")

    try:
        from stable_baselines3 import PPO
    except ImportError as exc:
        raise SystemExit("Missing stable-baselines3. Install with `pip install stable-baselines3 gymnasium`.") from exc

    env_kwargs = dict(
        render=args.viewer,
        max_episode_steps=args.max_steps,
        include_goal_obs=not args.no_goal_obs,
        seed=args.seed,
    )
    if args.model_path:
        env_kwargs["model_path"] = Path(args.model_path)

    env = AinexGymnasiumEnv(**env_kwargs)
    model = PPO.load(str(model_path))

    # Optional metadata print if present near the model.
    env_meta_path = model_path.parent / "env_interface.json"
    if env_meta_path.exists():
        try:
            env_meta = json.loads(env_meta_path.read_text())
            print(f"[play_policy] action_names={env_meta.get('action_names')}")
            print(f"[play_policy] obs_keys={env_meta.get('obs_keys')}")
        except Exception:
            pass

    try:
        for ep in range(args.episodes):
            obs, info = env.reset(seed=args.seed + ep)
            ep_return = 0.0
            print(
                f"[ep {ep}] reset ball=({info['ball_xy'][0]:+.3f},{info['ball_xy'][1]:+.3f}) "
                f"goal=({info['goal_xy'][0]:+.3f},{info['goal_xy'][1]:+.3f})"
            )
            for t in range(args.max_steps):
                action, _ = model.predict(obs, deterministic=not args.stochastic)
                obs, reward, terminated, truncated, info = env.step(int(action))
                ep_return += float(reward)
                action_name = env.action_names[int(action)]
                d = info["obs_dict"]
                print(
                    f"[ep {ep} step {t:02d}] action={action_name:<11s} "
                    f"r={reward:+.3f} "
                    f"ball_d={d['ball_distance']:.3f} "
                    f"ball_b={d['ball_bearing']:+.3f} "
                    f"goal_d={d.get('goal_distance', float('nan')):.3f} "
                    f"fell={int(info['is_fallen'])}"
                )
                if terminated or truncated:
                    print(
                        f"[ep {ep}] end terminated={terminated} truncated={truncated} "
                        f"return={ep_return:+.3f} metrics={info.get('episode_metrics', {})}"
                    )
                    break
    finally:
        env.close()


if __name__ == "__main__":
    main()
