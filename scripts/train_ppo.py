from __future__ import annotations

"""
Train a PPO policy (SB3) to select AINex action groups.

Examples:
  python3 scripts/train_ppo.py --timesteps 50000
  python3 scripts/train_ppo.py --timesteps 200000 --run-name ppo_test_01
  python3 scripts/train_ppo.py --timesteps 100000 --n-envs 4 --device cpu

Outputs:
  - policies/<run_name>/final_model.zip
  - policies/<run_name>/checkpoints/*.zip
  - policies/<run_name>/run_metadata.json
  - runs/sb3_tensorboard/<run_name>/... (TensorBoard logs)
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ainex_env import AinexGymnasiumEnv  # noqa: E402


def _now_tag() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def _make_run_dir(base_dir: Path, run_name: str) -> Path:
    run_dir = base_dir / run_name
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=False)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        return run_dir

    # Auto-suffix if a previous failed or completed run already created the directory.
    idx = 2
    while True:
        candidate = base_dir / f"{run_name}_{idx:02d}"
        if not candidate.exists():
            candidate.mkdir(parents=True, exist_ok=False)
            (candidate / "checkpoints").mkdir(parents=True, exist_ok=True)
            return candidate
        idx += 1


def _build_env_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "render": False,
        "max_episode_steps": args.max_episode_steps,
        "include_goal_obs": not args.no_goal_obs,
        "engine_kwargs": {"verbose": False},
    }
    if args.model_path:
        kwargs["model_path"] = Path(args.model_path)
    return kwargs


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PPO (SB3) on the AINex action-group env.")
    parser.add_argument("--timesteps", type=int, default=50_000, help="Total PPO training timesteps")
    parser.add_argument("--run-name", type=str, default="", help="Run folder name under policies/")
    parser.add_argument("--seed", type=int, default=0, help="RNG seed")
    parser.add_argument("--n-envs", type=int, default=0, help="Parallel env count (0=auto from CPU cores)")
    parser.add_argument("--vec-env", type=str, default="auto", choices=["auto", "dummy", "subproc"], help="Vector env backend")
    parser.add_argument("--max-episode-steps", type=int, default=40, help="Episode truncation horizon")
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--n-steps", type=int, default=512, help="PPO rollout steps per env")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--n-epochs", type=int, default=10)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--clip-range", type=float, default=0.2)
    parser.add_argument("--device", type=str, default="auto", help="SB3 device: auto/cpu/cuda")
    parser.add_argument("--model-path", type=str, default="", help="Optional MuJoCo XML path override")
    parser.add_argument("--no-goal-obs", action="store_true", help="Disable goal observation fields")
    parser.add_argument("--checkpoint-every", type=int, default=10_000, help="Checkpoint interval (timesteps)")
    parser.add_argument("--resume", type=str, default="", help="Path to existing SB3 .zip model to continue training")
    parser.add_argument("--policy-dir", type=Path, default=REPO_ROOT / "policies", help="Base directory for saved policies")
    parser.add_argument("--tb-dir", type=Path, default=REPO_ROOT / "runs" / "sb3_tensorboard", help="TensorBoard log directory")
    args = parser.parse_args()

    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecMonitor
    except ImportError as exc:
        raise SystemExit(
            "Missing RL dependencies. Install with:\n"
            "  pip install stable-baselines3 gymnasium tensorboard"
        ) from exc

    run_name = args.run_name or f"ppo_ainex_{_now_tag()}"
    args.policy_dir.mkdir(parents=True, exist_ok=True)
    args.tb_dir.mkdir(parents=True, exist_ok=True)
    run_dir = _make_run_dir(args.policy_dir, run_name)

    env_kwargs = _build_env_kwargs(args)
    cpu_count = max(1, os.cpu_count() or 1)
    n_envs = int(args.n_envs) if int(args.n_envs) > 0 else cpu_count
    if n_envs < 1:
        n_envs = 1

    def make_env(rank: int):
        def _thunk():
            return AinexGymnasiumEnv(**env_kwargs, seed=args.seed + rank)
        return _thunk

    vec_backend = args.vec_env
    if vec_backend == "auto":
        vec_backend = "subproc" if n_envs > 1 else "dummy"

    if vec_backend == "subproc" and n_envs > 1:
        # macOS uses spawn; explicitly request it for stability/predictability.
        vec_env = SubprocVecEnv([make_env(i) for i in range(n_envs)], start_method="spawn")
    else:
        vec_env = DummyVecEnv([make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)

    policy_kwargs = dict(net_arch=dict(pi=[128, 128], vf=[128, 128]))

    if args.resume:
        model = PPO.load(
            args.resume,
            env=vec_env,
            device=args.device,
            print_system_info=True,
        )
        # Allow CLI overrides on resume for quick iteration.
        model.learning_rate = args.learning_rate
        model.gamma = args.gamma
        model.gae_lambda = args.gae_lambda
        model.n_steps = args.n_steps
        model.batch_size = args.batch_size
        model.n_epochs = args.n_epochs
        model.ent_coef = args.ent_coef
        model.vf_coef = args.vf_coef
        model.clip_range = args.clip_range
    else:
        model = PPO(
            policy="MlpPolicy",
            env=vec_env,
            learning_rate=args.learning_rate,
            gamma=args.gamma,
            gae_lambda=args.gae_lambda,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=args.n_epochs,
            ent_coef=args.ent_coef,
            vf_coef=args.vf_coef,
            clip_range=args.clip_range,
            verbose=1,
            tensorboard_log=str(args.tb_dir),
            device=args.device,
            seed=args.seed,
            policy_kwargs=policy_kwargs,
        )

    checkpoint_callback = CheckpointCallback(
        save_freq=max(1, args.checkpoint_every // max(1, n_envs)),
        save_path=str(run_dir / "checkpoints"),
        name_prefix="ppo_ainex",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    metadata = {
        "run_name": run_name,
        "created_at": _now_tag(),
        "timesteps": args.timesteps,
        "env_kwargs": {
            k: (str(v) if isinstance(v, Path) else v) for k, v in env_kwargs.items()
        },
        "ppo_hparams": {
            "learning_rate": args.learning_rate,
            "gamma": args.gamma,
            "gae_lambda": args.gae_lambda,
            "n_steps": args.n_steps,
            "batch_size": args.batch_size,
            "n_epochs": args.n_epochs,
            "ent_coef": args.ent_coef,
            "vf_coef": args.vf_coef,
            "clip_range": args.clip_range,
            "n_envs": args.n_envs,
            "n_envs_resolved": n_envs,
            "vec_env": vec_backend,
            "cpu_count": cpu_count,
            "device": args.device,
            "resume": args.resume or None,
        },
    }

    with (run_dir / "run_metadata.json").open("w") as f:
        json.dump(metadata, f, indent=2)

    # Save env-facing metadata too (action ordering and observation layout).
    probe_env = AinexGymnasiumEnv(**env_kwargs, seed=args.seed)
    probe_meta = {
        "action_names": probe_env.action_names,
        "obs_keys": probe_env.core.obs_keys,
        "observation_space_shape": [int(x) for x in probe_env.observation_space.shape],
        "action_space_n": int(probe_env.action_space.n),
    }
    probe_env.close()
    with (run_dir / "env_interface.json").open("w") as f:
        json.dump(probe_meta, f, indent=2)

    try:
        model.learn(
            total_timesteps=args.timesteps,
            callback=checkpoint_callback,
            progress_bar=True,
            tb_log_name=run_name,
            reset_num_timesteps=not bool(args.resume),
        )
    finally:
        vec_env.close()

    final_path = run_dir / "final_model"
    model.save(str(final_path))
    print(f"[train_ppo] saved model: {final_path}.zip")
    print(f"[train_ppo] run dir: {run_dir}")
    print(f"[train_ppo] tensorboard: {args.tb_dir}")


if __name__ == "__main__":
    main()
