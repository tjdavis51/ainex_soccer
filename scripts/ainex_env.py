from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import mujoco
import numpy as np
try:
    import gymnasium as gym
except ImportError:  # Optional unless using the Gymnasium adapter
    gym = None

# Make repo root importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.actiongroup_engine import ActionGroup, ActionGroupEngine  # noqa: E402
from scripts.perception_standin import VisionStandIn, VisionStandInConfig  # noqa: E402


DEFAULT_MODEL_PATH = (
    REPO_ROOT / "assets" / "ainex" / "ainex_soccer_task.xml"
    if (REPO_ROOT / "assets" / "ainex" / "ainex_soccer_task.xml").exists()
    else REPO_ROOT / "assets" / "ainex" / "ainex_physics.xml"
)
DEFAULT_ACTION_DIR = REPO_ROOT / "assets" / "action_groups" / "csv"


@dataclass
class RewardConfig:
    dist_progress_scale: float = 4.0
    bearing_progress_scale: float = 0.5
    approach_close_distance_m: float = 0.22
    approach_close_scale: float = 0.20
    goal_progress_scale: float = 10.0
    kick_bonus: float = 4.0
    goal_bonus: float = 25.0
    fall_penalty: float = 8.0
    time_penalty: float = 0.05


@dataclass
class BallConfig:
    spawn_x_range: tuple[float, float] = (0.26, 0.40)
    spawn_y_range: tuple[float, float] = (-0.08, 0.08)
    spawn_z_m: float = 0.04
    radius_m: float = 0.04
    align_to_goal_y: bool = True
    goal_y_offset_range: tuple[float, float] = (-0.05, 0.05)
    kick_distance_thresh_m: float = 0.18
    kick_bearing_thresh_rad: float = math.radians(25.0)
    kick_speed_mps: float = 1.2
    drag_per_second: float = 1.8
    success_move_thresh_m: float = 0.35  # retained as a debug metric; not terminal reward target


@dataclass
class GoalConfig:
    center_x_range: tuple[float, float] = (1.38, 1.48)
    center_y_range: tuple[float, float] = (-0.05, 0.05)
    center_z_m: float = 0.18
    width_half_m: float = 0.30
    height_half_m: float = 0.18
    visual_goal_pane_name: str = "goal_pane"
    visual_goal_left_post_name: str = "goal_left_post"
    visual_goal_right_post_name: str = "goal_right_post"
    visual_goal_crossbar_name: str = "goal_crossbar"


class AinexActionEnv:
    """
    Minimal RL-style wrapper around MuJoCo + ActionGroupEngine.

    API:
      reset() -> (obs, info)
      step(action_id) -> (obs, reward, terminated, truncated, info)
    """

    BASE_OBS_KEYS = [
        "base_yaw",
        "base_linvel_x",
        "base_linvel_y",
        "base_angvel_z",
        "last_action_id",
        "ball_visible",
        "ball_distance",
        "ball_bearing",
        "is_fallen",
    ]
    GOAL_OBS_KEYS = [
        "goal_visible",
        "goal_distance",
        "goal_bearing",
        "ball_to_goal_distance",
    ]

    def __init__(
        self,
        *,
        model_path: Path | str = DEFAULT_MODEL_PATH,
        action_dir: Path | str = DEFAULT_ACTION_DIR,
        render: bool = False,
        max_episode_steps: int = 60,
        seed: Optional[int] = None,
        engine_kwargs: Optional[dict[str, Any]] = None,
        vision_config: Optional[VisionStandInConfig] = None,
        reward_config: Optional[RewardConfig] = None,
        ball_config: Optional[BallConfig] = None,
        goal_config: Optional[GoalConfig] = None,
        ball_body_name: Optional[str] = "soccer_ball",
        include_goal_obs: bool = True,
    ):
        self.model_path = Path(model_path)
        self.action_dir = Path(action_dir)
        self.render = bool(render)
        self.max_episode_steps = int(max_episode_steps)
        self.rng = np.random.default_rng(seed)
        self.ball_body_name = ball_body_name
        self.include_goal_obs = bool(include_goal_obs)

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)

        engine_defaults = dict(
            base_z=0.34,
            settle_seconds=0.7,
            transition_ms=500,
            abort_min_z=0.12,
            abort_min_up_z=0.30,
            sleep=False,
            safety_enabled=False,
        )
        if engine_kwargs:
            engine_defaults.update(engine_kwargs)
        self.engine = ActionGroupEngine(self.model, self.data, **engine_defaults)

        self.vision = VisionStandIn(self.model, self.data, config=vision_config or VisionStandInConfig())
        self.reward_cfg = reward_config or RewardConfig()
        self.ball_cfg = ball_config or BallConfig()
        self.goal_cfg = goal_config or GoalConfig()
        self._ball_body_id = self._try_body_id(self.ball_body_name) if self.ball_body_name else None
        self._ball_freejoint_qposadr = None
        self._ball_freejoint_dofadr = None
        if self._ball_body_id is not None:
            self._init_ball_freejoint_handles(self._ball_body_id)
        elif self.ball_body_name:
            print(
                f"[ainex_env] ball body '{self.ball_body_name}' not found in model "
                f"{self.model_path.name}; falling back to virtual ball."
            )
            self.ball_body_name = None

        self.actions = self._load_default_actions()
        self.action_names = list(self.actions.keys())
        self.action_index = {name: i for i, name in enumerate(self.action_names)}

        self._viewer_ctx = None
        self.viewer = None
        self._goal_geom_ids: dict[str, Optional[int]] = {
            "pane": self._try_geom_id(self.goal_cfg.visual_goal_pane_name),
            "left": self._try_geom_id(self.goal_cfg.visual_goal_left_post_name),
            "right": self._try_geom_id(self.goal_cfg.visual_goal_right_post_name),
            "crossbar": self._try_geom_id(self.goal_cfg.visual_goal_crossbar_name),
        }

        self.step_count = 0
        self.last_action_id = -1
        self._last_ball_distance = 0.0
        self._last_ball_abs_bearing = 0.0
        self._episode_start_ball_xy = np.zeros(2, dtype=float)
        self._episode_start_ball_to_goal_dist = 0.0
        self.virtual_ball_pos = np.array([0.35, 0.0, 0.0], dtype=float)
        self.virtual_ball_vel_xy = np.zeros(2, dtype=float)
        self.last_kick_success = False
        self.goal_pos_world = np.array([1.4, 0.0, self.goal_cfg.center_z_m], dtype=float)
        self._episode_min_ball_distance = float("inf")
        self._episode_min_abs_bearing = float("inf")
        self._episode_kick_attempts = 0
        self._episode_kick_successes = 0
        self._episode_return = 0.0
        self._episode_goal_scored = 0
        self._reset_count = 0
        # Per-action playback overrides help keep sensitive primitives (especially kicks)
        # from being over-amplified by the engine's global motion scaling.
        self.action_motion_scale_overrides: dict[str, float] = {
            "kick_left": 1.0,
            "kick_right": 1.0,
            "kick": 1.0,
        }
        self.action_transition_ms_overrides: dict[str, int] = {
            "kick_left": max(700, int(self.engine.transition_ms)),
            "kick_right": max(700, int(self.engine.transition_ms)),
            "kick": max(700, int(self.engine.transition_ms)),
        }

    def _try_body_id(self, body_name: Optional[str]) -> Optional[int]:
        if not body_name:
            return None
        body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        return body_id if body_id >= 0 else None

    def _try_geom_id(self, geom_name: Optional[str]) -> Optional[int]:
        if not geom_name:
            return None
        geom_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_name))
        return geom_id if geom_id >= 0 else None

    def _init_ball_freejoint_handles(self, body_id: int) -> None:
        jadr = int(self.model.body_jntadr[body_id])
        jnum = int(self.model.body_jntnum[body_id])
        if jnum < 1:
            raise RuntimeError(f"Ball body id={body_id} has no joint; expected freejoint.")
        j_id = jadr
        if int(self.model.jnt_type[j_id]) != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Ball body must use a freejoint for reset/randomization.")
        self._ball_freejoint_qposadr = int(self.model.jnt_qposadr[j_id])
        self._ball_freejoint_dofadr = int(self.model.jnt_dofadr[j_id])

    def _set_sim_ball_state(self, *, x: float, y: float, z: Optional[float] = None) -> None:
        if self._ball_body_id is None:
            return
        if self._ball_freejoint_qposadr is None or self._ball_freejoint_dofadr is None:
            raise RuntimeError("Sim ball freejoint handles are not initialized.")
        z_val = float(self.ball_cfg.spawn_z_m if z is None else z)
        qadr = self._ball_freejoint_qposadr
        dadr = self._ball_freejoint_dofadr
        self.data.qpos[qadr + 0] = float(x)
        self.data.qpos[qadr + 1] = float(y)
        self.data.qpos[qadr + 2] = z_val
        self.data.qpos[qadr + 3:qadr + 7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.data.qvel[dadr:dadr + 6] = 0.0
        mujoco.mj_forward(self.model, self.data)

    def _set_goal_state(self, *, x: float, y: float, z: Optional[float] = None) -> None:
        z_val = float(self.goal_cfg.center_z_m if z is None else z)
        self.goal_pos_world[:] = np.array([float(x), float(y), z_val], dtype=float)

        pane_id = self._goal_geom_ids.get("pane")
        if pane_id is not None:
            self.model.geom_pos[pane_id, 0:3] = np.array([x, y, z_val], dtype=float)
            self.model.geom_size[pane_id, 1] = float(self.goal_cfg.width_half_m)
            self.model.geom_size[pane_id, 2] = float(self.goal_cfg.height_half_m)

        left_id = self._goal_geom_ids.get("left")
        if left_id is not None:
            self.model.geom_pos[left_id, 0:3] = np.array([x, y + self.goal_cfg.width_half_m + 0.01, z_val], dtype=float)
            self.model.geom_size[left_id, 2] = float(self.goal_cfg.height_half_m)

        right_id = self._goal_geom_ids.get("right")
        if right_id is not None:
            self.model.geom_pos[right_id, 0:3] = np.array([x, y - self.goal_cfg.width_half_m - 0.01, z_val], dtype=float)
            self.model.geom_size[right_id, 2] = float(self.goal_cfg.height_half_m)

        crossbar_id = self._goal_geom_ids.get("crossbar")
        if crossbar_id is not None:
            self.model.geom_pos[crossbar_id, 0:3] = np.array([x, y, z_val + self.goal_cfg.height_half_m + 0.01], dtype=float)
            self.model.geom_size[crossbar_id, 1] = float(self.goal_cfg.width_half_m + 0.01)

        mujoco.mj_forward(self.model, self.data)

    @property
    def obs_keys(self) -> list[str]:
        if self.include_goal_obs:
            return self.BASE_OBS_KEYS + self.GOAL_OBS_KEYS
        return list(self.BASE_OBS_KEYS)

    # ----------------------------
    # Viewer lifecycle
    # ----------------------------
    def __enter__(self) -> "AinexActionEnv":
        self._ensure_viewer()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _ensure_viewer(self) -> None:
        if not self.render or self.viewer is not None:
            return
        import mujoco.viewer

        self._viewer_ctx = mujoco.viewer.launch_passive(
            self.model, self.data, show_left_ui=True, show_right_ui=True
        )
        self.viewer = self._viewer_ctx.__enter__()

    def close(self) -> None:
        if self._viewer_ctx is not None:
            self._viewer_ctx.__exit__(None, None, None)
            self._viewer_ctx = None
            self.viewer = None

    # ----------------------------
    # Action loading
    # ----------------------------
    def _load_default_actions(self) -> dict[str, ActionGroup]:
        candidates = {
            "stand": ["stand", "walk_ready", "stand_low"],
            "ready": ["walk_ready", "stand", "stand_low"],
            "step_forward": ["forward_step", "forward_one_step", "go_forward_low", "forward"],
            "turn_left": ["turn_left", "go_turn_left", "turn_left_30", "go_turn_left_low"],
            "turn_right": ["turn_right", "go_turn_right", "turn_right_30", "go_turn_right_low"],
            "kick_left": ["left_shot"],
            "kick_right": ["right_shot"],
        }
        loaded: dict[str, ActionGroup] = {}
        for name, stems in candidates.items():
            stem = self._first_existing_csv(stems)
            if stem is None:
                continue
            loaded[name] = self.engine.load_csv(self.action_dir / f"{stem}.csv")

        required = ["ready", "step_forward", "turn_left", "turn_right"]
        missing = [k for k in required if k not in loaded]
        if missing:
            raise FileNotFoundError(f"Missing required action groups for env: {missing}")

        # Generic kick alias prefers side-specific kicks if available.
        if "kick_left" in loaded:
            loaded["kick"] = loaded["kick_left"]
        elif "kick_right" in loaded:
            loaded["kick"] = loaded["kick_right"]

        # Keep action order stable and RL-friendly.
        ordered_names = [
            "stand",
            "ready",
            "step_forward",
            "turn_left",
            "turn_right",
            "kick_left",
            "kick_right",
            "kick",
        ]
        return {name: loaded[name] for name in ordered_names if name in loaded}

    def _first_existing_csv(self, stems: list[str]) -> Optional[str]:
        for stem in stems:
            p = self.action_dir / f"{stem}.csv"
            if p.exists():
                return stem
        return None

    # ----------------------------
    # Reset and step
    # ----------------------------
    def reset(self, *, seed: Optional[int] = None) -> tuple[np.ndarray, dict[str, Any]]:
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self._ensure_viewer()

        self.step_count = 0
        self.last_action_id = -1
        self.last_kick_success = False
        self.virtual_ball_vel_xy[:] = 0.0
        self._episode_kick_attempts = 0
        self._episode_kick_successes = 0
        self._episode_return = 0.0
        self._episode_goal_scored = 0
        self._randomize_goal(first_reset=(self._reset_count == 0))

        start_action = self.actions["ready"] if "ready" in self.actions else next(iter(self.actions.values()))
        ok = self.engine.reset_and_settle_to_action(start_action, viewer=self.viewer)
        if not ok:
            raise RuntimeError("Failed to reset and settle robot into initial pose.")
        self._randomize_ball(first_reset=(self._reset_count == 0))

        obs = self._get_observation()
        obs_dict = self.obs_to_dict(obs)
        self._last_ball_distance = float(obs_dict["ball_distance"])
        self._last_ball_abs_bearing = abs(float(obs_dict["ball_bearing"]))
        self._episode_start_ball_xy = self._get_ball_xy().copy()
        self._episode_start_ball_to_goal_dist = self._ball_to_goal_distance()
        self._episode_min_ball_distance = self._last_ball_distance
        self._episode_min_abs_bearing = self._last_ball_abs_bearing

        info = self._build_info(obs)
        info["reset_ok"] = True
        info["reset_index"] = int(self._reset_count)
        self._reset_count += 1
        return obs, info

    def step(self, action_id: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if action_id < 0 or action_id >= len(self.action_names):
            raise ValueError(f"Invalid action_id={action_id}; valid range is [0, {len(self.action_names)-1}]")

        action_name = self.action_names[action_id]
        action = self.actions[action_name]
        prev_ball_xy = self._get_ball_xy().copy()
        prev_ball_to_goal_dist = float(np.linalg.norm(prev_ball_xy - self.goal_pos_world[0:2]))
        if "kick" in action_name:
            self._episode_kick_attempts += 1

        ok = self._play_action_with_overrides(action_name, action)
        self.last_action_id = int(action_id)
        self.step_count += 1

        kicked_now = self._maybe_apply_virtual_kick(action_name)
        self._integrate_virtual_ball(self._estimate_action_duration_s(action))

        obs = self._get_observation()
        obs_dict = self.obs_to_dict(obs)

        is_fallen = bool(obs_dict["is_fallen"] > 0.5)
        dist = float(obs_dict["ball_distance"])
        abs_bearing = abs(float(obs_dict["ball_bearing"]))
        ball_to_goal_dist = self._ball_to_goal_distance()

        reward_dist = self.reward_cfg.dist_progress_scale * (self._last_ball_distance - dist)
        reward_bearing = self.reward_cfg.bearing_progress_scale * (self._last_ball_abs_bearing - abs_bearing)
        if min(self._last_ball_distance, dist) <= self.reward_cfg.approach_close_distance_m:
            reward_dist *= self.reward_cfg.approach_close_scale
            reward_bearing *= self.reward_cfg.approach_close_scale
        reward_goal_progress = self.reward_cfg.goal_progress_scale * (prev_ball_to_goal_dist - ball_to_goal_dist)
        reward_time = -self.reward_cfg.time_penalty
        reward_kick = 0.0
        reward_goal_bonus = 0.0
        reward_fall = 0.0

        ball_moved = float(np.linalg.norm(self._get_ball_xy() - prev_ball_xy))
        episode_ball_moved = float(np.linalg.norm(self._get_ball_xy() - self._episode_start_ball_xy))
        kick_success = bool(("kick" in action_name) and ball_moved > 0.05)
        if kick_success:
            reward_kick += self.reward_cfg.kick_bonus
            self._episode_kick_successes += 1
        goal_scored = self._is_goal_scored()

        terminated = False
        if not ok:
            terminated = True
        if is_fallen:
            terminated = True
            reward_fall -= self.reward_cfg.fall_penalty
        if goal_scored:
            terminated = True
            reward_goal_bonus += self.reward_cfg.goal_bonus
            self._episode_goal_scored = 1

        truncated = self.step_count >= self.max_episode_steps and not terminated

        reward = float(
            reward_dist
            + reward_bearing
            + reward_goal_progress
            + reward_time
            + reward_kick
            + reward_goal_bonus
            + reward_fall
        )

        self._last_ball_distance = dist
        self._last_ball_abs_bearing = abs_bearing
        self._episode_min_ball_distance = min(self._episode_min_ball_distance, dist)
        self._episode_min_abs_bearing = min(self._episode_min_abs_bearing, abs_bearing)
        self.last_kick_success = kick_success
        self._episode_return += reward

        info = self._build_info(obs)
        info.update(
            {
                "action_id": int(action_id),
                "action_name": action_name,
                "engine_ok": bool(ok),
                "ball_moved_this_step": ball_moved,
                "ball_moved_episode": episode_ball_moved,
                "kick_attempted": "kick" in action_name,
                "kick_success": kick_success,
                "goal_scored": goal_scored,
                "reward_dist": float(reward_dist),
                "reward_bearing": float(reward_bearing),
                "reward_goal_progress": float(reward_goal_progress),
                "reward_time": float(reward_time),
                "reward_kick": float(reward_kick),
                "reward_goal_bonus": float(reward_goal_bonus),
                "reward_fall": float(reward_fall),
                "episode_metrics": self._episode_metrics(),
            }
        )
        return obs, reward, terminated, truncated, info

    def _play_action_with_overrides(self, action_name: str, action: ActionGroup) -> bool:
        orig_motion_scale = float(self.engine.motion_scale)
        orig_transition_ms = int(self.engine.transition_ms)
        try:
            if action_name in self.action_motion_scale_overrides:
                self.engine.motion_scale = float(self.action_motion_scale_overrides[action_name])
            if action_name in self.action_transition_ms_overrides:
                self.engine.transition_ms = int(self.action_transition_ms_overrides[action_name])
            return self.engine.play_action(action, viewer=self.viewer, loop=False)
        finally:
            self.engine.motion_scale = orig_motion_scale
            self.engine.transition_ms = orig_transition_ms

    # ----------------------------
    # Observation helpers
    # ----------------------------
    def _get_ball_xy(self) -> np.ndarray:
        if self.ball_body_name is not None:
            obs = self.vision.observe_ball(ball_body_name=self.ball_body_name)
            return obs.ball_xy.copy()
        return np.array(self.virtual_ball_pos[0:2], dtype=float)

    def _get_ball_obs(self):
        if self.ball_body_name is not None:
            return self.vision.observe_ball(ball_body_name=self.ball_body_name)
        return self.vision.observe_ball(ball_pos_world=self.virtual_ball_pos)

    def _get_goal_obs(self):
        return self.vision.observe_goal(goal_pos_world=self.goal_pos_world)

    def _ball_to_goal_distance(self) -> float:
        return float(np.linalg.norm(self._get_ball_xy() - self.goal_pos_world[0:2]))

    def _get_ball_pos_xyz(self) -> np.ndarray:
        if self._ball_body_id is not None:
            return np.array(self.data.xpos[self._ball_body_id, 0:3], dtype=float)
        return np.array(self.virtual_ball_pos[0:3], dtype=float)

    def _is_goal_scored(self) -> bool:
        ball = self._get_ball_pos_xyz()
        gx, gy, gz = [float(v) for v in self.goal_pos_world]
        r = float(self.ball_cfg.radius_m)
        # Goal plane is at x ~= gx, robot attacks +x direction.
        crossed_plane = float(ball[0]) >= (gx - r)
        within_width = abs(float(ball[1]) - gy) <= (self.goal_cfg.width_half_m + r)
        below_bar = float(ball[2]) <= (gz + self.goal_cfg.height_half_m + r)
        above_ground = float(ball[2]) >= -r
        return bool(crossed_plane and within_width and below_bar and above_ground)

    def _base_up_z(self) -> float:
        quat = np.array(self.data.qpos[3:7], dtype=float)
        mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mat, quat)
        return float(mat[8])

    def is_fallen(self) -> bool:
        base_z = float(self.data.qpos[2]) if self.model.nq >= 3 else 0.0
        up_z = self._base_up_z() if self.model.nq >= 7 else 1.0
        return bool(base_z < 0.12 or up_z < 0.30)

    def _get_observation(self) -> np.ndarray:
        ball_obs = self._get_ball_obs()
        goal_obs = self._get_goal_obs() if self.include_goal_obs else None
        if self.model.nv >= 6:
            base_linvel_x = float(self.data.qvel[0])
            base_linvel_y = float(self.data.qvel[1])
            base_angvel_z = float(self.data.qvel[5])  # freejoint angular velocity wz
        else:
            base_linvel_x = 0.0
            base_linvel_y = 0.0
            base_angvel_z = 0.0

        last_action_val = float(self.last_action_id)
        if len(self.action_names) > 1 and self.last_action_id >= 0:
            last_action_val = float(self.last_action_id) / float(len(self.action_names) - 1)
        elif self.last_action_id < 0:
            last_action_val = -1.0

        obs = np.array(
            [
                float(ball_obs.robot_yaw),
                base_linvel_x,
                base_linvel_y,
                base_angvel_z,
                last_action_val,
                1.0 if ball_obs.ball_visible else 0.0,
                float(ball_obs.ball_distance),
                float(ball_obs.ball_bearing),
                1.0 if self.is_fallen() else 0.0,
            ]
            + (
                [
                    1.0 if goal_obs.visible else 0.0,
                    float(goal_obs.distance),
                    float(goal_obs.bearing),
                    float(self._ball_to_goal_distance()),
                ]
                if self.include_goal_obs
                else []
            ),
            dtype=np.float32,
        )
        return obs

    def obs_to_dict(self, obs: np.ndarray) -> dict[str, float]:
        return {k: float(v) for k, v in zip(self.obs_keys, np.asarray(obs, dtype=float).tolist())}

    def _build_info(self, obs: np.ndarray) -> dict[str, Any]:
        obs_dict = self.obs_to_dict(obs)
        base_pos = np.array(self.data.qpos[0:3], dtype=float)
        info: dict[str, Any] = {
            "step_count": int(self.step_count),
            "obs_dict": obs_dict,
            "base_pos_xyz": base_pos.copy(),
            "base_yaw": float(obs_dict["base_yaw"]),
            "ball_xy": self._get_ball_xy().copy(),
            "ball_vel_xy": self._get_ball_vel_xy(),
            "goal_xy": self.goal_pos_world[0:2].copy(),
            "goal_pos_xyz": self.goal_pos_world.copy(),
            "ball_to_goal_distance": self._ball_to_goal_distance(),
            "is_fallen": bool(obs_dict["is_fallen"] > 0.5),
            "goal_scored": self._is_goal_scored(),
            "episode_metrics": self._episode_metrics(),
        }
        return info

    # ----------------------------
    # Virtual ball model
    # ----------------------------
    def _randomize_ball(self, *, first_reset: bool = False) -> None:
        if first_reset:
            # Easy scripted first placement for debugging: nearly straight-on.
            x = 0.33
            y = float(np.clip(self.goal_pos_world[1] + 0.01, *self.ball_cfg.spawn_y_range))
            if self.ball_body_name is not None:
                self._set_sim_ball_state(x=x, y=y, z=self.ball_cfg.spawn_z_m)
                return
            self.virtual_ball_pos[:] = np.array([x, y, 0.0], dtype=float)
            self.virtual_ball_vel_xy[:] = 0.0
            return

        if self.ball_body_name is not None:
            x = float(self.rng.uniform(*self.ball_cfg.spawn_x_range))
            if self.ball_cfg.align_to_goal_y:
                y_center = float(self.goal_pos_world[1])
                y = float(y_center + self.rng.uniform(*self.ball_cfg.goal_y_offset_range))
                y = float(np.clip(y, *self.ball_cfg.spawn_y_range))
            else:
                y = float(self.rng.uniform(*self.ball_cfg.spawn_y_range))
            self._set_sim_ball_state(x=x, y=y, z=self.ball_cfg.spawn_z_m)
            return
        x = float(self.rng.uniform(*self.ball_cfg.spawn_x_range))
        if self.ball_cfg.align_to_goal_y:
            y_center = float(self.goal_pos_world[1])
            y = float(y_center + self.rng.uniform(*self.ball_cfg.goal_y_offset_range))
            y = float(np.clip(y, *self.ball_cfg.spawn_y_range))
        else:
            y = float(self.rng.uniform(*self.ball_cfg.spawn_y_range))
        self.virtual_ball_pos[:] = np.array([x, y, 0.0], dtype=float)
        self.virtual_ball_vel_xy[:] = 0.0

    def _randomize_goal(self, *, first_reset: bool = False) -> None:
        if first_reset:
            x = 1.43
            y = 0.00
        else:
            x = float(self.rng.uniform(*self.goal_cfg.center_x_range))
            y = float(self.rng.uniform(*self.goal_cfg.center_y_range))
        self._set_goal_state(x=x, y=y, z=self.goal_cfg.center_z_m)

    def _get_ball_vel_xy(self) -> np.ndarray:
        if self._ball_body_id is not None and self._ball_freejoint_dofadr is not None:
            dadr = self._ball_freejoint_dofadr
            return np.array(self.data.qvel[dadr:dadr + 2], dtype=float)
        return self.virtual_ball_vel_xy.copy()

    def _episode_metrics(self) -> dict[str, float]:
        return {
            "episode_return": float(self._episode_return),
            "min_ball_distance": float(self._episode_min_ball_distance),
            "min_abs_ball_bearing": float(self._episode_min_abs_bearing),
            "kick_attempts": float(self._episode_kick_attempts),
            "kick_successes": float(self._episode_kick_successes),
            "goal_scored": float(self._episode_goal_scored),
            "ball_to_goal_start": float(self._episode_start_ball_to_goal_dist),
            "ball_to_goal_now": float(self._ball_to_goal_distance()),
            "ball_to_goal_progress": float(self._episode_start_ball_to_goal_dist - self._ball_to_goal_distance()),
        }

    def _estimate_action_duration_s(self, action: ActionGroup) -> float:
        t = float(self.engine.transition_ms) / 1000.0
        for i in range(max(0, len(action.frames) - 1)):
            t += float(action.frames[i]["hold_ms"]) / 1000.0
        t += 0.20  # end settle in engine.play_action
        return t

    def _maybe_apply_virtual_kick(self, action_name: str) -> bool:
        if self.ball_body_name is not None:
            return False
        if "kick" not in action_name:
            return False

        ball_obs = self._get_ball_obs()
        if (not ball_obs.ball_visible):
            return False
        if ball_obs.ball_distance > self.ball_cfg.kick_distance_thresh_m:
            return False
        if abs(ball_obs.ball_bearing) > self.ball_cfg.kick_bearing_thresh_rad:
            return False

        _, _, forward_xy = self.vision.get_robot_pose_2d()
        lateral = np.array([-forward_xy[1], forward_xy[0]], dtype=float)

        # Small side bias to reflect left/right kick variants while keeping the interface generic.
        side_bias = 0.0
        if action_name == "kick_left":
            side_bias = 0.15
        elif action_name == "kick_right":
            side_bias = -0.15

        kick_dir = forward_xy + side_bias * lateral
        norm = float(np.linalg.norm(kick_dir))
        if norm < 1e-9:
            kick_dir = np.array([1.0, 0.0], dtype=float)
        else:
            kick_dir /= norm

        self.virtual_ball_vel_xy[:] = kick_dir * float(self.ball_cfg.kick_speed_mps)
        return True

    def _integrate_virtual_ball(self, dt: float) -> None:
        if self.ball_body_name is not None:
            return
        dt = max(0.0, float(dt))
        if dt <= 0.0:
            return
        self.virtual_ball_pos[0:2] += self.virtual_ball_vel_xy * dt
        decay = math.exp(-float(self.ball_cfg.drag_per_second) * dt)
        self.virtual_ball_vel_xy *= decay

    # ----------------------------
    # Convenience methods for controllers
    # ----------------------------
    def action_id(self, name: str) -> int:
        return int(self.action_index[name])

    def available_actions(self) -> list[str]:
        return list(self.action_names)


_GymBase = gym.Env if gym is not None else object


class AinexGymnasiumEnv(_GymBase):
    """
    Gymnasium-compatible adapter around AinexActionEnv.

    Keeps the core env free of a hard dependency on gymnasium while exposing
    `observation_space`/`action_space` for RL libraries.
    """

    metadata = {"render_modes": ["human", None]}

    def __init__(self, **ainex_env_kwargs):
        if gym is not None:
            super().__init__()
        self.core = AinexActionEnv(**ainex_env_kwargs)
        try:
            from gymnasium import spaces
        except ImportError as exc:
            raise ImportError(
                "gymnasium is required for AinexGymnasiumEnv. Install with `pip install gymnasium`."
            ) from exc

        obs_dim = len(self.core.obs_keys)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(len(self.core.action_names))

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict[str, Any]] = None):
        del options
        return self.core.reset(seed=seed)

    def step(self, action: int):
        return self.core.step(int(action))

    def close(self) -> None:
        self.core.close()

    def render(self):
        # Passive viewer is driven by env stepping; nothing extra required.
        return None

    @property
    def action_names(self) -> list[str]:
        return self.core.action_names
