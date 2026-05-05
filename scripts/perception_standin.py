from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import mujoco
import numpy as np

# Make repo root importable when run as a script.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.actiongroup_engine import ActionGroupEngine  # noqa: E402


MODEL_PATH = REPO_ROOT / "assets" / "ainex" / "ainex_physics.xml"
AG_DIR = REPO_ROOT / "assets" / "action_groups" / "csv"


@dataclass
class BallObservation:
    ball_visible: bool
    ball_distance: float
    ball_bearing: float
    robot_yaw: float
    robot_xy: np.ndarray
    ball_xy: np.ndarray


@dataclass
class TargetObservation:
    visible: bool
    distance: float
    bearing: float
    robot_yaw: float
    robot_xy: np.ndarray
    target_xy: np.ndarray


@dataclass
class VisionStandInConfig:
    torso_body_name: str = "torso"
    max_distance_m: float = 3.0
    fov_half_angle_rad: float = math.radians(70.0)
    distance_noise_std_m: float = 0.0
    bearing_noise_std_rad: float = 0.0
    dropout_prob: float = 0.0


class VisionStandIn:
    """
    Camera-like stand-in perception using MuJoCo ground truth positions.

    Returns ball visibility + planar distance + bearing in robot frame.
    """

    def __init__(self, model: mujoco.MjModel, data: mujoco.MjData, config: Optional[VisionStandInConfig] = None, *, rng_seed: Optional[int] = None):
        self.model = model
        self.data = data
        self.config = config or VisionStandInConfig()
        self.rng = np.random.default_rng(rng_seed)
        self._torso_body_id = self._require_body_id(self.config.torso_body_name)

    def _require_body_id(self, body_name: str) -> int:
        body_id = int(mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name))
        if body_id < 0:
            raise ValueError(f"Body '{body_name}' not found in model.")
        return body_id

    def _quat_wxyz_to_rotmat(self, quat_wxyz: np.ndarray) -> np.ndarray:
        mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mat, quat_wxyz.astype(float))
        return mat.reshape(3, 3)

    def get_robot_pose_2d(self) -> tuple[np.ndarray, float, np.ndarray]:
        """
        Returns (robot_xy, yaw, forward_xy_unit).

        Yaw/forward are derived from the torso freejoint quaternion.
        """
        if self.model.nq < 7 or self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise RuntimeError("Model does not appear to use a freejoint base at qpos[0:7].")

        robot_xy = np.array(self.data.xpos[self._torso_body_id, 0:2], dtype=float)
        quat = np.array(self.data.qpos[3:7], dtype=float)  # w, x, y, z
        rot = self._quat_wxyz_to_rotmat(quat)

        # Torso local +X axis as "forward" in world frame.
        forward_world = rot[:, 0]
        forward_xy = np.array([forward_world[0], forward_world[1]], dtype=float)
        norm = float(np.linalg.norm(forward_xy))
        if norm < 1e-9:
            yaw = 0.0
            forward_xy = np.array([1.0, 0.0], dtype=float)
        else:
            forward_xy /= norm
            yaw = math.atan2(float(forward_xy[1]), float(forward_xy[0]))

        return robot_xy, yaw, forward_xy

    def _target_xy_from_inputs(
        self,
        *,
        body_name: Optional[str],
        pos_world: Optional[np.ndarray],
        label: str,
    ) -> np.ndarray:
        if body_name is not None:
            body_id = self._require_body_id(body_name)
            return np.array(self.data.xpos[body_id, 0:2], dtype=float)

        if pos_world is None:
            raise ValueError(f"Provide either {label}_body_name or {label}_pos_world.")

        pos_world = np.asarray(pos_world, dtype=float)
        if pos_world.shape[0] < 2:
            raise ValueError(f"{label}_pos_world must have at least x,y.")
        return np.array(pos_world[0:2], dtype=float)

    def observe_target(
        self,
        *,
        target_body_name: Optional[str] = None,
        target_pos_world: Optional[np.ndarray] = None,
    ) -> TargetObservation:
        robot_xy, robot_yaw, forward_xy = self.get_robot_pose_2d()
        target_xy = self._target_xy_from_inputs(
            body_name=target_body_name,
            pos_world=target_pos_world,
            label="target",
        )

        rel_xy = target_xy - robot_xy
        distance = float(np.linalg.norm(rel_xy))

        if distance < 1e-9:
            bearing = 0.0
        else:
            rel_dir = rel_xy / distance
            dot = float(np.clip(np.dot(forward_xy, rel_dir), -1.0, 1.0))
            cross_z = float(forward_xy[0] * rel_dir[1] - forward_xy[1] * rel_dir[0])
            bearing = math.atan2(cross_z, dot)

        visible = (
            distance <= self.config.max_distance_m
            and abs(bearing) <= self.config.fov_half_angle_rad
        )

        if self.config.distance_noise_std_m > 0.0:
            distance = max(0.0, distance + float(self.rng.normal(0.0, self.config.distance_noise_std_m)))
        if self.config.bearing_noise_std_rad > 0.0:
            bearing = float(bearing + self.rng.normal(0.0, self.config.bearing_noise_std_rad))
            bearing = float((bearing + math.pi) % (2.0 * math.pi) - math.pi)
        if visible and self.config.dropout_prob > 0.0:
            visible = bool(self.rng.random() >= self.config.dropout_prob)

        return TargetObservation(
            visible=bool(visible),
            distance=float(distance),
            bearing=float(bearing),
            robot_yaw=float(robot_yaw),
            robot_xy=robot_xy,
            target_xy=target_xy,
        )

    def observe_ball(
        self,
        *,
        ball_body_name: Optional[str] = None,
        ball_pos_world: Optional[np.ndarray] = None,
    ) -> BallObservation:
        target = self.observe_target(
            target_body_name=ball_body_name,
            target_pos_world=ball_pos_world,
        )

        return BallObservation(
            ball_visible=bool(target.visible),
            ball_distance=float(target.distance),
            ball_bearing=float(target.bearing),
            robot_yaw=float(target.robot_yaw),
            robot_xy=target.robot_xy,
            ball_xy=target.target_xy,
        )

    def observe_goal(
        self,
        *,
        goal_body_name: Optional[str] = None,
        goal_pos_world: Optional[np.ndarray] = None,
    ) -> TargetObservation:
        return self.observe_target(
            target_body_name=goal_body_name,
            target_pos_world=goal_pos_world,
        )


def _build_default_engine(model: mujoco.MjModel, data: mujoco.MjData, *, sleep: bool = False) -> ActionGroupEngine:
    return ActionGroupEngine(
        model,
        data,
        base_z=0.34,
        settle_seconds=0.7,
        transition_ms=500,
        abort_min_z=0.12,
        abort_min_up_z=0.30,
        sleep=sleep,
        safety_enabled=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Print ball distance/bearing stand-in perception while robot stands.")
    parser.add_argument("--ball-x", type=float, default=0.40, help="Ball world X (m)")
    parser.add_argument("--ball-y", type=float, default=0.00, help="Ball world Y (m)")
    parser.add_argument("--steps", type=int, default=20, help="Number of print samples")
    parser.add_argument("--dt-steps", type=int, default=10, help="MuJoCo steps between prints")
    parser.add_argument("--action", type=str, default="walk_ready", help="Initial settle action CSV stem")
    parser.add_argument("--distance-noise", type=float, default=0.0, help="Stddev (m)")
    parser.add_argument("--bearing-noise", type=float, default=0.0, help="Stddev (rad)")
    parser.add_argument("--dropout", type=float, default=0.0, help="Dropout probability [0,1]")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    engine = _build_default_engine(model, data, sleep=False)

    action_path = AG_DIR / f"{args.action}.csv"
    if not action_path.exists():
        raise FileNotFoundError(f"Action CSV not found: {action_path}")
    start_action = engine.load_csv(action_path)
    engine.reset_and_settle_to_action(start_action, viewer=None)

    vision = VisionStandIn(
        model,
        data,
        VisionStandInConfig(
            distance_noise_std_m=float(args.distance_noise),
            bearing_noise_std_rad=float(args.bearing_noise),
            dropout_prob=float(args.dropout),
        ),
    )

    ball = np.array([args.ball_x, args.ball_y, 0.0], dtype=float)
    print(f"[perception] ball_world=({ball[0]:.3f}, {ball[1]:.3f}) action={args.action}")

    for i in range(args.steps):
        obs = vision.observe_ball(ball_pos_world=ball)
        base_pos = np.array(data.qpos[0:3], dtype=float)
        print(
            f"[{i:03d}] visible={int(obs.ball_visible)} "
            f"dist={obs.ball_distance:.3f}m "
            f"bearing={obs.ball_bearing:+.3f}rad "
            f"base_xy=({obs.robot_xy[0]:+.3f},{obs.robot_xy[1]:+.3f}) "
            f"base_z={base_pos[2]:.3f} "
            f"yaw={obs.robot_yaw:+.3f}"
        )
        for _ in range(max(1, args.dt_steps)):
            mujoco.mj_step(model, data)
        time.sleep(0.01)


if __name__ == "__main__":
    main()
