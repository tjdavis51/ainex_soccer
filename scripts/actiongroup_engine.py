# scripts/actiongroup_engine.py
from __future__ import annotations

import csv
import time
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Tuple

import mujoco
import numpy as np


@dataclass
class ActionGroup:
    name: str
    frames: list[dict]  # each: {"hold_ms": int, "servo": np.ndarray}


class ActionGroupEngine:
    """
    Action-group playback engine for MuJoCo AINex.

    IMPORTANT:
    The real robot interprets the CSV values as servo pulse widths (0..1000),
    then converts pulse -> joint radians using per-servo calibration:

        angle_rad = (pulse - init_pulse) / coef

    where coef is ENCODER_TICKS_PER_RADIAN, negated if the servo is flipped
    (min > max in the ROS yaml).

    This file now mirrors that logic, rather than linearly mapping 0..1000
    into the MuJoCo joint range.
    """

    # CSV contains 22 servos (ID1..ID22). Head is not included in CSV.
    NUM_CSV_SERVOS = 22

    # Real robot: 1000 encoder ticks correspond to 240 degrees.
    # From ROS: ENCODER_TICKS_PER_RADIAN = 180/pi/240*1000
    ENCODER_TICKS_PER_RADIAN = (180.0 / math.pi) / 240.0 * 1000.0

    # Real robot ID order (ID1..ID22) -> MuJoCo actuator index
    ID_TO_ACT = [
        11,  # ID1  -> 11_l_ank_roll
        5,   # ID2  -> 05_r_ank_roll
        10,  # ID3  -> 10_l_ank_pitch
        4,   # ID4  -> 04_r_ank_pitch
        9,   # ID5  -> 09_l_knee
        3,   # ID6  -> 03_r_knee
        8,   # ID7  -> 08_l_hip_pitch
        2,   # ID8  -> 02_r_hip_pitch
        7,   # ID9  -> 07_l_hip_roll
        1,   # ID10 -> 01_r_hip_roll
        6,   # ID11 -> 06_l_hip_yaw
        0,   # ID12 -> 00_r_hip_yaw
        19,  # ID13 -> 19_l_sho_pitch
        14,  # ID14 -> 14_r_sho_pitch
        20,  # ID15 -> 20_l_sho_roll
        15,  # ID16 -> 15_r_sho_roll
        21,  # ID17 -> 21_l_el_pitch
        16,  # ID18 -> 16_r_el_pitch
        22,  # ID19 -> 22_l_el_yaw
        17,  # ID20 -> 17_r_el_yaw
        23,  # ID21 -> 23_l_gripper
        18,  # ID22 -> 18_r_gripper
    ]

    SERVO_MIN = 0.0
    SERVO_MAX = 1000.0

    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        # --- Stability defaults for testing playback ---
        base_z: float = 0.1,
        settle_seconds: float = 0.5,
        transition_ms: int = 500,
        abort_min_z: float = 0.05,
        abort_min_up_z: float = 0.05,
        sleep: bool = True,

        # Toggle safety checks
        safety_enabled: bool = False,
        verbose: bool = True,

        # Motion scaling in PULSE space, applied relative to the first frame.
        # 1.0 = no scaling, 2.0 doubles deviations from frame 0, etc.
        motion_scale: float = 2.5,

        # Optional fine per-servo gain (multiplicative on delta-pulse around frame0)
        # Length 22 in ID order. 1.0 means unchanged.
        servo_gain: Optional[np.ndarray] = None,

        base_pitch_deg: float = 17.0,

        # Optional override to match ROS init/flip behavior.
        # Map servo_id (1..24) -> (init_pulse, coef_ticks_per_rad)
        joint_angles_convert_coef: Optional[Dict[int, Tuple[float, float]]] = None,
    ):
        self.model = model
        self.data = data

        self.base_pitch_deg = float(base_pitch_deg)
        self.base_z = float(base_z)
        self.settle_seconds = float(settle_seconds)
        self.transition_ms = int(transition_ms)

        self.abort_min_z = float(abort_min_z)
        self.abort_min_up_z = float(abort_min_up_z)

        self.sleep = bool(sleep)
        self.safety_enabled = bool(safety_enabled)
        self.verbose = bool(verbose)

        self.motion_scale = float(motion_scale)

        if servo_gain is None:
            servo_gain = np.ones(self.NUM_CSV_SERVOS, dtype=float)
            
        servo_gain[[3,5,7]] = 1
        servo_gain[[4,6,8]] = 1
            
        self.servo_gain = np.array(servo_gain, dtype=float)
        
        if len(self.servo_gain) != self.NUM_CSV_SERVOS:
            raise ValueError("servo_gain must have length 22")

        if self.model.nu <= max(self.ID_TO_ACT):
            raise ValueError(
                f"Model has nu={self.model.nu}, but mapping needs actuator {max(self.ID_TO_ACT)}"
            )

        # Build the pulse->angle conversion coefficients from your ROS yaml defaults.
        # coef is +ENCODER_TICKS_PER_RADIAN unless flipped, then negative.
        if joint_angles_convert_coef is None:
            joint_angles_convert_coef = self._default_joint_angles_convert_coef_from_yaml()
        self.joint_angles_convert_coef = joint_angles_convert_coef

        self._has_freejoint = (
            self.model.nq >= 7 and self.model.jnt_type[0] == mujoco.mjtJoint.mjJNT_FREE
        )

    @staticmethod
    def _default_joint_angles_convert_coef_from_yaml() -> Dict[int, Tuple[float, float]]:
        """
        Encodes the data you pasted from servo_controller.yaml.

        Return: {servo_id: (init_pulse, coef_ticks_per_rad)}
        """
        init_by_id = {
            1: 500,  2: 500,  3: 500,  4: 500,
            5: 240,  6: 760,  7: 500,  8: 500,
            9: 500, 10: 500, 11: 500, 12: 500,
            13: 875, 14: 125, 15: 500, 16: 500,
            17: 500, 18: 500, 19: 500, 20: 500,
            21: 500, 22: 500, 23: 500, 24: 500,
        }

        # flipped if min > max (your yaml shows this for l_sho_pitch id13 and r_sho_pitch id14)
        flipped_ids = {15, 16}

        coef = ActionGroupEngine.ENCODER_TICKS_PER_RADIAN
        out: Dict[int, Tuple[float, float]] = {}
        for sid, init in init_by_id.items():
            out[sid] = (float(init), -coef if sid in flipped_ids else coef)
        return out

    # ----------------------------
    # CSV loading
    # ----------------------------
    @staticmethod
    def load_csv(path: Path) -> ActionGroup:
        """
        Accepts either:
          - Index,Time,ID1..ID22
          - Index,Time,Servo1..Servo22

        Stores as a 22-vector in ID order.
        """
        frames: list[dict] = []
        with path.open("r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                hold_ms = int(round(float(row["Time"])))

                if "ID1" in row:
                    prefix = "ID"
                elif "Servo1" in row:
                    prefix = "Servo"
                else:
                    raise ValueError(
                        f"{path} does not contain ID1..ID22 or Servo1..Servo22 headers."
                    )

                vals = np.array([float(row[f"{prefix}{i}"]) for i in range(1, 23)], dtype=float)
                frames.append({"hold_ms": hold_ms, "servo": vals})

        return ActionGroup(name=path.stem, frames=frames)

    # ----------------------------
    # Mapping helpers
    # ----------------------------
    def _actuator_joint_info(self, act_idx: int):
        j_id = int(self.model.actuator_trnid[act_idx, 0])
        if j_id < 0:
            return None
        qadr = int(self.model.jnt_qposadr[j_id])
        rmin = float(self.model.jnt_range[j_id, 0])
        rmax = float(self.model.jnt_range[j_id, 1])
        return j_id, qadr, (rmin, rmax)

    def _clamp(self, x: float, r: tuple[float, float]) -> float:
        return float(np.clip(x, r[0], r[1]))

    def pulse2angle(self, servo_id: int, pulse: float) -> float:
        """
        Mirrors ROS pulse2angle:

            angle = (pulse - init) / coef

        where coef already has sign included (flipped servos use negative coef).
        """
        if servo_id not in self.joint_angles_convert_coef:
            raise KeyError(f"Missing joint_angles_convert_coef for servo_id={servo_id}")
        init_pulse, coef = self.joint_angles_convert_coef[servo_id]
        if abs(coef) < 1e-12:
            raise ZeroDivisionError(f"Bad coef for servo_id={servo_id}: {coef}")
        return (float(pulse) - float(init_pulse)) / float(coef)

    def _pulse_to_ctrl_angle(
        self,
        servo_id: int,
        pulse_value: float,
        joint_range: tuple[float, float],
    ) -> float:
        p = float(np.clip(pulse_value, self.SERVO_MIN, self.SERVO_MAX))
        ang = self.pulse2angle(servo_id, p)
        return self._clamp(ang, joint_range)

    def servo_frame_to_ctrl(self, servos: np.ndarray) -> np.ndarray:
        """
        Convert a 22-vector of pulses in ID order (ID1..ID22) into MuJoCo ctrl targets (radians).
        """
        ctrl = np.zeros(self.model.nu, dtype=float)

        for id_idx in range(self.NUM_CSV_SERVOS):
            act_idx = self.ID_TO_ACT[id_idx]
            info = self._actuator_joint_info(act_idx)
            if info is None:
                continue
            _, _, joint_range = info

            servo_id = id_idx + 1  # ID1..ID22 correspond to servo ids 1..22
            ctrl[act_idx] = self._pulse_to_ctrl_angle(
                servo_id,
                float(servos[id_idx]),
                joint_range,
            )

        # Keep head neutral (CSV leaves head out)
        if self.model.nu >= 14:
            ctrl[12] = 0.0
            ctrl[13] = 0.0

        return ctrl

    # ----------------------------
    # Reset / settle
    # ----------------------------
    def reset_base(self):
        if not self._has_freejoint:
            return

        self.data.qpos[0] = 0.0
        self.data.qpos[1] = 0.0
        self.data.qpos[2] = self.base_z

        theta = math.radians(self.base_pitch_deg)
        qw = math.cos(theta / 2.0)
        qx = 0.0
        qy = math.sin(theta / 2.0)
        qz = 0.0

        self.data.qpos[3] = qw
        self.data.qpos[4] = qx
        self.data.qpos[5] = qy
        self.data.qpos[6] = qz

        if self.model.nv >= 6:
            self.data.qvel[0:6] = 0.0

    def reset_sim(self):
        mujoco.mj_resetData(self.model, self.data)
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0

    def set_pose_from_frame(self, servos: np.ndarray):
        """
        Set qpos directly from pulse2angle conversion (real-robot style),
        then set ctrl to match.
        """
        self.data.ctrl[:] = 0.0
        self.data.qvel[:] = 0.0
        self.reset_base()

        for id_idx in range(self.NUM_CSV_SERVOS):
            act_idx = self.ID_TO_ACT[id_idx]
            info = self._actuator_joint_info(act_idx)
            if info is None:
                continue

            _, qadr, joint_range = info
            servo_id = id_idx + 1
            rad = self._pulse_to_ctrl_angle(
                servo_id,
                float(servos[id_idx]),
                joint_range,
            )

            self.data.qpos[qadr] = rad
            self.data.ctrl[act_idx] = rad

        if self.model.nu >= 14:
            self.data.ctrl[12] = 0.0
            self.data.ctrl[13] = 0.0

        mujoco.mj_forward(self.model, self.data)

    def settle(self, viewer=None, *, seconds: Optional[float] = None) -> bool:
        if seconds is None:
            seconds = self.settle_seconds

        steps = max(1, int(round(float(seconds) / self.model.opt.timestep)))
        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
            if self.sleep:
                time.sleep(self.model.opt.timestep)

            if not self.is_safe():
                return False

        return True

    def reset_and_settle_to_action(self, action: ActionGroup, viewer=None) -> bool:
        if not action.frames:
            return True

        self.reset_sim()
        self.set_pose_from_frame(action.frames[0]["servo"])
        return self.settle(viewer, seconds=self.settle_seconds)

    # ----------------------------
    # Safety
    # ----------------------------
    def is_safe(self) -> bool:
        if not self.safety_enabled:
            return True

        if not self._has_freejoint:
            return True

        z = float(self.data.qpos[2])
        if z < self.abort_min_z:
            if self.verbose:
                print(f"[ABORT] base z too low: {z:.3f} < {self.abort_min_z:.3f}")
            return False

        qw, qx, qy, qz = [float(x) for x in self.data.qpos[3:7]]
        quat = np.array([qw, qx, qy, qz], dtype=float)
        mat = np.zeros(9, dtype=float)
        mujoco.mju_quat2Mat(mat, quat)

        up_z = float(mat[8])
        if up_z < self.abort_min_up_z:
            if self.verbose:
                print(f"[ABORT] upright too low: up_z={up_z:.3f} < {self.abort_min_up_z:.3f}")
            return False

        return True

    # ----------------------------
    # Playback helpers
    # ----------------------------
    def _ramp_ctrl(self, ctrl_a: np.ndarray, ctrl_b: np.ndarray, hold_ms: int, viewer=None) -> bool:
        hold_s = max(0.0, hold_ms / 1000.0)
        steps = max(1, int(round(hold_s / self.model.opt.timestep)))

        print_every = max(1, steps // 10)

        fr = self.model.actuator_forcerange
        fr_lo = fr[:, 0]
        fr_hi = fr[:, 1]

        for k in range(steps):
            t = 0.0 if steps <= 1 else (k / (steps - 1))
            self.data.ctrl[:] = (1.0 - t) * ctrl_a + t * ctrl_b

            mujoco.mj_step(self.model, self.data)
            if viewer is not None:
                viewer.sync()
            if self.sleep:
                time.sleep(self.model.opt.timestep)

            if (k % print_every) == 0 or k == (steps - 1):
                errs = []
                for act_idx in range(self.model.nu):
                    j_id = int(self.model.actuator_trnid[act_idx, 0])
                    if j_id < 0:
                        continue
                    qadr = int(self.model.jnt_qposadr[j_id])
                    q = float(self.data.qpos[qadr])
                    u = float(self.data.ctrl[act_idx])
                    errs.append(abs(u - q))
                if errs and self.verbose:
                    print(
                        f"[TRACK] step {k+1}/{steps}  "
                        f"mean|ctrl-q|={float(np.mean(errs)):.4f}  "
                        f"max|ctrl-q|={float(np.max(errs)):.4f}"
                    )

                af = np.array(self.data.actuator_force, dtype=float)
                atol = 1e-2
                sat = (np.isclose(af, fr_hi, atol=atol) | np.isclose(af, fr_lo, atol=atol))
                if sat.any() and self.verbose:
                    idx = np.where(sat)[0]
                    print(
                        f"[SAT] step {k+1}/{steps}  "
                        f"saturated_acts={idx.tolist()}  "
                        f"forces={af[idx].round(3).tolist()}"
                    )

                if self._has_freejoint and self.verbose:
                    px, py, pz = [float(v) for v in self.data.qpos[0:3]]
                    qw, qx, qy, qz = [float(v) for v in self.data.qpos[3:7]]
                    print(
                        f"[BASE] pos=({px:.3f},{py:.3f},{pz:.3f})  "
                        f"quat=({qw:.3f},{qx:.3f},{qy:.3f},{qz:.3f})"
                    )

            if not self.is_safe():
                return False

        return True

    def _scale_pulses_relative_to_first(self, pulses: np.ndarray, pulse0: np.ndarray) -> np.ndarray:
        """
        Keep the first frame exactly as-is, scale deviations in pulse space.

        Also supports per-servo gains (servo_gain).
        """
        d = (pulses - pulse0) * float(self.motion_scale)
        d = d * self.servo_gain
        out = pulse0 + d
        return np.clip(out, self.SERVO_MIN, self.SERVO_MAX)

    # ----------------------------
    # Playback
    # ----------------------------
    def play_action(self, action: ActionGroup, viewer=None, *, loop: bool = False) -> bool:
        """
        Plays an action group.

        Behavior:
        - Frame 0 is used as an anchor and is not scaled.
        - All subsequent frames can be scaled in pulse space relative to frame 0.
        - Pulses are converted to joint radians using ROS-style pulse2angle.
        """
        if not action.frames:
            return True

        pulse0 = np.array(action.frames[0]["servo"], dtype=float)

        # 1) Transition from current ctrl -> first frame ctrl (unscaled)
        ctrl_first = self.servo_frame_to_ctrl(pulse0)
        ctrl_now = np.array(self.data.ctrl, dtype=float)

        delta0 = np.abs(ctrl_first - ctrl_now)
        if self.verbose:
            print(f"[CTRLΔ] (start->first) mean={float(delta0.mean()):.4f}  max={float(delta0.max()):.4f}")

        if not self._ramp_ctrl(ctrl_now, ctrl_first, self.transition_ms, viewer=viewer):
            return False

        # 2) Play frames (scaled relative to first in pulse space)
        for i in range(len(action.frames) - 1):
            frame_a = action.frames[i]
            frame_b = action.frames[i + 1]

            pulses_a = np.array(frame_a["servo"], dtype=float)
            pulses_b = np.array(frame_b["servo"], dtype=float)

            if i == 0:
                pulses_a_s = pulse0
            else:
                pulses_a_s = self._scale_pulses_relative_to_first(pulses_a, pulse0)
            pulses_b_s = self._scale_pulses_relative_to_first(pulses_b, pulse0)

            ctrl_a = self.servo_frame_to_ctrl(pulses_a_s)
            ctrl_b = self.servo_frame_to_ctrl(pulses_b_s)

            delta = np.abs(ctrl_b - ctrl_a)
            if self.verbose:
                print(
                    f"[CTRLΔ] frame {i}->{i+1}  mean={float(delta.mean()):.4f}  "
                    f"max={float(delta.max()):.4f}  hold_ms={int(frame_a['hold_ms'])}"
                )

            if not self._ramp_ctrl(ctrl_a, ctrl_b, int(frame_a["hold_ms"]), viewer=viewer):
                return False

        # 3) Settle at the end
        last_pulses = np.array(action.frames[-1]["servo"], dtype=float)
        last_pulses_s = self._scale_pulses_relative_to_first(last_pulses, pulse0)
        last_ctrl = self.servo_frame_to_ctrl(last_pulses_s)

        self.data.ctrl[:] = last_ctrl
        if not self.settle(viewer, seconds=0.20):
            return False

        # 4) Optional loop back
        if loop:
            if not self._ramp_ctrl(last_ctrl, ctrl_first, self.transition_ms, viewer=viewer):
                return False

        return True

    def transition_to(self, target_action: ActionGroup, viewer=None) -> bool:
        if not target_action.frames:
            return True
        ctrl_now = np.array(self.data.ctrl, dtype=float)
        ctrl_target = self.servo_frame_to_ctrl(target_action.frames[0]["servo"])
        return self._ramp_ctrl(ctrl_now, ctrl_target, self.transition_ms, viewer=viewer)
