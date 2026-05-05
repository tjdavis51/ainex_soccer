from __future__ import annotations

import argparse
import csv
import math
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import cv2
import numpy as np


@dataclass
class TargetEstimate:
    visible: bool
    distance_m: float
    bearing_rad: float
    confidence: float
    center_px: Tuple[float, float]
    radius_or_halfwidth_px: float


@dataclass
class CameraIntrinsics:
    fx_px: float = 520.0
    fy_px: float = 520.0
    cx_px: float = 320.0
    cy_px: float = 240.0


@dataclass
class RealPerceptionConfig:
    snapshot_url: str
    ball_diameter_m: float = 0.08
    goal_inner_width_m: float = 0.60

    # HSV thresholds in OpenCV scale: H in [0,179], S/V in [0,255]
    ball_hsv_low: Tuple[int, int, int] = (5, 90, 60)
    ball_hsv_high: Tuple[int, int, int] = (25, 255, 255)
    ball_hsv_alt_low: Tuple[int, int, int] = (-1, -1, -1)
    ball_hsv_alt_high: Tuple[int, int, int] = (-1, -1, -1)

    goal_hsv_low: Tuple[int, int, int] = (95, 60, 60)
    goal_hsv_high: Tuple[int, int, int] = (130, 255, 255)

    min_blob_area_px: float = 80.0
    ball_min_confidence: float = 0.40
    ball_min_circularity: float = 0.35
    ball_max_aspect_ratio: float = 2.0
    ball_use_tracking: bool = True
    ball_max_center_jump_px: float = 220.0
    bearing_noise_std_rad: float = 0.0
    distance_noise_std_m: float = 0.0
    dropout_prob: float = 0.0


class RealCameraPerception:
    """
    Produces camera-like observations used by the policy:
      ball_visible, ball_distance, ball_bearing,
      goal_visible, goal_distance, goal_bearing.

    Distance model:
    - Ball distance from apparent circle radius (known diameter).
    - Goal distance from apparent bbox width (known goal inner width).
    """

    def __init__(
        self,
        config: RealPerceptionConfig,
        intrinsics: Optional[CameraIntrinsics] = None,
        *,
        rng_seed: Optional[int] = None,
    ):
        self.cfg = config
        self.K = intrinsics or CameraIntrinsics()
        self.rng = np.random.default_rng(rng_seed)
        self._last_ball_center: Optional[Tuple[float, float]] = None

    def fetch_snapshot(self) -> np.ndarray:
        with urllib.request.urlopen(self.cfg.snapshot_url, timeout=5) as resp:
            payload = resp.read()
        arr = np.frombuffer(payload, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError("Failed to decode snapshot image.")
        return frame

    def _mask_hsv(self, frame_bgr: np.ndarray, low: Tuple[int, int, int], high: Tuple[int, int, int]) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, np.array(low, dtype=np.uint8), np.array(high, dtype=np.uint8))
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return mask

    def _largest_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
        c = max(contours, key=cv2.contourArea)
        if cv2.contourArea(c) < self.cfg.min_blob_area_px:
            return None
        return c

    def _build_ball_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        primary = self._mask_hsv(frame_bgr, self.cfg.ball_hsv_low, self.cfg.ball_hsv_high)
        if self.cfg.ball_hsv_alt_low[0] < 0 or self.cfg.ball_hsv_alt_high[0] < 0:
            return primary
        alt = self._mask_hsv(frame_bgr, self.cfg.ball_hsv_alt_low, self.cfg.ball_hsv_alt_high)
        return cv2.bitwise_or(primary, alt)

    def _select_ball_contour(self, mask: np.ndarray) -> Optional[np.ndarray]:
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        best = None
        best_score = -1.0
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < float(self.cfg.min_blob_area_px):
                continue

            perimeter = float(cv2.arcLength(c, True))
            if perimeter < 1e-6:
                continue
            circularity = float((4.0 * math.pi * area) / (perimeter * perimeter))
            if circularity < float(self.cfg.ball_min_circularity):
                continue

            x, y, w, h = cv2.boundingRect(c)
            aspect = float(max(w, h)) / float(max(1, min(w, h)))
            if aspect > float(self.cfg.ball_max_aspect_ratio):
                continue

            M = cv2.moments(c)
            if abs(M["m00"]) < 1e-6:
                continue
            cx = float(M["m10"] / M["m00"])
            cy = float(M["m01"] / M["m00"])

            score = area * (0.5 + 0.5 * max(0.0, circularity))
            if self.cfg.ball_use_tracking and self._last_ball_center is not None:
                dx = cx - self._last_ball_center[0]
                dy = cy - self._last_ball_center[1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > float(self.cfg.ball_max_center_jump_px):
                    continue
                score *= max(0.2, 1.0 - (dist / max(1.0, self.cfg.ball_max_center_jump_px)))

            if score > best_score:
                best_score = score
                best = c
        return best

    def _bearing_from_px(self, cx: float) -> float:
        # Same sign convention as sim: positive is target to robot-left.
        # Image x right => negative bearing in robot frame.
        return -math.atan2(cx - self.K.cx_px, self.K.fx_px)

    def _apply_noise_and_dropout(self, visible: bool, distance_m: float, bearing_rad: float) -> Tuple[bool, float, float]:
        if visible and self.cfg.dropout_prob > 0.0 and self.rng.random() < self.cfg.dropout_prob:
            visible = False
        if self.cfg.distance_noise_std_m > 0.0:
            distance_m = max(0.0, distance_m + float(self.rng.normal(0.0, self.cfg.distance_noise_std_m)))
        if self.cfg.bearing_noise_std_rad > 0.0:
            bearing_rad = float(bearing_rad + self.rng.normal(0.0, self.cfg.bearing_noise_std_rad))
            bearing_rad = float((bearing_rad + math.pi) % (2.0 * math.pi) - math.pi)
        return visible, distance_m, bearing_rad

    def estimate_ball(self, frame_bgr: np.ndarray) -> TargetEstimate:
        mask = self._build_ball_mask(frame_bgr)
        contour = self._select_ball_contour(mask)
        if contour is None:
            self._last_ball_center = None
            return TargetEstimate(False, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0)

        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        radius = float(max(radius, 1e-3))
        distance_m = float((self.cfg.ball_diameter_m * self.K.fx_px) / (2.0 * radius))
        bearing_rad = self._bearing_from_px(float(cx))

        area = float(cv2.contourArea(contour))
        circle_area = math.pi * radius * radius
        fill_ratio = float(np.clip(area / max(circle_area, 1e-6), 0.0, 1.0))
        perimeter = float(cv2.arcLength(contour, True))
        circularity = float((4.0 * math.pi * area) / max(perimeter * perimeter, 1e-6))
        circularity = float(np.clip(circularity, 0.0, 1.0))
        confidence = float(0.65 * fill_ratio + 0.35 * circularity)
        if confidence < float(self.cfg.ball_min_confidence):
            self._last_ball_center = None
            return TargetEstimate(False, 0.0, 0.0, confidence, (float(cx), float(cy)), radius)

        visible, distance_m, bearing_rad = self._apply_noise_and_dropout(True, distance_m, bearing_rad)
        self._last_ball_center = (float(cx), float(cy))
        return TargetEstimate(visible, distance_m, bearing_rad, confidence, (float(cx), float(cy)), radius)

    def estimate_goal(self, frame_bgr: np.ndarray) -> TargetEstimate:
        mask = self._mask_hsv(frame_bgr, self.cfg.goal_hsv_low, self.cfg.goal_hsv_high)
        contour = self._largest_contour(mask)
        if contour is None:
            return TargetEstimate(False, 0.0, 0.0, 0.0, (0.0, 0.0), 0.0)

        x, y, w, h = cv2.boundingRect(contour)
        half_w = max(0.5 * float(w), 1e-3)
        cx = float(x) + half_w
        cy = float(y) + 0.5 * float(h)

        distance_m = float((self.cfg.goal_inner_width_m * self.K.fx_px) / max(float(w), 1e-3))
        bearing_rad = self._bearing_from_px(cx)

        area = float(cv2.contourArea(contour))
        confidence = float(np.clip(area / max(float(w) * float(h), 1e-6), 0.0, 1.0))

        visible, distance_m, bearing_rad = self._apply_noise_and_dropout(True, distance_m, bearing_rad)
        return TargetEstimate(visible, distance_m, bearing_rad, confidence, (cx, cy), half_w)

    def observe(self, frame_bgr: Optional[np.ndarray] = None) -> Dict[str, Union[float, bool]]:
        frame = self.fetch_snapshot() if frame_bgr is None else frame_bgr
        ball = self.estimate_ball(frame)
        goal = self.estimate_goal(frame)
        return {
            "ball_visible": bool(ball.visible),
            "ball_distance": float(ball.distance_m),
            "ball_bearing": float(ball.bearing_rad),
            "goal_visible": bool(goal.visible),
            "goal_distance": float(goal.distance_m),
            "goal_bearing": float(goal.bearing_rad),
            "ball_confidence": float(ball.confidence),
            "goal_confidence": float(goal.confidence),
        }


def _parse_hsv(text: str) -> Tuple[int, int, int]:
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("HSV must be h,s,v")
    return (vals[0], vals[1], vals[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Real camera perception debug tool for AINex soccer.")
    parser.add_argument("--url", required=True, help="Snapshot URL, e.g. http://IP:8080/snapshot?topic=/camera/image_raw")
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--max-fps", type=float, default=8.0)
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--csv-log", type=Path, default=None)

    parser.add_argument("--fx", type=float, default=520.0)
    parser.add_argument("--fy", type=float, default=520.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)

    parser.add_argument("--ball-diameter", type=float, default=0.08)
    parser.add_argument("--goal-inner-width", type=float, default=0.60)
    parser.add_argument("--ball-hsv-low", type=str, default="5,90,60")
    parser.add_argument("--ball-hsv-high", type=str, default="25,255,255")
    parser.add_argument("--ball-hsv-alt-low", type=str, default="", help="Optional second HSV low range h,s,v")
    parser.add_argument("--ball-hsv-alt-high", type=str, default="", help="Optional second HSV high range h,s,v")
    parser.add_argument("--goal-hsv-low", type=str, default="95,60,60")
    parser.add_argument("--goal-hsv-high", type=str, default="130,255,255")
    parser.add_argument("--ball-min-confidence", type=float, default=0.40)
    parser.add_argument("--ball-min-circularity", type=float, default=0.35)
    parser.add_argument("--ball-max-aspect-ratio", type=float, default=2.0)
    parser.add_argument("--no-ball-tracking", action="store_true")
    parser.add_argument("--ball-max-center-jump", type=float, default=220.0)
    args = parser.parse_args()

    if args.ball_hsv_alt_low and args.ball_hsv_alt_high:
        ball_alt_low = _parse_hsv(args.ball_hsv_alt_low)
        ball_alt_high = _parse_hsv(args.ball_hsv_alt_high)
    else:
        ball_alt_low = (-1, -1, -1)
        ball_alt_high = (-1, -1, -1)

    cfg = RealPerceptionConfig(
        snapshot_url=args.url,
        ball_diameter_m=float(args.ball_diameter),
        goal_inner_width_m=float(args.goal_inner_width),
        ball_hsv_low=_parse_hsv(args.ball_hsv_low),
        ball_hsv_high=_parse_hsv(args.ball_hsv_high),
        ball_hsv_alt_low=ball_alt_low,
        ball_hsv_alt_high=ball_alt_high,
        goal_hsv_low=_parse_hsv(args.goal_hsv_low),
        goal_hsv_high=_parse_hsv(args.goal_hsv_high),
        ball_min_confidence=float(args.ball_min_confidence),
        ball_min_circularity=float(args.ball_min_circularity),
        ball_max_aspect_ratio=float(args.ball_max_aspect_ratio),
        ball_use_tracking=bool(not args.no_ball_tracking),
        ball_max_center_jump_px=float(args.ball_max_center_jump),
    )
    intr = CameraIntrinsics(fx_px=args.fx, fy_px=args.fy, cx_px=args.cx, cy_px=args.cy)
    perception = RealCameraPerception(cfg, intr)

    writer = None
    f = None
    if args.csv_log is not None:
        args.csv_log.parent.mkdir(parents=True, exist_ok=True)
        f = args.csv_log.open("w", newline="")
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "step",
                "ball_visible",
                "ball_distance",
                "ball_bearing",
                "ball_confidence",
                "goal_visible",
                "goal_distance",
                "goal_bearing",
                "goal_confidence",
            ],
        )
        writer.writeheader()

    min_dt = 1.0 / max(1e-6, float(args.max_fps))

    try:
        for i in range(int(args.steps)):
            t0 = time.time()
            frame = perception.fetch_snapshot()
            obs = perception.observe(frame)

            print(
                f"[{i:03d}] "
                f"ball(v={int(obs['ball_visible'])}, d={obs['ball_distance']:.3f}, b={obs['ball_bearing']:+.3f}, c={obs['ball_confidence']:.2f}) "
                f"goal(v={int(obs['goal_visible'])}, d={obs['goal_distance']:.3f}, b={obs['goal_bearing']:+.3f}, c={obs['goal_confidence']:.2f})"
            )

            if writer is not None:
                writer.writerow({"step": i, **obs})

            if args.show:
                text = (
                    f"ball d={obs['ball_distance']:.2f} b={obs['ball_bearing']:+.2f} "
                    f"goal d={obs['goal_distance']:.2f} b={obs['goal_bearing']:+.2f}"
                )
                cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.imshow("AINex perception", frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            dt = time.time() - t0
            time.sleep(max(0.0, min_dt - dt))
    finally:
        if f is not None:
            f.close()
        if args.show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
