from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path
from typing import List, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.real.perception_camera import CameraIntrinsics, RealCameraPerception, RealPerceptionConfig


def _parse_hsv(text: str) -> Tuple[int, int, int]:
    vals = [int(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("HSV must be h,s,v")
    return (vals[0], vals[1], vals[2])


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate ball distance scale using known ground-truth distance.")
    parser.add_argument("--url", required=True)
    parser.add_argument("--true-distance", type=float, required=True, help="Measured robot-to-ball distance (meters)")
    parser.add_argument("--samples", type=int, default=80)
    parser.add_argument("--max-fps", type=float, default=8.0)
    parser.add_argument("--fx", type=float, default=520.0)
    parser.add_argument("--fy", type=float, default=520.0)
    parser.add_argument("--cx", type=float, default=320.0)
    parser.add_argument("--cy", type=float, default=240.0)
    parser.add_argument("--ball-diameter", type=float, default=0.08)
    parser.add_argument("--ball-hsv-low", type=str, default="5,90,60")
    parser.add_argument("--ball-hsv-high", type=str, default="25,255,255")
    args = parser.parse_args()

    cfg = RealPerceptionConfig(
        snapshot_url=args.url,
        ball_diameter_m=float(args.ball_diameter),
        ball_hsv_low=_parse_hsv(args.ball_hsv_low),
        ball_hsv_high=_parse_hsv(args.ball_hsv_high),
    )
    intr = CameraIntrinsics(fx_px=args.fx, fy_px=args.fy, cx_px=args.cx, cy_px=args.cy)
    percep = RealCameraPerception(cfg, intr)

    values: List[float] = []
    min_dt = 1.0 / max(1e-6, float(args.max_fps))

    for i in range(int(args.samples)):
        t0 = time.time()
        frame = percep.fetch_snapshot()
        ball = percep.estimate_ball(frame)
        if ball.visible:
            values.append(float(ball.distance_m))
            print(f"[{i:03d}] est={ball.distance_m:.3f}m bearing={ball.bearing_rad:+.3f} conf={ball.confidence:.2f}")
        else:
            print(f"[{i:03d}] est=NONE")
        dt = time.time() - t0
        time.sleep(max(0.0, min_dt - dt))

    if not values:
        raise SystemExit("No visible ball detections collected; tune HSV or camera placement first.")

    median_est = float(statistics.median(values))
    mean_est = float(statistics.mean(values))
    ratio = float(args.true_distance) / max(median_est, 1e-9)

    print("\nCalibration summary")
    print(f"samples_visible={len(values)}")
    print(f"true_distance_m={float(args.true_distance):.3f}")
    print(f"median_est_m={median_est:.3f}")
    print(f"mean_est_m={mean_est:.3f}")
    print(f"recommended_distance_scale={ratio:.3f}")
    print("Apply by multiplying estimated ball_distance by this scale in your runner.")


if __name__ == "__main__":
    main()
