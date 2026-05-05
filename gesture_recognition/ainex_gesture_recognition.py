#!/usr/bin/env python3
"""
Real-time hand-gesture recognition from the AINex web_video_server snapshot URL
using MediaPipe Tasks and a trained .task model.

This version is adapted from your working webcam script so it uses:
- your trained .task file
- the Pi's snapshot URL instead of cv2.VideoCapture(camera_index)

Usage:
  python gesture_ros_compressed.py \
    --model /home/pi/gesture_recognition/FISTBUMP_HIGHFIVE_HANDSHAKE.task \
    --url "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw"

Notes:
- This uses VIDEO running mode and recognize_for_video(), just like your laptop script.
- It fetches a fresh snapshot each loop to avoid MJPEG buffering lag.
- If no hand/gesture is detected, it prints: NONE
"""

from __future__ import annotations

import argparse
import sys
import time
import urllib.request
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe GestureRecognizer runner using AINex snapshot URL (prints gesture to stdout)."
    )
    p.add_argument("--model", required=True, help="Path to a MediaPipe Tasks gesture recognizer .task model file.")
    p.add_argument(
        "--url",
        required=True,
        help='Snapshot URL, e.g. "http://192.168.149.1:8080/snapshot?topic=/camera/image_raw"',
    )
    p.add_argument("--max-fps", type=float, default=5.0, help="How often to fetch/process a new snapshot.")
    p.add_argument("--mirror", action="store_true", help="Mirror frames horizontally.")
    p.add_argument("--min-score", type=float, default=0.0, help="Only print gestures with score >= this threshold.")
    p.add_argument(
        "--print-on-change",
        action="store_true",
        help="Only print when the estimated gesture label changes (reduces spam).",
    )
    p.add_argument(
        "--print-every",
        type=int,
        default=1,
        help="Print every N processed frames (ignored if --print-on-change is set).",
    )
    p.add_argument("--show", action="store_true", help="Show the latest snapshot with overlay text.")
    return p.parse_args()


def fetch_snapshot(url: str) -> np.ndarray:
    with urllib.request.urlopen(url, timeout=5) as resp:
        data = resp.read()

    arr = np.frombuffer(data, dtype=np.uint8)
    frame_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if frame_bgr is None:
        raise RuntimeError("Failed to decode JPEG snapshot from URL.")
    return frame_bgr


def build_recognizer(model_path: str) -> vision.GestureRecognizer:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.GestureRecognizer.create_from_options(options)


def top_gesture_label_and_score(
    result: vision.GestureRecognizerResult,
) -> Optional[Tuple[str, float]]:
    # result.gestures is a list of hands; each hand is a list of Category objects sorted by score desc.
    if not result.gestures:
        return None
    if not result.gestures[0]:
        return None
    top = result.gestures[0][0]
    return (top.category_name, float(top.score))


def main() -> int:
    args = parse_args()

    try:
        recognizer = build_recognizer(args.model)
    except Exception as e:
        print(
            f"ERROR: Failed to create GestureRecognizer from model '{args.model}'.\n{e}",
            file=sys.stderr,
        )
        return 2

    last_label: Optional[str] = None
    frame_idx = 0
    start_time = time.time()
    min_dt = 1.0 / max(1e-6, args.max_fps)

    print(f"Loaded model: {args.model}", flush=True)
    print(f"Polling snapshots from: {args.url}", flush=True)

    try:
        while True:
            loop_start = time.time()

            try:
                frame_bgr = fetch_snapshot(args.url)
            except Exception as e:
                print(f"ERROR: Failed to fetch snapshot.\n{e}", file=sys.stderr, flush=True)
                time.sleep(0.2)
                continue

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Timestamp must be monotonically increasing for VIDEO mode (milliseconds).
            timestamp_ms = int((time.time() - start_time) * 1000)

            try:
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"ERROR: recognize_for_video failed.\n{e}", file=sys.stderr, flush=True)
                return 3

            top = top_gesture_label_and_score(result)

            should_print = False
            out = "NONE"

            if top is not None:
                label, score = top
                if score >= args.min_score:
                    out = f"{label}\t{score:.3f}"
                else:
                    out = "NONE"

            if args.print_on_change:
                current_label = out.split("\t", 1)[0]
                if current_label != last_label:
                    should_print = True
                    last_label = current_label
            else:
                if args.print_every <= 1 or (frame_idx % args.print_every == 0):
                    should_print = True

            if should_print:
                print(out, flush=True)

            if args.show:
                overlay = out.replace("\t", "  ")
                cv2.putText(
                    frame_bgr,
                    overlay,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                cv2.imshow("Gesture Snapshot", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1

            elapsed = time.time() - loop_start
            sleep_time = max(0.0, min_dt - elapsed)
            time.sleep(sleep_time)

    except KeyboardInterrupt:
        return 0
    finally:
        try:
            recognizer.close()
        except Exception:
            pass
        if args.show:
            cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())