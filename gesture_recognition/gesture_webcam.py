#!/usr/bin/env python3
"""
Real-time hand-gesture recognition from a webcam/USB camera using MediaPipe Tasks.

- Python 3.12 compatible (requires a MediaPipe wheel that supports 3.12, e.g. mediapipe>=0.10.13).
- Prints the top "estimated" gesture (label + score) to stdout.

Usage:
  python gesture_webcam.py --model /path/to/your_gesture.task
  python gesture_webcam.py --model your.task --camera 1 --min-score 0.6

Notes:
- This uses VIDEO running mode and recognize_for_video(), which is appropriate for camera frames.
- If no hand/gesture is detected, it prints: NONE
"""
from __future__ import annotations

import argparse
import sys
import time
from typing import Optional, Tuple

import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="MediaPipe GestureRecognizer webcam runner (prints gesture to stdout).")
    p.add_argument("--model", required=True, help="Path to a MediaPipe Tasks gesture recognizer .task model file.")
    p.add_argument("--camera", type=int, default=0, help="OpenCV camera index (0 is usually the built-in webcam).")
    p.add_argument("--width", type=int, default=640, help="Requested capture width.")
    p.add_argument("--height", type=int, default=480, help="Requested capture height.")
    p.add_argument("--fps", type=int, default=30, help="Requested capture FPS (best-effort).")
    p.add_argument("--mirror", action="store_true", help="Mirror frames horizontally (often nicer for webcams).")
    p.add_argument("--min-score", type=float, default=0.0, help="Only print gestures with score >= this threshold.")
    p.add_argument("--print-on-change", action="store_true",
                   help="Only print when the estimated gesture label changes (reduces spam).")
    p.add_argument("--print-every", type=int, default=1,
                   help="Print every N frames (ignored if --print-on-change is set).")
    return p.parse_args()


def open_camera(index: int, width: int, height: int, fps: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}. Try --camera 1,2,... or check permissions.")

    # Best-effort property sets; some cameras ignore these.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
    cap.set(cv2.CAP_PROP_FPS, float(fps))
    return cap


def build_recognizer(model_path: str) -> vision.GestureRecognizer:
    base_options = python.BaseOptions(model_asset_path=model_path)
    options = vision.GestureRecognizerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
    )
    return vision.GestureRecognizer.create_from_options(options)


def top_gesture_label_and_score(result: vision.GestureRecognizerResult) -> Optional[Tuple[str, float]]:
    # result.gestures is a list of hands; each hand is a list of Category objects sorted by score desc.
    if not result.gestures:
        return None
    if not result.gestures[0]:
        return None
    top = result.gestures[0][0]
    # Category fields: category_name, score, index, display_name (display_name often empty)
    return (top.category_name, float(top.score))


def main() -> int:
    args = parse_args()

    try:
        recognizer = build_recognizer(args.model)
    except Exception as e:
        print(f"ERROR: Failed to create GestureRecognizer from model '{args.model}'.\n{e}", file=sys.stderr)
        return 2

    try:
        cap = open_camera(args.camera, args.width, args.height, args.fps)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    last_label: Optional[str] = None
    frame_idx = 0
    start_time = time.time()

    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or frame_bgr is None:
                print("ERROR: Failed to read frame from camera.", file=sys.stderr)
                return 3

            if args.mirror:
                frame_bgr = cv2.flip(frame_bgr, 1)

            # MediaPipe expects SRGB (RGB) uint8.
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Timestamp must be monotonically increasing for VIDEO mode (milliseconds).
            timestamp_ms = int((time.time() - start_time) * 1000)

            result = recognizer.recognize_for_video(mp_image, timestamp_ms)
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
                current_label = out.split("\t", 1)[0]  # "NONE" or label
                if current_label != last_label:
                    should_print = True
                    last_label = current_label
            else:
                if args.print_every <= 1 or (frame_idx % args.print_every == 0):
                    should_print = True

            if should_print:
                print(out, flush=True)

            frame_idx += 1

    except KeyboardInterrupt:
        return 0
    finally:
        cap.release()
        try:
            recognizer.close()
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
