#!/usr/bin/env python3
"""
Gesture recognition + response demo with probabilistic high-five response.

Behavior:
- fistbump: fixed action (default: wave)
- handshake: fixed action (default: wave)
- highfive: primary action most of the time, alternate action with configurable probability
"""

from __future__ import annotations

import argparse
import random
import subprocess
import sys
import time
import urllib.request
from typing import Optional, Tuple

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="MediaPipe GestureRecognizer runner using AINex snapshot URL with probabilistic high-five response."
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
    p.add_argument("--print-on-change", action="store_true", help="Only print when gesture label changes.")
    p.add_argument("--print-every", type=int, default=1, help="Print every N frames (ignored with --print-on-change).")
    p.add_argument("--show", action="store_true", help="Show snapshot window with overlay.")
    p.add_argument("--quiet", action="store_true", help="Minimal stdout; suppress periodic gesture prints.")

    # Trigger logic
    p.add_argument("--trigger-score", type=float, default=0.70, help="Minimum score required to begin persistence timing.")
    p.add_argument("--persist-sec", type=float, default=2.0, help="How long the trigger gesture must persist before firing.")
    p.add_argument("--cooldown-sec", type=float, default=5.0, help="Cooldown after a trigger before another can fire.")
    p.add_argument("--container-id", default="82df027dddb8", help="Docker container ID running ROS.")
    p.add_argument("--dry-run", action="store_true", help="Do not publish action, only print what would trigger.")

    # Fixed gesture actions
    p.add_argument("--fistbump-action", default="wave")
    p.add_argument("--handshake-action", default="wave")

    # High-five probabilistic action
    p.add_argument("--highfive-primary-action", default="greet")
    p.add_argument("--highfive-alt-action", default="psych")
    p.add_argument(
        "--highfive-alt-prob",
        type=float,
        default=0.10,
        help="Probability [0,1] to use alternate highfive action instead of primary.",
    )
    p.add_argument("--seed", type=int, default=0, help="Random seed for reproducible demo behavior.")
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
    if not result.gestures or not result.gestures[0]:
        return None
    top = result.gestures[0][0]
    return (top.category_name, float(top.score))


def trigger_action(container_id: str, action_name: str, dry_run: bool = False, quiet: bool = False) -> None:
    cmd = [
        "docker",
        "exec",
        container_id,
        "bash",
        "-lc",
        f"source /opt/ros/noetic/setup.bash && rostopic pub -1 /app/set_action std_msgs/String \"data: '{action_name}'\"",
    ]

    if dry_run:
        if not quiet:
            print(f"DRY RUN trigger: {' '.join(cmd)}", flush=True)
        return

    try:
        subprocess.run(cmd, check=True)
        if not quiet:
            print(f"TRIGGERED ACTION: {action_name}", flush=True)
    except subprocess.CalledProcessError as e:
        print(f"ERROR: Failed to trigger action '{action_name}'.\n{e}", file=sys.stderr, flush=True)


def choose_action_for_gesture(label: str, args: argparse.Namespace, rng: random.Random) -> Optional[str]:
    if label == "fistbump":
        return args.fistbump_action
    if label == "handshake":
        return args.handshake_action
    if label == "highfive":
        alt_p = min(1.0, max(0.0, float(args.highfive_alt_prob)))
        if rng.random() < alt_p:
            return args.highfive_alt_action
        return args.highfive_primary_action
    return None


def main() -> int:
    args = parse_args()
    rng = random.Random(args.seed)

    try:
        recognizer = build_recognizer(args.model)
    except Exception as e:
        print(f"ERROR: Failed to create GestureRecognizer from model '{args.model}'.\n{e}", file=sys.stderr)
        return 2

    last_label: Optional[str] = None
    frame_idx = 0
    start_time = time.time()
    min_dt = 1.0 / max(1e-6, args.max_fps)
    candidate_start_time: Optional[float] = None
    candidate_label: Optional[str] = None
    last_trigger_time: float = -1e9
    last_action_fired: str = ""

    if not args.quiet:
        print(f"Loaded model: {args.model}", flush=True)
        print(f"Polling snapshots from: {args.url}", flush=True)
        print(
            f"Response config: score>={args.trigger_score}, persist={args.persist_sec}s, cooldown={args.cooldown_sec}s",
            flush=True,
        )
        print(
            "Gesture map: "
            f"fistbump->{args.fistbump_action}, "
            f"handshake->{args.handshake_action}, "
            f"highfive->{args.highfive_primary_action}/{args.highfive_alt_action} "
            f"(alt_prob={args.highfive_alt_prob:.2f})",
            flush=True,
        )

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
            timestamp_ms = int((time.time() - start_time) * 1000)

            try:
                result = recognizer.recognize_for_video(mp_image, timestamp_ms)
            except Exception as e:
                print(f"ERROR: recognize_for_video failed.\n{e}", file=sys.stderr, flush=True)
                return 3

            top = top_gesture_label_and_score(result)
            should_print = False
            out = "NONE"
            detected_label: Optional[str] = None
            detected_score: float = 0.0

            if top is not None:
                label, score = top
                detected_label = label
                detected_score = score
                if score >= args.min_score:
                    out = f"{label}\t{score:.3f}"

            now = time.time()
            action_for_label = None if detected_label is None else choose_action_for_gesture(detected_label, args, rng)
            trigger_match = action_for_label is not None and detected_score >= args.trigger_score

            if trigger_match:
                if candidate_label != detected_label:
                    candidate_label = detected_label
                    candidate_start_time = now
                elif candidate_start_time is not None and (now - candidate_start_time) >= args.persist_sec:
                    if (now - last_trigger_time) >= args.cooldown_sec:
                        selected_action = choose_action_for_gesture(detected_label, args, rng) or ""
                        if selected_action:
                            trigger_action(
                                args.container_id,
                                selected_action,
                                dry_run=args.dry_run,
                                quiet=args.quiet,
                            )
                            last_action_fired = selected_action
                            last_trigger_time = now
                            candidate_start_time = None
                            candidate_label = None
            else:
                candidate_start_time = None
                candidate_label = None

            if args.print_on_change:
                current_label = out.split("\t", 1)[0]
                if current_label != last_label:
                    should_print = True
                    last_label = current_label
            else:
                if args.print_every <= 1 or (frame_idx % args.print_every == 0):
                    should_print = True

            if should_print and (not args.quiet):
                print(out, flush=True)

            if args.show:
                overlay = out.replace("\t", "  ")
                if trigger_match and candidate_start_time is not None and candidate_label is not None:
                    persist_progress = min(1.0, (now - candidate_start_time) / max(1e-6, args.persist_sec))
                    persist_text = f"{candidate_label} persist {persist_progress:.0%}"
                else:
                    persist_text = "persist 0%"

                cooldown_remaining = max(0.0, args.cooldown_sec - (now - last_trigger_time))
                cooldown_text = f"cooldown {cooldown_remaining:.1f}s"
                action_text = f"last_action {last_action_fired}" if last_action_fired else "last_action none"

                cv2.putText(frame_bgr, overlay, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, persist_text, (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, cooldown_text, (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 200, 255), 2, cv2.LINE_AA)
                cv2.putText(frame_bgr, action_text, (10, 135), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (220, 220, 220), 2, cv2.LINE_AA)
                cv2.imshow("Gesture Snapshot", frame_bgr)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
            elapsed = time.time() - loop_start
            time.sleep(max(0.0, min_dt - elapsed))

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
