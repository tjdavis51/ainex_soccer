from __future__ import annotations

import argparse
import curses
import json
import os
import queue
import shlex
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional


DEFAULT_KEYMAP = {
    "w": "forward",
    "a": "turn_left",
    "d": "turn_right",
    "s": "stand",
    "r": "walk_ready",
    "j": "left_shot",
    "k": "right_shot",
    " ": "stand",
}


@dataclass
class PublishConfig:
    host: str
    user: str
    container_id: str
    ros_topic: str = "/app/set_action"
    ros_msg_type: str = "std_msgs/String"
    ros_setup_script: str = "/opt/ros/noetic/setup.bash"
    ssh_port: int = 22
    timeout_sec: float = 5.0
    local_mode: bool = False
    identity_file: str = ""
    connect_timeout_sec: float = 2.0
    ssh_control_path: str = ""


class LatestOnlySender:
    def __init__(self, cfg: PublishConfig, *, min_interval_sec: float) -> None:
        self.cfg = cfg
        self.min_interval_sec = float(min_interval_sec)
        self._q: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1)
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._stop = threading.Event()
        self._last_sent_action = ""
        self._last_sent_time = 0.0
        self._ssh_control_path = self._resolve_control_path(cfg.ssh_control_path)

    def _resolve_control_path(self, path: str) -> str:
        if not path:
            path = "~/.ssh/ainex_mux_%h_%p_%r"
        return os.path.expanduser(path)

    def start(self) -> None:
        self._thread.start()

    def close(self) -> None:
        self._stop.set()
        self.submit(None)
        self._thread.join(timeout=1.0)

    def submit(self, action_name: Optional[str]) -> None:
        try:
            self._q.put_nowait(action_name)
        except queue.Full:
            try:
                _ = self._q.get_nowait()
            except queue.Empty:
                pass
            self._q.put_nowait(action_name)

    def _build_publish_inner(self, action_name: str) -> str:
        escaped = action_name.replace("'", "\\'")
        return (
            f"source {shlex.quote(self.cfg.ros_setup_script)} && "
            f"rostopic pub -1 {shlex.quote(self.cfg.ros_topic)} {shlex.quote(self.cfg.ros_msg_type)} "
            f"\"data: '{escaped}'\""
        )

    def _run_publish(self, action_name: str) -> None:
        inner = self._build_publish_inner(action_name)
        if self.cfg.local_mode:
            cmd = ["bash", "-lc", inner]
        else:
            docker_inner = (
                f"docker exec {shlex.quote(self.cfg.container_id)} bash -lc {shlex.quote(inner)}"
            )
            cmd = [
                "ssh",
                "-p",
                str(self.cfg.ssh_port),
                "-o",
                "BatchMode=yes",
                "-o",
                "StrictHostKeyChecking=accept-new",
                "-o",
                f"ConnectTimeout={int(max(1, self.cfg.connect_timeout_sec))}",
                "-o",
                "ControlMaster=auto",
                "-o",
                "ControlPersist=60",
                "-o",
                f"ControlPath={self._ssh_control_path}",
                f"{self.cfg.user}@{self.cfg.host}",
                docker_inner,
            ]
            if self.cfg.identity_file:
                cmd[1:1] = ["-i", self.cfg.identity_file]
        try:
            subprocess.run(
                cmd,
                check=False,
                timeout=self.cfg.timeout_sec,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except (subprocess.TimeoutExpired, OSError, ValueError):
            return

    def _worker(self) -> None:
        while not self._stop.is_set():
            action = self._q.get()
            if action is None:
                continue

            now = time.time()
            if action == self._last_sent_action and (now - self._last_sent_time) < self.min_interval_sec:
                continue

            self._run_publish(action)
            self._last_sent_action = action
            self._last_sent_time = time.time()


def _load_keymap(path: Optional[Path]) -> Dict[str, str]:
    if path is None:
        return dict(DEFAULT_KEYMAP)
    payload = json.loads(path.read_text())
    mapping: Dict[str, str] = {}
    for k, v in payload.items():
        if isinstance(k, str) and len(k) > 0 and isinstance(v, str) and len(v) > 0:
            mapping[k] = v
    return mapping


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="AINex keyboard action-group controller over SSH.")
    p.add_argument("--host", default="192.168.149.1")
    p.add_argument("--user", default="pi")
    p.add_argument("--port", type=int, default=22)
    p.add_argument("--container-id", default="")
    p.add_argument("--local-mode", action="store_true", help="Publish locally on this machine (no ssh).")
    p.add_argument("--topic", default="/app/set_action")
    p.add_argument("--ros-msg-type", default="std_msgs/String")
    p.add_argument("--ros-setup-script", default="/opt/ros/noetic/setup.bash")
    p.add_argument("--keymap-json", type=Path, default=None)
    p.add_argument("--repeat-interval", type=float, default=0.35, help="Minimum seconds between same-action publishes.")
    p.add_argument("--send-timeout", type=float, default=5.0)
    p.add_argument("--ssh-connect-timeout", type=float, default=2.0)
    p.add_argument("--identity-file", default="", help="SSH private key path (recommended).")
    p.add_argument("--ssh-control-path", default="~/.ssh/ainex_mux_%h_%p_%r")
    return p.parse_args()


def _run_ui(stdscr: "curses._CursesWindow", sender: LatestOnlySender, keymap: Dict[str, str]) -> None:
    curses.noecho()
    curses.cbreak()
    stdscr.nodelay(True)
    stdscr.keypad(True)

    while True:
        code = stdscr.getch()
        if code == -1:
            time.sleep(0.01)
            continue

        if code in (27,):  # ESC
            break
        if code in (ord("q"), ord("Q")):
            break

        try:
            ch = chr(code)
        except (TypeError, ValueError):
            continue

        action = keymap.get(ch.lower())
        if action:
            sender.submit(action)


def main() -> None:
    args = _parse_args()
    keymap = _load_keymap(args.keymap_json)
    cfg = PublishConfig(
        host=args.host,
        user=args.user,
        container_id=args.container_id,
        ros_topic=args.topic,
        ros_msg_type=args.ros_msg_type,
        ros_setup_script=args.ros_setup_script,
        ssh_port=args.port,
        timeout_sec=args.send_timeout,
        local_mode=bool(args.local_mode),
        identity_file=args.identity_file,
        connect_timeout_sec=args.ssh_connect_timeout,
        ssh_control_path=args.ssh_control_path,
    )
    sender = LatestOnlySender(cfg, min_interval_sec=float(args.repeat_interval))
    sender.start()
    try:
        try:
            curses.wrapper(_run_ui, sender, keymap)
        except KeyboardInterrupt:
            pass
    finally:
        sender.close()


if __name__ == "__main__":
    main()
