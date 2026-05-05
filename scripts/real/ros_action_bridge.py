from __future__ import annotations

import shlex
import subprocess
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class RosBridgeConfig:
    topic: str = "/app/set_action"
    msg_type: str = "std_msgs/String"
    ros_setup_script: str = "/opt/ros/noetic/setup.bash"
    container_id: str = ""
    dry_run: bool = False
    timeout_sec: float = 8.0


class ROSActionBridge:
    """
    Publish action-group names to a ROS topic.

    Modes:
    - Local ROS shell: publishes directly.
    - Docker ROS: wraps publish command with `docker exec <container> bash -lc ...`.
    """

    def __init__(self, config: Optional[RosBridgeConfig] = None):
        self.cfg = config or RosBridgeConfig()

    def _pub_shell_command(self, action_name: str) -> str:
        escaped = action_name.replace("'", "\\'")
        return (
            f"source {shlex.quote(self.cfg.ros_setup_script)} && "
            f"rostopic pub -1 {shlex.quote(self.cfg.topic)} {shlex.quote(self.cfg.msg_type)} "
            f"\"data: '{escaped}'\""
        )

    def _build_command(self, action_name: str) -> List[str]:
        inner = self._pub_shell_command(action_name)
        if self.cfg.container_id:
            return ["docker", "exec", self.cfg.container_id, "bash", "-lc", inner]
        return ["bash", "-lc", inner]

    def send_action(self, action_name: str) -> bool:
        cmd = self._build_command(action_name)

        if self.cfg.dry_run:
            print(f"[ros_bridge] DRY RUN: {' '.join(cmd)}")
            return True

        try:
            subprocess.run(cmd, check=True, timeout=self.cfg.timeout_sec)
            return True
        except subprocess.TimeoutExpired:
            print(f"[ros_bridge] timeout while sending action '{action_name}'")
            return False
        except subprocess.CalledProcessError as exc:
            print(f"[ros_bridge] failed action '{action_name}': {exc}")
            return False
