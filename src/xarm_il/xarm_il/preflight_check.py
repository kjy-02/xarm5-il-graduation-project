"""Short ROS 2 preflight check for cameras, joints, TF-derived TCP, and safety state."""

from __future__ import annotations

import argparse
import json
import time

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String


class PreflightCheckNode(Node):
    def __init__(self, camera_topics: list[str], joint_names: list[str]) -> None:
        super().__init__("preflight_check")
        self.camera_counts = {topic: 0 for topic in camera_topics}
        self.image_shapes: dict[str, tuple[int, int, str]] = {}
        self.joint_names = tuple(joint_names)
        self.joint_messages = 0
        self.joint_missing: set[str] = set(joint_names)
        self.tcp_messages = 0
        self.safety_status: dict[str, object] | None = None
        for topic in camera_topics:
            self.create_subscription(
                Image,
                topic,
                lambda message, source=topic: self._image_callback(source, message),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            JointState, "/joint_states", self._joint_callback, qos_profile_sensor_data
        )
        self.create_subscription(PoseStamped, "/xarm_il/tcp_pose", self._tcp_callback, 10)
        self.create_subscription(String, "/xarm_il/safety_status", self._status_callback, 10)

    def _image_callback(self, topic: str, message: Image) -> None:
        self.camera_counts[topic] += 1
        self.image_shapes[topic] = (message.width, message.height, message.encoding)

    def _joint_callback(self, message: JointState) -> None:
        self.joint_messages += 1
        available = set(message.name)
        self.joint_missing = set(self.joint_names) - available

    def _tcp_callback(self, _: PoseStamped) -> None:
        self.tcp_messages += 1

    def _status_callback(self, message: String) -> None:
        try:
            self.safety_status = json.loads(message.data)
        except json.JSONDecodeError:
            self.safety_status = {"raw": message.data}

    def report(self, duration_s: float) -> dict[str, object]:
        errors = []
        for topic, count in self.camera_counts.items():
            if count == 0:
                errors.append(f"no camera messages: {topic}")
        if self.joint_messages == 0:
            errors.append("no /joint_states messages")
        elif self.joint_missing:
            errors.append(f"missing configured joints: {sorted(self.joint_missing)}")
        if self.tcp_messages == 0:
            errors.append("no /xarm_il/tcp_pose; check link_base -> link_eef TF")
        if self.safety_status is None:
            errors.append("no /xarm_il/safety_status; is safety_mux running?")
        return {
            "pass": not errors,
            "duration_s": duration_s,
            "errors": errors,
            "camera_message_counts": self.camera_counts,
            "camera_shapes": self.image_shapes,
            "joint_messages": self.joint_messages,
            "tcp_messages": self.tcp_messages,
            "safety_status": self.safety_status,
        }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--camera-topic", action="append", default=[], help="Repeat for each configured camera"
    )
    parser.add_argument(
        "--joint-name",
        action="append",
        default=[],
        help="Repeat for each configured xArm joint",
    )
    parser.add_argument("--duration", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    camera_topics = args.camera_topic or ["/camera_wrist/color/image_raw"]
    joint_names = args.joint_name or [f"joint{index}" for index in range(1, 6)]
    rclpy.init()
    node = PreflightCheckNode(camera_topics, joint_names)
    started = time.monotonic()
    try:
        while rclpy.ok() and time.monotonic() - started < args.duration:
            rclpy.spin_once(node, timeout_sec=0.1)
    finally:
        report = node.report(time.monotonic() - started)
        node.destroy_node()
        rclpy.shutdown()
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if not report["pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
