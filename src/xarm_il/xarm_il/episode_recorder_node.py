"""Record synchronized camera, proprioception, and executed action episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String

from .control import recording_safety_reason
from .storage import EpisodeWriter, next_episode_index


@dataclass
class TimedSample:
    value: Any
    receipt_s: float
    source_s: float


def stamp_to_seconds(stamp: Any, fallback: float) -> float:
    value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
    return fallback if value <= 0.0 else value


class EpisodeRecorderNode(Node):
    def __init__(self) -> None:
        super().__init__("episode_recorder")
        defaults = {
            "dataset_root": "datasets/xarm5_drag",
            "task": "Drag the object into the side basket.",
            "fps": 10.0,
            "episode_timeout_s": 30.0,
            "max_sensor_age_s": 0.25,
            "jpeg_quality": 90,
            "min_episode_frames": 20,
            "joint_names": ["joint1", "joint2", "joint3", "joint4", "joint5"],
            "camera_names": ["wrist"],
            "camera_topics": ["/camera_wrist/color/image_raw"],
            "joint_state_topic": "/joint_states",
            "action_topic": "/xarm_il/executed_twist",
            "tcp_pose_topic": "/xarm_il/tcp_pose",
            "event_topic": "/xarm_il/record_event",
            "safety_status_topic": "/xarm_il/safety_status",
            "require_tcp_pose": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.dataset_root = Path(str(self.get_parameter("dataset_root").value)).expanduser()
        self.task = str(self.get_parameter("task").value)
        self.fps = float(self.get_parameter("fps").value)
        self.timeout_s = float(self.get_parameter("episode_timeout_s").value)
        self.max_age_s = float(self.get_parameter("max_sensor_age_s").value)
        self.jpeg_quality = int(self.get_parameter("jpeg_quality").value)
        self.min_frames = int(self.get_parameter("min_episode_frames").value)
        self.joint_names = tuple(str(x) for x in self.get_parameter("joint_names").value)
        self.require_tcp = bool(self.get_parameter("require_tcp_pose").value)
        self.camera_names = tuple(str(x) for x in self.get_parameter("camera_names").value)
        camera_topics = tuple(str(x) for x in self.get_parameter("camera_topics").value)
        if self.fps <= 0.0 or self.timeout_s <= 0.0 or self.max_age_s <= 0.0:
            raise ValueError("Recorder rates and time limits must be positive")
        if not 1 <= self.jpeg_quality <= 100:
            raise ValueError("jpeg_quality must be in [1, 100]")
        if self.min_frames <= 0:
            raise ValueError("min_episode_frames must be positive")
        if not self.camera_names or len(self.camera_names) != len(camera_topics):
            raise ValueError("camera_names and camera_topics must be non-empty and equal length")
        if len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be unique")
        self.bridge = CvBridge()

        self.images: dict[str, TimedSample | None] = {name: None for name in self.camera_names}
        self.joints: TimedSample | None = None
        self.action: TimedSample | None = None
        self.tcp_pose: TimedSample | None = None
        self.safety_status = "unknown"
        self.writer: EpisodeWriter | None = None
        self.recording_started_s = 0.0
        self.skipped_frames = 0
        self.last_source_times: np.ndarray | None = None

        for name, topic in zip(self.camera_names, camera_topics):
            self.create_subscription(
                Image,
                topic,
                lambda msg, camera=name: self._image_callback(camera, msg),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_callback,
            qos_profile_sensor_data,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("action_topic").value),
            self._action_callback,
            20,
        )
        self.create_subscription(
            PoseStamped,
            str(self.get_parameter("tcp_pose_topic").value),
            self._tcp_callback,
            20,
        )
        self.create_subscription(
            String, str(self.get_parameter("event_topic").value), self._event_callback, 10
        )
        self.create_subscription(
            String,
            str(self.get_parameter("safety_status_topic").value),
            lambda msg: setattr(self, "safety_status", msg.data),
            10,
        )
        self.timer = self.create_timer(1.0 / self.fps, self._sample_timer)
        self.get_logger().info(
            f"Recorder ready at {self.fps:g} Hz. Space=start, Right=save, Left=discard. "
            f"Dataset root: {self.dataset_root}"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _image_callback(self, name: str, message: Image) -> None:
        now = self._now_s()
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to decode {name} image: {exc}")
            return
        self.images[name] = TimedSample(
            np.ascontiguousarray(image), now, stamp_to_seconds(message.header.stamp, now)
        )

    def _joint_callback(self, message: JointState) -> None:
        now = self._now_s()
        index = {name: i for i, name in enumerate(message.name)}
        if any(name not in index for name in self.joint_names):
            missing = [name for name in self.joint_names if name not in index]
            self.get_logger().warning(f"JointState is missing configured joints: {missing}")
            return
        position = np.array([message.position[index[name]] for name in self.joint_names], dtype=np.float32)
        velocity = np.zeros(len(self.joint_names), dtype=np.float32)
        for out_idx, name in enumerate(self.joint_names):
            msg_idx = index[name]
            if msg_idx < len(message.velocity):
                velocity[out_idx] = message.velocity[msg_idx]
        self.joints = TimedSample(
            (position, velocity), now, stamp_to_seconds(message.header.stamp, now)
        )

    def _action_callback(self, message: TwistStamped) -> None:
        now = self._now_s()
        action = np.array(
            [message.twist.linear.x, message.twist.linear.y, message.twist.linear.z],
            dtype=np.float32,
        )
        self.action = TimedSample(action, now, stamp_to_seconds(message.header.stamp, now))

    def _tcp_callback(self, message: PoseStamped) -> None:
        now = self._now_s()
        p = message.pose.position
        q = message.pose.orientation
        pose = np.array([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float32)
        self.tcp_pose = TimedSample(pose, now, stamp_to_seconds(message.header.stamp, now))

    def _event_callback(self, message: String) -> None:
        event = message.data.strip().lower()
        if event == "start":
            self._start_episode()
        elif event == "save":
            self._save_episode()
        elif event == "discard":
            self._discard_episode("operator requested re-record")
        elif event == "stop":
            self._discard_episode("recording stopped")
        else:
            self.get_logger().warning(f"Unknown recorder event: {event!r}")

    def _start_episode(self) -> None:
        if self.writer is not None:
            self.get_logger().warning("Already recording; start request ignored")
            return
        safety = self._safety_dict()
        if safety is None:
            self.get_logger().warning("Cannot start recording before valid safety status arrives")
            return
        safety_reason = recording_safety_reason(safety)
        if safety_reason is not None:
            self.get_logger().warning(f"Cannot start recording: {safety_reason}")
            return
        fresh, reason = self._all_inputs_fresh(self._now_s())
        if not fresh:
            self.get_logger().warning(f"Cannot start recording: {reason}")
            return
        episode_index = next_episode_index(self.dataset_root)
        self.writer = EpisodeWriter(
            self.dataset_root,
            episode_index,
            self.joint_names,
            self.camera_names,
            self.task,
            self.fps,
        )
        self.recording_started_s = self._now_s()
        self.skipped_frames = 0
        self.last_source_times = None
        self.get_logger().warning(f"Recording episode {episode_index:06d}")

    def _save_episode(self) -> None:
        if self.writer is None:
            self.get_logger().warning("No active episode to save")
            return
        if self.writer.count < self.min_frames:
            self._discard_episode(
                f"only {self.writer.count} frames; minimum is {self.min_frames}"
            )
            return
        writer = self.writer
        self.writer = None
        path = writer.finalize(
            {
                "success": True,
                "skipped_frames": self.skipped_frames,
                "last_safety_status": self.safety_status,
            }
        )
        self.get_logger().warning(f"Saved successful episode: {path}")

    def _discard_episode(self, reason: str) -> None:
        if self.writer is None:
            return
        index = self.writer.episode_index
        count = self.writer.count
        self.writer.discard()
        self.writer = None
        self.get_logger().warning(
            f"Discarded episode {index:06d} ({count} frames): {reason}"
        )

    def _all_inputs_fresh(self, now: float) -> tuple[bool, str]:
        values: list[tuple[str, TimedSample | None]] = [
            *((name, self.images[name]) for name in self.camera_names),
            ("joint_state", self.joints),
            ("action", self.action),
        ]
        if self.require_tcp:
            values.append(("tcp_pose", self.tcp_pose))
        for name, sample in values:
            if sample is None:
                return False, f"missing {name}"
            if now - sample.receipt_s > self.max_age_s:
                return False, f"stale {name}: {now - sample.receipt_s:.3f}s"
        return True, ""

    def _safety_dict(self) -> dict[str, Any] | None:
        try:
            value = json.loads(self.safety_status)
        except (json.JSONDecodeError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _sample_timer(self) -> None:
        if self.writer is None:
            return
        now = self._now_s()
        if now - self.recording_started_s >= self.timeout_s:
            self._discard_episode(f"timeout after {self.timeout_s:.1f}s")
            return
        safety = self._safety_dict()
        if safety is None:
            self._discard_episode("safety status became invalid")
            return
        safety_reason = recording_safety_reason(safety)
        if safety_reason is not None:
            self._discard_episode(f"unsafe control state: {safety_reason}")
            return
        fresh, reason = self._all_inputs_fresh(now)
        if not fresh:
            self.skipped_frames += 1
            if self.skipped_frames % max(1, int(self.fps)) == 1:
                self.get_logger().warning(f"Skipping dataset frame: {reason}")
            return

        assert self.joints is not None and self.action is not None
        joint_position, joint_velocity = self.joints.value
        tcp_pose = self.tcp_pose.value if self.tcp_pose is not None else np.full(7, np.nan)
        image_samples = []
        for name in self.camera_names:
            sample = self.images[name]
            assert sample is not None
            image_samples.append((name, sample))
        source_times = np.asarray(
            [
                *(sample.source_s for _, sample in image_samples),
                self.joints.source_s,
                self.action.source_s,
            ],
            dtype=np.float64,
        )
        if self.last_source_times is not None and np.any(source_times <= self.last_source_times):
            self.skipped_frames += 1
            return

        encoded: dict[str, bytes] = {}
        for name, sample in image_samples:
            ok, jpeg = cv2.imencode(
                ".jpg", sample.value, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality]
            )
            if not ok:
                self.skipped_frames += 1
                self.get_logger().error(f"JPEG encoding failed for {name}")
                return
            encoded[name] = jpeg.tobytes()

        source = str(safety.get("source", "unknown"))
        self.writer.append(
            record_time=now,
            source_times=source_times,
            joint_position=joint_position,
            joint_velocity=joint_velocity,
            tcp_pose=tcp_pose,
            action=self.action.value,
            action_source=source,
            jpeg_images=encoded,
        )
        self.last_source_times = source_times

    def destroy_node(self) -> bool:
        self._discard_episode("node shutdown")
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = EpisodeRecorderNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
