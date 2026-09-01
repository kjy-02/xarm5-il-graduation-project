"""ROS 2 inference node that publishes guarded Cartesian policy commands."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import rclpy
import torch
from cv_bridge import CvBridge
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import String
from std_srvs.srv import SetBool

from .dataset import NormalizationStats, preprocess_bgr_images
from .model import ActionChunkTransformer, ModelConfig


class PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("policy_node")
        defaults = {
            "checkpoint_path": "",
            "camera_topics": ["/camera_wrist/color/image_raw"],
            "joint_state_topic": "/joint_states",
            "policy_topic": "/xarm_il/policy_twist",
            "mode_topic": "/xarm_il/control_mode",
            "enable_service": "/xarm_il/enable_policy",
            "command_frame": "link_base",
            "control_rate_hz": 10.0,
            "max_observation_age_s": 0.25,
            "max_policy_run_s": 15.0,
            "execute_steps_per_chunk": 3,
            "max_xy_speed": 0.04,
            "max_z_speed": 0.015,
            "device": "auto",
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        checkpoint_path = Path(str(self.get_parameter("checkpoint_path").value)).expanduser()
        if not str(checkpoint_path) or str(checkpoint_path) == ".":
            raise ValueError("checkpoint_path must point to a trained best.pt file")
        requested_device = str(self.get_parameter("device").value)
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        if requested_device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self.device = torch.device(requested_device)
        checkpoint = self._load_checkpoint(checkpoint_path)
        if checkpoint.get("format_version") != "xarm_il_policy_v1":
            raise ValueError(f"Unsupported policy checkpoint: {checkpoint_path}")
        self.model_config = ModelConfig(**checkpoint["model_config"])
        self.model = ActionChunkTransformer(self.model_config, pretrained_backbone=False)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.to(self.device).eval()
        self.stats = NormalizationStats.from_dict(checkpoint["normalization"])
        self.camera_names = tuple(str(name) for name in checkpoint["camera_names"])
        self.joint_names = tuple(str(name) for name in checkpoint["joint_names"])
        self.image_size = tuple(int(value) for value in checkpoint["image_size"])

        camera_topics = tuple(str(x) for x in self.get_parameter("camera_topics").value)
        if len(camera_topics) != len(self.camera_names):
            raise ValueError(
                f"Checkpoint expects {len(self.camera_names)} cameras {self.camera_names}, "
                f"but camera_topics has {len(camera_topics)} entries"
            )
        self.command_frame = str(self.get_parameter("command_frame").value)
        self.control_rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_age_s = float(self.get_parameter("max_observation_age_s").value)
        self.max_run_s = float(self.get_parameter("max_policy_run_s").value)
        self.execute_steps = int(self.get_parameter("execute_steps_per_chunk").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)
        if self.control_rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        if self.max_age_s <= 0.0 or self.max_run_s <= 0.0:
            raise ValueError("observation age and policy run limits must be positive")
        if self.max_xy_speed <= 0.0 or self.max_z_speed <= 0.0:
            raise ValueError("Cartesian speed limits must be positive")
        if self.execute_steps <= 0 or self.execute_steps > self.model_config.horizon:
            raise ValueError("execute_steps_per_chunk must be in [1, model horizon]")

        self.bridge = CvBridge()
        self.images: dict[str, tuple[np.ndarray, float] | None] = {
            name: None for name in self.camera_names
        }
        self.joint_state: tuple[np.ndarray, float] | None = None
        self.enabled = False
        self.selected = False
        self.enabled_at_s = 0.0
        self.action_queue: list[np.ndarray] = []

        for name, topic in zip(self.camera_names, camera_topics):
            self.create_subscription(
                Image,
                topic,
                lambda message, camera=name: self._image_callback(camera, message),
                qos_profile_sensor_data,
            )
        self.create_subscription(
            JointState,
            str(self.get_parameter("joint_state_topic").value),
            self._joint_callback,
            qos_profile_sensor_data,
        )
        mode_topic = str(self.get_parameter("mode_topic").value)
        self.create_subscription(String, mode_topic, self._mode_callback, 10)
        self.command_pub = self.create_publisher(
            TwistStamped, str(self.get_parameter("policy_topic").value), 10
        )
        self.mode_pub = self.create_publisher(String, mode_topic, 10)
        self.create_service(
            SetBool, str(self.get_parameter("enable_service").value), self._enable_callback
        )
        self.timer = self.create_timer(1.0 / self.control_rate_hz, self._on_timer)
        self.get_logger().info(
            f"Loaded policy {checkpoint_path} on {self.device}; disabled by default. "
            "Enable only in a cleared workspace using /xarm_il/enable_policy."
        )

    def _load_checkpoint(self, path: Path) -> dict[str, Any]:
        if not path.is_file():
            raise FileNotFoundError(path)
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:  # PyTorch versions before the weights_only argument
            return torch.load(path, map_location=self.device)

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _image_callback(self, name: str, message: Image) -> None:
        try:
            image = self.bridge.imgmsg_to_cv2(message, desired_encoding="bgr8")
        except Exception as exc:
            self.get_logger().error(f"Failed to convert {name} image: {exc}")
            return
        self.images[name] = (np.ascontiguousarray(image), self._now_s())

    def _joint_callback(self, message: JointState) -> None:
        index = {name: i for i, name in enumerate(message.name)}
        missing = [name for name in self.joint_names if name not in index]
        if missing:
            self.get_logger().warning(f"Policy JointState is missing joints: {missing}")
            return
        position = np.asarray(
            [message.position[index[name]] for name in self.joint_names], dtype=np.float32
        )
        velocity = np.zeros(len(self.joint_names), dtype=np.float32)
        for output_index, name in enumerate(self.joint_names):
            message_index = index[name]
            if message_index < len(message.velocity):
                velocity[output_index] = message.velocity[message_index]
        self.joint_state = (np.concatenate([position, velocity]), self._now_s())

    def _mode_callback(self, message: String) -> None:
        was_selected = self.selected
        self.selected = message.data.strip().lower() == "policy"
        if was_selected and not self.selected:
            self.action_queue.clear()
            self.enabled = False
            self.get_logger().warning("Policy disabled because control mode left policy")

    def _enable_callback(self, request: SetBool.Request, response: SetBool.Response) -> SetBool.Response:
        self.action_queue.clear()
        if request.data:
            fresh, reason = self._observations_fresh(self._now_s())
            if not fresh:
                self.enabled = False
                response.success = False
                response.message = f"Policy was not enabled: {reason}"
                return response
            self.enabled = True
            self.enabled_at_s = self._now_s()
            self.mode_pub.publish(String(data="policy"))
            response.message = "Policy enabled; safety mux mode requested"
            self.get_logger().warning("POLICY ENABLED")
        else:
            self.enabled = False
            self.mode_pub.publish(String(data="idle"))
            self._publish_velocity(np.zeros(3, dtype=np.float32))
            response.message = "Policy disabled; idle mode requested"
            self.get_logger().warning("Policy disabled")
        response.success = True
        return response

    def _observations_fresh(self, now_s: float) -> tuple[bool, str]:
        for name, item in self.images.items():
            if item is None:
                return False, f"missing image {name}"
            if now_s - item[1] > self.max_age_s:
                return False, f"stale image {name}"
        if self.joint_state is None:
            return False, "missing joint state"
        if now_s - self.joint_state[1] > self.max_age_s:
            return False, "stale joint state"
        return True, ""

    def _infer_chunk(self) -> list[np.ndarray]:
        assert self.joint_state is not None
        images = [self.images[name][0] for name in self.camera_names if self.images[name] is not None]
        image_array = preprocess_bgr_images(images, self.image_size)
        state = self.joint_state[0]
        normalized_state = (state - self.stats.state_mean) / self.stats.state_std
        image_tensor = torch.from_numpy(image_array).unsqueeze(0).to(self.device)
        state_tensor = torch.from_numpy(normalized_state.astype(np.float32)).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            normalized_action = self.model(image_tensor, state_tensor)[0].cpu().numpy()
        action = normalized_action * self.stats.action_std + self.stats.action_mean
        if not np.all(np.isfinite(action)):
            raise RuntimeError("policy produced NaN or infinite actions")
        return [row.astype(np.float32) for row in action[: self.execute_steps]]

    def _clip_velocity(self, velocity: np.ndarray) -> np.ndarray:
        output = velocity.copy()
        xy_norm = float(np.linalg.norm(output[:2]))
        if xy_norm > self.max_xy_speed:
            output[:2] *= self.max_xy_speed / xy_norm
        output[2] = np.clip(output[2], -self.max_z_speed, self.max_z_speed)
        return output

    def _publish_velocity(self, velocity: np.ndarray) -> None:
        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.command_frame
        message.twist.linear.x = float(velocity[0])
        message.twist.linear.y = float(velocity[1])
        message.twist.linear.z = float(velocity[2])
        self.command_pub.publish(message)

    def _on_timer(self) -> None:
        if not self.enabled or not self.selected:
            return
        now_s = self._now_s()
        if now_s - self.enabled_at_s >= self.max_run_s:
            self.enabled = False
            self.action_queue.clear()
            self._publish_velocity(np.zeros(3, dtype=np.float32))
            self.mode_pub.publish(String(data="idle"))
            self.get_logger().warning("Policy auto-disabled at maximum run time")
            return
        fresh, reason = self._observations_fresh(now_s)
        if not fresh:
            self.enabled = False
            self.action_queue.clear()
            self._publish_velocity(np.zeros(3, dtype=np.float32))
            self.mode_pub.publish(String(data="idle"))
            self.get_logger().error(f"Policy disabled because observations became invalid: {reason}")
            return
        if not self.action_queue:
            try:
                started = time.monotonic()
                self.action_queue = self._infer_chunk()
                latency_ms = (time.monotonic() - started) * 1000.0
                if latency_ms > 1000.0 / self.control_rate_hz:
                    self.get_logger().warning(
                        f"Policy inference ({latency_ms:.1f} ms) exceeds control period"
                    )
                fresh_after_inference, reason = self._observations_fresh(self._now_s())
                if not fresh_after_inference:
                    raise RuntimeError(f"observation expired during inference: {reason}")
            except Exception as exc:
                self.enabled = False
                self._publish_velocity(np.zeros(3, dtype=np.float32))
                self.mode_pub.publish(String(data="idle"))
                self.get_logger().error(f"Policy inference failed; disabled: {exc}")
                return
        velocity = self._clip_velocity(self.action_queue.pop(0))
        self._publish_velocity(velocity)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node._publish_velocity(np.zeros(3, dtype=np.float32))
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
