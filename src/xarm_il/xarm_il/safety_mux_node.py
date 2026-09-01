"""Command source multiplexer and independent Cartesian safety envelope."""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.duration import Duration
from rclpy.node import Node
from std_msgs.msg import Bool, String
from std_srvs.srv import Trigger
from tf2_ros import Buffer, TransformException, TransformListener

from .control import WorkspaceBounds, clamp_cartesian_velocity, enforce_workspace


@dataclass
class TimedCommand:
    velocity: np.ndarray
    received_s: float
    frame_id: str


class SafetyMuxNode(Node):
    VALID_MODES = {"idle", "teleop", "policy"}

    def __init__(self) -> None:
        super().__init__("safety_mux")
        defaults = {
            "teleop_topic": "/xarm_il/teleop_twist",
            "policy_topic": "/xarm_il/policy_twist",
            "mode_topic": "/xarm_il/control_mode",
            "deadman_topic": "/xarm_il/deadman",
            "servo_topic": "/servo_server/delta_twist_cmds",
            "executed_topic": "/xarm_il/executed_twist",
            "tcp_pose_topic": "/xarm_il/tcp_pose",
            "status_topic": "/xarm_il/safety_status",
            "command_frame": "link_base",
            "eef_frame": "link_eef",
            "initial_mode": "teleop",
            "publish_rate_hz": 20.0,
            "command_timeout_s": 0.25,
            "max_xy_speed": 0.04,
            "max_z_speed": 0.015,
            "workspace_min": [0.20, -0.35, 0.05],
            "workspace_max": [0.65, 0.35, 0.50],
            "workspace_margin": 0.005,
            "workspace_lookahead_s": 0.25,
            "require_tf": True,
            "dry_run": True,
            "start_servo_service": "/servo_server/start_servo",
            "auto_start_servo": True,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

        self.command_frame = str(self.get_parameter("command_frame").value)
        self.eef_frame = str(self.get_parameter("eef_frame").value)
        self.mode = str(self.get_parameter("initial_mode").value)
        if self.mode not in self.VALID_MODES:
            raise ValueError(f"initial_mode must be one of {sorted(self.VALID_MODES)}")
        self.rate_hz = float(self.get_parameter("publish_rate_hz").value)
        self.timeout_s = float(self.get_parameter("command_timeout_s").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)
        self.margin = float(self.get_parameter("workspace_margin").value)
        self.lookahead = float(self.get_parameter("workspace_lookahead_s").value)
        self.require_tf = bool(self.get_parameter("require_tf").value)
        self.dry_run = bool(self.get_parameter("dry_run").value)
        if self.rate_hz <= 0.0:
            raise ValueError("publish_rate_hz must be positive")
        if self.timeout_s <= 0.0:
            raise ValueError("command_timeout_s must be positive")
        if self.max_xy_speed <= 0.0 or self.max_z_speed <= 0.0:
            raise ValueError("Cartesian speed limits must be positive")
        if self.margin < 0.0 or self.lookahead < 0.0:
            raise ValueError("workspace margin and lookahead must be non-negative")
        self.bounds = WorkspaceBounds.from_sequences(
            self.get_parameter("workspace_min").value,
            self.get_parameter("workspace_max").value,
        )

        self._commands: dict[str, TimedCommand | None] = {"teleop": None, "policy": None}
        self._deadman = False
        self._deadman_received_s = -1e9
        self._last_status = ""
        self._last_status_publish_s = -1e9
        self._last_tf_warning_s = -1e9
        self._servo_start_requested = False

        self.tf_buffer = Buffer(cache_time=Duration(seconds=2.0))
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("teleop_topic").value),
            lambda msg: self._command_callback("teleop", msg),
            10,
        )
        self.create_subscription(
            TwistStamped,
            str(self.get_parameter("policy_topic").value),
            lambda msg: self._command_callback("policy", msg),
            10,
        )
        self.create_subscription(
            String, str(self.get_parameter("mode_topic").value), self._mode_callback, 10
        )
        self.create_subscription(
            Bool, str(self.get_parameter("deadman_topic").value), self._deadman_callback, 10
        )
        self.servo_pub = self.create_publisher(
            TwistStamped, str(self.get_parameter("servo_topic").value), 10
        )
        self.executed_pub = self.create_publisher(
            TwistStamped, str(self.get_parameter("executed_topic").value), 10
        )
        self.pose_pub = self.create_publisher(
            PoseStamped, str(self.get_parameter("tcp_pose_topic").value), 10
        )
        self.status_pub = self.create_publisher(
            String, str(self.get_parameter("status_topic").value), 10
        )

        self._auto_start_servo = bool(self.get_parameter("auto_start_servo").value)
        self._servo_client = self.create_client(
            Trigger, str(self.get_parameter("start_servo_service").value)
        )
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)
        self.get_logger().info(
            f"Safety mux ready in {self.mode!r} mode; output={self.servo_pub.topic_name}, "
            f"workspace={self.bounds.minimum.tolist()}..{self.bounds.maximum.tolist()} m"
        )

    def _now_s(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _command_callback(self, source: str, message: TwistStamped) -> None:
        velocity = np.array(
            [message.twist.linear.x, message.twist.linear.y, message.twist.linear.z],
            dtype=np.float64,
        )
        self._commands[source] = TimedCommand(velocity, self._now_s(), message.header.frame_id)

    def _mode_callback(self, message: String) -> None:
        requested = message.data.strip().lower()
        if requested not in self.VALID_MODES:
            self.get_logger().warning(f"Ignoring invalid control mode: {requested!r}")
            return
        if requested != self.mode:
            self.mode = requested
            self.get_logger().warning(f"Control mode changed to: {self.mode}")

    def _deadman_callback(self, message: Bool) -> None:
        self._deadman = bool(message.data)
        self._deadman_received_s = self._now_s()
        if self._deadman and self.mode == "policy":
            # A human pressing the deadman always takes control and latches teleop mode.
            self.mode = "teleop"
            self.get_logger().warning("Human takeover: policy -> teleop")

    def _lookup_tcp(self) -> np.ndarray | None:
        try:
            transform = self.tf_buffer.lookup_transform(
                self.command_frame, self.eef_frame, rclpy.time.Time()
            )
        except TransformException as exc:
            now = self._now_s()
            if now - self._last_tf_warning_s > 2.0:
                self.get_logger().warning(f"TCP transform unavailable; command held at zero: {exc}")
                self._last_tf_warning_s = now
            return None
        translation = transform.transform.translation
        rotation = transform.transform.rotation
        pose = PoseStamped()
        pose.header = transform.header
        pose.pose.position.x = translation.x
        pose.pose.position.y = translation.y
        pose.pose.position.z = translation.z
        pose.pose.orientation = rotation
        self.pose_pub.publish(pose)
        return np.array([translation.x, translation.y, translation.z], dtype=np.float64)

    def _try_start_servo(self) -> None:
        if self.dry_run or not self._auto_start_servo or self._servo_start_requested:
            return
        if not self._servo_client.service_is_ready():
            return
        self._servo_client.call_async(Trigger.Request())
        self._servo_start_requested = True
        self.get_logger().info("Requested MoveIt Servo start")

    def _select_velocity(self, now_s: float) -> tuple[np.ndarray, str, bool]:
        if self.mode == "idle":
            return np.zeros(3), "idle", False
        if now_s - self._deadman_received_s > self.timeout_s:
            return np.zeros(3), f"{self.mode}:keyboard_watchdog", True
        if self.mode == "teleop" and not self._deadman:
            return np.zeros(3), "teleop:no_deadman", False
        command = self._commands[self.mode]
        if command is None or now_s - command.received_s > self.timeout_s:
            return np.zeros(3), f"{self.mode}:stale", True
        if command.frame_id and command.frame_id != self.command_frame:
            return np.zeros(3), f"{self.mode}:wrong_frame", True
        return command.velocity.copy(), self.mode, False

    def _on_timer(self) -> None:
        self._try_start_servo()
        now_s = self._now_s()
        velocity, source, stale = self._select_velocity(now_s)
        velocity = clamp_cartesian_velocity(velocity, self.max_xy_speed, self.max_z_speed)
        tcp = self._lookup_tcp()
        blocked: dict[str, bool] = {}
        if tcp is None and self.require_tf:
            velocity.fill(0.0)
            source += ":no_tf"
        elif tcp is not None:
            velocity, blocked = enforce_workspace(
                velocity, tcp, self.bounds, self.lookahead, self.margin
            )

        output = TwistStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.command_frame
        output.twist.linear.x = float(velocity[0])
        output.twist.linear.y = float(velocity[1])
        output.twist.linear.z = float(velocity[2])
        if not self.dry_run:
            self.servo_pub.publish(output)
        else:
            output.twist.linear.x = 0.0
            output.twist.linear.y = 0.0
            output.twist.linear.z = 0.0
            source += ":dry_run"
        self.executed_pub.publish(output)

        status = json.dumps(
            {
                "mode": self.mode,
                "source": source,
                "stale": stale,
                "blocked": blocked,
                "dry_run": self.dry_run,
            },
            sort_keys=True,
        )
        if status != self._last_status or now_s - self._last_status_publish_s >= 1.0:
            self.status_pub.publish(String(data=status))
            self._last_status = status
            self._last_status_publish_s = now_s

    def publish_stop(self) -> None:
        """Send an explicit final halt in addition to MoveIt Servo's command timeout."""
        output = TwistStamped()
        output.header.stamp = self.get_clock().now().to_msg()
        output.header.frame_id = self.command_frame
        if not self.dry_run:
            self.servo_pub.publish(output)
        self.executed_pub.publish(output)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = SafetyMuxNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.publish_stop()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
