"""Continuous, deadman-gated keyboard teleoperation for MoveIt Servo."""

from __future__ import annotations

import queue
import threading
from typing import Any

import numpy as np
import rclpy
from geometry_msgs.msg import TwistStamped
from rclpy.node import Node
from std_msgs.msg import Bool, String

from .control import keyboard_target_velocity, smooth_velocity


class KeyboardTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("keyboard_teleop")
        self.declare_parameter("output_topic", "/xarm_il/teleop_twist")
        self.declare_parameter("event_topic", "/xarm_il/record_event")
        self.declare_parameter("mode_topic", "/xarm_il/control_mode")
        self.declare_parameter("deadman_topic", "/xarm_il/deadman")
        self.declare_parameter("frame_id", "link_base")
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("max_xy_speed", 0.03)
        self.declare_parameter("max_z_speed", 0.01)
        self.declare_parameter("smoothing_response", 0.35)
        self.declare_parameter("axis_signs", [1.0, 1.0, 1.0])

        self.frame_id = str(self.get_parameter("frame_id").value)
        self.rate_hz = float(self.get_parameter("control_rate_hz").value)
        self.max_xy_speed = float(self.get_parameter("max_xy_speed").value)
        self.max_z_speed = float(self.get_parameter("max_z_speed").value)
        self.response = float(self.get_parameter("smoothing_response").value)
        self.axis_signs = tuple(float(x) for x in self.get_parameter("axis_signs").value)
        if self.rate_hz <= 0.0:
            raise ValueError("control_rate_hz must be positive")
        if self.max_xy_speed <= 0.0 or self.max_z_speed <= 0.0:
            raise ValueError("Cartesian speed limits must be positive")
        if not 0.0 < self.response <= 1.0:
            raise ValueError("smoothing_response must be in (0, 1]")

        self.twist_pub = self.create_publisher(
            TwistStamped, str(self.get_parameter("output_topic").value), 10
        )
        self.event_pub = self.create_publisher(
            String, str(self.get_parameter("event_topic").value), 10
        )
        self.mode_pub = self.create_publisher(
            String, str(self.get_parameter("mode_topic").value), 10
        )
        self.deadman_pub = self.create_publisher(
            Bool, str(self.get_parameter("deadman_topic").value), 10
        )

        self._pressed: set[str] = set()
        self._special_pressed: set[str] = set()
        self._lock = threading.Lock()
        self._event_queue: queue.SimpleQueue[tuple[str, str]] = queue.SimpleQueue()
        self._velocity = np.zeros(3, dtype=np.float64)
        self._listener: Any | None = None
        self._start_keyboard_listener()
        self.timer = self.create_timer(1.0 / self.rate_hz, self._on_timer)
        self.get_logger().info(
            "Keyboard ready: hold Left Shift + WASD/RF. Space=start, Right=save, "
            "Left=discard, F8=teleop, F9=policy, Backspace=idle, Esc=stop."
        )

    @staticmethod
    def _special_name(key: Any) -> str:
        text = str(key)
        if text.startswith("Key."):
            return text[4:]
        return text

    def _start_keyboard_listener(self) -> None:
        try:
            from pynput import keyboard
        except ImportError as exc:
            raise RuntimeError(
                "pynput is required. Install it with: sudo apt install python3-pynput"
            ) from exc

        def on_press(key: Any) -> None:
            try:
                char = key.char.lower()
            except AttributeError:
                name = self._special_name(key)
                with self._lock:
                    first_press = name not in self._special_pressed
                    self._special_pressed.add(name)
                if not first_press:
                    return
                commands = {
                    "space": ("event", "start"),
                    "right": ("event", "save"),
                    "left": ("event", "discard"),
                    "esc": ("stop", "stop"),
                    "f8": ("mode", "teleop"),
                    "f9": ("mode", "policy"),
                    "backspace": ("mode", "idle"),
                }
                if name in commands:
                    self._event_queue.put(commands[name])
                if name in {"shift", "shift_l", "shift_r"}:
                    self._event_queue.put(("mode", "teleop"))
                return
            if char in {"w", "a", "s", "d", "r", "f"}:
                with self._lock:
                    self._pressed.add(char)

        def on_release(key: Any) -> None:
            try:
                char = key.char.lower()
            except AttributeError:
                name = self._special_name(key)
                with self._lock:
                    self._special_pressed.discard(name)
                return
            with self._lock:
                self._pressed.discard(char)

        self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        self._listener.daemon = True
        self._listener.start()

    def _deadman_active(self) -> bool:
        return bool(self._special_pressed.intersection({"shift", "shift_l", "shift_r"}))

    def _drain_events(self) -> None:
        while True:
            try:
                kind, value = self._event_queue.get_nowait()
            except queue.Empty:
                return
            if kind == "event":
                self.event_pub.publish(String(data=value))
                self.get_logger().info(f"Recorder event: {value}")
            elif kind == "mode":
                self.mode_pub.publish(String(data=value))
                self.get_logger().info(f"Control mode requested: {value}")
            elif kind == "stop":
                self.event_pub.publish(String(data="stop"))
                self.mode_pub.publish(String(data="idle"))
                with self._lock:
                    self._pressed.clear()
                    self._special_pressed.clear()
                self.get_logger().warning("Stop requested; control mode set to idle")

    def _on_timer(self) -> None:
        self._drain_events()
        with self._lock:
            pressed = set(self._pressed)
            deadman = self._deadman_active()

        target = keyboard_target_velocity(
            pressed,
            deadman,
            self.max_xy_speed,
            self.max_z_speed,
            self.axis_signs,
        )
        if deadman:
            self._velocity = smooth_velocity(self._velocity, target, self.response)
        else:
            # Deadman release must stop immediately; do not smooth this transition.
            self._velocity.fill(0.0)

        message = TwistStamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = self.frame_id
        message.twist.linear.x = float(self._velocity[0])
        message.twist.linear.y = float(self._velocity[1])
        message.twist.linear.z = float(self._velocity[2])
        self.twist_pub.publish(message)
        self.deadman_pub.publish(Bool(data=deadman))

    def destroy_node(self) -> bool:
        if self._listener is not None:
            self._listener.stop()
        return super().destroy_node()


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = KeyboardTeleopNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
