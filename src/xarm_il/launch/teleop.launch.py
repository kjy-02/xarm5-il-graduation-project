"""Launch keyboard control and the independent safety command mux."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution([FindPackageShare("xarm_il"), "config", "xarm5_il.yaml"])
    config = LaunchConfiguration("config")
    dry_run = LaunchConfiguration("dry_run")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument(
                "dry_run",
                default_value="true",
                description="Publish zero executed actions and do not command MoveIt Servo",
            ),
            Node(
                package="xarm_il",
                executable="keyboard_teleop",
                name="keyboard_teleop",
                output="screen",
                parameters=[config],
            ),
            Node(
                package="xarm_il",
                executable="safety_mux",
                name="safety_mux",
                output="screen",
                parameters=[config, {"dry_run": ParameterValue(dry_run, value_type=bool)}],
            ),
        ]
    )
