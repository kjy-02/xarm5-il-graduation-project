"""Launch keyboard teleoperation, safety mux, and episode recording."""

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
    dataset_root = LaunchConfiguration("dataset_root")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("dataset_root", default_value="~/xarm_il_data/xarm5_drag"),
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
            Node(
                package="xarm_il",
                executable="episode_recorder",
                name="episode_recorder",
                output="screen",
                parameters=[config, {"dataset_root": dataset_root}],
            ),
        ]
    )
