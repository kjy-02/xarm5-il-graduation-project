"""Launch guarded learned-policy evaluation with keyboard takeover available."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution([FindPackageShare("xarm_il"), "config", "xarm5_il.yaml"])
    config = LaunchConfiguration("config")
    checkpoint = LaunchConfiguration("checkpoint")
    dry_run = LaunchConfiguration("dry_run")
    return LaunchDescription(
        [
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("checkpoint"),
            DeclareLaunchArgument("dry_run", default_value="true"),
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
                parameters=[
                    config,
                    {
                        "initial_mode": "idle",
                        "dry_run": ParameterValue(dry_run, value_type=bool),
                    },
                ],
            ),
            Node(
                package="xarm_il",
                executable="policy_node",
                name="policy_node",
                output="screen",
                parameters=[config, {"checkpoint_path": checkpoint}],
            ),
        ]
    )
