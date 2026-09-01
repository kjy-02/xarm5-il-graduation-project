"""Bring up a real xArm5 MoveIt Servo stack and the collection nodes."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    robot_ip = LaunchConfiguration("robot_ip")
    config = LaunchConfiguration("config")
    dry_run = LaunchConfiguration("dry_run")
    dataset_root = LaunchConfiguration("dataset_root")
    default_config = PathJoinSubstitution([FindPackageShare("xarm_il"), "config", "xarm5_il.yaml"])
    xarm_servo_launch = PathJoinSubstitution(
        [FindPackageShare("xarm_moveit_servo"), "launch", "xarm_moveit_servo_realmove.launch.py"]
    )
    collect_launch = PathJoinSubstitution(
        [FindPackageShare("xarm_il"), "launch", "collect.launch.py"]
    )
    return LaunchDescription(
        [
            DeclareLaunchArgument("robot_ip"),
            DeclareLaunchArgument("config", default_value=default_config),
            DeclareLaunchArgument("dry_run", default_value="true"),
            DeclareLaunchArgument("dataset_root", default_value="~/xarm_il_data/xarm5_drag"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(xarm_servo_launch),
                launch_arguments={"robot_ip": robot_ip, "dof": "5"}.items(),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(collect_launch),
                launch_arguments={
                    "config": config,
                    "dry_run": dry_run,
                    "dataset_root": dataset_root,
                }.items(),
            ),
        ]
    )
