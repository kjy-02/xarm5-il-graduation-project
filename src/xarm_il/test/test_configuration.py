import re
from pathlib import Path

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parents[1]


def _parameters(config: dict, node: str) -> dict:
    return config[node]["ros__parameters"]


def test_default_configuration_is_internally_consistent() -> None:
    config = yaml.safe_load((PACKAGE_ROOT / "config" / "xarm5_il.yaml").read_text())
    keyboard = _parameters(config, "keyboard_teleop")
    safety = _parameters(config, "safety_mux")
    recorder = _parameters(config, "episode_recorder")
    policy = _parameters(config, "policy_node")

    assert len(keyboard["axis_signs"]) == 3
    assert keyboard["max_xy_speed"] <= safety["max_xy_speed"]
    assert keyboard["max_z_speed"] <= safety["max_z_speed"]
    assert policy["max_xy_speed"] <= safety["max_xy_speed"]
    assert policy["max_z_speed"] <= safety["max_z_speed"]
    assert safety["command_timeout_s"] > 2.0 / safety["publish_rate_hz"]
    assert all(
        minimum < maximum
        for minimum, maximum in zip(safety["workspace_min"], safety["workspace_max"])
    )
    assert safety["dry_run"] is True
    assert len(recorder["camera_names"]) == len(recorder["camera_topics"])
    assert len(policy["camera_topics"]) == len(recorder["camera_topics"])
    assert recorder["fps"] == policy["control_rate_hz"]
    assert len(recorder["joint_names"]) == 5


def test_console_entry_point_targets_exist() -> None:
    setup_text = (PACKAGE_ROOT / "setup.py").read_text(encoding="utf-8")
    expected_modules = (
        "keyboard_teleop_node",
        "safety_mux_node",
        "episode_recorder_node",
        "preflight_check",
        "policy_node",
        "train_policy",
        "validate_dataset",
        "visualize_episode",
        "smoke_test_pipeline",
    )
    for module in expected_modules:
        assert f"xarm_il.{module}:main" in setup_text
        assert (PACKAGE_ROOT / "xarm_il" / f"{module}.py").is_file()


def test_motion_launches_default_to_dry_run() -> None:
    for name in (
        "teleop.launch.py",
        "collect.launch.py",
        "policy_eval.launch.py",
        "system_collect.launch.py",
        "system_policy_eval.launch.py",
    ):
        launch_text = (PACKAGE_ROOT / "launch" / name).read_text(encoding="utf-8")
        assert re.search(
            r'DeclareLaunchArgument\(\s*"dry_run",\s*default_value="true"', launch_text
        )
