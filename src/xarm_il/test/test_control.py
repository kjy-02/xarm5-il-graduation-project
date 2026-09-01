import numpy as np
import pytest

from xarm_il.control import (
    WorkspaceBounds,
    clamp_cartesian_velocity,
    enforce_workspace,
    keyboard_target_velocity,
    recording_safety_reason,
    smooth_velocity,
)


def test_deadman_release_is_zero() -> None:
    velocity = keyboard_target_velocity({"w", "r"}, False, 0.03, 0.01)
    np.testing.assert_array_equal(velocity, np.zeros(3))


def test_diagonal_keyboard_speed_is_normalized() -> None:
    velocity = keyboard_target_velocity({"w", "d"}, True, 0.03, 0.01)
    assert np.linalg.norm(velocity[:2]) == pytest.approx(0.03)


def test_axis_signs_are_applied() -> None:
    velocity = keyboard_target_velocity({"w", "d", "r"}, True, 0.03, 0.01, [-1, 1, -1])
    assert velocity[0] < 0.0
    assert velocity[1] > 0.0
    assert velocity[2] < 0.0


def test_smoothing_and_clamping() -> None:
    np.testing.assert_allclose(smooth_velocity([0, 0, 0], [1, 0, 0], 0.25), [0.25, 0, 0])
    clamped = clamp_cartesian_velocity([3.0, 4.0, 2.0], 0.05, 0.01)
    assert np.linalg.norm(clamped[:2]) == pytest.approx(0.05)
    assert clamped[2] == pytest.approx(0.01)


def test_workspace_blocks_only_outward_component() -> None:
    bounds = WorkspaceBounds.from_sequences([0.2, -0.3, 0.1], [0.6, 0.3, 0.5])
    velocity, blocked = enforce_workspace(
        [-0.02, 0.01, 0.0], [0.202, 0.0, 0.2], bounds, lookahead_s=0.25, margin=0.005
    )
    assert velocity[0] == 0.0
    assert velocity[1] == pytest.approx(0.01)
    assert blocked == {"x": True, "y": False, "z": False}


def test_invalid_workspace_is_rejected() -> None:
    with pytest.raises(ValueError):
        WorkspaceBounds.from_sequences([0, 0, 0], [0, 1, 1])


def test_recording_requires_real_fresh_teleop() -> None:
    valid = {"dry_run": False, "mode": "teleop", "stale": False, "source": "teleop"}
    assert recording_safety_reason(valid) is None
    assert recording_safety_reason({**valid, "dry_run": True}) == "dry_run is enabled"
    assert recording_safety_reason({**valid, "mode": "policy"}) is not None
    assert recording_safety_reason({**valid, "stale": True}) is not None
    assert recording_safety_reason({**valid, "source": "teleop:no_tf"}) is not None
