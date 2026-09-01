"""Pure control helpers shared by ROS nodes and tests."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class WorkspaceBounds:
    """Axis-aligned TCP workspace in the command frame, in metres."""

    minimum: np.ndarray
    maximum: np.ndarray

    @classmethod
    def from_sequences(
        cls, minimum: Sequence[float], maximum: Sequence[float]
    ) -> "WorkspaceBounds":
        lo = np.asarray(minimum, dtype=np.float64)
        hi = np.asarray(maximum, dtype=np.float64)
        if lo.shape != (3,) or hi.shape != (3,):
            raise ValueError("Workspace bounds must contain exactly three values")
        if np.any(lo >= hi):
            raise ValueError(f"Each workspace minimum must be less than maximum: {lo}, {hi}")
        return cls(lo, hi)


def keyboard_target_velocity(
    pressed: Iterable[str],
    deadman_active: bool,
    max_xy_speed: float,
    max_z_speed: float,
    axis_signs: Sequence[float] = (1.0, 1.0, 1.0),
) -> np.ndarray:
    """Convert held keys to a bounded Cartesian velocity [vx, vy, vz]."""

    if not deadman_active:
        return np.zeros(3, dtype=np.float64)

    keys = set(pressed)
    direction = np.array(
        [
            float("w" in keys) - float("s" in keys),
            float("d" in keys) - float("a" in keys),
            float("r" in keys) - float("f" in keys),
        ],
        dtype=np.float64,
    )
    signs = np.asarray(axis_signs, dtype=np.float64)
    if signs.shape != (3,) or np.any(np.abs(signs) != 1.0):
        raise ValueError("axis_signs must be three values, each either -1 or 1")
    direction *= signs

    xy_norm = np.linalg.norm(direction[:2])
    if xy_norm > 1.0:
        direction[:2] /= xy_norm
    direction[2] = float(np.clip(direction[2], -1.0, 1.0))
    return np.array(
        [
            direction[0] * max_xy_speed,
            direction[1] * max_xy_speed,
            direction[2] * max_z_speed,
        ],
        dtype=np.float64,
    )


def smooth_velocity(
    previous: Sequence[float], target: Sequence[float], response: float
) -> np.ndarray:
    """First-order velocity smoothing; response=1 follows target immediately."""

    if not 0.0 < response <= 1.0:
        raise ValueError("response must be in (0, 1]")
    prev = np.asarray(previous, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    return prev + response * (tgt - prev)


def clamp_cartesian_velocity(
    velocity: Sequence[float], max_xy_speed: float, max_z_speed: float
) -> np.ndarray:
    """Clamp horizontal magnitude and vertical speed independently."""

    result = np.asarray(velocity, dtype=np.float64).copy()
    if result.shape != (3,):
        raise ValueError("velocity must contain exactly three values")
    if not np.all(np.isfinite(result)):
        return np.zeros(3, dtype=np.float64)
    xy_norm = np.linalg.norm(result[:2])
    if xy_norm > max_xy_speed > 0.0:
        result[:2] *= max_xy_speed / xy_norm
    result[2] = np.clip(result[2], -max_z_speed, max_z_speed)
    return result


def enforce_workspace(
    velocity: Sequence[float],
    tcp_position: Sequence[float],
    bounds: WorkspaceBounds,
    lookahead_s: float,
    margin: float,
) -> tuple[np.ndarray, Mapping[str, bool]]:
    """Block velocity components that would cross an axis-aligned workspace."""

    vel = np.asarray(velocity, dtype=np.float64).copy()
    pos = np.asarray(tcp_position, dtype=np.float64)
    if vel.shape != (3,) or pos.shape != (3,):
        raise ValueError("velocity and tcp_position must each contain three values")
    projected = pos + vel * max(lookahead_s, 0.0)
    blocked: dict[str, bool] = {}
    labels = ("x", "y", "z")
    for axis, label in enumerate(labels):
        low = bounds.minimum[axis] + margin
        high = bounds.maximum[axis] - margin
        outward_low = vel[axis] < 0.0 and (pos[axis] <= low or projected[axis] < low)
        outward_high = vel[axis] > 0.0 and (pos[axis] >= high or projected[axis] > high)
        is_blocked = bool(outward_low or outward_high)
        blocked[label] = is_blocked
        if is_blocked:
            vel[axis] = 0.0
    return vel, blocked


def recording_safety_reason(status: Mapping[str, object]) -> str | None:
    """Return why a safety status is unsuitable for demonstration recording."""
    if bool(status.get("dry_run", True)):
        return "dry_run is enabled"
    if str(status.get("mode", "")) != "teleop":
        return "control is not in teleop mode"
    if bool(status.get("stale", True)):
        return "the teleop command or keyboard watchdog is stale"
    source = str(status.get("source", ""))
    if "no_tf" in source or "wrong_frame" in source or "keyboard_watchdog" in source:
        return f"unsafe command source: {source}"
    return None
