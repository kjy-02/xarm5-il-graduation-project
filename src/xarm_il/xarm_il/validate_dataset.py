"""Validate xArm HDF5 episodes before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np

from .storage import FORMAT_VERSION, list_episode_files


REQUIRED_DATASETS = (
    "timestamp/record",
    "timestamp/source",
    "observation/joint_position",
    "observation/joint_velocity",
    "observation/tcp_pose",
    "action/twist",
    "action/source",
)


def inspect_episode(
    path: str | Path,
    max_xy_speed: float = 0.041,
    max_z_speed: float = 0.016,
    max_source_skew_s: float = 0.20,
) -> dict[str, Any]:
    path = Path(path)
    errors: list[str] = []
    warnings: list[str] = []
    result: dict[str, Any] = {"path": str(path), "errors": errors, "warnings": warnings}
    try:
        episode = h5py.File(path, "r")
    except (OSError, ValueError) as exc:
        errors.append(f"cannot open HDF5: {exc}")
        return result

    with episode:
        if episode.attrs.get("format_version") != FORMAT_VERSION:
            errors.append(f"format_version is not {FORMAT_VERSION}")
        if not bool(episode.attrs.get("complete", False)):
            errors.append("episode is not marked complete")
        if not bool(episode.attrs.get("success", False)):
            errors.append("episode is not marked successful")
        for name in REQUIRED_DATASETS:
            if name not in episode:
                errors.append(f"missing dataset: {name}")
        if errors:
            return result
        try:
            camera_names = tuple(json.loads(episode.attrs["camera_names_json"]))
            joint_names = tuple(json.loads(episode.attrs["joint_names_json"]))
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"invalid metadata JSON: {exc}")
            return result
        if not camera_names or len(set(camera_names)) != len(camera_names):
            errors.append("camera_names must be non-empty and unique")
        if not joint_names or len(set(joint_names)) != len(joint_names):
            errors.append("joint_names must be non-empty and unique")
        lengths = {name: len(episode[name]) for name in REQUIRED_DATASETS}
        for camera in camera_names:
            name = f"observation/images/{camera}"
            if name not in episode:
                errors.append(f"missing dataset: {name}")
            else:
                lengths[name] = len(episode[name])
        unique_lengths = set(lengths.values())
        if len(unique_lengths) != 1:
            errors.append(f"dataset lengths disagree: {lengths}")
            return result
        frames = next(iter(unique_lengths), 0)
        result.update(frames=frames, cameras=list(camera_names), joints=list(joint_names))
        if frames == 0:
            errors.append("episode has no frames")
            return result

        record_time = np.asarray(episode["timestamp/record"], dtype=np.float64)
        if not np.all(np.isfinite(record_time)):
            errors.append("record timestamps contain NaN or infinity")
        if frames > 1 and np.any(np.diff(record_time) <= 0.0):
            errors.append("record timestamps are not strictly increasing")
        duration = float(record_time[-1] - record_time[0]) if frames > 1 else 0.0
        result["duration_s"] = duration
        result["actual_fps"] = (frames - 1) / duration if duration > 0.0 else 0.0

        arrays = {
            "joint_position": np.asarray(episode["observation/joint_position"]),
            "joint_velocity": np.asarray(episode["observation/joint_velocity"]),
            "tcp_pose": np.asarray(episode["observation/tcp_pose"]),
            "action": np.asarray(episode["action/twist"]),
        }
        expected_shapes = {
            "joint_position": (frames, len(joint_names)),
            "joint_velocity": (frames, len(joint_names)),
            "tcp_pose": (frames, 7),
            "action": (frames, 3),
        }
        for name, array in arrays.items():
            if array.shape != expected_shapes[name]:
                errors.append(f"{name} shape {array.shape} != {expected_shapes[name]}")
            if not np.all(np.isfinite(array)):
                errors.append(f"{name} contains NaN or infinity")

        action = arrays["action"]
        if action.shape == expected_shapes["action"]:
            xy_speed = np.linalg.norm(action[:, :2], axis=1)
            result["max_xy_speed"] = float(xy_speed.max())
            result["max_abs_z_speed"] = float(np.abs(action[:, 2]).max())
            if float(np.linalg.norm(action, axis=1).max()) < 1.0e-5:
                errors.append("all recorded actions are zero (possibly collected in dry_run mode)")
            if np.any(xy_speed > max_xy_speed + 1.0e-6):
                errors.append(f"XY action exceeds {max_xy_speed:g} m/s")
            if np.any(np.abs(action[:, 2]) > max_z_speed + 1.0e-6):
                errors.append(f"Z action exceeds {max_z_speed:g} m/s")

        tcp_pose = arrays["tcp_pose"]
        if tcp_pose.shape == expected_shapes["tcp_pose"] and np.all(np.isfinite(tcp_pose)):
            quaternion_norm = np.linalg.norm(tcp_pose[:, 3:], axis=1)
            if np.any(np.abs(quaternion_norm - 1.0) > 0.05):
                warnings.append("TCP quaternion norm differs from 1 by more than 0.05")

        source_time = np.asarray(episode["timestamp/source"], dtype=np.float64)
        expected_source_shape = (frames, len(camera_names) + 2)
        if source_time.shape != expected_source_shape:
            errors.append(f"source timestamp shape is invalid: {source_time.shape}")
        elif not np.all(np.isfinite(source_time)):
            errors.append("source timestamps contain NaN or infinity")
        else:
            skew = np.ptp(source_time, axis=1)
            result["max_source_skew_s"] = float(skew.max())
            result["mean_source_skew_s"] = float(skew.mean())
            if np.any(skew > max_source_skew_s):
                warnings.append(
                    f"source timestamp skew exceeds {max_source_skew_s:g}s "
                    f"(max {float(skew.max()):.3f}s)"
                )
            duplicate_rows = np.any(np.diff(source_time, axis=0) <= 0.0, axis=1)
            duplicate_ratio = float(duplicate_rows.mean()) if len(duplicate_rows) else 0.0
            result["duplicate_source_ratio"] = duplicate_ratio
            if duplicate_ratio > 0.01:
                warnings.append(f"{duplicate_ratio:.1%} of frame transitions reuse a sensor sample")

        try:
            source_columns = json.loads(episode.attrs["source_time_columns_json"])
            expected_columns = [*camera_names, "joint_state", "executed_action"]
            if source_columns != expected_columns:
                errors.append(
                    f"source timestamp columns {source_columns} != {expected_columns}"
                )
        except (KeyError, json.JSONDecodeError, TypeError) as exc:
            errors.append(f"invalid source timestamp metadata: {exc}")

        action_sources = [
            value.decode("utf-8") if isinstance(value, bytes) else str(value)
            for value in episode["action/source"][:]
        ]
        non_teleop = sum(not value.startswith("teleop") for value in action_sources)
        non_teleop_ratio = non_teleop / frames
        result["non_teleop_action_ratio"] = non_teleop_ratio
        if non_teleop_ratio > 0.20:
            warnings.append(f"{non_teleop_ratio:.1%} of frames are not labeled teleop")

        check_indices = sorted({0, frames // 2, frames - 1})
        for camera in camera_names:
            dataset = episode[f"observation/images/{camera}"]
            for index in check_indices:
                decoded = cv2.imdecode(np.asarray(dataset[index], dtype=np.uint8), cv2.IMREAD_COLOR)
                if decoded is None:
                    errors.append(f"cannot decode {camera} JPEG at frame {index}")
                    break
        configured_fps = float(episode.attrs.get("fps", 0.0))
        if result["actual_fps"] and configured_fps:
            relative_error = abs(result["actual_fps"] - configured_fps) / configured_fps
            if relative_error > 0.20:
                warnings.append(
                    f"actual FPS {result['actual_fps']:.2f} differs from configured {configured_fps:.2f}"
                )
    return result


def validate_files(
    files: Sequence[str | Path],
    max_xy_speed: float = 0.041,
    max_z_speed: float = 0.016,
    max_source_skew_s: float = 0.20,
) -> dict[str, Any]:
    episodes = [
        inspect_episode(path, max_xy_speed, max_z_speed, max_source_skew_s) for path in files
    ]
    reference_cameras: list[str] | None = None
    reference_joints: list[str] | None = None
    for episode in episodes:
        if episode["errors"]:
            continue
        cameras = episode.get("cameras")
        joints = episode.get("joints")
        if reference_cameras is None:
            reference_cameras = cameras
            reference_joints = joints
            continue
        if cameras != reference_cameras:
            episode["errors"].append(
                f"camera names/order differ from dataset reference: {cameras} != {reference_cameras}"
            )
        if joints != reference_joints:
            episode["errors"].append(
                f"joint names/order differ from dataset reference: {joints} != {reference_joints}"
            )
    return {
        "valid": bool(episodes) and all(not item["errors"] for item in episodes),
        "episode_count": len(episodes),
        "frame_count": sum(int(item.get("frames", 0)) for item in episodes),
        "error_count": sum(len(item["errors"]) for item in episodes),
        "warning_count": sum(len(item["warnings"]) for item in episodes),
        "episodes": episodes,
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--max-xy-speed", type=float, default=0.041)
    parser.add_argument("--max-z-speed", type=float, default=0.016)
    parser.add_argument("--max-source-skew", type=float, default=0.20)
    parser.add_argument("--report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    report = validate_files(
        list_episode_files(args.data),
        max_xy_speed=args.max_xy_speed,
        max_z_speed=args.max_z_speed,
        max_source_skew_s=args.max_source_skew,
    )
    output = json.dumps(report, indent=2, ensure_ascii=False)
    print(output)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(output, encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
