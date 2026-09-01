from pathlib import Path

import cv2
import h5py
import numpy as np
import pytest

from xarm_il.dataset import (
    ChunkedEpisodeDataset,
    compute_normalization_stats,
    split_episode_files,
)
from xarm_il.storage import EpisodeWriter, list_episode_files, next_episode_index
from xarm_il.validate_dataset import validate_files


def _jpeg(value: int) -> bytes:
    image = np.full((32, 48, 3), value, dtype=np.uint8)
    ok, payload = cv2.imencode(".jpg", image)
    assert ok
    return payload.tobytes()


def _write_episode(root: Path, episode_index: int, frames: int = 6) -> Path:
    writer = EpisodeWriter(
        root,
        episode_index,
        [f"joint{i}" for i in range(1, 6)],
        ["wrist"],
        "test task",
        10.0,
    )
    for frame in range(frames):
        t = frame / 10.0
        writer.append(
            record_time=10.0 + t,
            source_times=[10.0 + t] * 3,
            joint_position=np.arange(5, dtype=np.float32) + t,
            joint_velocity=np.full(5, 0.1, dtype=np.float32),
            tcp_pose=[0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
            action=[0.01, 0.001 * frame, 0.0],
            action_source="teleop",
            jpeg_images={"wrist": _jpeg(20 + frame)},
        )
    return writer.finalize()


def test_episode_round_trip_and_validation(tmp_path: Path) -> None:
    path = _write_episode(tmp_path, 0)
    assert path == list_episode_files(tmp_path)[0]
    assert next_episode_index(tmp_path) == 1
    report = validate_files([path])
    assert report["valid"], report
    with h5py.File(path, "r") as episode:
        assert episode["observation/joint_position"].shape == (6, 5)
        assert len(episode["observation/images/wrist"]) == 6


def test_dataset_chunk_padding_and_shapes(tmp_path: Path) -> None:
    path = _write_episode(tmp_path, 0)
    stats = compute_normalization_stats([path])
    dataset = ChunkedEpisodeDataset([path], stats, horizon=4, image_size=(64, 48))
    sample = dataset[len(dataset) - 1]
    assert tuple(sample["images"].shape) == (1, 3, 48, 64)
    assert tuple(sample["state"].shape) == (10,)
    assert tuple(sample["actions"].shape) == (4, 3)
    assert sample["action_mask"].tolist() == [True, False, False, False]


def test_split_is_reproducible_and_by_episode(tmp_path: Path) -> None:
    files = [_write_episode(tmp_path, index) for index in range(5)]
    first = split_episode_files(files, 0.2, seed=9)
    second = split_episode_files(files, 0.2, seed=9)
    assert first == second
    assert len(first[0]) == 4
    assert len(first[1]) == 1
    assert set(first[0]).isdisjoint(first[1])


def test_partial_episode_is_removed_on_discard(tmp_path: Path) -> None:
    writer = EpisodeWriter(tmp_path, 0, ["joint1"], ["wrist"], "test", 10.0)
    writer.discard()
    assert not list(tmp_path.glob("*.partial.h5"))


def test_zero_action_episode_is_rejected(tmp_path: Path) -> None:
    path = _write_episode(tmp_path, 0)
    with h5py.File(path, "r+") as episode:
        episode["action/twist"][:] = 0.0
    report = validate_files([path])
    assert not report["valid"]
    assert any("all recorded actions are zero" in error for error in report["episodes"][0]["errors"])


def test_mixed_camera_configurations_are_rejected(tmp_path: Path) -> None:
    first = _write_episode(tmp_path, 0)
    writer = EpisodeWriter(
        tmp_path,
        1,
        [f"joint{i}" for i in range(1, 6)],
        ["wrist", "overhead"],
        "test task",
        10.0,
    )
    for frame in range(2):
        t = frame / 10.0
        writer.append(
            record_time=20.0 + t,
            source_times=[20.0 + t] * 4,
            joint_position=np.arange(5, dtype=np.float32),
            joint_velocity=np.zeros(5, dtype=np.float32),
            tcp_pose=[0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
            action=[0.01, 0.0, 0.0],
            action_source="teleop",
            jpeg_images={"wrist": _jpeg(30), "overhead": _jpeg(40)},
        )
    second = writer.finalize()
    report = validate_files([first, second])
    assert not report["valid"]
    assert any(
        "camera names/order differ" in error for error in report["episodes"][1]["errors"]
    )
    stats = compute_normalization_stats([first])
    with pytest.raises(ValueError, match="Camera names/order mismatch"):
        ChunkedEpisodeDataset([first, second], stats, horizon=2, image_size=(32, 32))
