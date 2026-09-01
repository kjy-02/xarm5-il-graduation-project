"""PyTorch dataset utilities for versioned xArm imitation-learning episodes."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import cv2
import h5py
import numpy as np

from .storage import FORMAT_VERSION, read_episode_metadata


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True)
class NormalizationStats:
    state_mean: np.ndarray
    state_std: np.ndarray
    action_mean: np.ndarray
    action_std: np.ndarray

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "state_mean": self.state_mean.tolist(),
            "state_std": self.state_std.tolist(),
            "action_mean": self.action_mean.tolist(),
            "action_std": self.action_std.tolist(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "NormalizationStats":
        return cls(
            state_mean=np.asarray(data["state_mean"], dtype=np.float32),
            state_std=np.asarray(data["state_std"], dtype=np.float32),
            action_mean=np.asarray(data["action_mean"], dtype=np.float32),
            action_std=np.asarray(data["action_std"], dtype=np.float32),
        )


def split_episode_files(
    files: Sequence[str | Path], validation_fraction: float, seed: int
) -> tuple[list[Path], list[Path]]:
    """Split by whole episode to prevent neighboring frames leaking into validation."""
    paths = [Path(path) for path in files]
    if not paths:
        raise ValueError("No episode files were provided")
    if not 0.0 <= validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in [0, 1)")
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(paths))
    if len(paths) == 1 or validation_fraction == 0.0:
        validation_count = 0
    else:
        validation_count = max(1, int(round(len(paths) * validation_fraction)))
        validation_count = min(validation_count, len(paths) - 1)
    validation = [paths[index] for index in indices[:validation_count]]
    train = [paths[index] for index in indices[validation_count:]]
    return train, validation


def compute_normalization_stats(files: Sequence[str | Path]) -> NormalizationStats:
    """Compute state/action statistics using training episodes only."""
    state_blocks: list[np.ndarray] = []
    action_blocks: list[np.ndarray] = []
    for path in files:
        with h5py.File(path, "r") as episode:
            state_blocks.append(
                np.concatenate(
                    [
                        np.asarray(episode["observation/joint_position"], dtype=np.float32),
                        np.asarray(episode["observation/joint_velocity"], dtype=np.float32),
                    ],
                    axis=1,
                )
            )
            action_blocks.append(np.asarray(episode["action/twist"], dtype=np.float32))
    if not state_blocks:
        raise ValueError("Cannot compute statistics without episodes")
    states = np.concatenate(state_blocks, axis=0)
    actions = np.concatenate(action_blocks, axis=0)
    if len(states) == 0:
        raise ValueError("Cannot compute statistics from empty episodes")
    return NormalizationStats(
        state_mean=states.mean(axis=0),
        state_std=np.maximum(states.std(axis=0), 1.0e-6),
        action_mean=actions.mean(axis=0),
        action_std=np.maximum(actions.std(axis=0), 1.0e-6),
    )


def save_normalization_stats(path: str | Path, stats: NormalizationStats) -> None:
    Path(path).write_text(json.dumps(stats.to_dict(), indent=2), encoding="utf-8")


def decode_jpeg(payload: np.ndarray) -> np.ndarray:
    image = cv2.imdecode(np.asarray(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Failed to decode a JPEG frame")
    return image


def preprocess_bgr_images(images: Sequence[np.ndarray], image_size: tuple[int, int]) -> np.ndarray:
    """Return ImageNet-normalized RGB images as [camera, channel, height, width]."""
    width, height = image_size
    output = []
    for image in images:
        resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
        output.append(np.transpose(normalized, (2, 0, 1)))
    return np.stack(output, axis=0).astype(np.float32)


class ChunkedEpisodeDataset:
    """Frame observations paired with the current and future action chunk."""

    def __init__(
        self,
        files: Sequence[str | Path],
        stats: NormalizationStats,
        horizon: int = 20,
        image_size: tuple[int, int] = (224, 224),
        camera_names: Sequence[str] | None = None,
    ) -> None:
        try:
            import torch
            from torch.utils.data import Dataset
        except ImportError as exc:  # pragma: no cover - installation error path
            raise RuntimeError("PyTorch is required for model training") from exc

        del Dataset  # Document the expected protocol without dynamic inheritance.
        self._torch = torch
        self.files = [Path(path) for path in files]
        self.stats = stats
        self.horizon = int(horizon)
        self.image_size = tuple(int(value) for value in image_size)
        if not self.files:
            raise ValueError("Dataset needs at least one episode")
        if self.horizon <= 0:
            raise ValueError("horizon must be positive")

        first_metadata = read_episode_metadata(self.files[0])
        self.camera_names = tuple(camera_names or first_metadata["camera_names"])
        self.joint_names = tuple(first_metadata["joint_names"])
        self._index: list[tuple[int, int]] = []
        for file_index, path in enumerate(self.files):
            metadata = read_episode_metadata(path)
            if metadata.get("format_version") != FORMAT_VERSION:
                raise ValueError(f"Unsupported dataset format in {path}")
            if not bool(metadata.get("complete", False)):
                raise ValueError(f"Episode is incomplete: {path}")
            if tuple(metadata["joint_names"]) != self.joint_names:
                raise ValueError(f"Joint name mismatch in {path}")
            if tuple(metadata["camera_names"]) != self.camera_names:
                raise ValueError(
                    f"Camera names/order mismatch in {path}: "
                    f"{metadata['camera_names']} != {list(self.camera_names)}"
                )
            self._index.extend((file_index, frame) for frame in range(metadata["num_frames"]))
        if not self._index:
            raise ValueError("All provided episodes are empty")

    @property
    def state_dim(self) -> int:
        return int(self.stats.state_mean.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.stats.action_mean.shape[0])

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        file_index, frame_index = self._index[index]
        with h5py.File(self.files[file_index], "r") as episode:
            bgr_images = [
                decode_jpeg(episode[f"observation/images/{name}"][frame_index])
                for name in self.camera_names
            ]
            images = preprocess_bgr_images(bgr_images, self.image_size)
            state = np.concatenate(
                [
                    episode["observation/joint_position"][frame_index],
                    episode["observation/joint_velocity"][frame_index],
                ]
            ).astype(np.float32)
            actions_dataset = episode["action/twist"]
            available = min(self.horizon, len(actions_dataset) - frame_index)
            actions = np.zeros((self.horizon, self.action_dim), dtype=np.float32)
            actions[:available] = actions_dataset[frame_index : frame_index + available]
            mask = np.zeros(self.horizon, dtype=np.bool_)
            mask[:available] = True

        normalized_state = (state - self.stats.state_mean) / self.stats.state_std
        normalized_actions = (actions - self.stats.action_mean) / self.stats.action_std
        return {
            "images": self._torch.from_numpy(images),
            "state": self._torch.from_numpy(normalized_state.astype(np.float32)),
            "actions": self._torch.from_numpy(normalized_actions.astype(np.float32)),
            "action_mask": self._torch.from_numpy(mask),
            "episode_index": file_index,
            "frame_index": frame_index,
        }
