"""Versioned HDF5 episode storage used by recording, training, and validation."""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import h5py
import numpy as np


FORMAT_VERSION = "xarm_il_hdf5_v1"
EPISODE_PATTERN = re.compile(r"episode_(\d{6})\.h5$")
PARTIAL_EPISODE_PATTERN = re.compile(r"episode_(\d{6})\.partial\.h5$")


def list_episode_files(root: str | Path) -> list[Path]:
    return sorted(Path(root).glob("episode_*.h5"))


def next_episode_index(root: str | Path) -> int:
    indices = []
    for path in Path(root).glob("episode_*"):
        match = EPISODE_PATTERN.match(path.name) or PARTIAL_EPISODE_PATTERN.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return max(indices, default=-1) + 1


class EpisodeWriter:
    """Incrementally writes JPEG-compressed observations to a temporary HDF5 file."""

    def __init__(
        self,
        root: str | Path,
        episode_index: int,
        joint_names: Sequence[str],
        camera_names: Sequence[str],
        task: str,
        fps: float,
    ) -> None:
        self.root = Path(root).expanduser()
        self.root.mkdir(parents=True, exist_ok=True)
        self.episode_index = int(episode_index)
        self.joint_names = tuple(joint_names)
        self.camera_names = tuple(camera_names)
        self.task = task
        self.fps = float(fps)
        if not self.joint_names or len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError("joint_names must be non-empty and unique")
        if not self.camera_names or len(set(self.camera_names)) != len(self.camera_names):
            raise ValueError("camera_names must be non-empty and unique")
        if self.fps <= 0.0:
            raise ValueError("fps must be positive")
        self.partial_path = self.root / f"episode_{self.episode_index:06d}.partial.h5"
        self.final_path = self.root / f"episode_{self.episode_index:06d}.h5"
        if self.partial_path.exists() or self.final_path.exists():
            raise FileExistsError(f"Episode path already exists: {self.final_path}")
        self._file = h5py.File(self.partial_path, "w")
        self._count = 0
        self._closed = False
        self._created_wall_time = time.time()
        self._create_layout()

    @property
    def count(self) -> int:
        return self._count

    def _create_layout(self) -> None:
        root = self._file
        root.attrs["format_version"] = FORMAT_VERSION
        root.attrs["episode_index"] = self.episode_index
        root.attrs["task"] = self.task
        root.attrs["fps"] = self.fps
        root.attrs["joint_names_json"] = json.dumps(self.joint_names)
        root.attrs["camera_names_json"] = json.dumps(self.camera_names)
        root.attrs["created_unix_s"] = self._created_wall_time
        root.attrs["complete"] = False

        self._record_time = root.create_dataset("timestamp/record", shape=(0,), maxshape=(None,), dtype="f8")
        source_width = len(self.camera_names) + 2  # cameras, joint state, executed action
        root.attrs["source_time_columns_json"] = json.dumps(
            [*self.camera_names, "joint_state", "executed_action"]
        )
        self._source_time = root.create_dataset(
            "timestamp/source",
            shape=(0, source_width),
            maxshape=(None, source_width),
            dtype="f8",
        )
        self._joint_pos = root.create_dataset(
            "observation/joint_position",
            shape=(0, len(self.joint_names)),
            maxshape=(None, len(self.joint_names)),
            dtype="f4",
        )
        self._joint_vel = root.create_dataset(
            "observation/joint_velocity",
            shape=(0, len(self.joint_names)),
            maxshape=(None, len(self.joint_names)),
            dtype="f4",
        )
        self._tcp_pose = root.create_dataset(
            "observation/tcp_pose", shape=(0, 7), maxshape=(None, 7), dtype="f4"
        )
        self._action = root.create_dataset(
            "action/twist", shape=(0, 3), maxshape=(None, 3), dtype="f4"
        )
        string_dtype = h5py.string_dtype(encoding="utf-8")
        self._source = root.create_dataset(
            "action/source", shape=(0,), maxshape=(None,), dtype=string_dtype
        )
        image_group = root.require_group("observation/images")
        jpeg_dtype = h5py.vlen_dtype(np.dtype("uint8"))
        self._images = {
            name: image_group.create_dataset(name, shape=(0,), maxshape=(None,), dtype=jpeg_dtype)
            for name in self.camera_names
        }

    @staticmethod
    def _grow(dataset: h5py.Dataset, size: int) -> None:
        dataset.resize((size,) + dataset.shape[1:])

    def append(
        self,
        *,
        record_time: float,
        source_times: Sequence[float],
        joint_position: Sequence[float],
        joint_velocity: Sequence[float],
        tcp_pose: Sequence[float],
        action: Sequence[float],
        action_source: str,
        jpeg_images: Mapping[str, bytes | np.ndarray],
    ) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed episode")
        if set(jpeg_images) != set(self.camera_names):
            raise ValueError(
                f"Image keys {sorted(jpeg_images)} do not match cameras {sorted(self.camera_names)}"
            )
        new_size = self._count + 1
        datasets = [
            self._record_time,
            self._source_time,
            self._joint_pos,
            self._joint_vel,
            self._tcp_pose,
            self._action,
            self._source,
            *self._images.values(),
        ]
        for dataset in datasets:
            self._grow(dataset, new_size)

        idx = self._count
        self._record_time[idx] = float(record_time)
        self._source_time[idx] = np.asarray(source_times, dtype=np.float64)
        self._joint_pos[idx] = np.asarray(joint_position, dtype=np.float32)
        self._joint_vel[idx] = np.asarray(joint_velocity, dtype=np.float32)
        self._tcp_pose[idx] = np.asarray(tcp_pose, dtype=np.float32)
        self._action[idx] = np.asarray(action, dtype=np.float32)
        self._source[idx] = action_source
        for name, payload in jpeg_images.items():
            if isinstance(payload, bytes):
                payload = np.frombuffer(payload, dtype=np.uint8)
            self._images[name][idx] = np.asarray(payload, dtype=np.uint8)
        self._count = new_size

    def finalize(self, extra_metadata: Mapping[str, Any] | None = None) -> Path:
        if self._closed:
            raise RuntimeError("Episode is already closed")
        self._file.attrs["complete"] = True
        self._file.attrs["success"] = True
        self._file.attrs["num_frames"] = self._count
        self._file.attrs["finished_unix_s"] = time.time()
        if extra_metadata:
            for key, value in extra_metadata.items():
                if isinstance(value, (dict, list, tuple)):
                    value = json.dumps(value)
                self._file.attrs[key] = value
        self._file.flush()
        self._file.close()
        self._closed = True
        os.replace(self.partial_path, self.final_path)
        self._append_manifest()
        return self.final_path

    def discard(self) -> None:
        if not self._closed:
            self._file.close()
            self._closed = True
        self.partial_path.unlink(missing_ok=True)

    def _append_manifest(self) -> None:
        manifest_path = self.root / "manifest.jsonl"
        entry = {
            "episode_index": self.episode_index,
            "path": self.final_path.name,
            "num_frames": self._count,
            "fps": self.fps,
            "task": self.task,
            "success": True,
        }
        with manifest_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def __enter__(self) -> "EpisodeWriter":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if not self._closed:
            self.discard()


def read_episode_metadata(path: str | Path) -> dict[str, Any]:
    with h5py.File(path, "r") as handle:
        attrs = dict(handle.attrs)
        attrs["joint_names"] = json.loads(attrs.pop("joint_names_json"))
        attrs["camera_names"] = json.loads(attrs.pop("camera_names_json"))
        attrs["num_frames"] = int(handle["timestamp/record"].shape[0])
        return attrs
