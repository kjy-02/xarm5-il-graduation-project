"""Create synthetic data and exercise storage, validation, dataset, and model paths."""

from __future__ import annotations

import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import ChunkedEpisodeDataset, compute_normalization_stats
from .model import ActionChunkTransformer, ModelConfig, masked_action_loss
from .storage import EpisodeWriter
from .train_policy import main as train_main
from .validate_dataset import validate_files


def _jpeg(image: np.ndarray) -> bytes:
    ok, payload = cv2.imencode(".jpg", image)
    if not ok:
        raise RuntimeError("Synthetic JPEG encoding failed")
    return payload.tobytes()


def run_smoke_test(root: Path) -> None:
    rng = np.random.default_rng(7)
    writer = EpisodeWriter(
        root,
        episode_index=0,
        joint_names=[f"joint{i}" for i in range(1, 6)],
        camera_names=["wrist", "overhead"],
        task="synthetic smoke test",
        fps=10.0,
    )
    for frame in range(24):
        wrist = rng.integers(0, 256, (64, 80, 3), dtype=np.uint8)
        overhead = rng.integers(0, 256, (72, 96, 3), dtype=np.uint8)
        t = frame / 10.0
        q = np.sin(np.arange(5, dtype=np.float32) + t) * 0.1
        qdot = np.cos(np.arange(5, dtype=np.float32) + t) * 0.01
        action = np.asarray([0.01 * np.sin(t), 0.01 * np.cos(t), 0.002], dtype=np.float32)
        writer.append(
            record_time=100.0 + t,
            source_times=[100.0 + t] * 4,
            joint_position=q,
            joint_velocity=qdot,
            tcp_pose=[0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 1.0],
            action=action,
            action_source="teleop",
            jpeg_images={"wrist": _jpeg(wrist), "overhead": _jpeg(overhead)},
        )
    episode_path = writer.finalize({"success": True})
    report = validate_files([episode_path])
    if not report["valid"]:
        raise AssertionError(report)

    stats = compute_normalization_stats([episode_path])
    dataset = ChunkedEpisodeDataset(
        [episode_path], stats, horizon=4, image_size=(64, 64)
    )
    batch = next(iter(DataLoader(dataset, batch_size=2, shuffle=False)))
    config = ModelConfig(
        camera_count=2,
        state_dim=10,
        action_dim=3,
        horizon=4,
        d_model=64,
        nhead=4,
        decoder_layers=1,
        dropout=0.0,
    )
    model = ActionChunkTransformer(config, pretrained_backbone=False)
    prediction = model(batch["images"], batch["state"])
    loss, _ = masked_action_loss(
        prediction, batch["actions"], batch["action_mask"], smoothness_weight=0.01
    )
    loss.backward()
    if prediction.shape != (2, 4, 3) or not torch.isfinite(loss):
        raise AssertionError("Model smoke test produced invalid output")

    training_output = root / "training_output"
    train_main(
        [
            "--data",
            str(root),
            "--output",
            str(training_output),
            "--epochs",
            "1",
            "--batch-size",
            "4",
            "--horizon",
            "4",
            "--image-width",
            "64",
            "--image-height",
            "64",
            "--validation-fraction",
            "0",
            "--workers",
            "0",
            "--d-model",
            "64",
            "--nhead",
            "4",
            "--decoder-layers",
            "1",
            "--device",
            "cpu",
        ]
    )
    checkpoint_path = training_output / "best.pt"
    if not checkpoint_path.is_file():
        raise AssertionError("Miniature training run did not write best.pt")
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
    restored = ActionChunkTransformer(ModelConfig(**checkpoint["model_config"]))
    restored.load_state_dict(checkpoint["model_state"])
    restored.eval()
    with torch.inference_mode():
        restored_action = restored(batch["images"], batch["state"])
    if restored_action.shape != (2, 4, 3) or not torch.all(torch.isfinite(restored_action)):
        raise AssertionError("Restored checkpoint produced invalid output")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="xarm_il_smoke_") as temporary:
        run_smoke_test(Path(temporary))
    print("PASS: storage -> validation -> dataset -> model -> backward -> checkpoint")


if __name__ == "__main__":
    main()
