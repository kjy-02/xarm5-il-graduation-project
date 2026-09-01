"""Offline training entry point for the xArm action-chunk policy."""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .dataset import (
    ChunkedEpisodeDataset,
    compute_normalization_stats,
    split_episode_files,
)
from .model import ActionChunkTransformer, ModelConfig, masked_action_loss
from .storage import list_episode_files
from .validate_dataset import validate_files


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Directory containing episode_*.h5")
    parser.add_argument("--output", type=Path, required=True, help="Checkpoint output directory")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--horizon", type=int, default=20)
    parser.add_argument("--image-width", type=int, default=224)
    parser.add_argument("--image-height", type=int, default=224)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--nhead", type=int, default=8)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--smoothness-weight", type=float, default=0.01)
    parser.add_argument("--pretrained-backbone", action="store_true")
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args(argv)


def _select_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _run_epoch(
    model: ActionChunkTransformer,
    loader: DataLoader,
    device: torch.device,
    smoothness_weight: float,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    totals = {"loss": 0.0, "imitation": 0.0, "smoothness": 0.0}
    sample_count = 0
    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for batch in loader:
            images = batch["images"].to(device, non_blocking=True)
            state = batch["state"].to(device, non_blocking=True)
            target = batch["actions"].to(device, non_blocking=True)
            mask = batch["action_mask"].to(device, non_blocking=True)
            prediction = model(images, state)
            loss, components = masked_action_loss(
                prediction, target, mask, smoothness_weight=smoothness_weight
            )
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            batch_size = int(images.shape[0])
            totals["loss"] += float(loss.detach()) * batch_size
            totals["imitation"] += float(components["imitation"]) * batch_size
            totals["smoothness"] += float(components["smoothness"]) * batch_size
            sample_count += batch_size
    return {key: value / max(sample_count, 1) for key, value in totals.items()}


def _checkpoint(
    path: Path,
    model: ActionChunkTransformer,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    stats: dict[str, list[float]],
    camera_names: tuple[str, ...],
    joint_names: tuple[str, ...],
    image_size: tuple[int, int],
    train_files: list[Path],
    validation_files: list[Path],
    metrics: dict[str, float],
) -> None:
    payload = {
        "format_version": "xarm_il_policy_v1",
        "epoch": epoch,
        "model_config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "normalization": stats,
        "camera_names": list(camera_names),
        "joint_names": list(joint_names),
        "image_size": list(image_size),
        "train_files": [item.name for item in train_files],
        "validation_files": [item.name for item in validation_files],
        "metrics": metrics,
    }
    torch.save(payload, path)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    if args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("epochs and batch-size must be positive")
    if args.workers < 0:
        raise ValueError("workers must be non-negative")
    if args.image_width <= 0 or args.image_height <= 0 or args.horizon <= 0:
        raise ValueError("image dimensions and horizon must be positive")
    if args.learning_rate <= 0.0 or args.weight_decay < 0.0:
        raise ValueError("learning-rate must be positive and weight-decay non-negative")
    if args.d_model <= 0 or args.nhead <= 0 or args.decoder_layers <= 0:
        raise ValueError("Transformer dimensions and layer count must be positive")
    if not 0.0 <= args.dropout < 1.0 or args.smoothness_weight < 0.0:
        raise ValueError("dropout must be in [0, 1) and smoothness-weight non-negative")
    if args.output.exists() and not args.output.is_dir():
        raise FileExistsError(f"Training output path is not a directory: {args.output}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Training output directory is not empty: {args.output}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    files = list_episode_files(args.data)
    validation_report = validate_files(files)
    if not validation_report["valid"]:
        problems = [
            f"{item['path']}: {error}"
            for item in validation_report["episodes"]
            for error in item["errors"]
        ]
        raise ValueError("Dataset validation failed:\n" + "\n".join(problems))
    train_files, validation_files = split_episode_files(
        files, args.validation_fraction, args.seed
    )
    stats = compute_normalization_stats(train_files)
    image_size = (args.image_width, args.image_height)
    train_dataset = ChunkedEpisodeDataset(
        train_files, stats, horizon=args.horizon, image_size=image_size
    )
    validation_dataset = (
        ChunkedEpisodeDataset(
            validation_files,
            stats,
            horizon=args.horizon,
            image_size=image_size,
            camera_names=train_dataset.camera_names,
        )
        if validation_files
        else None
    )
    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
        persistent_workers=args.workers > 0,
    )
    validation_loader = (
        DataLoader(
            validation_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=pin_memory,
            persistent_workers=args.workers > 0,
        )
        if validation_dataset is not None
        else None
    )

    config = ModelConfig(
        camera_count=len(train_dataset.camera_names),
        state_dim=train_dataset.state_dim,
        action_dim=train_dataset.action_dim,
        horizon=args.horizon,
        d_model=args.d_model,
        nhead=args.nhead,
        decoder_layers=args.decoder_layers,
        dropout=args.dropout,
    )
    device = _select_device(args.device)
    model = ActionChunkTransformer(config, pretrained_backbone=args.pretrained_backbone).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    args.output.mkdir(parents=True, exist_ok=True)
    run_config = vars(args).copy()
    run_config.update(
        {
            "data": str(args.data),
            "output": str(args.output),
            "device_used": str(device),
            "train_episodes": [path.name for path in train_files],
            "validation_episodes": [path.name for path in validation_files],
            "model_config": config.to_dict(),
            "normalization": stats.to_dict(),
        }
    )
    (args.output / "run_config.json").write_text(
        json.dumps(run_config, indent=2), encoding="utf-8"
    )

    best_metric = float("inf")
    history_path = args.output / "history.jsonl"
    print(
        f"device={device} train_frames={len(train_dataset)} "
        f"validation_frames={len(validation_dataset) if validation_dataset else 0}"
    )
    for epoch in range(1, args.epochs + 1):
        started = time.monotonic()
        train_metrics = _run_epoch(
            model, train_loader, device, args.smoothness_weight, optimizer
        )
        if validation_loader is not None:
            validation_metrics = _run_epoch(
                model, validation_loader, device, args.smoothness_weight, optimizer=None
            )
        else:
            validation_metrics = train_metrics
        scheduler.step()
        record = {
            "epoch": epoch,
            "learning_rate": scheduler.get_last_lr()[0],
            "duration_s": time.monotonic() - started,
            **{f"train_{key}": value for key, value in train_metrics.items()},
            **{f"validation_{key}": value for key, value in validation_metrics.items()},
        }
        with history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record) + "\n")
        _checkpoint(
            args.output / "last.pt",
            model,
            optimizer,
            epoch,
            stats.to_dict(),
            train_dataset.camera_names,
            train_dataset.joint_names,
            image_size,
            train_files,
            validation_files,
            record,
        )
        metric = validation_metrics["loss"]
        if metric < best_metric:
            best_metric = metric
            _checkpoint(
                args.output / "best.pt",
                model,
                optimizer,
                epoch,
                stats.to_dict(),
                train_dataset.camera_names,
                train_dataset.joint_names,
                image_size,
                train_files,
                validation_files,
                record,
            )
        print(
            f"epoch={epoch:03d} train={train_metrics['loss']:.5f} "
            f"validation={validation_metrics['loss']:.5f} "
            f"time={record['duration_s']:.1f}s"
        )


if __name__ == "__main__":
    main()
