"""Render a recorded episode as a side-by-side MP4 for human review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import h5py
import numpy as np

from .dataset import decode_jpeg


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--height", type=int, default=480)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    output_path = args.output or args.episode.with_suffix(".mp4")
    with h5py.File(args.episode, "r") as episode:
        camera_names = tuple(json.loads(episode.attrs["camera_names_json"]))
        frame_count = len(episode["timestamp/record"])
        fps = float(episode.attrs.get("fps", 10.0))
        if not camera_names or frame_count == 0:
            raise ValueError("Episode has no camera frames")
        first_images = [
            decode_jpeg(episode[f"observation/images/{name}"][0]) for name in camera_names
        ]
        widths = [max(1, round(image.shape[1] * args.height / image.shape[0])) for image in first_images]
        canvas_size = (sum(widths), args.height)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(
            str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, canvas_size
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open video writer: {output_path}")
        try:
            actions = episode["action/twist"]
            for frame_index in range(frame_count):
                panels = []
                for name, width in zip(camera_names, widths):
                    image = decode_jpeg(episode[f"observation/images/{name}"][frame_index])
                    panel = cv2.resize(image, (width, args.height), interpolation=cv2.INTER_AREA)
                    cv2.putText(
                        panel,
                        name,
                        (16, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )
                    panels.append(panel)
                canvas = np.concatenate(panels, axis=1)
                action = actions[frame_index]
                label = (
                    f"frame {frame_index + 1}/{frame_count}  "
                    f"v=({action[0]:+.3f}, {action[1]:+.3f}, {action[2]:+.3f}) m/s"
                )
                cv2.putText(
                    canvas,
                    label,
                    (16, args.height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(canvas)
        finally:
            writer.release()
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
