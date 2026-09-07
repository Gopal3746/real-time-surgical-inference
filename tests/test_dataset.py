from pathlib import Path

import cv2
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from surgphase.dataset import Cholec80Dataset
from surgphase.manifest import (
    ManifestRecord,
    write_manifest,
)


def create_test_video(
    video_path: Path,
    fps: float = 25.0,
    frame_count: int = 50,
) -> None:
    width = 64
    height = 48

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not create temporary test video"
        )

    try:
        for frame_number in range(frame_count):
            value = min(
                frame_number * 5,
                255,
            )

            frame = np.full(
                (height, width, 3),
                value,
                dtype=np.uint8,
            )

            writer.write(frame)
    finally:
        writer.release()


def create_test_manifest(
    manifest_path: Path,
    video_path: Path,
    frame_count: int = 50,
) -> None:
    records = [
        ManifestRecord(
            video_id="video01",
            video_number=1,
            split="train",
            frame=frame_number,
            timestamp_seconds=frame_number / 25.0,
            phase="Preparation",
            phase_index=0,
            video_path=str(video_path),
        )
        for frame_number in range(frame_count)
    ]

    write_manifest(
        records=records,
        output_path=manifest_path,
    )


def test_dataset_samples_at_one_fps(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video01.mp4"
    manifest_path = tmp_path / "manifest.csv"

    create_test_video(video_path)
    create_test_manifest(
        manifest_path,
        video_path,
    )

    dataset = Cholec80Dataset(
        manifest_path=manifest_path,
        split="train",
        sample_fps=1.0,
    )

    assert len(dataset) == 2

    assert dataset.records[0].frame == 0
    assert dataset.records[1].frame == 25


def test_dataset_returns_model_ready_sample(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video01.mp4"
    manifest_path = tmp_path / "manifest.csv"

    create_test_video(video_path)
    create_test_manifest(
        manifest_path,
        video_path,
    )

    dataset = Cholec80Dataset(
        manifest_path=manifest_path,
        split="train",
    )

    sample = dataset[0]

    image = sample["image"]

    assert isinstance(
        image,
        torch.Tensor,
    )

    assert image.shape == (
        3,
        224,
        224,
    )

    assert image.dtype == torch.float32

    assert sample["label"] == 0
    assert sample["video_id"] == "video01"
    assert sample["frame"] == 0
    assert sample["timestamp_seconds"] == 0.0


def test_dataset_rejects_invalid_split(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="Invalid split",
    ):
        Cholec80Dataset(
            manifest_path=tmp_path / "manifest.csv",
            split="invalid",
        )


def test_dataset_rejects_invalid_sample_fps(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        Cholec80Dataset(
            manifest_path=tmp_path / "manifest.csv",
            split="train",
            sample_fps=30.0,
        )


def test_dataset_works_with_dataloader(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "video01.mp4"
    manifest_path = tmp_path / "manifest.csv"

    create_test_video(video_path)
    create_test_manifest(
        manifest_path,
        video_path,
    )

    dataset = Cholec80Dataset(
        manifest_path=manifest_path,
        split="train",
        sample_fps=1.0,
    )

    loader = DataLoader(
        dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(loader))

    assert batch["image"].shape == (
        2,
        3,
        224,
        224,
    )

    assert batch["label"].shape == (2,)

    assert torch.equal(
        batch["frame"],
        torch.tensor([0, 25]),
    )