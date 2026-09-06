from pathlib import Path

import cv2
import numpy as np
import pytest

from surgphase.video import (
    get_video_metadata,
    iter_sampled_frames,
    read_frame,
)


def create_test_video(
    video_path: Path,
    fps: float = 10.0,
    frame_count: int = 20,
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
                frame_number * 10,
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


def test_get_video_metadata(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "test.mp4"

    create_test_video(video_path)

    metadata = get_video_metadata(video_path)

    assert metadata.fps == pytest.approx(
        10.0,
        abs=0.1,
    )

    assert metadata.frame_count == 20
    assert metadata.width == 64
    assert metadata.height == 48

    assert metadata.duration_seconds == pytest.approx(
        2.0,
        abs=0.1,
    )


def test_read_frame(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "test.mp4"

    create_test_video(video_path)

    frame = read_frame(
        video_path=video_path,
        frame_number=5,
    )

    assert frame.shape == (
        48,
        64,
        3,
    )

    assert frame.dtype == np.uint8


def test_read_frame_rejects_negative_frame(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "test.mp4"

    create_test_video(video_path)

    with pytest.raises(
        ValueError,
        match="non-negative",
    ):
        read_frame(
            video_path=video_path,
            frame_number=-1,
        )


def test_read_frame_rejects_out_of_range_frame(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "test.mp4"

    create_test_video(
        video_path,
        frame_count=5,
    )

    with pytest.raises(
        IndexError,
        match="Could not read frame",
    ):
        read_frame(
            video_path=video_path,
            frame_number=100,
        )


def test_iter_sampled_frames(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "test.mp4"

    create_test_video(
        video_path,
        fps=10.0,
        frame_count=20,
    )

    samples = list(
        iter_sampled_frames(
            video_path=video_path,
            sample_fps=1.0,
        )
    )

    assert len(samples) == 2

    assert samples[0][0] == 0
    assert samples[1][0] == 10


def test_sample_fps_cannot_exceed_video_fps(
    tmp_path: Path,
) -> None:
    video_path = tmp_path / "test.mp4"

    create_test_video(
        video_path,
        fps=10.0,
    )

    with pytest.raises(
        ValueError,
        match="cannot exceed",
    ):
        list(
            iter_sampled_frames(
                video_path=video_path,
                sample_fps=30.0,
            )
        )