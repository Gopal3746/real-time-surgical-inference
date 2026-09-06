from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float


def open_video(video_path: Path) -> cv2.VideoCapture:
    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video file does not exist: {video_path}"
        )

    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        capture.release()

        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    return capture


def get_video_metadata(
    video_path: Path,
) -> VideoMetadata:
    capture = open_video(video_path)

    try:
        fps = float(
            capture.get(cv2.CAP_PROP_FPS)
        )

        frame_count = int(
            capture.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        width = int(
            capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        )

        height = int(
            capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        )
    finally:
        capture.release()

    if fps <= 0:
        raise ValueError(
            f"Invalid video FPS: {fps}"
        )

    duration_seconds = frame_count / fps

    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=width,
        height=height,
        duration_seconds=duration_seconds,
    )


def read_frame(
    video_path: Path,
    frame_number: int,
) -> NDArray[np.uint8]:
    if frame_number < 0:
        raise ValueError(
            f"Frame number must be non-negative: {frame_number}"
        )

    capture = open_video(video_path)

    try:
        capture.set(
            cv2.CAP_PROP_POS_FRAMES,
            frame_number,
        )

        success, frame = capture.read()
    finally:
        capture.release()

    if not success or frame is None:
        raise IndexError(
            f"Could not read frame {frame_number} "
            f"from {video_path}"
        )

    return cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )


def iter_sampled_frames(
    video_path: Path,
    sample_fps: float = 1.0,
):
    if sample_fps <= 0:
        raise ValueError(
            f"Sample FPS must be positive: {sample_fps}"
        )

    metadata = get_video_metadata(video_path)

    if sample_fps > metadata.fps:
        raise ValueError(
            "Sample FPS cannot exceed source video FPS"
        )

    frame_step = max(
        1,
        round(metadata.fps / sample_fps),
    )

    for frame_number in range(
        0,
        metadata.frame_count,
        frame_step,
    ):
        yield (
            frame_number,
            read_frame(
                video_path=video_path,
                frame_number=frame_number,
            ),
        )