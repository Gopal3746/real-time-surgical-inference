from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from numpy.typing import NDArray

from surgphase.annotations import PHASES
from surgphase.preprocessing import (
    FrameTransform,
    build_frame_transform,
    preprocess_frame,
)
from surgphase.temporal import (
    TemporalPhaseAggregator,
)
from surgphase.triton_client import (
    TritonPhaseClient,
)
from surgphase.video import (
    get_video_metadata,
    iter_sampled_frames,
)


@dataclass(frozen=True)
class StreamingPrediction:
    frame_number: int
    timestamp_seconds: float
    raw_phase_index: int
    raw_phase: str
    raw_confidence: float
    smoothed_phase_index: int
    smoothed_phase: str
    smoothed_confidence: float
    inference_latency_ms: float


def infer_stream_frame(
    frame: NDArray[np.uint8],
    *,
    frame_number: int,
    source_fps: float,
    client: TritonPhaseClient,
    aggregator: TemporalPhaseAggregator,
    transform: FrameTransform | None = None,
) -> StreamingPrediction:
    if frame_number < 0:
        raise ValueError(
            "Frame number must be non-negative"
        )

    if source_fps <= 0:
        raise ValueError(
            "Source FPS must be positive"
        )

    if transform is None:
        transform = build_frame_transform()

    image = preprocess_frame(
        frame,
        transform=transform,
    )

    batch = image.unsqueeze(0)

    result = client.infer(
        batch
    )

    logits = result.logits.squeeze(0)

    probabilities = torch.softmax(
        logits,
        dim=0,
    )

    raw_phase_index = int(
        probabilities.argmax().item()
    )

    raw_confidence = float(
        probabilities[
            raw_phase_index
        ].item()
    )

    temporal = aggregator.update(
        logits
    )

    return StreamingPrediction(
        frame_number=frame_number,
        timestamp_seconds=(
            frame_number / source_fps
        ),
        raw_phase_index=raw_phase_index,
        raw_phase=PHASES[
            raw_phase_index
        ],
        raw_confidence=raw_confidence,
        smoothed_phase_index=(
            temporal.phase_index
        ),
        smoothed_phase=temporal.phase,
        smoothed_confidence=(
            temporal.confidence
        ),
        inference_latency_ms=(
            result.latency_ms
        ),
    )


def stream_video(
    video_path: Path,
    *,
    client: TritonPhaseClient,
    sample_fps: float = 1.0,
    temporal_window: int = 5,
    transform: FrameTransform | None = None,
) -> Iterator[StreamingPrediction]:
    metadata = get_video_metadata(
        video_path
    )

    aggregator = TemporalPhaseAggregator(
        window_size=temporal_window
    )

    if transform is None:
        transform = build_frame_transform()

    for (
        frame_number,
        frame,
    ) in iter_sampled_frames(
        video_path=video_path,
        sample_fps=sample_fps,
    ):
        yield infer_stream_frame(
            frame,
            frame_number=frame_number,
            source_fps=metadata.fps,
            client=client,
            aggregator=aggregator,
            transform=transform,
        )