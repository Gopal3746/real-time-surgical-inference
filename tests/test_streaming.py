import numpy as np
import torch

from surgphase.streaming import (
    infer_stream_frame,
)
from surgphase.temporal import (
    TemporalPhaseAggregator,
)
from surgphase.triton_client import (
    TritonInferenceResult,
)


class FakePhaseClient:
    def __init__(
        self,
        phases: list[int],
    ) -> None:
        self.phases = phases
        self.index = 0

    def infer(
        self,
        inputs: torch.Tensor,
    ) -> TritonInferenceResult:
        phase = self.phases[
            self.index
        ]

        self.index += 1

        logits = torch.zeros(
            1,
            7,
        )

        logits[0, phase] = 10.0

        return TritonInferenceResult(
            logits=logits,
            latency_ms=2.5,
        )


def create_frame() -> np.ndarray:
    return np.zeros(
        (
            48,
            64,
            3,
        ),
        dtype=np.uint8,
    )


def test_stream_frame_prediction() -> None:
    client = FakePhaseClient(
        [2]
    )

    aggregator = TemporalPhaseAggregator(
        window_size=3
    )

    prediction = infer_stream_frame(
        create_frame(),
        frame_number=25,
        source_fps=25.0,
        client=client,
        aggregator=aggregator,
    )

    assert (
        prediction.frame_number
        == 25
    )

    assert (
        prediction.timestamp_seconds
        == 1.0
    )

    assert (
        prediction.raw_phase_index
        == 2
    )

    assert (
        prediction.smoothed_phase_index
        == 2
    )

    assert (
        prediction.inference_latency_ms
        == 2.5
    )


def test_stream_temporal_smoothing() -> None:
    client = FakePhaseClient(
        [
            2,
            2,
            1,
        ]
    )

    aggregator = TemporalPhaseAggregator(
        window_size=3
    )

    frame = create_frame()

    infer_stream_frame(
        frame,
        frame_number=0,
        source_fps=25.0,
        client=client,
        aggregator=aggregator,
    )

    infer_stream_frame(
        frame,
        frame_number=25,
        source_fps=25.0,
        client=client,
        aggregator=aggregator,
    )

    prediction = infer_stream_frame(
        frame,
        frame_number=50,
        source_fps=25.0,
        client=client,
        aggregator=aggregator,
    )

    assert (
        prediction.raw_phase_index
        == 1
    )

    assert (
        prediction.smoothed_phase_index
        == 2
    )