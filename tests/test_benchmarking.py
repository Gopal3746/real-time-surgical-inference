import numpy as np
import pytest
import torch

from surgphase.benchmarking import (
    LatencySummary,
    run_concurrent_stream_benchmark,
    summarize_latencies,
)
from surgphase.triton_client import (
    TritonInferenceResult,
)


class FakeBenchmarkClient:
    def infer(
        self,
        inputs: torch.Tensor,
    ) -> TritonInferenceResult:
        logits = torch.zeros(
            inputs.shape[0],
            7,
        )

        logits[:, 2] = 10.0

        return TritonInferenceResult(
            logits=logits,
            latency_ms=2.0,
        )


def create_frames(
    count: int = 4,
) -> tuple[np.ndarray, ...]:
    return tuple(
        np.full(
            (
                48,
                64,
                3,
            ),
            fill_value=index,
            dtype=np.uint8,
        )
        for index in range(count)
    )


def test_summarize_latencies() -> None:
    result = summarize_latencies(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
        ]
    )

    assert isinstance(
        result,
        LatencySummary,
    )

    assert result.mean_ms == pytest.approx(
        3.0
    )

    assert result.p50_ms == pytest.approx(
        3.0
    )

    assert result.p95_ms == pytest.approx(
        4.8
    )

    assert result.p99_ms == pytest.approx(
        4.96
    )


def test_summarize_rejects_empty_samples() -> None:
    with pytest.raises(
        ValueError,
        match="empty",
    ):
        summarize_latencies([])


def test_concurrent_stream_benchmark() -> None:
    result = run_concurrent_stream_benchmark(
        frames=create_frames(),
        client_factory=FakeBenchmarkClient,
        concurrency=2,
        requests_per_stream=3,
        source_fps=25.0,
        temporal_window=3,
    )

    assert result.concurrency == 2

    assert (
        result.requests_per_stream
        == 3
    )

    assert result.total_requests == 6

    assert result.duration_seconds > 0.0

    assert result.throughput_fps > 0.0

    assert (
        result.triton_latency.mean_ms
        == pytest.approx(2.0)
    )

    assert (
        result.triton_latency.p99_ms
        == pytest.approx(2.0)
    )

    assert (
        result.end_to_end_latency.mean_ms
        > 0.0
    )


def test_benchmark_rejects_zero_concurrency() -> None:
    with pytest.raises(
        ValueError,
        match="Concurrency",
    ):
        run_concurrent_stream_benchmark(
            frames=create_frames(),
            client_factory=FakeBenchmarkClient,
            concurrency=0,
        )


def test_benchmark_rejects_empty_frames() -> None:
    with pytest.raises(
        ValueError,
        match="at least one frame",
    ):
        run_concurrent_stream_benchmark(
            frames=[],
            client_factory=FakeBenchmarkClient,
        )


def test_benchmark_rejects_zero_requests() -> None:
    with pytest.raises(
        ValueError,
        match="Requests per stream",
    ):
        run_concurrent_stream_benchmark(
            frames=create_frames(),
            client_factory=FakeBenchmarkClient,
            requests_per_stream=0,
        )
