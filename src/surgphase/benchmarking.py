from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from time import perf_counter

import numpy as np
from numpy.typing import NDArray

from surgphase.preprocessing import (
    build_frame_transform,
)
from surgphase.streaming import (
    infer_stream_frame,
)
from surgphase.temporal import (
    TemporalPhaseAggregator,
)
from surgphase.triton_client import (
    TritonPhaseClient,
)
from surgphase.video import iter_sampled_frames


@dataclass(frozen=True)
class LatencySummary:
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float


@dataclass(frozen=True)
class StreamingBenchmark:
    concurrency: int
    requests_per_stream: int
    total_requests: int
    duration_seconds: float
    throughput_fps: float
    triton_latency: LatencySummary
    end_to_end_latency: LatencySummary


@dataclass(frozen=True)
class _WorkerResult:
    triton_latencies_ms: tuple[float, ...]
    end_to_end_latencies_ms: tuple[float, ...]


def summarize_latencies(
    latencies_ms: Sequence[float],
) -> LatencySummary:
    if not latencies_ms:
        raise ValueError(
            "Latency samples cannot be empty"
        )

    values = np.asarray(
        latencies_ms,
        dtype=np.float64,
    )

    if np.any(values < 0):
        raise ValueError(
            "Latency samples cannot be negative"
        )

    return LatencySummary(
        mean_ms=float(
            np.mean(values)
        ),
        p50_ms=float(
            np.percentile(
                values,
                50,
            )
        ),
        p95_ms=float(
            np.percentile(
                values,
                95,
            )
        ),
        p99_ms=float(
            np.percentile(
                values,
                99,
            )
        ),
    )


def load_benchmark_frames(
    video_path: Path,
    *,
    sample_fps: float = 1.0,
    frame_limit: int = 32,
) -> tuple[NDArray[np.uint8], ...]:
    if frame_limit <= 0:
        raise ValueError(
            "Frame limit must be positive"
        )

    frames = tuple(
        frame
        for _, frame in islice(
            iter_sampled_frames(
                video_path=video_path,
                sample_fps=sample_fps,
            ),
            frame_limit,
        )
    )

    if not frames:
        raise ValueError(
            "Video produced no benchmark frames"
        )

    return frames


def _run_stream_worker(
    *,
    frames: Sequence[NDArray[np.uint8]],
    requests: int,
    source_fps: float,
    temporal_window: int,
    client_factory: Callable[
        [],
        TritonPhaseClient,
    ],
) -> _WorkerResult:
    client = client_factory()

    aggregator = TemporalPhaseAggregator(
        window_size=temporal_window
    )

    transform = build_frame_transform()

    triton_latencies: list[float] = []
    end_to_end_latencies: list[float] = []

    for request_index in range(
        requests
    ):
        frame = frames[
            request_index % len(frames)
        ]

        start = perf_counter()

        prediction = infer_stream_frame(
            frame,
            frame_number=request_index,
            source_fps=source_fps,
            client=client,
            aggregator=aggregator,
            transform=transform,
        )

        elapsed_ms = (
            perf_counter() - start
        ) * 1000.0

        triton_latencies.append(
            prediction.inference_latency_ms
        )

        end_to_end_latencies.append(
            elapsed_ms
        )

    return _WorkerResult(
        triton_latencies_ms=tuple(
            triton_latencies
        ),
        end_to_end_latencies_ms=tuple(
            end_to_end_latencies
        ),
    )


def run_concurrent_stream_benchmark(
    *,
    frames: Sequence[NDArray[np.uint8]],
    client_factory: Callable[
        [],
        TritonPhaseClient,
    ],
    concurrency: int = 1,
    requests_per_stream: int = 100,
    source_fps: float = 25.0,
    temporal_window: int = 5,
) -> StreamingBenchmark:
    if not frames:
        raise ValueError(
            "Benchmark requires at least one frame"
        )

    if concurrency <= 0:
        raise ValueError(
            "Concurrency must be positive"
        )

    if requests_per_stream <= 0:
        raise ValueError(
            "Requests per stream must be positive"
        )

    if source_fps <= 0:
        raise ValueError(
            "Source FPS must be positive"
        )

    if temporal_window <= 0:
        raise ValueError(
            "Temporal window must be positive"
        )

    start = perf_counter()

    with ThreadPoolExecutor(
        max_workers=concurrency
    ) as executor:
        futures = [
            executor.submit(
                _run_stream_worker,
                frames=frames,
                requests=(
                    requests_per_stream
                ),
                source_fps=source_fps,
                temporal_window=(
                    temporal_window
                ),
                client_factory=(
                    client_factory
                ),
            )
            for _ in range(concurrency)
        ]

        worker_results = [
            future.result()
            for future in futures
        ]

    duration_seconds = (
        perf_counter() - start
    )

    triton_latencies = [
        latency
        for result in worker_results
        for latency
        in result.triton_latencies_ms
    ]

    end_to_end_latencies = [
        latency
        for result in worker_results
        for latency
        in result.end_to_end_latencies_ms
    ]

    total_requests = (
        concurrency
        * requests_per_stream
    )

    throughput_fps = (
        total_requests
        / duration_seconds
    )

    return StreamingBenchmark(
        concurrency=concurrency,
        requests_per_stream=(
            requests_per_stream
        ),
        total_requests=total_requests,
        duration_seconds=(
            duration_seconds
        ),
        throughput_fps=throughput_fps,
        triton_latency=(
            summarize_latencies(
                triton_latencies
            )
        ),
        end_to_end_latency=(
            summarize_latencies(
                end_to_end_latencies
            )
        ),
    )
