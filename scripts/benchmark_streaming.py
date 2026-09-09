import argparse
import json
from dataclasses import asdict
from pathlib import Path

from surgphase.benchmarking import (
    load_benchmark_frames,
    run_concurrent_stream_benchmark,
)
from surgphase.triton_client import (
    DEFAULT_TRITON_URL,
    TritonPhaseClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark concurrent surgical "
            "video streams through Triton."
        )
    )

    parser.add_argument(
        "--video",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--url",
        default=DEFAULT_TRITON_URL,
    )

    parser.add_argument(
        "--model-name",
        default="surgical_phase",
    )

    parser.add_argument(
        "--model-version",
        default="1",
    )

    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--requests-per-stream",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--frame-pool-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    probe_client = TritonPhaseClient(
        url=args.url,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    print(
        f"Checking Triton at "
        f"{args.url}..."
    )

    probe_client.check_ready()

    print("Triton ready.")

    print(
        f"Loading up to "
        f"{args.frame_pool_size} "
        f"benchmark frames..."
    )

    frames = load_benchmark_frames(
        video_path=args.video,
        sample_fps=args.sample_fps,
        frame_limit=args.frame_pool_size,
    )

    print(
        f"Loaded {len(frames)} frames."
    )

    def client_factory() -> TritonPhaseClient:
        return TritonPhaseClient(
            url=args.url,
            model_name=args.model_name,
            model_version=(
                args.model_version
            ),
        )

    print()
    print(
        f"Starting benchmark with "
        f"concurrency={args.concurrency}"
    )

    result = (
        run_concurrent_stream_benchmark(
            frames=frames,
            client_factory=client_factory,
            concurrency=args.concurrency,
            requests_per_stream=(
                args.requests_per_stream
            ),
            source_fps=25.0,
            temporal_window=(
                args.temporal_window
            ),
        )
    )

    print()
    print("Streaming benchmark")
    print("-------------------")

    print(
        f"Concurrency:       "
        f"{result.concurrency}"
    )

    print(
        f"Total requests:    "
        f"{result.total_requests:,}"
    )

    print(
        f"Duration:          "
        f"{result.duration_seconds:.3f} s"
    )

    print(
        f"Throughput:        "
        f"{result.throughput_fps:.2f} fps"
    )

    print()
    print("Triton request latency")

    print(
        f"Mean:              "
        f"{result.triton_latency.mean_ms:.3f} ms"
    )

    print(
        f"P50:               "
        f"{result.triton_latency.p50_ms:.3f} ms"
    )

    print(
        f"P95:               "
        f"{result.triton_latency.p95_ms:.3f} ms"
    )

    print(
        f"P99:               "
        f"{result.triton_latency.p99_ms:.3f} ms"
    )

    print()
    print("End-to-end frame latency")

    print(
        f"Mean:              "
        f"{result.end_to_end_latency.mean_ms:.3f} ms"
    )

    print(
        f"P50:               "
        f"{result.end_to_end_latency.p50_ms:.3f} ms"
    )

    print(
        f"P95:               "
        f"{result.end_to_end_latency.p95_ms:.3f} ms"
    )

    print(
        f"P99:               "
        f"{result.end_to_end_latency.p99_ms:.3f} ms"
    )

    if args.output is not None:
        args.output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        args.output.write_text(
            json.dumps(
                asdict(result),
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        print()
        print(
            f"Results written to "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()
