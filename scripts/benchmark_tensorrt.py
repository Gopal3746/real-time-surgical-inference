import argparse
import json
from dataclasses import asdict
from pathlib import Path

from surgphase.tensorrt_tools import (
    benchmark_tensorrt_engine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark a TensorRT surgical "
            "phase inference engine."
        )
    )

    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--warmup-ms",
        type=int,
        default=500,
    )

    parser.add_argument(
        "--duration",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--iterations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=None,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    result = benchmark_tensorrt_engine(
        engine_path=args.engine,
        batch_size=args.batch_size,
        warmup_ms=args.warmup_ms,
        duration_seconds=args.duration,
        iterations=args.iterations,
    )

    print(
        f"Batch size:       "
        f"{result.batch_size}"
    )

    print(
        f"Throughput:       "
        f"{result.throughput_qps:.2f} qps"
    )

    print(
        f"Mean latency:     "
        f"{result.latency_mean_ms:.3f} ms"
    )

    print(
        f"Median latency:   "
        f"{result.latency_median_ms:.3f} ms"
    )

    print(
        f"P95 latency:      "
        f"{result.latency_p95_ms:.3f} ms"
    )

    print(
        f"P99 latency:      "
        f"{result.latency_p99_ms:.3f} ms"
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

        print(
            f"Results written to "
            f"{args.output}"
        )


if __name__ == "__main__":
    main()