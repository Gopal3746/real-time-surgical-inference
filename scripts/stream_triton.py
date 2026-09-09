import argparse
import time
from pathlib import Path

from surgphase.streaming import stream_video
from surgphase.triton_client import (
    DEFAULT_TRITON_URL,
    TritonPhaseClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Stream surgical video frames "
            "through Triton Inference Server."
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
        "--sample-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--temporal-window",
        type=int,
        default=5,
    )

    parser.add_argument(
        "--realtime",
        action="store_true",
        help=(
            "Pace requests according to "
            "the requested sampling rate."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    client = TritonPhaseClient(
        url=args.url,
        model_name=args.model_name,
        model_version=args.model_version,
    )

    print(
        f"Checking Triton at "
        f"{args.url}..."
    )

    client.check_ready()

    print("Triton server ready.")
    print()

    interval_seconds = (
        1.0 / args.sample_fps
    )

    for prediction in stream_video(
        video_path=args.video,
        client=client,
        sample_fps=args.sample_fps,
        temporal_window=(
            args.temporal_window
        ),
    ):
        start = time.monotonic()

        print(
            f"t={prediction.timestamp_seconds:8.2f}s "
            f"frame={prediction.frame_number:7d} | "
            f"raw={prediction.raw_phase:<28} "
            f"({prediction.raw_confidence:.3f}) | "
            f"smoothed={prediction.smoothed_phase:<28} "
            f"({prediction.smoothed_confidence:.3f}) | "
            f"triton={prediction.inference_latency_ms:.2f} ms"
        )

        if args.realtime:
            elapsed = (
                time.monotonic() - start
            )

            sleep_seconds = max(
                0.0,
                interval_seconds - elapsed,
            )

            time.sleep(
                sleep_seconds
            )


if __name__ == "__main__":
    main()