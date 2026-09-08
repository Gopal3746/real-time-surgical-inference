import argparse
from pathlib import Path

from surgphase.triton_repository import (
    DEFAULT_MAX_BATCH_SIZE,
    DEFAULT_MODEL_NAME,
    DEFAULT_MODEL_VERSION,
    get_repository_paths,
    prepare_triton_repository,
    render_triton_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a Triton model repository "
            "for the surgical phase TensorRT engine."
        )
    )

    parser.add_argument(
        "--engine",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--repository",
        type=Path,
        default=Path(
            "deployment/model_repository"
        ),
    )

    parser.add_argument(
        "--model-name",
        default=DEFAULT_MODEL_NAME,
    )

    parser.add_argument(
        "--version",
        type=int,
        default=DEFAULT_MODEL_VERSION,
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=DEFAULT_MAX_BATCH_SIZE,
    )

    parser.add_argument(
        "--instance-count",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--queue-delay-us",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the model repository layout "
            "and config without copying an engine."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        paths = get_repository_paths(
            repository=args.repository,
            model_name=args.model_name,
            version=args.version,
        )

        print("Triton repository layout")
        print("------------------------")
        print(paths.config_path)
        print(paths.engine_path)
        print()
        print("config.pbtxt")
        print("------------")

        print(
            render_triton_config(
                model_name=args.model_name,
                max_batch_size=(
                    args.max_batch_size
                ),
                instance_count=(
                    args.instance_count
                ),
                max_queue_delay_microseconds=(
                    args.queue_delay_us
                ),
            )
        )

        return

    paths = prepare_triton_repository(
        engine_source=args.engine,
        repository=args.repository,
        model_name=args.model_name,
        version=args.version,
        max_batch_size=args.max_batch_size,
        instance_count=args.instance_count,
        max_queue_delay_microseconds=(
            args.queue_delay_us
        ),
    )

    print("Prepared Triton model repository")
    print(
        f"Config: {paths.config_path}"
    )
    print(
        f"Engine: {paths.engine_path}"
    )


if __name__ == "__main__":
    main()