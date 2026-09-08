import argparse
from pathlib import Path

from surgphase.tensorrt_tools import (
    build_engine_command,
    build_tensorrt_engine,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a TensorRT engine from "
            "the surgical phase ONNX model."
        )
    )

    parser.add_argument(
        "--onnx",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--engine",
        type=Path,
        default=Path(
            "models/surgical_phase.plan"
        ),
    )

    parser.add_argument(
        "--min-batch",
        type=int,
        default=1,
    )

    parser.add_argument(
        "--opt-batch",
        type=int,
        default=8,
    )

    parser.add_argument(
        "--max-batch",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--workspace-mib",
        type=int,
        default=4096,
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Print the trtexec command "
            "without executing it."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.dry_run:
        command = build_engine_command(
            onnx_path=args.onnx,
            engine_path=args.engine,
            min_batch=args.min_batch,
            opt_batch=args.opt_batch,
            max_batch=args.max_batch,
            workspace_mib=(
                args.workspace_mib
            ),
        )

        print(" ".join(command))
        return

    engine_path = build_tensorrt_engine(
        onnx_path=args.onnx,
        engine_path=args.engine,
        min_batch=args.min_batch,
        opt_batch=args.opt_batch,
        max_batch=args.max_batch,
        workspace_mib=args.workspace_mib,
    )

    print(
        f"TensorRT engine saved to "
        f"{engine_path}"
    )


if __name__ == "__main__":
    main()