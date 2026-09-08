import argparse
from pathlib import Path

import torch

from surgphase.model import SurgicalPhaseClassifier
from surgphase.onnx_export import (
    DEFAULT_INPUT_SHAPE,
    compare_pytorch_onnx,
    export_model_to_onnx,
)
from surgphase.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export the trained surgical phase "
            "classifier to ONNX."
        )
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the PyTorch checkpoint.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "models/surgical_phase.onnx"
        ),
    )

    parser.add_argument(
        "--example-batch-size",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
    )

    parser.add_argument(
        "--parity-batch-size",
        type=int,
        default=4,
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if (
        args.parity_batch_size
        > args.max_batch_size
    ):
        raise ValueError(
            "Parity batch size cannot exceed "
            "maximum batch size"
        )

    model = SurgicalPhaseClassifier(
        pretrained=False,
    )

    metadata = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
    )

    print(
        f"Loaded checkpoint from epoch "
        f"{metadata.epoch}"
    )

    output_path = export_model_to_onnx(
        model=model,
        output_path=args.output,
        example_batch_size=(
            args.example_batch_size
        ),
        dynamic_batch=True,
        max_batch_size=args.max_batch_size,
    )

    print(
        f"Exported ONNX model to "
        f"{output_path}"
    )

    parity_inputs = torch.randn(
        args.parity_batch_size,
        *DEFAULT_INPUT_SHAPE,
        dtype=torch.float32,
    )

    result = compare_pytorch_onnx(
        model=model,
        model_path=output_path,
        inputs=parity_inputs,
    )

    print(
        f"Parity batch size: "
        f"{result.batch_size}"
    )

    print(
        f"Maximum absolute difference: "
        f"{result.max_abs_diff:.8f}"
    )

    print(
        f"Maximum relative difference: "
        f"{result.max_rel_diff:.8f}"
    )

    print(
        "PyTorch and ONNX outputs match."
    )


if __name__ == "__main__":
    main()