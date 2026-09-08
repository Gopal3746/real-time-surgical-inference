from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

INPUT_NAME = "input"
OUTPUT_NAME = "logits"

DEFAULT_INPUT_SHAPE = (
    3,
    224,
    224,
)

DEFAULT_MAX_BATCH_SIZE = 32


@dataclass(frozen=True)
class OnnxParityResult:
    batch_size: int
    max_abs_diff: float
    max_rel_diff: float


def export_model_to_onnx(
    model: torch.nn.Module,
    output_path: Path,
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
    example_batch_size: int = 2,
    dynamic_batch: bool = True,
    max_batch_size: int = DEFAULT_MAX_BATCH_SIZE,
) -> Path:
    if example_batch_size <= 0:
        raise ValueError(
            "Example batch size must be positive"
        )

    if (
        len(input_shape) != 3
        or any(
            dimension <= 0
            for dimension in input_shape
        )
    ):
        raise ValueError(
            f"Invalid input shape: {input_shape}"
        )

    if max_batch_size <= 0:
        raise ValueError(
            "Maximum batch size must be positive"
        )

    if (
        dynamic_batch
        and example_batch_size > max_batch_size
    ):
        raise ValueError(
            "Example batch size cannot exceed "
            "maximum batch size"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = model.cpu()
    model.eval()

    example_input = torch.randn(
        example_batch_size,
        *input_shape,
        dtype=torch.float32,
    )

    dynamic_shapes = None

    if dynamic_batch:
        batch_dimension = torch.export.Dim(
            "batch",
            min=1,
            max=max_batch_size,
        )

        dynamic_shapes = (
            {
                0: batch_dimension,
            },
        )

    torch.onnx.export(
        model,
        (example_input,),
        f=str(output_path),
        input_names=[
            INPUT_NAME,
        ],
        output_names=[
            OUTPUT_NAME,
        ],
        dynamo=True,
        dynamic_shapes=dynamic_shapes,
        external_data=False,
    )

    onnx_model = onnx.load(
        str(output_path)
    )

    onnx.checker.check_model(
        onnx_model
    )

    return output_path


def run_onnx_inference(
    model_path: Path,
    inputs: torch.Tensor,
) -> np.ndarray:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"ONNX model does not exist: {model_path}"
        )

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    input_array = (
        inputs
        .detach()
        .cpu()
        .to(torch.float32)
        .numpy()
    )

    outputs = session.run(
        [
            OUTPUT_NAME,
        ],
        {
            INPUT_NAME: input_array,
        },
    )

    return np.asarray(
        outputs[0]
    )


def compare_pytorch_onnx(
    model: torch.nn.Module,
    model_path: Path,
    inputs: torch.Tensor,
    *,
    rtol: float = 1e-3,
    atol: float = 1e-4,
) -> OnnxParityResult:
    model = model.cpu()
    model.eval()

    cpu_inputs = (
        inputs
        .detach()
        .cpu()
        .to(torch.float32)
    )

    with torch.inference_mode():
        pytorch_output = (
            model(cpu_inputs)
            .detach()
            .cpu()
            .numpy()
        )

    onnx_output = run_onnx_inference(
        model_path=model_path,
        inputs=cpu_inputs,
    )

    absolute_difference = np.abs(
        pytorch_output - onnx_output
    )

    denominator = np.maximum(
        np.abs(pytorch_output),
        1e-8,
    )

    relative_difference = (
        absolute_difference / denominator
    )

    max_abs_diff = float(
        absolute_difference.max()
    )

    max_rel_diff = float(
        relative_difference.max()
    )

    if not np.allclose(
        pytorch_output,
        onnx_output,
        rtol=rtol,
        atol=atol,
    ):
        raise ValueError(
            "ONNX output does not match PyTorch output: "
            f"max_abs_diff={max_abs_diff:.6g}, "
            f"max_rel_diff={max_rel_diff:.6g}"
        )

    return OnnxParityResult(
        batch_size=cpu_inputs.shape[0],
        max_abs_diff=max_abs_diff,
        max_rel_diff=max_rel_diff,
    )