from pathlib import Path

import onnx
import onnxruntime as ort
import pytest
import torch

from surgphase.onnx_export import (
    INPUT_NAME,
    OnnxParityResult,
    compare_pytorch_onnx,
    export_model_to_onnx,
    run_onnx_inference,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Conv2d(
                3,
                4,
                kernel_size=3,
                padding=1,
            ),
            torch.nn.ReLU(),
            torch.nn.AdaptiveAvgPool2d(
                (1, 1)
            ),
            torch.nn.Flatten(),
            torch.nn.Linear(
                4,
                7,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)


@pytest.fixture(scope="module")
def exported_model(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[
    TinyClassifier,
    Path,
]:
    torch.manual_seed(42)

    model = TinyClassifier()

    directory = tmp_path_factory.mktemp(
        "onnx"
    )

    model_path = (
        directory / "model.onnx"
    )

    export_model_to_onnx(
        model=model,
        output_path=model_path,
        input_shape=(
            3,
            8,
            8,
        ),
        example_batch_size=2,
        max_batch_size=8,
    )

    return model, model_path


def test_export_creates_valid_onnx(
    exported_model: tuple[
        TinyClassifier,
        Path,
    ],
) -> None:
    _, model_path = exported_model

    assert model_path.exists()

    model = onnx.load(
        str(model_path)
    )

    onnx.checker.check_model(
        model
    )


def test_onnx_matches_pytorch(
    exported_model: tuple[
        TinyClassifier,
        Path,
    ],
) -> None:
    model, model_path = exported_model

    inputs = torch.randn(
        4,
        3,
        8,
        8,
    )

    result = compare_pytorch_onnx(
        model=model,
        model_path=model_path,
        inputs=inputs,
    )

    assert isinstance(
        result,
        OnnxParityResult,
    )

    assert result.batch_size == 4

    assert result.max_abs_diff < 1e-4


def test_onnx_supports_dynamic_batch(
    exported_model: tuple[
        TinyClassifier,
        Path,
    ],
) -> None:
    _, model_path = exported_model

    session = ort.InferenceSession(
        str(model_path),
        providers=[
            "CPUExecutionProvider",
        ],
    )

    input_metadata = (
        session.get_inputs()[0]
    )

    assert (
        input_metadata.name
        == INPUT_NAME
    )

    assert isinstance(
        input_metadata.shape[0],
        str,
    )

    batch_one = run_onnx_inference(
        model_path=model_path,
        inputs=torch.randn(
            1,
            3,
            8,
            8,
        ),
    )

    batch_four = run_onnx_inference(
        model_path=model_path,
        inputs=torch.randn(
            4,
            3,
            8,
            8,
        ),
    )

    assert batch_one.shape == (
        1,
        7,
    )

    assert batch_four.shape == (
        4,
        7,
    )


def test_export_rejects_invalid_batch_size(
    tmp_path: Path,
) -> None:
    model = TinyClassifier()

    with pytest.raises(
        ValueError,
        match="positive",
    ):
        export_model_to_onnx(
            model=model,
            output_path=(
                tmp_path / "model.onnx"
            ),
            input_shape=(
                3,
                8,
                8,
            ),
            example_batch_size=0,
        )