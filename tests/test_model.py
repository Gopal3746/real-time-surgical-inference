import numpy as np
import pytest
import torch

from surgphase.annotations import PHASES
from surgphase.device import get_device
from surgphase.model import (
    MODEL_NAME,
    NUM_PHASES,
    SurgicalPhaseClassifier,
)
from surgphase.preprocessing import preprocess_frame


@pytest.fixture(scope="module")
def model() -> SurgicalPhaseClassifier:
    classifier = SurgicalPhaseClassifier(
        pretrained=False,
    )

    classifier.eval()

    return classifier


def test_model_configuration() -> None:
    assert MODEL_NAME == "efficientnet-b0"
    assert NUM_PHASES == 7
    assert NUM_PHASES == len(PHASES)

def test_device_selection() -> None:
    device = get_device()

    assert device.type in {
        "cpu",
        "cuda",
        "mps",
    }

def test_model_output_shape(
    model: SurgicalPhaseClassifier,
) -> None:
    inputs = torch.randn(
        2,
        3,
        224,
        224,
    )

    with torch.inference_mode():
        outputs = model(inputs)

    assert outputs.shape == (
        2,
        NUM_PHASES,
    )


def test_model_outputs_are_finite(
    model: SurgicalPhaseClassifier,
) -> None:
    inputs = torch.randn(
        1,
        3,
        224,
        224,
    )

    with torch.inference_mode():
        outputs = model(inputs)

    assert torch.isfinite(outputs).all()


def test_preprocessed_frame_runs_through_model(
    model: SurgicalPhaseClassifier,
) -> None:
    frame = np.zeros(
        (480, 854, 3),
        dtype=np.uint8,
    )

    tensor = preprocess_frame(frame)

    batch = tensor.unsqueeze(0)

    assert batch.shape == (
        1,
        3,
        224,
        224,
    )

    with torch.inference_mode():
        logits = model(batch)

    assert logits.shape == (
        1,
        NUM_PHASES,
    )