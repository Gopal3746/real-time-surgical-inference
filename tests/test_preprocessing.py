import numpy as np
import pytest
import torch

from surgphase.preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    build_frame_transform,
    preprocess_frame,
)


def test_preprocess_frame_shape() -> None:
    frame = np.zeros(
        (48, 64, 3),
        dtype=np.uint8,
    )

    output = preprocess_frame(frame)

    assert output.shape == (
        3,
        224,
        224,
    )


def test_preprocess_frame_dtype() -> None:
    frame = np.zeros(
        (48, 64, 3),
        dtype=np.uint8,
    )

    output = preprocess_frame(frame)

    assert output.dtype == torch.float32
    assert type(output) is torch.Tensor


def test_preprocess_frame_normalization() -> None:
    frame = np.zeros(
        (32, 32, 3),
        dtype=np.uint8,
    )

    frame[..., 0] = 255
    frame[..., 1] = 128
    frame[..., 2] = 0

    output = preprocess_frame(frame)

    expected_red = (
        1.0 - IMAGENET_MEAN[0]
    ) / IMAGENET_STD[0]

    expected_green = (
        (128.0 / 255.0) - IMAGENET_MEAN[1]
    ) / IMAGENET_STD[1]

    expected_blue = (
        0.0 - IMAGENET_MEAN[2]
    ) / IMAGENET_STD[2]

    assert output[0, 0, 0].item() == pytest.approx(
        expected_red,
        abs=1e-4,
    )

    assert output[1, 0, 0].item() == pytest.approx(
        expected_green,
        abs=1e-4,
    )

    assert output[2, 0, 0].item() == pytest.approx(
        expected_blue,
        abs=1e-4,
    )


def test_preprocess_frame_is_deterministic() -> None:
    rng = np.random.default_rng(42)

    frame = rng.integers(
        low=0,
        high=256,
        size=(48, 64, 3),
        dtype=np.uint8,
    )

    transform = build_frame_transform()

    first = preprocess_frame(
        frame,
        transform=transform,
    )

    second = preprocess_frame(
        frame,
        transform=transform,
    )

    assert torch.equal(
        first,
        second,
    )


def test_preprocess_rejects_grayscale_frame() -> None:
    frame = np.zeros(
        (48, 64),
        dtype=np.uint8,
    )

    with pytest.raises(
        ValueError,
        match="3 dimensions",
    ):
        preprocess_frame(frame)


def test_preprocess_rejects_non_uint8_frame() -> None:
    frame = np.zeros(
        (48, 64, 3),
        dtype=np.float32,
    )

    with pytest.raises(
        TypeError,
        match="uint8",
    ):
        preprocess_frame(frame)


def test_transform_rejects_invalid_image_size() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        build_frame_transform(
            image_size=(0, 224),
        )