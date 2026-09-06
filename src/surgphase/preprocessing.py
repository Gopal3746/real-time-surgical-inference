from collections.abc import Callable

import numpy as np
import torch
from monai.transforms import (
    Compose,
    EnsureChannelFirst,
    EnsureType,
    NormalizeIntensity,
    Resize,
    ScaleIntensityRange,
)
from numpy.typing import NDArray

DEFAULT_IMAGE_SIZE = (224, 224)

IMAGENET_MEAN = (
    0.485,
    0.456,
    0.406,
)

IMAGENET_STD = (
    0.229,
    0.224,
    0.225,
)

FrameTransform = Callable[
    [NDArray[np.uint8]],
    torch.Tensor,
]


def build_frame_transform(
    image_size: tuple[int, int] = DEFAULT_IMAGE_SIZE,
) -> Compose:
    height, width = image_size

    if height <= 0 or width <= 0:
        raise ValueError(
            f"Image dimensions must be positive: {image_size}"
        )

    return Compose(
        [
            EnsureChannelFirst(
                channel_dim=-1,
            ),
            ScaleIntensityRange(
                a_min=0.0,
                a_max=255.0,
                b_min=0.0,
                b_max=1.0,
                clip=True,
            ),
            Resize(
                spatial_size=image_size,
                mode="bilinear",
                anti_aliasing=True,
            ),
            NormalizeIntensity(
                subtrahend=IMAGENET_MEAN,
                divisor=IMAGENET_STD,
                channel_wise=True,
            ),
            EnsureType(
                data_type="tensor",
                dtype=torch.float32,
                track_meta=False,
            ),
        ]
    )


def validate_frame(
    frame: NDArray[np.uint8],
) -> None:
    if not isinstance(frame, np.ndarray):
        raise TypeError(
            "Frame must be a NumPy array"
        )

    if frame.ndim != 3:
        raise ValueError(
            f"Frame must have 3 dimensions, got shape {frame.shape}"
        )

    if frame.shape[2] != 3:
        raise ValueError(
            "Frame must contain exactly 3 RGB channels"
        )

    if frame.dtype != np.uint8:
        raise TypeError(
            f"Frame dtype must be uint8, got {frame.dtype}"
        )


def preprocess_frame(
    frame: NDArray[np.uint8],
    transform: FrameTransform | None = None,
) -> torch.Tensor:
    validate_frame(frame)

    if transform is None:
        transform = build_frame_transform()

    output = transform(frame)

    if not isinstance(output, torch.Tensor):
        raise TypeError(
            "Preprocessing pipeline did not return a tensor"
        )

    return output