from collections import deque
from dataclasses import dataclass

import torch

from surgphase.annotations import PHASES
from surgphase.model import NUM_PHASES


@dataclass(frozen=True)
class TemporalPrediction:
    phase_index: int
    phase: str
    confidence: float
    window_size: int


class TemporalPhaseAggregator:
    def __init__(
        self,
        window_size: int = 5,
    ) -> None:
        if window_size <= 0:
            raise ValueError(
                f"Window size must be positive: {window_size}"
            )

        self.window_size = window_size

        self._logits: deque[torch.Tensor] = deque(
            maxlen=window_size
        )

    def __len__(self) -> int:
        return len(self._logits)

    def reset(self) -> None:
        self._logits.clear()

    def update(
        self,
        logits: torch.Tensor,
    ) -> TemporalPrediction:
        if logits.ndim == 2:
            if logits.shape[0] != 1:
                raise ValueError(
                    "Temporal aggregation expects "
                    "a single prediction at a time"
                )

            logits = logits.squeeze(0)

        if logits.ndim != 1:
            raise ValueError(
                "Logits must have shape [num_classes] "
                "or [1, num_classes]"
            )

        if logits.shape[0] != NUM_PHASES:
            raise ValueError(
                f"Expected {NUM_PHASES} classes, "
                f"got {logits.shape[0]}"
            )

        self._logits.append(
            logits.detach().cpu()
        )

        stacked = torch.stack(
            tuple(self._logits)
        )

        mean_logits = stacked.mean(
            dim=0
        )

        probabilities = torch.softmax(
            mean_logits,
            dim=0,
        )

        phase_index = int(
            probabilities.argmax().item()
        )

        confidence = float(
            probabilities[phase_index].item()
        )

        return TemporalPrediction(
            phase_index=phase_index,
            phase=PHASES[phase_index],
            confidence=confidence,
            window_size=len(self._logits),
        )


@torch.inference_mode()
def infer_clip(
    model: torch.nn.Module,
    frames: torch.Tensor,
    device: torch.device,
) -> TemporalPrediction:
    if frames.ndim != 4:
        raise ValueError(
            "Frames must have shape "
            "[time, channels, height, width]"
        )

    if frames.shape[0] == 0:
        raise ValueError(
            "Clip must contain at least one frame"
        )

    model.eval()

    frames = frames.to(
        device,
        non_blocking=True,
    )

    logits = model(frames)

    if logits.ndim != 2:
        raise ValueError(
            "Model output must have shape "
            "[time, num_classes]"
        )

    if logits.shape[1] != NUM_PHASES:
        raise ValueError(
            f"Expected {NUM_PHASES} classes, "
            f"got {logits.shape[1]}"
        )

    mean_logits = logits.mean(
        dim=0,
    )

    probabilities = torch.softmax(
        mean_logits,
        dim=0,
    )

    phase_index = int(
        probabilities.argmax().item()
    )

    return TemporalPrediction(
        phase_index=phase_index,
        phase=PHASES[phase_index],
        confidence=float(
            probabilities[phase_index].item()
        ),
        window_size=frames.shape[0],
    )