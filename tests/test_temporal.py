import pytest
import torch

from surgphase.annotations import PHASES
from surgphase.temporal import (
    TemporalPhaseAggregator,
    infer_clip,
)


class ControlledClassifier(
    torch.nn.Module
):
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        logits = torch.zeros(
            (
                inputs.shape[0],
                7,
            ),
            device=inputs.device,
        )

        predicted_classes = (
            inputs[:, 0, 0, 0]
            .to(torch.long)
        )

        logits.scatter_(
            1,
            predicted_classes.unsqueeze(1),
            10.0,
        )

        return logits


def make_logits(
    phase_index: int,
) -> torch.Tensor:
    logits = torch.zeros(7)

    logits[phase_index] = 10.0

    return logits


def test_temporal_aggregator_prediction() -> None:
    aggregator = TemporalPhaseAggregator(
        window_size=3
    )

    prediction = aggregator.update(
        make_logits(2)
    )

    assert prediction.phase_index == 2
    assert prediction.phase == PHASES[2]
    assert prediction.window_size == 1

    assert 0.0 <= prediction.confidence <= 1.0


def test_temporal_aggregator_smooths_noise() -> None:
    aggregator = TemporalPhaseAggregator(
        window_size=3
    )

    aggregator.update(
        make_logits(2)
    )

    aggregator.update(
        make_logits(2)
    )

    prediction = aggregator.update(
        make_logits(1)
    )

    assert prediction.phase_index == 2


def test_temporal_window_rolls_forward() -> None:
    aggregator = TemporalPhaseAggregator(
        window_size=2
    )

    aggregator.update(
        make_logits(0)
    )

    aggregator.update(
        make_logits(1)
    )

    prediction = aggregator.update(
        make_logits(1)
    )

    assert len(aggregator) == 2
    assert prediction.phase_index == 1


def test_temporal_reset() -> None:
    aggregator = TemporalPhaseAggregator(
        window_size=3
    )

    aggregator.update(
        make_logits(4)
    )

    assert len(aggregator) == 1

    aggregator.reset()

    assert len(aggregator) == 0


def test_temporal_rejects_invalid_window() -> None:
    with pytest.raises(
        ValueError,
        match="positive",
    ):
        TemporalPhaseAggregator(
            window_size=0
        )


def test_infer_clip() -> None:
    model = ControlledClassifier()

    frames = torch.zeros(
        3,
        3,
        4,
        4,
    )

    frames[0, 0, 0, 0] = 3
    frames[1, 0, 0, 0] = 3
    frames[2, 0, 0, 0] = 1

    prediction = infer_clip(
        model=model,
        frames=frames,
        device=torch.device("cpu"),
    )

    assert prediction.phase_index == 3
    assert prediction.phase == PHASES[3]
    assert prediction.window_size == 3


def test_infer_clip_rejects_empty_clip() -> None:
    model = ControlledClassifier()

    frames = torch.empty(
        0,
        3,
        224,
        224,
    )

    with pytest.raises(
        ValueError,
        match="at least one",
    ):
        infer_clip(
            model=model,
            frames=frames,
            device=torch.device("cpu"),
        )