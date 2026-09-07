from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader

from surgphase.evaluation import (
    evaluate_classifier,
)
from surgphase.training import (
    EpochMetrics,
    load_checkpoint,
    save_checkpoint,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.linear = torch.nn.Linear(
            4,
            7,
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(inputs)


class PredictionClassifier(
    torch.nn.Module
):
    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        predicted_classes = (
            inputs[:, 0]
            .to(torch.long)
        )

        logits = torch.full(
            (
                inputs.shape[0],
                7,
            ),
            -10.0,
            device=inputs.device,
        )

        logits.scatter_(
            1,
            predicted_classes.unsqueeze(1),
            10.0,
        )

        return logits


def test_load_checkpoint_restores_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    model = TinyClassifier()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    original = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    checkpoint_path = (
        tmp_path / "model.pt"
    )

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=5,
        validation_metrics=EpochMetrics(
            loss=0.4,
            accuracy=0.8,
            samples=10,
        ),
    )

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()

    metadata = load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
    )

    assert metadata.epoch == 5

    assert metadata.validation_loss == pytest.approx(
        0.4
    )

    assert (
        metadata.validation_accuracy
        == pytest.approx(0.8)
    )

    for name, parameter in (
        model.named_parameters()
    ):
        assert torch.equal(
            parameter,
            original[name],
        )


def test_load_checkpoint_rejects_missing_file(
    tmp_path: Path,
) -> None:
    model = TinyClassifier()

    with pytest.raises(
        FileNotFoundError,
        match="does not exist",
    ):
        load_checkpoint(
            checkpoint_path=(
                tmp_path / "missing.pt"
            ),
            model=model,
        )


def test_evaluate_classifier_metrics() -> None:
    samples = [
        {
            "image": torch.tensor(
                [0.0, 0.0, 0.0, 0.0]
            ),
            "label": 0,
        },
        {
            "image": torch.tensor(
                [1.0, 0.0, 0.0, 0.0]
            ),
            "label": 1,
        },
        {
            "image": torch.tensor(
                [0.0, 0.0, 0.0, 0.0]
            ),
            "label": 1,
        },
        {
            "image": torch.tensor(
                [2.0, 0.0, 0.0, 0.0]
            ),
            "label": 2,
        },
    ]

    loader = DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
    )

    model = PredictionClassifier()

    metrics = evaluate_classifier(
        model=model,
        loader=loader,
        device=torch.device("cpu"),
    )

    assert metrics.samples == 4

    assert metrics.accuracy == pytest.approx(
        0.75
    )

    assert len(metrics.per_phase) == 7

    assert len(
        metrics.confusion_matrix
    ) == 7

    assert metrics.per_phase[0].precision == (
        pytest.approx(0.5)
    )

    assert metrics.per_phase[0].recall == (
        pytest.approx(1.0)
    )

    assert metrics.per_phase[1].precision == (
        pytest.approx(1.0)
    )

    assert metrics.per_phase[1].recall == (
        pytest.approx(0.5)
    )


def test_evaluate_rejects_empty_loader() -> None:
    loader = DataLoader(
        [],
        batch_size=1,
    )

    model = PredictionClassifier()

    with pytest.raises(
        ValueError,
        match="no samples",
    ):
        evaluate_classifier(
            model=model,
            loader=loader,
            device=torch.device("cpu"),
        )