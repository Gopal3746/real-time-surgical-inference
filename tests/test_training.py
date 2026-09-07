from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surgphase.training import (
    EpochMetrics,
    evaluate_one_epoch,
    save_checkpoint,
    train_one_epoch,
)


class TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()

        self.network = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(
                3 * 4 * 4,
                2,
            ),
        )

    def forward(
        self,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(inputs)


def create_loader() -> DataLoader:
    samples = [
        {
            "image": torch.zeros(
                3,
                4,
                4,
            ),
            "label": 0,
        },
        {
            "image": torch.ones(
                3,
                4,
                4,
            ),
            "label": 1,
        },
        {
            "image": torch.full(
                (3, 4, 4),
                0.25,
            ),
            "label": 0,
        },
        {
            "image": torch.full(
                (3, 4, 4),
                0.75,
            ),
            "label": 1,
        },
    ]

    return DataLoader(
        samples,
        batch_size=2,
        shuffle=False,
    )


def test_train_one_epoch_updates_model() -> None:
    torch.manual_seed(42)

    model = TinyClassifier()

    loader = create_loader()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    criterion = torch.nn.CrossEntropyLoss()

    before = {
        name: parameter.detach().clone()
        for name, parameter
        in model.named_parameters()
    }

    metrics = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert metrics.samples == 4
    assert metrics.loss > 0.0

    assert 0.0 <= metrics.accuracy <= 1.0

    changed = any(
        not torch.equal(
            before[name],
            parameter.detach(),
        )
        for name, parameter
        in model.named_parameters()
    )

    assert changed


def test_evaluate_one_epoch() -> None:
    torch.manual_seed(42)

    model = TinyClassifier()

    loader = create_loader()

    criterion = torch.nn.CrossEntropyLoss()

    metrics = evaluate_one_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert metrics.samples == 4
    assert metrics.loss > 0.0

    assert 0.0 <= metrics.accuracy <= 1.0

    assert model.training is False


def test_save_checkpoint(
    tmp_path: Path,
) -> None:
    model = TinyClassifier()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    metrics = EpochMetrics(
        loss=0.75,
        accuracy=0.5,
        samples=4,
    )

    checkpoint_path = (
        tmp_path / "best_model.pt"
    )

    save_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_metrics=metrics,
    )

    assert checkpoint_path.exists()

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    assert checkpoint["epoch"] == 3

    assert (
        checkpoint["validation_loss"]
        == 0.75
    )

    assert (
        checkpoint["validation_accuracy"]
        == 0.5
    )

    assert "model_state_dict" in checkpoint

    assert (
        "optimizer_state_dict"
        in checkpoint
    )


def test_epoch_metrics() -> None:
    metrics = EpochMetrics(
        loss=1.25,
        accuracy=0.75,
        samples=100,
    )

    assert metrics.loss == 1.25
    assert metrics.accuracy == 0.75
    assert metrics.samples == 100