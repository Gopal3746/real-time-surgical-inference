from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    samples: int


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: torch.nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.train()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        optimizer.zero_grad(
            set_to_none=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        optimizer.step()

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1,
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Training loader produced no samples"
        )

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
    )


@torch.inference_mode()
def evaluate_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in loader:
        images = batch["image"].to(
            device,
            non_blocking=True,
        )

        labels = batch["label"].to(
            device,
            dtype=torch.long,
            non_blocking=True,
        )

        logits = model(images)

        loss = criterion(
            logits,
            labels,
        )

        batch_size = labels.size(0)

        total_loss += (
            loss.item() * batch_size
        )

        predictions = logits.argmax(
            dim=1,
        )

        total_correct += (
            predictions == labels
        ).sum().item()

        total_samples += batch_size

    if total_samples == 0:
        raise ValueError(
            "Validation loader produced no samples"
        )

    return EpochMetrics(
        loss=total_loss / total_samples,
        accuracy=total_correct / total_samples,
        samples=total_samples,
    )


def save_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    validation_metrics: EpochMetrics,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": (
                optimizer.state_dict()
            ),
            "validation_loss": (
                validation_metrics.loss
            ),
            "validation_accuracy": (
                validation_metrics.accuracy
            ),
        },
        checkpoint_path,
    )