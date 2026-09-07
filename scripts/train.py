import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from surgphase.dataset import Cholec80Dataset
from surgphase.device import get_device
from surgphase.model import SurgicalPhaseClassifier
from surgphase.training import (
    evaluate_one_epoch,
    save_checkpoint,
    train_one_epoch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train the Cholec80 surgical "
            "phase classifier."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the Cholec80 manifest CSV.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
    )

    parser.add_argument(
        "--sample-fps",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(
            "models/best_model.pt"
        ),
    )

    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.epochs <= 0:
        raise ValueError(
            "Epochs must be positive"
        )

    if args.batch_size <= 0:
        raise ValueError(
            "Batch size must be positive"
        )

    device = get_device()

    print(
        f"Training device: {device}"
    )

    train_dataset = Cholec80Dataset(
        manifest_path=args.manifest,
        split="train",
        sample_fps=args.sample_fps,
    )

    val_dataset = Cholec80Dataset(
        manifest_path=args.manifest,
        split="val",
        sample_fps=args.sample_fps,
    )

    pin_memory = (
        device.type == "cuda"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )

    print(
        f"Training samples:   "
        f"{len(train_dataset):,}"
    )

    print(
        f"Validation samples: "
        f"{len(val_dataset):,}"
    )

    model = SurgicalPhaseClassifier(
        pretrained=not args.no_pretrained,
    ).to(device)

    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    best_validation_loss = float("inf")

    for epoch in range(
        1,
        args.epochs + 1,
    ):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        validation_metrics = (
            evaluate_one_epoch(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
            )
        )

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"train_loss={train_metrics.loss:.4f} | "
            f"train_acc={train_metrics.accuracy:.4f} | "
            f"val_loss={validation_metrics.loss:.4f} | "
            f"val_acc={validation_metrics.accuracy:.4f}"
        )

        if (
            validation_metrics.loss
            < best_validation_loss
        ):
            best_validation_loss = (
                validation_metrics.loss
            )

            save_checkpoint(
                checkpoint_path=args.checkpoint,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                validation_metrics=(
                    validation_metrics
                ),
            )

            print(
                "Saved best checkpoint to "
                f"{args.checkpoint}"
            )


if __name__ == "__main__":
    main()