import argparse
from pathlib import Path

from torch.utils.data import DataLoader

from surgphase.dataset import Cholec80Dataset
from surgphase.device import get_device
from surgphase.evaluation import evaluate_classifier
from surgphase.model import SurgicalPhaseClassifier
from surgphase.training import load_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a trained Cholec80 "
            "surgical phase classifier."
        )
    )

    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--split",
        choices=(
            "train",
            "val",
            "test",
        ),
        default="test",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
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

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    device = get_device()

    dataset = Cholec80Dataset(
        manifest_path=args.manifest,
        split=args.split,
        sample_fps=args.sample_fps,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = SurgicalPhaseClassifier(
        pretrained=False,
    )

    metadata = load_checkpoint(
        checkpoint_path=args.checkpoint,
        model=model,
    )

    model = model.to(device)

    print(f"Device: {device}")
    print(
        f"Checkpoint epoch: "
        f"{metadata.epoch}"
    )
    print(
        f"Checkpoint val accuracy: "
        f"{metadata.validation_accuracy:.4f}"
    )

    metrics = evaluate_classifier(
        model=model,
        loader=loader,
        device=device,
    )

    print()
    print(
        f"Evaluation samples: "
        f"{metrics.samples:,}"
    )
    print(
        f"Accuracy:        "
        f"{metrics.accuracy:.4f}"
    )
    print(
        f"Macro precision: "
        f"{metrics.macro_precision:.4f}"
    )
    print(
        f"Macro recall:    "
        f"{metrics.macro_recall:.4f}"
    )
    print(
        f"Macro F1:        "
        f"{metrics.macro_f1:.4f}"
    )

    print()
    print("Per-phase metrics")
    print("-----------------")

    for phase in metrics.per_phase:
        print(
            f"{phase.phase:<28} "
            f"P={phase.precision:.3f} "
            f"R={phase.recall:.3f} "
            f"F1={phase.f1:.3f} "
            f"n={phase.support}"
        )

    print()
    print("Confusion matrix")

    for row in metrics.confusion_matrix:
        print(
            " ".join(
                f"{value:6d}"
                for value in row
            )
        )


if __name__ == "__main__":
    main()