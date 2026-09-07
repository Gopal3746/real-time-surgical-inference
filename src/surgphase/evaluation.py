from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader

from surgphase.annotations import PHASES
from surgphase.model import NUM_PHASES


@dataclass(frozen=True)
class PhaseMetrics:
    phase: str
    precision: float
    recall: float
    f1: float
    support: int


@dataclass(frozen=True)
class EvaluationMetrics:
    accuracy: float
    macro_precision: float
    macro_recall: float
    macro_f1: float
    samples: int
    per_phase: tuple[PhaseMetrics, ...]
    confusion_matrix: tuple[
        tuple[int, ...],
        ...,
    ]


@torch.inference_mode()
def evaluate_classifier(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> EvaluationMetrics:
    model.eval()

    targets: list[int] = []
    predictions: list[int] = []

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

        if logits.ndim != 2:
            raise ValueError(
                "Model output must have shape "
                "[batch_size, num_classes]"
            )

        if logits.shape[1] != NUM_PHASES:
            raise ValueError(
                f"Expected {NUM_PHASES} output classes, "
                f"got {logits.shape[1]}"
            )

        predicted = logits.argmax(
            dim=1,
        )

        targets.extend(
            labels.cpu().tolist()
        )

        predictions.extend(
            predicted.cpu().tolist()
        )

    if not targets:
        raise ValueError(
            "Evaluation loader produced no samples"
        )

    labels = list(
        range(NUM_PHASES)
    )

    accuracy = accuracy_score(
        targets,
        predictions,
    )

    (
        precision,
        recall,
        f1,
        support,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )

    (
        macro_precision,
        macro_recall,
        macro_f1,
        _,
    ) = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        average="macro",
        zero_division=0,
    )

    matrix = confusion_matrix(
        targets,
        predictions,
        labels=labels,
    )

    per_phase = tuple(
        PhaseMetrics(
            phase=phase,
            precision=float(
                precision[index]
            ),
            recall=float(
                recall[index]
            ),
            f1=float(
                f1[index]
            ),
            support=int(
                support[index]
            ),
        )
        for index, phase in enumerate(PHASES)
    )

    matrix_tuple = tuple(
        tuple(
            int(value)
            for value in row
        )
        for row in np.asarray(matrix)
    )

    return EvaluationMetrics(
        accuracy=float(accuracy),
        macro_precision=float(
            macro_precision
        ),
        macro_recall=float(
            macro_recall
        ),
        macro_f1=float(
            macro_f1
        ),
        samples=len(targets),
        per_phase=per_phase,
        confusion_matrix=matrix_tuple,
    )