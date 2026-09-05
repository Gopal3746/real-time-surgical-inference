from dataclasses import dataclass
from pathlib import Path

PHASES = (
    "Preparation",
    "CalotTriangleDissection",
    "ClippingCutting",
    "GallbladderDissection",
    "GallbladderRetraction",
    "CleaningCoagulation",
    "GallbladderPackaging",
)

PHASE_TO_INDEX = {
    phase: index
    for index, phase in enumerate(PHASES)
}


@dataclass(frozen=True)
class PhaseAnnotation:
    frame: int
    phase: str
    phase_index: int


def parse_phase_annotation(
    annotation_path: Path,
) -> list[PhaseAnnotation]:
    if not annotation_path.exists():
        raise FileNotFoundError(
            f"Annotation file does not exist: {annotation_path}"
        )

    annotations: list[PhaseAnnotation] = []

    with annotation_path.open(
        "r",
        encoding="utf-8",
    ) as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()

            if not line:
                continue

            parts = line.split()

            if parts[0].lower() == "frame":
                continue

            if len(parts) != 2:
                raise ValueError(
                    f"Invalid annotation at line {line_number}: {line}"
                )

            frame_text, phase = parts

            try:
                frame = int(frame_text)
            except ValueError as error:
                raise ValueError(
                    f"Invalid frame number at line {line_number}: "
                    f"{frame_text}"
                ) from error

            if phase not in PHASE_TO_INDEX:
                raise ValueError(
                    f"Unknown phase at line {line_number}: {phase}"
                )

            annotations.append(
                PhaseAnnotation(
                    frame=frame,
                    phase=phase,
                    phase_index=PHASE_TO_INDEX[phase],
                )
            )

    return annotations