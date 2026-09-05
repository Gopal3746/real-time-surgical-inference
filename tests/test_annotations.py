from pathlib import Path

import pytest

from surgphase.annotations import (
    PHASES,
    PhaseAnnotation,
    parse_phase_annotation,
)


def test_parse_phase_annotation(
    tmp_path: Path,
) -> None:
    annotation_file = tmp_path / "video01-phase.txt"

    annotation_file.write_text(
        "Frame\tPhase\n"
        "0\tPreparation\n"
        "25\tPreparation\n"
        "50\tCalotTriangleDissection\n",
        encoding="utf-8",
    )

    annotations = parse_phase_annotation(annotation_file)

    assert annotations == [
        PhaseAnnotation(
            frame=0,
            phase="Preparation",
            phase_index=0,
        ),
        PhaseAnnotation(
            frame=25,
            phase="Preparation",
            phase_index=0,
        ),
        PhaseAnnotation(
            frame=50,
            phase="CalotTriangleDissection",
            phase_index=1,
        ),
    ]


def test_parse_rejects_unknown_phase(
    tmp_path: Path,
) -> None:
    annotation_file = tmp_path / "video01-phase.txt"

    annotation_file.write_text(
        "Frame\tPhase\n"
        "0\tUnknownPhase\n",
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Unknown phase",
    ):
        parse_phase_annotation(annotation_file)


def test_cholec80_has_seven_phases() -> None:
    assert len(PHASES) == 7