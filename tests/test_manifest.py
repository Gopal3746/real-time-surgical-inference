import csv
from pathlib import Path

import pytest

from surgphase.manifest import (
    ManifestRecord,
    get_split,
    iter_manifest_records,
    parse_video_number,
    read_manifest,
    write_manifest,
)


def test_get_split() -> None:
    assert get_split(1) == "train"
    assert get_split(32) == "train"

    assert get_split(33) == "val"
    assert get_split(40) == "val"

    assert get_split(41) == "test"
    assert get_split(80) == "test"


def test_get_split_rejects_invalid_video() -> None:
    with pytest.raises(
        ValueError,
        match="between 1 and 80",
    ):
        get_split(81)


def test_parse_video_number() -> None:
    path = Path("video07-phase.txt")

    assert parse_video_number(path) == 7


def test_iter_manifest_records(
    tmp_path: Path,
) -> None:
    annotation_dir = tmp_path / "phase_annotations"
    video_dir = tmp_path / "videos"

    annotation_dir.mkdir()
    video_dir.mkdir()

    annotation_file = (
        annotation_dir / "video01-phase.txt"
    )

    annotation_file.write_text(
        "Frame\tPhase\n"
        "0\tPreparation\n"
        "25\tCalotTriangleDissection\n",
        encoding="utf-8",
    )

    video_file = video_dir / "video01.mp4"
    video_file.touch()

    records = list(
        iter_manifest_records(
            annotation_dir=annotation_dir,
            video_dir=video_dir,
        )
    )

    assert len(records) == 2

    assert records[0].video_id == "video01"
    assert records[0].split == "train"
    assert records[0].frame == 0
    assert records[0].timestamp_seconds == 0.0

    assert records[1].frame == 25
    assert records[1].timestamp_seconds == 1.0
    assert records[1].phase_index == 1


def test_write_manifest(
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "manifest.csv"

    records = [
        ManifestRecord(
            video_id="video01",
            video_number=1,
            split="train",
            frame=25,
            timestamp_seconds=1.0,
            phase="Preparation",
            phase_index=0,
            video_path="video01.mp4",
        )
    ]

    count = write_manifest(
        records=records,
        output_path=output_path,
    )

    assert count == 1
    assert output_path.exists()

    with output_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 1
    assert rows[0]["video_id"] == "video01"
    assert rows[0]["split"] == "train"
    assert rows[0]["phase"] == "Preparation"

def test_read_manifest(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.csv"

    records = [
        ManifestRecord(
            video_id="video01",
            video_number=1,
            split="train",
            frame=25,
            timestamp_seconds=1.0,
            phase="Preparation",
            phase_index=0,
            video_path="/tmp/video01.mp4",
        )
    ]

    write_manifest(
        records=records,
        output_path=manifest_path,
    )

    loaded = read_manifest(manifest_path)

    assert loaded == records