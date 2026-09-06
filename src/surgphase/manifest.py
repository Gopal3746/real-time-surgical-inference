import csv
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from pathlib import Path

from surgphase.annotations import parse_phase_annotation

ANNOTATION_PATTERN = re.compile(
    r"video(?P<video_number>\d{2})-phase\.txt$"
)

ANNOTATION_FPS = 25.0


@dataclass(frozen=True)
class ManifestRecord:
    video_id: str
    video_number: int
    split: str
    frame: int
    timestamp_seconds: float
    phase: str
    phase_index: int
    video_path: str


def get_split(video_number: int) -> str:
    if 1 <= video_number <= 32:
        return "train"

    if 33 <= video_number <= 40:
        return "val"

    if 41 <= video_number <= 80:
        return "test"

    raise ValueError(
        f"Video number must be between 1 and 80: {video_number}"
    )


def parse_video_number(annotation_path: Path) -> int:
    match = ANNOTATION_PATTERN.fullmatch(annotation_path.name)

    if match is None:
        raise ValueError(
            f"Invalid annotation filename: {annotation_path.name}"
        )

    return int(match.group("video_number"))


def iter_manifest_records(
    annotation_dir: Path,
    video_dir: Path,
) -> Iterator[ManifestRecord]:
    if not annotation_dir.is_dir():
        raise NotADirectoryError(
            f"Annotation directory does not exist: {annotation_dir}"
        )

    if not video_dir.is_dir():
        raise NotADirectoryError(
            f"Video directory does not exist: {video_dir}"
        )

    annotation_files = sorted(
        annotation_dir.glob("video*-phase.txt")
    )

    if not annotation_files:
        raise FileNotFoundError(
            f"No phase annotation files found in: {annotation_dir}"
        )

    for annotation_path in annotation_files:
        video_number = parse_video_number(annotation_path)

        video_id = f"video{video_number:02d}"
        video_path = video_dir / f"{video_id}.mp4"

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Missing video for {annotation_path.name}: {video_path}"
            )

        split = get_split(video_number)

        annotations = parse_phase_annotation(annotation_path)

        for annotation in annotations:
            yield ManifestRecord(
                video_id=video_id,
                video_number=video_number,
                split=split,
                frame=annotation.frame,
                timestamp_seconds=annotation.frame / ANNOTATION_FPS,
                phase=annotation.phase,
                phase_index=annotation.phase_index,
                video_path=str(video_path),
            )


def write_manifest(
    records: Iterable[ManifestRecord],
    output_path: Path,
) -> int:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "video_id",
        "video_number",
        "split",
        "frame",
        "timestamp_seconds",
        "phase",
        "phase_index",
        "video_path",
    ]

    count = 0

    with output_path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for record in records:
            writer.writerow(
                {
                    "video_id": record.video_id,
                    "video_number": record.video_number,
                    "split": record.split,
                    "frame": record.frame,
                    "timestamp_seconds": (
                        record.timestamp_seconds
                    ),
                    "phase": record.phase,
                    "phase_index": record.phase_index,
                    "video_path": record.video_path,
                }
            )

            count += 1

    return count