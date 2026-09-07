from pathlib import Path

from torch.utils.data import Dataset

from surgphase.manifest import (
    ANNOTATION_FPS,
    ManifestRecord,
    read_manifest,
)
from surgphase.preprocessing import (
    FrameTransform,
    build_frame_transform,
    preprocess_frame,
)
from surgphase.video import read_frame

VALID_SPLITS = {
    "train",
    "val",
    "test",
}


class Cholec80Dataset(Dataset):
    def __init__(
        self,
        manifest_path: Path,
        split: str,
        sample_fps: float = 1.0,
        transform: FrameTransform | None = None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(
                f"Invalid split: {split}. "
                f"Expected one of {sorted(VALID_SPLITS)}"
            )

        if sample_fps <= 0:
            raise ValueError(
                f"Sample FPS must be positive: {sample_fps}"
            )

        if sample_fps > ANNOTATION_FPS:
            raise ValueError(
                "Sample FPS cannot exceed annotation FPS "
                f"({ANNOTATION_FPS})"
            )

        frame_step = max(
            1,
            round(
                ANNOTATION_FPS / sample_fps
            ),
        )

        records = read_manifest(manifest_path)

        self.records: list[ManifestRecord] = [
            record
            for record in records
            if (
                record.split == split
                and record.frame % frame_step == 0
            )
        ]

        if not self.records:
            raise ValueError(
                "No manifest records found for "
                f"split={split}, sample_fps={sample_fps}"
            )

        self.transform = (
            transform
            if transform is not None
            else build_frame_transform()
        )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(
        self,
        index: int,
    ) -> dict[str, object]:
        record = self.records[index]

        frame = read_frame(
            video_path=Path(record.video_path),
            frame_number=record.frame,
        )

        image = preprocess_frame(
            frame,
            transform=self.transform,
        )

        return {
            "image": image,
            "label": record.phase_index,
            "video_id": record.video_id,
            "frame": record.frame,
            "timestamp_seconds": record.timestamp_seconds,
        }