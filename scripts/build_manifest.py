import argparse
from pathlib import Path

from surgphase.manifest import (
    iter_manifest_records,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the Cholec80 phase annotation manifest."
    )

    parser.add_argument(
        "--annotations",
        type=Path,
        required=True,
        help="Directory containing Cholec80 phase annotation files.",
    )

    parser.add_argument(
        "--videos",
        type=Path,
        required=True,
        help="Directory containing Cholec80 MP4 videos.",
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/manifest.csv"),
        help="Output CSV manifest path.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    records = iter_manifest_records(
        annotation_dir=args.annotations,
        video_dir=args.videos,
    )

    count = write_manifest(
        records=records,
        output_path=args.output,
    )

    print(
        f"Wrote {count:,} manifest records to "
        f"{args.output}"
    )


if __name__ == "__main__":
    main()