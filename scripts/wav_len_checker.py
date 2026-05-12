#!/usr/bin/env python3
import argparse
import contextlib
import csv
import shutil
import wave
from pathlib import Path


def get_duration_seconds(wav_path: Path):
    with contextlib.closing(wave.open(str(wav_path), "r")) as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate), frames


def collect_long_files(input_dir: Path, min_seconds: float):
    long_files = []
    total_files = 0
    for wav_path in input_dir.rglob("*.wav"):
        total_files += 1
        try:
            duration, frames = get_duration_seconds(wav_path)
        except wave.Error:
            print(f"Warning: failed to read WAV header: {wav_path}")
            continue
        if duration >= min_seconds:
            long_files.append(
                {
                    "path": str(wav_path),
                    "filename": wav_path.name,
                    "duration_seconds": duration,
                    "num_frames": frames,
                }
            )
    return long_files, total_files


def write_csv(rows, output_csv: Path):
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["path", "filename", "duration_seconds", "num_frames"]
        )
        writer.writeheader()
        writer.writerows(rows)


def move_files(rows, move_dir: Path):
    move_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for row in rows:
        src = Path(row["path"])
        dst = move_dir / src.name
        if dst.exists():
            # Avoid accidental overwrite by suffixing numerically.
            stem, suffix = src.stem, src.suffix
            i = 1
            while (move_dir / f"{stem}_{i}{suffix}").exists():
                i += 1
            dst = move_dir / f"{stem}_{i}{suffix}"
        shutil.move(str(src), str(dst))
        moved += 1
    return moved


def parse_args():
    parser = argparse.ArgumentParser(
        description="Find or move WAV files longer than a threshold (e.g. 21.6s)."
    )
    parser.add_argument("--input-dir", required=True, help="Root directory to scan recursively.")
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=21.6,
        help="Threshold in seconds. Files >= threshold are selected.",
    )
    parser.add_argument(
        "--mode",
        choices=["list", "move"],
        default="list",
        help="list: report only, move: move matching files to --move-dir.",
    )
    parser.add_argument(
        "--move-dir",
        default=None,
        help="Destination directory used when --mode move.",
    )
    parser.add_argument(
        "--output-csv",
        default=None,
        help="Optional CSV output for matching files.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"--input-dir is not a directory: {input_dir}")

    rows, total_files = collect_long_files(input_dir, args.min_seconds)
    print(f"Total WAV files scanned: {total_files}")
    print(f"Files >= {args.min_seconds}s: {len(rows)}")

    if args.output_csv:
        write_csv(rows, Path(args.output_csv))
        print(f"Saved report: {args.output_csv}")

    if args.mode == "move":
        if not args.move_dir:
            raise ValueError("--move-dir is required when --mode move")
        moved = move_files(rows, Path(args.move_dir))
        print(f"Moved files: {moved}")
    else:
        for row in rows[:20]:
            print(f"{row['path']}\t{row['duration_seconds']:.2f}s")
        if len(rows) > 20:
            print(f"... and {len(rows) - 20} more")


if __name__ == "__main__":
    main()
