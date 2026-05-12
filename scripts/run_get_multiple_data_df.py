#!/usr/bin/env python3
import argparse
import csv
from pathlib import Path

from extract_embeddings import get_multiple_data_df


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create grouped donor clips CSV for CVTDS/CATDS scoring."
    )
    parser.add_argument(
        "--wav-dir",
        required=True,
        help="Directory containing donor .wav files (ignored if --manifest-path is set).",
    )
    parser.add_argument(
        "--num-hours",
        type=float,
        required=True,
        help="Hours per group. Example: 0.006 (about 21.6s).",
    )
    parser.add_argument(
        "--num-sets",
        type=int,
        required=True,
        help="Number of groups to sample.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path. Example: /work/result/gujarati_21_20000_full.csv",
    )
    parser.add_argument(
        "--manifest-path",
        default=None,
        help="Optional manifest TSV to sample from instead of listing --wav-dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.manifest_path is None:
        wav_dir = Path(args.wav_dir)
        if not wav_dir.exists() or not wav_dir.is_dir():
            raise ValueError(f"--wav-dir does not exist or is not a directory: {wav_dir}")
        file_count = len([p for p in wav_dir.iterdir() if p.is_file()])
        print(f"WAV directory: {wav_dir}")
        print(f"Found files: {file_count}")
    else:
        manifest = Path(args.manifest_path)
        if not manifest.exists():
            raise ValueError(f"--manifest-path does not exist: {manifest}")
        print(f"Sampling from manifest: {manifest}")

    print(
        f"Generating groups with num_hours={args.num_hours}, "
        f"num_sets={args.num_sets} ..."
    )
    grouped = get_multiple_data_df(
        args.wav_dir,
        args.num_hours,
        args.num_sets,
        args.manifest_path,
    )

    if grouped is None or not isinstance(grouped, list):
        raise RuntimeError("get_multiple_data_df did not return a list.")
    if len(grouped) == 0:
        raise RuntimeError("No groups were created. Check num_hours/num_sets/input data.")

    with output_path.open("w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["index", "data"])
        for i, item in enumerate(grouped):
            writer.writerow([i, item])

    print(f"Saved grouped clips CSV: {output_path}")
    print(f"Groups saved: {len(grouped)}")


if __name__ == "__main__":
    main()
