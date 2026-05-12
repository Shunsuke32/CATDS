#!/usr/bin/env python3
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from tqdm import tqdm


def convert_m4a_to_wav(input_file: Path, output_dir: Path = None):
    if output_dir is None:
        output_file = input_file.with_suffix(".wav")
    else:
        output_file = output_dir / (input_file.stem + ".wav")
        output_dir.mkdir(parents=True, exist_ok=True)

    ffmpeg_cmd = [
        "ffmpeg",
        "-i",
        str(input_file),
        "-acodec",
        "pcm_s16le",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-y",
        str(output_file),
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, capture_output=True)
        return True, input_file
    except subprocess.CalledProcessError:
        return False, input_file


def parse_args():
    parser = argparse.ArgumentParser(description="Convert M4A files to 16kHz mono WAV using ffmpeg.")
    parser.add_argument("--input-dir", required=True, help="Directory containing .m4a files")
    parser.add_argument("--output-dir", default=None, help="Optional destination directory for .wav files")
    parser.add_argument("--workers", type=int, default=8, help="Parallel workers")
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise ValueError(f"input directory does not exist: {input_dir}")

    output_dir = Path(args.output_dir) if args.output_dir else None
    m4a_files = list(input_dir.rglob("*.m4a"))
    if not m4a_files:
        print("No M4A files found.")
        return

    print(f"Found {len(m4a_files)} M4A files. Converting...")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = (executor.submit(convert_m4a_to_wav, p, output_dir) for p in m4a_files)
        for future in tqdm(futures, total=len(m4a_files), desc="Converting"):
            results.append(future.result())

    failed = [str(path) for ok, path in results if not ok]
    print(f"Done. success={len(results) - len(failed)}, failed={len(failed)}")
    if failed:
        print("Failed files:")
        for p in failed:
            print(p)


if __name__ == "__main__":
    main()
