#!/usr/bin/env python3
import argparse
from pathlib import Path

import librosa
import soundfile as sf


def resample_wav_files(directory_path, target_sr=16000):
    directory = Path(directory_path)
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"Directory not found: {directory}")

    processed = 0
    skipped = 0
    for wav_file in directory.rglob("*.wav"):
        try:
            audio_data, original_sr = librosa.load(wav_file, sr=None, mono=True)
            if original_sr == target_sr:
                skipped += 1
                continue

            resampled_audio = librosa.resample(
                audio_data, orig_sr=original_sr, target_sr=target_sr
            )
            sf.write(wav_file, resampled_audio, target_sr)
            processed += 1
            print(f"Converted: {wav_file} ({original_sr} -> {target_sr})")
        except Exception as e:
            print(f"Error processing {wav_file}: {e}")

    print(f"Finished. converted={processed}, skipped={skipped}")


def parse_args():
    parser = argparse.ArgumentParser(description="Resample WAV files recursively.")
    parser.add_argument("--input-dir", required=True, help="Directory containing WAV files.")
    parser.add_argument("--target-sr", type=int, default=16000, help="Target sample rate.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    resample_wav_files(args.input_dir, args.target_sr)
