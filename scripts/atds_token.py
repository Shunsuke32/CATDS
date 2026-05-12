#!/usr/bin/env python3
import argparse
import re
from pathlib import Path

import fairseq
import joblib
import numpy as np
import pandas as pd
import sentencepiece as spm
import torch
import torchaudio
import torch.nn.functional as F
from scipy.spatial import distance
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute raw CVTDS/CATDS score per donor group."
    )
    parser.add_argument("--groups-csv", required=True, help="Grouped clips CSV from run_get_multiple_data_df.py")
    parser.add_argument("--donor-wav-dir", required=True, help="Root directory containing donor wav files")
    parser.add_argument("--target-clustered-parquet", required=True, help="Target clustered parquet (wav_file, cluster_id)")
    parser.add_argument("--kmeans-model", required=True, help="Target k-means model .joblib path")
    parser.add_argument("--checkpoint-path", required=True, help="XLSR model checkpoint for embeddings")
    parser.add_argument("--target-lang", required=True, help="Target language name label")
    parser.add_argument("--donor-lang", required=True, help="Donor language name label")
    parser.add_argument("--output-atds-csv", required=True, help="Output CSV for raw group scores")
    parser.add_argument("--output-piece-counts-csv", required=True, help="Output CSV for per-group token sums")
    parser.add_argument("--tmp-dir", default="/work/tmp", help="Temporary directory for sentencepiece files")
    parser.add_argument("--spm-model", default=None, help="Optional existing sentencepiece model (.model)")
    parser.add_argument("--spm-prefix", default="10k_piece", help="Output prefix (without extension) when training sentencepiece")
    parser.add_argument("--spm-vocab-size", type=int, default=10001, help="SentencePiece vocab size")
    parser.add_argument("--spm-ident-norm", action="store_true", help="Use identity normalization in SentencePiece")
    return parser.parse_args()


def parse_group_csv(groups_csv: Path):
    df = pd.read_csv(groups_csv)
    groups = {}
    for i, row in df.iterrows():
        group_id = int(row["index"]) if "index" in row else i
        text = str(row["data"])
        wavs = []
        for line in text.splitlines():
            line = line.strip()
            if not line or line.startswith("path"):
                continue
            tokens = line.replace(",", " ").split()
            wav_token = None
            for token in tokens:
                candidate = token.strip("\"' ")
                if candidate.endswith(".wav"):
                    wav_token = candidate
                    break
            if wav_token is not None:
                wavs.append(wav_token)
        groups[group_id] = wavs
    return groups


def get_model(checkpoint_path):
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = models[0]
    model.eval()
    model.to("cuda")
    return model


def collapse_duplicates(chars: str):
    return re.sub(r"(.)\1+", r"\1", chars, 0, re.MULTILINE)


def cluster_df_to_utterances(cluster_df):
    char_offset = 34
    cluster_df = cluster_df.copy()
    cluster_df["cluster_char"] = [chr(int(cid) + char_offset) for cid in cluster_df["cluster_id"]]
    utt_df = cluster_df.groupby("wav_file")["cluster_char"].apply("".join).reset_index()
    utt_df["cluster_char"] = utt_df["cluster_char"].apply(collapse_duplicates)
    return utt_df


def train_or_load_spm(target_utts_df, args):
    if args.spm_model:
        model_path = Path(args.spm_model)
        if not model_path.exists():
            raise ValueError(f"--spm-model not found: {model_path}")
        print(f"Loading existing sentencepiece model: {model_path}")
        return spm.SentencePieceProcessor(model_file=str(model_path))

    tmp_dir = Path(args.tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tgt_utts_path = tmp_dir / "tgt_utts.txt"
    spm_prefix = tmp_dir / args.spm_prefix
    model_path = Path(str(spm_prefix) + ".model")

    tgt_utts = "\n".join(target_utts_df["cluster_char"].tolist()) + "\n"
    tgt_utts_path.write_text(tgt_utts)
    if tgt_utts_path.stat().st_size == 0:
        raise ValueError("Target utterance text is empty; cannot train sentencepiece.")

    spm.SentencePieceTrainer.train(
        input=str(tgt_utts_path),
        model_prefix=str(spm_prefix),
        vocab_size=args.spm_vocab_size,
        character_coverage=1.0,
        model_type="unigram",
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name="identity" if args.spm_ident_norm else "nmt_nfkc",
    )
    print(f"Trained sentencepiece model: {model_path}")
    return spm.SentencePieceProcessor(model_file=str(model_path))


def encode_piece_counts(utt_strings, sp):
    counts = {}
    token_sum = 0
    for text in utt_strings:
        ids = sp.encode(text, out_type=int)
        token_sum += len(ids)
        for pid in ids:
            counts[pid] = counts.get(pid, 0) + 1

    if not counts:
        return {}, token_sum

    max_count = max(counts.values())
    normalized = {pid: val / max_count for pid, val in counts.items()}
    return normalized, token_sum


def cosine_similarity_from_dicts(lhs, rhs):
    keys = sorted(set(lhs.keys()) | set(rhs.keys()))
    if not keys:
        return np.nan
    lhs_vec = np.array([lhs.get(k, 0.0) for k in keys], dtype=float)
    rhs_vec = np.array([rhs.get(k, 0.0) for k in keys], dtype=float)
    if np.all(lhs_vec == 0.0) or np.all(rhs_vec == 0.0):
        return np.nan
    return 1.0 - distance.cosine(lhs_vec, rhs_vec)


def extract_group_clustered_df(wav_files, donor_root, model, get_speech_timestamps, vad_model, km_model):
    rows = []
    embedding_cols = [f"e{i:03}" for i in range(1024)]

    for wav_file in wav_files:
        wav_path = donor_root / wav_file
        if not wav_path.exists():
            continue

        wav_data, sample_rate = torchaudio.load(str(wav_path))
        speech_timestamps = get_speech_timestamps(wav_data, vad_model, sampling_rate=sample_rate)
        speech_intervals = pd.DataFrame(speech_timestamps)
        if len(speech_intervals) == 0:
            continue

        with torch.no_grad():
            normed = F.layer_norm(wav_data, wav_data.shape)
            encoder_out = model(normed.to("cuda"), features_only=True, mask=False)
            layer12 = (
                encoder_out["layer_results"][12 - 1][0]
                .transpose(0, 1)
                .squeeze(0)
                .cpu()
                .numpy()
            )

        emb_df = pd.DataFrame(layer12, columns=embedding_cols)
        wav_dur = wav_data.shape[1] / sample_rate
        start_times = np.linspace(0.00, wav_dur, layer12.shape[0], endpoint=False)
        end_times = np.concatenate([start_times[1:], np.array([wav_dur])])
        emb_df["code_start_time"] = start_times
        emb_df["code_end_time"] = end_times

        speech_intervals["start"] /= sample_rate
        speech_intervals["end"] /= sample_rate

        speech_codes = speech_intervals.merge(emb_df, how="cross").query(
            "code_start_time >= start and code_end_time <= end"
        )
        if speech_codes.empty:
            continue

        vecs = np.array(speech_codes[embedding_cols]).astype(float)
        cluster_ids = km_model.predict(vecs)
        rows.extend(
            {
                "wav_file": wav_file,
                "cluster_id": int(cid),
            }
            for cid in cluster_ids
        )

    if not rows:
        return pd.DataFrame(columns=["wav_file", "cluster_id"])
    return pd.DataFrame(rows)


def main():
    args = parse_args()
    groups_csv = Path(args.groups_csv)
    donor_root = Path(args.donor_wav_dir)
    target_clustered = Path(args.target_clustered_parquet)
    out_atds = Path(args.output_atds_csv)
    out_counts = Path(args.output_piece_counts_csv)

    if not groups_csv.exists():
        raise ValueError(f"--groups-csv not found: {groups_csv}")
    if not donor_root.exists():
        raise ValueError(f"--donor-wav-dir not found: {donor_root}")
    if not target_clustered.exists():
        raise ValueError(f"--target-clustered-parquet not found: {target_clustered}")

    out_atds.parent.mkdir(parents=True, exist_ok=True)
    out_counts.parent.mkdir(parents=True, exist_ok=True)

    grouped_paths = parse_group_csv(groups_csv)
    print(f"Loaded groups: {len(grouped_paths)}")

    print("Loading target clustered parquet...")
    target_cluster_df = pd.read_parquet(target_clustered, columns=["wav_file", "cluster_id"])
    target_utts_df = cluster_df_to_utterances(target_cluster_df)

    print("Preparing sentencepiece model...")
    sp = train_or_load_spm(target_utts_df, args)

    target_piece_norm, _ = encode_piece_counts(target_utts_df["cluster_char"].tolist(), sp)

    print("Loading k-means model...")
    km_model = joblib.load(args.kmeans_model)
    print("Loading XLSR model and VAD...")
    model = get_model(args.checkpoint_path)
    vad_model, vad_utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        force_reload=False,
        onnx=False,
    )
    get_speech_timestamps, _, _, _, _ = vad_utils

    score_rows = []
    count_rows = []

    for group_id, wav_files in tqdm(grouped_paths.items(), total=len(grouped_paths)):
        donor_cluster_df = extract_group_clustered_df(
            wav_files, donor_root, model, get_speech_timestamps, vad_model, km_model
        )
        donor_utts_df = cluster_df_to_utterances(donor_cluster_df)
        donor_piece_norm, donor_token_sum = encode_piece_counts(
            donor_utts_df["cluster_char"].tolist(), sp
        )
        score = cosine_similarity_from_dicts(target_piece_norm, donor_piece_norm)

        score_rows.append(
            {
                "group_id": int(group_id),
                "target_lang": args.target_lang,
                "donor_lang": args.donor_lang,
                "atds": float(score) if pd.notna(score) else np.nan,
            }
        )
        count_rows.append(
            {
                "group_id": int(group_id),
                "piece_counts_sum": int(donor_token_sum),
            }
        )

    atds_df = pd.DataFrame(score_rows).sort_values("group_id")
    counts_df = pd.DataFrame(count_rows).sort_values("group_id")

    atds_df.to_csv(out_atds, index=False)
    counts_df.to_csv(out_counts, index=False)
    print(f"Saved raw scores: {out_atds}")
    print(f"Saved token sums: {out_counts}")


if __name__ == "__main__":
    main()
