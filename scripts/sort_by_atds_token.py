#!/usr/bin/env python3
import argparse
import random
from pathlib import Path

import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Normalize CATDS scores with a quadratic correction and export top-N groups as manifest rows."
    )
    parser.add_argument("--atds-csv", required=True, help="CSV produced by atds_token.py")
    parser.add_argument("--counts-csv", required=True, help="Token count CSV produced by atds_token.py")
    parser.add_argument("--groups-csv", required=True, help="Grouped donor CSV produced by run_get_multiple_data_df.py")
    parser.add_argument("--output-manifest", required=True, help="Output TSV path for selected groups")
    parser.add_argument("--top-n", type=int, required=True, help="Number of groups to select")
    parser.add_argument("--coef-a", type=float, required=True, help="Quadratic coefficient a for a*x^2 + b*x + c")
    parser.add_argument("--coef-b", type=float, required=True, help="Quadratic coefficient b for a*x^2 + b*x + c")
    parser.add_argument("--coef-c", type=float, required=True, help="Quadratic coefficient c for a*x^2 + b*x + c")
    parser.add_argument("--equation-floor", type=float, default=1e-8, help="Minimum denominator floor to avoid division by zero")
    parser.add_argument("--manifest-root", default=None, help="Optional first line for fairseq TSV manifest root path")
    parser.add_argument("--shuffle", action="store_true", help="Shuffle ranking before selecting top-N")
    parser.add_argument("--output-ranking-csv", default=None, help="Optional CSV path to save normalized ranking details")
    return parser.parse_args()


def parse_group_rows(data_text):
    rows = []
    for line in str(data_text).splitlines():
        line = line.strip()
        if not line or line.startswith("path"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        wav = None
        frames = None
        for i, part in enumerate(parts):
            cleaned = part.strip("\"',")
            if cleaned.endswith(".wav"):
                wav = cleaned
                if i + 1 < len(parts):
                    frames = parts[i + 1].strip("\"',")
                break
        if wav is not None and frames is not None:
            rows.append(f"{wav}\t{frames}")
    return rows


def main():
    args = parse_args()

    atds_df = pd.read_csv(args.atds_csv)
    counts_df = pd.read_csv(args.counts_csv)
    groups_df = pd.read_csv(args.groups_csv)

    if "group_id" not in atds_df.columns:
        raise ValueError("atds-csv must contain 'group_id' column")
    if "group_id" not in counts_df.columns:
        raise ValueError("counts-csv must contain 'group_id' column")

    merged = atds_df.merge(counts_df, on="group_id", how="inner")
    merged = merged.dropna(subset=["atds", "piece_counts_sum"]).copy()
    merged["piece_counts_sum"] = merged["piece_counts_sum"].astype(float)

    denom = (
        args.coef_a * merged["piece_counts_sum"] ** 2
        + args.coef_b * merged["piece_counts_sum"]
        + args.coef_c
    )
    denom = denom.where(denom.abs() >= args.equation_floor, args.equation_floor)
    merged["normalized_catds"] = merged["atds"] / denom

    ranking = list(
        merged.sort_values("normalized_catds", ascending=False)[
            ["group_id", "atds", "piece_counts_sum", "normalized_catds"]
        ].itertuples(index=False, name=None)
    )
    if args.shuffle:
        random.shuffle(ranking)

    selected_group_ids = [int(row[0]) for row in ranking[: args.top_n]]
    group_text_map = {}
    for _, row in groups_df.iterrows():
        gid = int(row["index"]) if "index" in row else int(_)
        group_text_map[gid] = row["data"]

    output_rows = []
    seen = set()
    for gid in selected_group_ids:
        if gid not in group_text_map:
            continue
        for row in parse_group_rows(group_text_map[gid]):
            if row not in seen:
                seen.add(row)
                output_rows.append(row)

    out_manifest = Path(args.output_manifest)
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with out_manifest.open("w", encoding="utf-8") as f:
        if args.manifest_root:
            f.write(f"{args.manifest_root}\n")
        f.write("\n".join(output_rows))
        if output_rows:
            f.write("\n")

    if args.output_ranking_csv:
        Path(args.output_ranking_csv).parent.mkdir(parents=True, exist_ok=True)
        merged.sort_values("normalized_catds", ascending=False).to_csv(
            args.output_ranking_csv, index=False
        )

    print(f"Selected groups: {len(selected_group_ids)}")
    print(f"Manifest rows written: {len(output_rows)}")
    print(f"Saved manifest: {out_manifest}")
    print(
        "Normalization equation: "
        f"score / ({args.coef_a}*x^2 + {args.coef_b}*x + {args.coef_c}), "
        "x=piece_counts_sum"
    )


if __name__ == "__main__":
    main()
