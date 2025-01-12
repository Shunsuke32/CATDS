import subprocess

import numpy as np
import pandas as pd
import sentencepiece as spm

from pathlib import Path
from scipy.spatial import distance

import os
import sys

def make_all_clusters_df(langs_dir):
    clustered_parquets = Path(f"/workspace/{langs_dir}").glob("*clustered*")

    all_clusters_df = pd.concat([
        pd.read_parquet(p).assign(lang=p.name.split("_")[0]) for p in
        clustered_parquets
    ])

    # Skip first 34 (reserved) chars of unicode table
    char_offset = 34
    all_clusters_df["cluster_char"] = [ chr(i + char_offset) for i in all_clusters_df.cluster_id ]
    
    return all_clusters_df

def make_all_utts_df(all_clusters_df):
    import re

    def merge_duplicates(line):
        return re.sub(r"(.)\1+", r"\1", line, 0, re.MULTILINE)

    all_utts_df = all_clusters_df.groupby(["lang", "wav_file"])["cluster_char"].apply(''.join).reset_index()
    all_utts_df.cluster_char = all_utts_df.cluster_char.apply(merge_duplicates)

    return all_utts_df

def train_and_encode_spm(all_utts_df, target_lang, ident_norm=False):
    print(f"Number of utterances for {target_lang}: {len(all_utts_df[all_utts_df.lang == target_lang])}")
    
    with open('/workspace/tmp/tgt_utts.txt', 'w') as w1, open('/workspace/tmp/all_utts.txt', 'w') as w2:
        tgt_utts = "\n".join(all_utts_df[all_utts_df.lang == target_lang].cluster_char.to_list()) + "\n"
        w1.write(tgt_utts)
        w2.writelines("\n".join(all_utts_df.cluster_char.to_list()) + "\n")
    
    print(f"Size of tgt_utts.txt: {os.path.getsize('/workspace/tmp/tgt_utts.txt')} bytes")
    print(f"First 100 characters of tgt_utts.txt: {tgt_utts[:100]}")

    if os.path.getsize('/workspace/tmp/tgt_utts.txt') == 0:
        raise ValueError("The target utterances file is empty. Cannot proceed with training.")
    # Use SentencePiece Python API instead of command-line tool
    spm.SentencePieceTrainer.train(
        input='/workspace/tmp/tgt_utts.txt',
        model_prefix='/workspace/tmp/10k_piece',
        vocab_size=10001,
        character_coverage=1.0,
        model_type='unigram',
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name='identity' if ident_norm else 'nmt_nfkc'
    )
    
    # Load trained model
    s = spm.SentencePieceProcessor(model_file='/workspace/tmp/10k_piece.model')
    
    # Encode all utterances
    all_utts_df["utt_piece_ids"] = all_utts_df.cluster_char.apply(lambda x: s.encode(x, out_type=int))
    
    return all_utts_df

def make_piece_freqs_matrix(all_utts_df, target_lang):
    piece_counts_matrix = all_utts_df \
        .explode('utt_piece_ids')[['lang', 'utt_piece_ids']] \
        .groupby('lang')['utt_piece_ids'] \
        .value_counts() \
        .to_frame('count') \
        .reset_index() \
        .pivot(index='utt_piece_ids', columns='lang', values='count') \
        .fillna(0)
    
    # Normalize counts in each language column to most frequent piece
    for c in piece_counts_matrix.columns:
        piece_counts_matrix[c] /= piece_counts_matrix[c].max()

    return piece_counts_matrix[[target_lang] + [ c for c in piece_counts_matrix.columns if c != target_lang ]]

def make_ATDS_matrix(piece_freqs_matrix):
    langs = list(piece_freqs_matrix.columns)

    dists_df = pd.DataFrame([], columns=["ref_lang"] + langs)

    for r in langs:

        row_data = {"ref_lang":r}

        for c in langs:
            if r == c:
                row_data[c] = [None]
            else:
                row_data[c] = [ 1 - distance.cosine(piece_freqs_matrix[r].to_list(), piece_freqs_matrix[c].to_list()) ]

        dists_df = dists_df.append(pd.DataFrame(row_data))

    return dists_df

def get_best_donors_by_ATDS(ATDS_matrix, target_lang):
    atds_df = ATDS_matrix.rename(columns={'ref_lang':'target_lang'}) \
        .melt(id_vars=['target_lang'], var_name="donor_lang", value_name="atds") \
        .sort_values('target_lang') \
        .query(f"target_lang=='{target_lang}' and target_lang!=donor_lang") \
        .sort_values('atds', ascending=False) \

    atds_df.atds = atds_df.atds.apply(lambda x: round(x, 4))

    return atds_df

def run_all(langs_dir, target_lang, ident_norm=False):
    all_clusters_df = make_all_clusters_df(langs_dir)
    
    all_utts_df = make_all_utts_df(all_clusters_df)
    all_utts_df = train_and_encode_spm(all_utts_df, target_lang, ident_norm=ident_norm)
    
    piece_freqs_matrix = make_piece_freqs_matrix(all_utts_df, target_lang)
    
    atds_matrix = make_ATDS_matrix(piece_freqs_matrix)  
    best_donors = get_best_donors_by_ATDS(atds_matrix, target_lang)

    return atds_matrix, best_donors

def save_results(atds_matrix, best_donors, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ATDS行列を保存
    atds_df = pd.DataFrame(atds_matrix)
    atds_df.to_csv(os.path.join(output_dir, 'atds_matrix.csv'), index=False)
    
    # 最適なドナーを保存
    best_donors_df = pd.DataFrame(best_donors)
    best_donors_df.to_csv(os.path.join(output_dir, 'best_donors.csv'), index=False)

# スクリプトを実行する部分
if __name__ == "__main__":

    if len(sys.argv) != 3:
        print("Usage: python ATDS.py <langs_dir> <target_lang>")
        sys.exit(1)

    langs_dir = sys.argv[1]
    target_lang = sys.argv[2]

    output_dir = "result"  # 結果を保存するディレクトリ

    atds_matrix, best_donors = run_all(langs_dir, target_lang)
    save_results(atds_matrix, best_donors, output_dir)
    print("Results saved in the 'result' folder.")