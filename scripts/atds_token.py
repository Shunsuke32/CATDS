import pandas as pd
import torch
import torchaudio
import torch.nn.functional as F
from pathlib import Path
import numpy as np
from tqdm import tqdm
import time
import joblib
import fairseq
import sentencepiece as spm
from scipy.spatial import distance
import os
import sys

def convert_csv_to_grouped_paths(csv_path):
    df = pd.read_csv(csv_path)
    grouped_paths = {}
    
    for index, row in df.iterrows():
        current_group = index
        grouped_paths[current_group] = []
        
        data_lines = row['data'].split('\n')
        for line in data_lines:
            line = line.strip()
            if line.startswith('path'):
                continue
            if line:
                parts = line.split()
                if len(parts) >= 2:
                    path = parts[-2]
                    grouped_paths[current_group].append(path)
    
    return grouped_paths

def get_model(checkpoint_path):
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([checkpoint_path])
    model = models[0]
    model.eval()
    model.to('cuda')
    return model

def getembeddings(data_list, wav_dir, output_parquet):
    print('Done!')
    data = data_list
    print('Begin Extraction')
    embedding_cols = [f"e{i:03}" for i in range(1024)]
    final_speech_df = pd.DataFrame(columns=["wav_file", "code_start_time", "code_end_time"] + embedding_cols)
    
    for wav_file in tqdm(data_list, total=len(data_list)):
        wav_data, sample_rate = torchaudio.load(wav_dir + "/" + wav_file)
        speech_timestamps = get_speech_timestamps(wav_data, vad_model, sampling_rate=sample_rate)
        speech_intervals = pd.DataFrame(speech_timestamps)

        if len(speech_intervals) == 0:
            continue

        with torch.no_grad():
            normed_wav = F.layer_norm(wav_data, wav_data.shape)
            encoder_out = model(normed_wav.to('cuda'), features_only=True, mask=False)
            layer12_embeddings = encoder_out['layer_results'][12 - 1][0].transpose(0,1).squeeze(0).cpu().numpy()
            
        embeddings_df = pd.DataFrame(layer12_embeddings, columns=embedding_cols)
        wav_dur_in_seconds = wav_data.shape[1]/sample_rate

        start_times = np.linspace(0.00, wav_dur_in_seconds, layer12_embeddings.shape[0], endpoint=False)
        end_times = np.concatenate([start_times[1:], np.array([wav_dur_in_seconds])])
        embeddings_df["code_start_time"] = start_times
        embeddings_df["code_end_time"] = end_times

        speech_intervals.start /= sample_rate
        speech_intervals.end /= sample_rate

        speech_codes = speech_intervals.merge(embeddings_df, how="cross").query("code_start_time >= start and code_end_time <= end")
        speech_codes["wav_file"] = wav_file
        cur_speech_df = speech_codes[["wav_file", "code_start_time", "code_end_time"] + embedding_cols]

        final_speech_df = pd.concat([final_speech_df, cur_speech_df], ignore_index=True, sort=False)

    print(f'Saving to {output_parquet}')
    final_speech_df.to_parquet(output_parquet)

def infer_kmeans(km_model, input_file, output_file):
    km_model = joblib.load(km_model)
    lang_embeds_file = input_file
    print(f"Reading {lang_embeds_file}... ")

    embedding_cols = [f"e{i:03}" for i in range(1024)]
    lang_df = pd.read_parquet(
        lang_embeds_file,
        columns=["wav_file"] + embedding_cols
    )

    lang_df["cluster_id"] = km_model.predict(np.array(lang_df[embedding_cols]).astype(float))
    lang_df[["wav_file", "cluster_id"]].to_parquet(output_file)

def make_all_clusters_df(langs_dir):
    clustered_parquets = Path(f"/work/{langs_dir}").glob("*clustered*")
    all_clusters_df = pd.concat([
        pd.read_parquet(p).assign(lang=p.name.split("_")[0]) for p in clustered_parquets
    ])
    char_offset = 34
    all_clusters_df["cluster_char"] = [chr(i + char_offset) for i in all_clusters_df.cluster_id]
    return all_clusters_df

def make_all_utts_df(all_clusters_df):
    import re
    def merge_duplicates(line):
        return re.sub(r"(.)\1+", r"\1", line, 0, re.MULTILINE)
    
    all_utts_df = all_clusters_df.groupby(["lang", "wav_file"])["cluster_char"].apply(''.join).reset_index()
    all_utts_df.cluster_char = all_utts_df.cluster_char.apply(merge_duplicates)
    return all_utts_df

def train_and_encode_spm(all_utts_df, target_lang, ident_norm=False):
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
    
    piece_counts_sums = {}
    for c in piece_counts_matrix.columns:
        piece_counts_sums[c] = piece_counts_matrix[c].sum()
    
    for c in piece_counts_matrix.columns:
        piece_counts_matrix[c] /= piece_counts_matrix[c].max()
    
    return piece_counts_matrix[[target_lang] + [c for c in piece_counts_matrix.columns if c != target_lang]], piece_counts_sums

def make_ATDS_matrix(piece_freqs_matrix):
    langs = list(piece_freqs_matrix.columns)
    dists_df = pd.DataFrame([], columns=["ref_lang"] + langs)

    for r in langs:
        row_data = {"ref_lang": r}
        for c in langs:
            if r == c:
                row_data[c] = [None]
            else:
                row_data[c] = [1 - distance.cosine(piece_freqs_matrix[r].to_list(), piece_freqs_matrix[c].to_list())]
        dists_df = pd.concat([dists_df, pd.DataFrame(row_data)], ignore_index=True)
    
    return dists_df

def get_best_donors_by_ATDS(ATDS_matrix, target_lang):
    atds_df = ATDS_matrix.rename(columns={'ref_lang':'target_lang'}) \
        .melt(id_vars=['target_lang'], var_name="donor_lang", value_name="atds") \
        .sort_values('target_lang') \
        .query(f"target_lang=='{target_lang}' and target_lang!=donor_lang") \
        .sort_values('atds', ascending=False)
    
    atds_df.atds = atds_df.atds.apply(lambda x: round(x, 4))
    return atds_df

def run_all(langs_dir, target_lang, ident_norm=False):
    all_clusters_df = make_all_clusters_df(langs_dir)
    all_utts_df = make_all_utts_df(all_clusters_df)
    all_utts_df = train_and_encode_spm(all_utts_df, target_lang, ident_norm=ident_norm)
    piece_freqs_matrix, piece_counts_sums = make_piece_freqs_matrix(all_utts_df, target_lang)
    atds_matrix = make_ATDS_matrix(piece_freqs_matrix)
    best_donors = get_best_donors_by_ATDS(atds_matrix, target_lang)
    return atds_matrix, best_donors, piece_counts_sums

if __name__ == "__main__":
    csv_path = "result/bengali_21_20000_full.csv"
    grouped_paths = convert_csv_to_grouped_paths(csv_path)
    
    print('Getting model from /work/checkpoints/xlsr2_300m.pt')
    model = get_model("/work/checkpoints/xlsr2_300m.pt")
    print('Done!')
    print('Loading VAD')
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_timestamps, _, _, VADIterator, collect_chunks) = vad_utils
    print('Done!')

    ident_norm=False
    spm.SentencePieceTrainer.train(
        input='/work/tmp/tgt_utts.txt',
        model_prefix='/work/tmp/10k_piece',
        vocab_size=10001,
        character_coverage=1.0,
        model_type='unigram',
        bos_id=-1,
        eos_id=-1,
        normalization_rule_name='identity' if ident_norm else 'nmt_nfkc'
    )
    s = spm.SentencePieceProcessor(model_file='/work/tmp/10k_piece.model')

    ATDS_dict = {}
    counts_sum_dict = {}

    for num_group, wav_list in grouped_paths.items():
        getembeddings(wav_list, "/work/data/IndicSUPERB/kb_data_clean_m4a/bengali/train/audio", "/work/tmp/bengali.parquet")#/work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audioこれとか全部ドナー言語にするやつ
        infer_kmeans("/work/tmp/k-means_punjabi.joblib", "/work/tmp/bengali.parquet", "/work/tmp/bengali_clustered.parquet")
        atds_matrix, best_donors, piece_counts_sums = run_all("tmp", "punjabi")
        
        ATDS_dict[f"{num_group}"] = best_donors['atds'].iloc[0]
        counts_sum_dict[f"{num_group}"] = piece_counts_sums["bengali"]
        
        print(f"Group {num_group}: ATDS = {ATDS_dict[f'{num_group}']}, Sum = {counts_sum_dict[f'{num_group}']}")
        
        try:
            os.remove("/work/tmp/bengali.parquet")
            os.remove("/work/tmp/bengali_clustered.parquet")
        except Exception as e:
            print(f"Error removing temporary files: {str(e)}")

    atds_df = pd.DataFrame.from_dict(ATDS_dict, orient='index', columns=['atds'])
    atds_df.to_csv('/work/result/ATDS_bengali_21_20000_full.csv')

    counts_sum_df = pd.DataFrame.from_dict(counts_sum_dict, orient='index', columns=['piece_counts_sum'])
    counts_sum_df.to_csv('/work/result/piece_counts_sums_bengali_21_20000_full.csv')