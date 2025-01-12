import torch
import torchaudio
import torch.nn.functional as F

from pathlib import Path
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
import time

import fairseq

torch.set_num_threads(1)
RANDOM_STATE = int(time.time())

def get_model(checkpoint_path):
    models, _, _ = fairseq.checkpoint_utils.load_model_ensemble_and_task([ checkpoint_path ])
    model = models[0]
    model.eval()
    model.to('cuda')
    return model

def get_data_df(wav_dir, num_hours, manifest_path=None):
    if manifest_path is None:
        data_path = Path(wav_dir)
        audio_df = pd.DataFrame({
            'path' : sorted(data_path.glob("*.wav"))
        }).sample(frac=1.0, random_state=RANDOM_STATE)
        audio_df['num_frames'] = audio_df.path.apply(lambda p: torchaudio.info(p).num_frames)
        audio_df['path'] = audio_df['path'].apply(lambda p: str(p.name))
    else:
        print(f"Loading manifest from {manifest_path}")
        audio_df = pd.read_csv(manifest_path, sep="\t", skiprows=1, header=None, names=["path", "num_frames"])

    total_frames = int(16_000 * 60 * 60 * num_hours)
    audio_nhours_df = audio_df.query(f"num_frames.cumsum() <= (16_000 * 60 * 60 * {str(num_hours)})")

    print(f"Selected {len(audio_nhours_df)} files, totaling {audio_nhours_df['num_frames'].sum() / 16000 / 60 / 60:.4f} hours")

    return audio_nhours_df

def get_multiple_data_df(wav_dir, num_hours, num_sets, manifest_path=None): 
    #get_data_dfの複数回施行バージョン、num_setで何個やるか指定可能
    if manifest_path is None:
        data_path = Path(wav_dir)
        audio_df = pd.DataFrame({
            'path': sorted(data_path.glob("*.wav"))
        }).sample(frac=1.0, random_state=RANDOM_STATE)
        audio_df['num_frames'] = audio_df.path.apply(lambda p: torchaudio.info(p).num_frames)
        audio_df['path'] = audio_df['path'].apply(lambda p: str(p.name))
    else:
        print(f"Loading manifest from {manifest_path}")
        audio_df = pd.read_csv(manifest_path, sep="\t", skiprows=1, header=None, names=["path", "num_frames"])

    frames_per_set = int(16_000 * 60 * 60 * num_hours)
    datasets = []
    remaining_df = audio_df.copy()

    for i in range(num_sets):
        if remaining_df.empty:
            print(f"Warning: Ran out of data after creating {i} datasets")
            break

        set_df = remaining_df.loc[remaining_df['num_frames'].cumsum() <= frames_per_set]
        
        if set_df.empty:
            print(f"Warning: Could not create dataset {i+1}. Not enough continuous data left.")
            break

        datasets.append(set_df)
        print(f"Created dataset {i+1} with {len(set_df)} files, "
              f"totaling {set_df['num_frames'].sum() / 16000 / 60 / 60:.4f} hours")

        remaining_df = remaining_df.iloc[len(set_df):]

    return datasets

def run(args):
    print(f'Getting model from {args.checkpoint_path}')
    model = get_model(args.checkpoint_path)
    print('Done!')

    print(f'Getting {args.num_hours} hours of data from {args.wav_dir} for {args.num_sets} sets')
    datasets = get_multiple_data_df(args.wav_dir, args.num_hours, args.num_sets, args.manifest_path)
    print('Done!')

    print('Loading VAD')
    vad_model, vad_utils = torch.hub.load(repo_or_dir='snakers4/silero-vad', model='silero_vad', force_reload=False, onnx=False)
    (get_speech_timestamps, _, _, VADIterator, collect_chunks) = vad_utils
    print('Done!')
    
    print('Begin Extraction')
    embedding_cols = [f"e{i:03}" for i in range(1024)]
    final_speech_df = pd.DataFrame(columns=["wav_file", "code_start_time", "code_end_time"] + embedding_cols)
    
    for dataset_idx, data_df in enumerate(datasets):
        print(f'Processing dataset {dataset_idx + 1} of {len(datasets)}')
        for idx, vals in tqdm(data_df.iterrows(), total=data_df.shape[0]):
            wav_file = vals['path']
            wav_data, sample_rate = torchaudio.load(args.wav_dir + "/" + wav_file)

            speech_timestamps = get_speech_timestamps(wav_data, vad_model, sampling_rate=sample_rate)
            speech_intervals = pd.DataFrame(speech_timestamps)

            # If no voice activity detected, skip clip
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

    print(f'Saving to {args.output_parquet}')
    final_speech_df.to_parquet(args.output_parquet)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Extract Codebook Indices Based on a Trained Model')

    parser.add_argument('--checkpoint-path', required=True, type=str,
                help='path to model checkpoint (eg. checkpoints/xlsr2_300m.pt)')
    parser.add_argument('--wav-dir', required=True, type=str,
                help='path to wav data directory (eg. data/IndicSUPERB/punjabi/audio)')
    parser.add_argument('--manifest-path', required=False, type=str,
                help='optional. Select data from tsv manifest instead of directory listing')
    parser.add_argument('--num-hours', default=5, type=float,
                help='number of hours to subset (default=5, can be a float)')
    parser.add_argument('--num-sets', default=1000, type=int,
                help='number of datasets (default=1000)')
    # --output-parquetを削除し、新しい引数を追加
    parser.add_argument('--output-prefix', required=True, type=str,
                help='prefix for output parquet files (eg. "hindi" will create hindi1_.parquet)')
    parser.add_argument('--output-dir', default='.', type=str,
                help='directory to save output parquet files (default: current directory)')

    args = parser.parse_args()

    run(args)
