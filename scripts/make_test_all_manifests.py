import wave
import os
import re
from pathlib import Path

def get_wav_duration(wav_path):
    """WAVファイルの再生時間(秒)を計算する"""
    with wave.open(wav_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        return frames / float(rate), frames

def extract_m4a_files(input_path):
    """テキストファイルから.m4aファイル名を抽出してWAVに変換"""
    pattern = r'\d+-\d+-[a-z]\.m4a'
    
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        m4a_files = re.findall(pattern, content)
        return [f.replace('.m4a', '.wav') for f in dict.fromkeys(m4a_files)]
    except Exception as e:
        print(f"エラー: {str(e)}")
        return []

def create_manifest(input_path, audio_dir, output_dir):
    """テストデータのマニフェストを作成"""
    wav_files = extract_m4a_files(input_path)
    
    tsv_lines = []
    selected_files = []
    
    for wav_file in wav_files:
        wav_path = os.path.join(audio_dir, wav_file)
        if os.path.exists(wav_path):
            _, frames = get_wav_duration(wav_path)
            tsv_lines.append(f"{wav_file}\t{frames}")
            selected_files.append(wav_file)
    
    # TSVファイル作成
    tsv_path = os.path.join(output_dir, "test_all.tsv")
    os.makedirs(output_dir, exist_ok=True)
    with open(tsv_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(tsv_lines))
    
    return selected_files

def create_text_files(input_path, wav_files, output_dir):
    """WRDファイルとLTRファイルを作成"""
    try:
        # WRDファイル作成
        text_pairs = {}
        with open(input_path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    text_pairs[parts[0].replace('.m4a', '.wav')] = parts[1]
        
        wrd_lines = [text_pairs[wav] for wav in wav_files if wav in text_pairs]
        wrd_path = os.path.join(output_dir, "test_all.wrd")
        with open(wrd_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(wrd_lines))
        
        # LTRファイル作成
        ltr_lines = []
        for line in wrd_lines:
            if line.strip() == '.':
                continue
            words = line.split()
            spaced_words = [' '.join(list(word)) for word in words]
            ltr_lines.append('| ' + ' | '.join(spaced_words) + ' |')
        
        ltr_path = os.path.join(output_dir, "test_all.ltr")
        with open(ltr_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(ltr_lines))
            
    except Exception as e:
        print(f"エラー: {str(e)}")

if __name__ == "__main__":
    input_path = "data/IndicSUPERB/kb_data_clean_m4a/punjabi/test_known/transcription_n2w.txt"
    audio_dir = "data/IndicSUPERB/kb_data_clean_m4a/punjabi/test_known/audio"
    output_dir = "data/manifests/finetune/punjabi"
    
    selected_files = create_manifest(input_path, audio_dir, output_dir)
    create_text_files(input_path, selected_files, output_dir)