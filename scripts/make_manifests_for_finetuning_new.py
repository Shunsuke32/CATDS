import re
from pathlib import Path
import random
import wave 
import os

def convert_to_wav(filename):
    """
    .m4aファイル名を.wavファイル名に変換する関数
    """
    return filename.replace('.m4a', '.wav')

def extract_m4a_files(input_path, output_path=None):
    """
    テキストファイルから.m4aファイル名を抽出する関数
    """
    pattern = r'\d+-\d+-[a-z]\.m4a'
    
    try:
        with open(input_path, 'r', encoding='utf-8') as file:
            content = file.read()
        
        m4a_files = re.findall(pattern, content)
        m4a_files = list(dict.fromkeys(m4a_files))
        wav_files = [convert_to_wav(f) for f in m4a_files]
        
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as out_file:
                for file_name in wav_files:
                    out_file.write(f"{file_name}\n")
            print(f"{len(wav_files)}件のファイルを{output_path}に保存しました。")
            
        return wav_files
        
    except FileNotFoundError:
        print(f"エラー: {input_path}が見つかりません。")
        return []
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return []

def get_wav_duration(wav_path):
    """WAVファイルの再生時間(秒)を計算する"""
    with wave.open(wav_path, 'rb') as wav_file:
        frames = wav_file.getnframes()
        rate = wav_file.getframerate()
        duration = frames / float(rate)
        return duration

def make_manifests(file_list, train_dir, test_dir, valid_dir):
    """
    WAVファイルリストからファイルを選択し、3つのグループに分類する
    各グループで異なるbase_dirを使用
    """
    random.shuffle(file_list)
    
    groups = [
        {'total_duration': 0, 'selected_files': [], 'limit': 3600, 'base_dir': train_dir},  # 訓練用1時間
        {'total_duration': 0, 'selected_files': [], 'limit': 7200, 'base_dir': test_dir},   # テスト用2時間
        {'total_duration': 0, 'selected_files': [], 'limit': 3600, 'base_dir': valid_dir}   # 検証用1時間
    ]
    
    for filename in file_list:
        assigned = False
        for group in groups:
            filepath = os.path.join(group['base_dir'], filename)
            if not os.path.exists(filepath):
                continue
                
            duration = get_wav_duration(filepath)
            
            if group['total_duration'] + duration <= group['limit']:
                with wave.open(filepath, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                group['total_duration'] += duration
                group['selected_files'].append((filename, frames))
                assigned = True
                break
        
        if not assigned:
            continue
    
    results_tsv = []
    results_files = []
    
    for group in groups:
        tsv_lines = []
        files = []
        for filename, frames in group['selected_files']:
            tsv_lines.append(f"{filename}\t{frames}")
            files.append(filename)
        results_tsv.append('\n'.join(tsv_lines))
        results_files.append(files)
    
    return results_tsv, results_files

def create_text_only_file(input_path, wav_files_list, output_path):
    """
    テキストファイルからWAVリストに対応する文章のみを抽出する
    """
    try:
        # WAVファイル名をM4Aファイル名に変換
        m4a_files = [w.replace('.wav', '.m4a') for w in wav_files_list]
        
        # 元のテキストファイルからファイル名と文章のペアを作成
        text_pairs = {}
        with open(input_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                    
                parts = line.split('\t')
                if len(parts) == 2:
                    file_name, text = parts
                    text_pairs[file_name] = text
        
        # WAVリストの順序に従って文章を抽出
        extracted_texts = []
        for m4a_file in m4a_files:
            if m4a_file in text_pairs:
                extracted_texts.append(text_pairs[m4a_file])
        
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 文章をファイルに書き出し
        with open(output_path, 'w', encoding='utf-8') as out_file:
            for text in extracted_texts:
                out_file.write(f"{text}\n")
        
        print(f"{len(extracted_texts)}件のテキストを{output_path}に保存しました。")
        return True
        
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return False

def format_punjabi_text(input_text):
    """パンジャブ語テキストをフォーマットする"""
    lines = [line.strip() for line in input_text.split('\n') if line.strip()]
    
    formatted_lines = []
    for line in lines:
        if line.strip() == '.':
            continue
            
        words = line.split()
        spaced_words = []
        
        for word in words:
            chars = ' '.join(list(word))
            spaced_words.append(chars)
            
        formatted_line = '| ' + ' | '.join(spaced_words) + ' |'
        formatted_lines.append(formatted_line)
    
    return '\n'.join(formatted_lines)

def process_file(input_path, output_path):
    """テキストファイルを処理する"""
    try:
        with open(input_path, 'r', encoding='utf-8') as f:
            input_text = f.read()
        
        formatted_text = format_punjabi_text(input_text)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(formatted_text)
            
        print(f"変換が完了しました。出力ファイル: {output_path}")
        
    except FileNotFoundError:
        print(f"エラー: 入力ファイル '{input_path}' が見つかりません。")
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")

if __name__ == "__main__":
    # 入力パスの設定
    input_paths = {
        'train': "data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/transcription_n2w.txt",
        'test': "data/IndicSUPERB/kb_data_clean_m4a/punjabi/test_known/transcription_n2w.txt",
        'valid': "data/IndicSUPERB/kb_data_clean_m4a/punjabi/valid/transcription_n2w.txt"
    }
    
    # 各データセット用のベースディレクトリ設定
    base_dirs = {
        'train': "data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio",
        'test': "data/IndicSUPERB/kb_data_clean_m4a/punjabi/test_known/audio",
        'valid': "data/IndicSUPERB/kb_data_clean_m4a/punjabi/valid/audio"
    }
    
    output_dir = "data/manifests/finetune/punjabi"
    
    # 出力ファイルパスの設定
    output_paths = {
        'tsv': {
            'train': os.path.join(output_dir, "train-1h_2.tsv"),
            'test': os.path.join(output_dir, "test-2h_known_2.tsv"),
            'valid': os.path.join(output_dir, "valid-1h_2.tsv")
        },
        'wrd': {
            'train': os.path.join(output_dir, "train-1h_2.wrd"),
            'test': os.path.join(output_dir, "test-2h_known_2.wrd"),
            'valid': os.path.join(output_dir, "valid-1h_2.wrd")
        },
        'ltr': {
            'train': os.path.join(output_dir, "train-1h_2.ltr"),
            'test': os.path.join(output_dir, "test-2h_known_2.ltr"),
            'valid': os.path.join(output_dir, "valid-1h_2.ltr")
        }
    }
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    # 各データセット用のWAVファイルリストを取得
    wav_files = {
        'train': extract_m4a_files(input_paths['train']),
        'test': extract_m4a_files(input_paths['test']),
        'valid': extract_m4a_files(input_paths['valid'])
    }
    
    # マニフェストの作成（trainはtrainから、testはtestから、validはvalidから）
    def create_manifest_for_split(split_name, target_duration):
        wav_list = wav_files[split_name]
        base_dir = base_dirs[split_name]
        random.shuffle(wav_list)
        
        selected_files = []
        total_duration = 0
        
        for filename in wav_list:
            filepath = os.path.join(base_dir, filename)
            if not os.path.exists(filepath):
                continue
                
            duration = get_wav_duration(filepath)
            if total_duration + duration <= target_duration:
                with wave.open(filepath, 'rb') as wav_file:
                    frames = wav_file.getnframes()
                total_duration += duration
                selected_files.append((filename, frames))
        
        tsv_lines = [f"{filename}\t{frames}" for filename, frames in selected_files]
        files = [filename for filename, _ in selected_files]
        
        return '\n'.join(tsv_lines), files
    
    # 各分割用のマニフェストを作成
    manifests = {}
    file_lists = {}
    
    # train (1時間)
    manifests['train'], file_lists['train'] = create_manifest_for_split('train', 3600)
    
    # test (2時間)
    manifests['test'], file_lists['test'] = create_manifest_for_split('test', 7200)
    
    # valid (1時間)
    manifests['valid'], file_lists['valid'] = create_manifest_for_split('valid', 3600)
    
    # TSVファイルの書き込み
    for split_name in ['train', 'test', 'valid']:
        with open(output_paths['tsv'][split_name], 'w', encoding='utf-8') as f:
            f.write(manifests[split_name])
        print(f"TSVファイルを保存しました: {output_paths['tsv'][split_name]}")
    
    # WRDファイルの作成
    for split_name in ['train', 'test', 'valid']:
        create_text_only_file(
            input_paths[split_name],
            file_lists[split_name],
            output_paths['wrd'][split_name]
        )
    
    # LTRファイルの作成
    for split_name in ['train', 'test', 'valid']:
        process_file(
            output_paths['wrd'][split_name],
            output_paths['ltr'][split_name]
        )