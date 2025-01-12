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

def make_manifests(file_list, base_dir):
    """
    WAVファイルリストからファイルを選択し、3つのグループに分類する
    """
    if not file_list:
        print("警告: 入力ファイルリストが空です")
        return [], []
        
    print(f"処理開始: 合計{len(file_list)}個のWAVファイル")
    random.shuffle(file_list)
    
    groups = [
        {'total_duration': 0, 'selected_files': [], 'limit': 3600, 'name': 'train-1h'},  # 1時間
        {'total_duration': 0, 'selected_files': [], 'limit': 7200, 'name': 'test-2h'},   # 2時間
        {'total_duration': 0, 'selected_files': [], 'limit': 3600, 'name': 'valid-1h'}   # 1時間
    ]
    
    skipped_files = 0
    processed_files = 0
    
    for filename in file_list:
        filepath = os.path.join(base_dir, filename)
        if not os.path.exists(filepath):
            print(f"警告: ファイルが見つかりません: {filepath}")
            skipped_files += 1
            continue
            
        try:
            duration = get_wav_duration(filepath)
            
            for group in groups:
                if group['total_duration'] + duration <= group['limit']:
                    with wave.open(filepath, 'rb') as wav_file:
                        frames = wav_file.getnframes()
                    group['total_duration'] += duration
                    group['selected_files'].append((filename, frames))
                    processed_files += 1
                    break
        except Exception as e:
            print(f"エラー: ファイル {filepath} の処理中にエラーが発生しました: {str(e)}")
            skipped_files += 1
    
    print(f"\n処理サマリー:")
    print(f"処理されたファイル: {processed_files}")
    print(f"スキップされたファイル: {skipped_files}")
    
    results_tsv = []
    results_files = []
    
    for group in groups:
        tsv_lines = []
        files = []
        print(f"\n{group['name']}グループ:")
        print(f"選択されたファイル数: {len(group['selected_files'])}")
        print(f"合計時間: {group['total_duration']:.2f}秒")
        
        for filename, frames in group['selected_files']:
            tsv_lines.append(f"{filename}\t{frames}")
            files.append(filename)
            
        tsv_content = '\n'.join(tsv_lines)
        results_tsv.append(tsv_content)
        results_files.append(files)
        
        # TSVの内容を確認
        print(f"TSV行数: {len(tsv_lines)}")
        if len(tsv_lines) > 0:
            print(f"TSVサンプル（最初の行）: {tsv_lines[0]}")
    
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
    input_path = "data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/transcription_n2w.txt"
    base_dir = "data/IndicSUPERB/kb_data_clean_m4a/punjabi/train/audio"
    output_dir = "data/manifests/finetune/punjabi"
    
    # 出力ファイルパスの設定
    output_paths = {
        'tsv': {
            'train': os.path.join(output_dir, "train-1h.tsv"),
            'test': os.path.join(output_dir, "test-2h.tsv"),
            'valid': os.path.join(output_dir, "valid-1h.tsv")
        },
        'wrd': {
            'train': os.path.join(output_dir, "train-1h.wrd"),
            'test': os.path.join(output_dir, "test-2h.wrd"),
            'valid': os.path.join(output_dir, "valid-1h.wrd")
        },
        'ltr': {
            'train': os.path.join(output_dir, "train-1h.ltr"),
            'test': os.path.join(output_dir, "test-2h.ltr"),
            'valid': os.path.join(output_dir, "valid-1h.ltr")
        }
    }
    
    # ディレクトリが存在しない場合は作成
    os.makedirs(output_dir, exist_ok=True)
    
    print("1. 入力ファイルの確認")
    if not os.path.exists(input_path):
        print(f"エラー: 入力ファイルが見つかりません: {input_path}")
        exit(1)
    print(f"入力ファイルを確認: {input_path}")
    
    print("\n2. WAVファイルリストの取得")
    wav_files = extract_m4a_files(input_path)
    print(f"抽出されたWAVファイル数: {len(wav_files)}")
    
    print("\n3. マニフェストの作成")
    tsv_results, file_lists = make_manifests(wav_files, base_dir)
    
    print("\n4. TSVファイルの書き込み")
    for i, (key, path) in enumerate(output_paths['tsv'].items()):
        try:
            if tsv_results[i].strip():  # 空でないことを確認
                with open(path, 'w', encoding='utf-8') as f:
                    f.write(tsv_results[i])
                print(f"TSVファイルを保存しました: {path}")
                # ファイルの内容を確認
                with open(path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.strip().split('\n')
                    print(f"  - 行数: {len(lines)}")
                    print(f"  - ファイルサイズ: {os.path.getsize(path)} bytes")
            else:
                print(f"警告: {key}のTSV内容が空のため、ファイルを作成しませんでした")
        except Exception as e:
            print(f"エラー: {path}の保存中にエラーが発生しました: {str(e)}")