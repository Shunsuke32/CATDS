import os
from pathlib import Path
import librosa
import soundfile as sf

def resample_wav_files(directory_path, target_sr=16000):
    """
    指定されたディレクトリ内のすべてのWAVファイルを16000Hzにリサンプリングします。
    変換したファイルは元のファイルを上書きします。
    
    Parameters:
    directory_path (str): 処理するディレクトリのパス
    target_sr (int): 目標のサンプリングレート（デフォルト: 16000Hz）
    """
    # Pathオブジェクトを作成
    directory = Path(directory_path)
    
    # WAVファイルを再帰的に検索
    for wav_file in directory.rglob('*.wav'):
        try:
            print(f'処理中: {wav_file}')
            
            # オーディオデータを読み込み
            audio_data, original_sr = librosa.load(wav_file, sr=None)
            
            # 既にターゲットのサンプリングレートの場合はスキップ
            if original_sr == target_sr:
                print(f'スキップ: {wav_file} は既に {target_sr}Hz です')
                continue
            
            # リサンプリング実行
            resampled_audio = librosa.resample(
                audio_data,
                orig_sr=original_sr,
                target_sr=target_sr
            )
            
            # 元のファイルを上書き
            sf.write(wav_file, resampled_audio, target_sr)
            
            print(f'完了: {original_sr}Hz から {target_sr}Hz に変換: {wav_file}')
            
        except Exception as e:
            print(f'エラー: {wav_file} の処理中にエラーが発生しました: {str(e)}')

# 使用例
if __name__ == "__main__":
    # 処理したいディレクトリのパスを指定
    directory_path = "work/data/IndicSUPERB/kb_data_clean_m4a"
    
    # 確認メッセージを表示
    confirm = input("この処理は元のファイルを上書きします。続行しますか？ (y/n): ")
    if confirm.lower() == 'y':
        resample_wav_files(directory_path)
    else:
        print("処理を中止しました")