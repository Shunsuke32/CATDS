import os
from pathlib import Path
import wave
import contextlib

def find_long_wav_files(directory_path, min_duration_seconds=21.6):
    """
    指定されたディレクトリ内の特定の長さ以上のWAVファイルを探す
    
    Parameters:
    directory_path (str): 検索するディレクトリのパス
    min_duration_seconds (float): 最小の長さ（秒）
    
    Returns:
    tuple: (長いファイルのリスト, 合計ファイル数, 条件を満たすファイル数)
    """
    long_files = []
    total_files = 0
    matching_files = 0
    
    try:
        # ディレクトリ内のすべてのWAVファイルを再帰的に検索
        for wav_path in Path(directory_path).rglob("*.wav"):
            total_files += 1
            try:
                with contextlib.closing(wave.open(str(wav_path), 'r')) as wav_file:
                    # WAVファイルの長さを計算
                    frames = wav_file.getnframes()
                    rate = wav_file.getframerate()
                    duration = frames / float(rate)
                    
                    if duration >= min_duration_seconds:
                        matching_files += 1
                        long_files.append({
                            'path': str(wav_path),
                            'duration': duration
                        })
            except wave.Error:
                print(f"警告: {wav_path}を開けませんでした")
                continue
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return [], 0, 0
    
    return long_files, total_files, matching_files

def main():
    directory = input("WAVファイルを検索するディレクトリのパスを入力してください: ")
    min_duration = float(input("最小の長さを秒単位で入力してください（デフォルト: 18）: ") or 18)
    
    long_files, total_files, matching_files = find_long_wav_files(directory, min_duration)
    
    print(f"\n検索結果:")
    print(f"検索されたWAVファイルの総数: {total_files}")
    print(f"{min_duration}秒以上のファイル数: {matching_files}")
    
    if long_files:
        print("\n条件を満たすファイル:")
        for file in long_files:
            print(f"ファイル: {file['path']}")
            print(f"長さ: {file['duration']:.2f}秒")
            print("-" * 50)

if __name__ == "__main__":
    main()