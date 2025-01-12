import os
from pathlib import Path
import wave
import contextlib
import shutil

def find_and_move_long_wav_files(directory_path, output_directory, min_duration_seconds=21.6):
    """
    指定された長さ以上のWAVファイルを探し、指定されたディレクトリに移動する
    
    Parameters:
    directory_path (str): 検索するディレクトリのパス
    output_directory (str): 移動先のディレクトリパス
    min_duration_seconds (float): 最小の長さ（秒）
    
    Returns:
    tuple: (移動したファイルのリスト, 合計ファイル数, 移動したファイル数)
    """
    moved_files = []
    total_files = 0
    matching_files = 0
    
    # 出力ディレクトリが存在しない場合は作成
    os.makedirs(output_directory, exist_ok=True)
    
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
                        # 移動先のパスを生成
                        output_path = os.path.join(output_directory, wav_path.name)
                        
                        # ファイルを移動
                        shutil.move(str(wav_path), output_path)
                        
                        moved_files.append({
                            'original_path': str(wav_path),
                            'new_path': output_path,
                            'duration': duration
                        })
            except wave.Error:
                print(f"警告: {wav_path}を開けませんでした")
                continue
            
    except Exception as e:
        print(f"エラーが発生しました: {str(e)}")
        return [], 0, 0
    
    return moved_files, total_files, matching_files

def main():
    directory = input("WAVファイルを検索するディレクトリのパスを入力してください: ")
    min_duration = float(input("最小の長さを秒単位で入力してください（デフォルト: 18）: ") or 18)
    output_dir = "/work/data/IndicSUPERB/kb_data_clean_m4a/tmp_malayalam_long_files"
    
    moved_files, total_files, matching_files = find_and_move_long_wav_files(directory, output_dir, min_duration)
    
    print(f"\n処理結果:")
    print(f"検索されたWAVファイルの総数: {total_files}")
    print(f"{min_duration}秒以上のファイル数: {matching_files}")
    print(f"移動先ディレクトリ: {output_dir}")
    
    if moved_files:
        print("\n移動したファイル:")
        for file in moved_files:
            print(f"元のパス: {file['original_path']}")
            print(f"新しいパス: {file['new_path']}")
            print(f"長さ: {file['duration']:.2f}秒")
            print("-" * 50)

if __name__ == "__main__":
    main()