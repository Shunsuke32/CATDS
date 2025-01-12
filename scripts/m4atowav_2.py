import os
from pydub import AudioSegment
import glob

def convert_m4a_to_wav(input_dir):
    """
    指定されたディレクトリ内のすべてのm4aファイルを16kHz monoのWAVファイルに変換します。
    
    Parameters:
    input_dir (str): 変換したいm4aファイルがあるディレクトリのパス
    """
    # m4aファイルを検索
    m4a_files = glob.glob(os.path.join(input_dir, "*.m4a"))
    
    if not m4a_files:
        print(f"警告: {input_dir} にm4aファイルが見つかりませんでした。")
        return
    
    # 出力用のディレクトリを作成
    output_dir = os.path.join(input_dir, "converted_wav")
    os.makedirs(output_dir, exist_ok=True)
    
    # 各ファイルを変換
    for m4a_file in m4a_files:
        try:
            # ファイル名から拡張子を除いた部分を取得
            base_name = os.path.splitext(os.path.basename(m4a_file))[0]
            
            # 出力ファイルパスを設定
            wav_file = os.path.join(output_dir, f"{base_name}.wav")
            
            # m4aファイルを読み込み
            audio = AudioSegment.from_file(m4a_file, format="m4a")
            
            # モノラルに変換
            audio = audio.set_channels(1)
            
            # サンプリングレートを16kHzに変更
            audio = audio.set_frame_rate(16000)
            
            # WAVファイルとして保存
            audio.export(wav_file, format="wav", parameters=["-acodec", "pcm_s16le"])
            
            print(f"変換成功: {m4a_file} -> {wav_file}")
            
        except Exception as e:
            print(f"エラー: {m4a_file} の変換に失敗しました。")
            print(f"エラー内容: {str(e)}")

if __name__ == "__main__":
    # スクリプトを実行するディレクトリのパスを指定
    directory = input("変換したいm4aファイルがあるディレクトリのパスを入力してください: ")
    convert_m4a_to_wav(directory)