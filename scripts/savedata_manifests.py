# デバッグ機能マシマシのまま

import pandas as pd
import argparse
import os
import io

def read_input_data(input_file):
    """Read input CSV file and parse the DataFrame string representation"""
    try:
        # CSVファイルを読み込む
        raw_data = pd.read_csv(input_file)
        
        all_entries = []
        # 'data'列の各行を処理
        for df_str in raw_data['data']:
            # 文字列をDataFrameに変換
            # StringIOを使用してテキストをDataFrameとして読み込む
            df = pd.read_csv(io.StringIO(df_str), delim_whitespace=True)
            if not df.empty:
                all_entries.extend(zip(df['path'], df['num_frames']))
        
        print(f"Parsed {len(all_entries)} entries from input file")
        if len(all_entries) > 0:
            print("Sample entries:")
            for entry in all_entries[:3]:
                print(f"{entry[0]}\t{entry[1]}")
        
        return all_entries
        
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        print("Attempting alternative parsing method...")
        try:
            # 別の解析方法を試す
            df_str = raw_data['data'].iloc[0]
            lines = df_str.split('\n')
            entries = []
            for line in lines[1:]:  # ヘッダーをスキップ
                parts = line.strip().split()
                if len(parts) >= 2 and parts[0].endswith('.wav'):
                    wav_file = parts[0]
                    num_frames = int(parts[-1])
                    entries.append((wav_file, num_frames))
            
            print(f"Parsed {len(entries)} entries using alternative method")
            if len(entries) > 0:
                print("Sample entries:")
                for entry in entries[:3]:
                    print(f"{entry[0]}\t{entry[1]}")
            
            return entries
            
        except Exception as e2:
            print(f"Alternative parsing also failed: {str(e2)}")
            return None

def save_manifest(file_info, output_file):
    """Save manifest to a TSV file"""
    if not file_info:
        print("No entries to save")
        return False
        
    try:
        # 出力ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        # 拡張子を .tsv に変更
        output_file = os.path.splitext(output_file)[0] + '.tsv'
        
        print(f"Saving manifest as TSV to: {output_file}")
        
        with open(output_file, 'w', encoding='utf-8') as f:
            # ヘッダーを追加（オプション）
            f.write("filename\tframes\n")
            # データ行を書き込み
            for filename, frames in file_info:
                f.write(f"{filename}\t{frames}\n")
        return True
    except Exception as e:
        print(f"Error saving manifest: {str(e)}")
        return False
def main():
    parser = argparse.ArgumentParser(
        description='Create manifest file from DataFrame containing audio file information.'
    )
    
    parser.add_argument('--input', '-i', required=True,
                      help='Input CSV file path containing audio file information')
    parser.add_argument('--output', '-o', required=True,
                      help='Output manifest file path')
    parser.add_argument('--debug', action='store_true',
                      help='Enable debug output')
    
    args = parser.parse_args()
    
    # 入力ファイルの存在確認
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' does not exist.")
        return
        
    print(f"Reading input file: {args.input}")
    
    # 入力ファイルの最初の数行を表示
    if args.debug:
        print("\nFirst few lines of input file:")
        with open(args.input, 'r') as f:
            print(f.read(500))
    
    # データの読み込み
    entries = read_input_data(args.input)
    if entries is None:
        return
    
    # マニフェストの保存
    if save_manifest(entries, args.output):
        print(f"Successfully created manifest at: {args.output}")
        print(f"Total entries saved: {len(entries)}")
    else:
        print("Failed to create manifest")

if __name__ == "__main__":
    main()