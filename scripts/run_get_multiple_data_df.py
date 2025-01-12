import os
import sys
import csv
from extract_embeddings import get_multiple_data_df

def check_directory(path):
    print(f"Checking directory: {path}")
    if not os.path.exists(path):
        print(f"Directory does not exist: {path}")
        try:
            os.makedirs(path)
            print(f"Created directory: {path}")
        except Exception as e:
            print(f"Error creating directory: {str(e)}")
            return False
    elif not os.path.isdir(path):
        print(f"Error: Path is not a directory: {path}")
        return False
    else:
        print(f"Directory already exists: {path}")
    return True

def count_files(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

# 関数のパラメータを設定
wav_dir = "/work/data/IndicSUPERB/kb_data_clean_m4a/malayalam/train/audio"
num_hours = 0.006
num_sets = 20000
manifest_path = None

# 現在の作業ディレクトリを表示
print(f"Current working directory: {os.getcwd()}")

# 入力ディレクトリの存在確認
if not check_directory(wav_dir):
    sys.exit(1)

# ファイル数の確認
file_count = count_files(wav_dir)
print(f"Number of files in directory: {file_count}")

# get_multiple_data_df関数の呼び出し
try:
    print("Calling get_multiple_data_df function...")
    result = get_multiple_data_df(wav_dir, num_hours, num_sets, manifest_path)
    print("Function call completed.")
    print(f"Type of result: {type(result)}")
    
    # 結果の確認と処理
    if result is None:
        print("Error: get_multiple_data_df returned None")
    elif isinstance(result, list):
        print(f"Number of items in the list: {len(result)}")
        if not result:
            print("Warning: get_multiple_data_df returned an empty list")
        else:
            print("\nFirst few items of the result:")
            print(result[:5])
            
            # 保存先ディレクトリの確認と作成
            output_dir = "/work/result"
            if check_directory(output_dir):
                # CSVファイルとして保存
                output_file = os.path.join(output_dir, "malayalam_21_20000_full.csv")
                try:
                    with open(output_file, 'w', newline='') as csvfile:
                        writer = csv.writer(csvfile)
                        # ヘッダーを書き込む（必要に応じて調整）
                        writer.writerow(["index", "data"])
                        # データを書き込む
                        for i, item in enumerate(result):
                            writer.writerow([i, item])
                    print(f"\nData has been saved to {output_file}")
                    print(f"File exists: {os.path.exists(output_file)}")
                    print(f"File size: {os.path.getsize(output_file)} bytes")
                except Exception as save_error:
                    print(f"Error saving CSV file: {str(save_error)}")
            else:
                print("Failed to create or access the output directory.")
    else:
        print("Error: result is not a list as expected")
        
except Exception as e:
    print(f"An error occurred: {str(e)}")
    print("\nStack trace:")
    import traceback
    traceback.print_exc()

print("\nScript execution completed.")
print(f"Final check - Output directory exists: {os.path.exists('/work/result')}")
print(f"Final check - Output file exists: {os.path.exists('/work/result/malayalam_21_20000_full.csv')}")