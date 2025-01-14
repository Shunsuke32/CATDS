# import pandas as pd
# import random

# def sort_atds(random_shuffle=False):
#     df = pd.read_csv("/work/result/ATDS_hindi_18_6000.csv")
#     d = df.to_dict()
#     ATDS_dict = d["atds"]
#     #print(ATDS_dict)
#     ATDS_sorted_list = sorted(ATDS_dict.items(), key=lambda x:x[1], reverse=True)
#     print(ATDS_sorted_list)
#     wavfile_dict = pd.read_csv("/work/result/hindi18sec_6000_data.csv").to_dict()["data"]
#     print(wavfile_dict[20])

#     num_group = []
#     if random_shuffle == True:
#         random.shuffle(ATDS_sorted_list)
#     ATDS_filtered_list = ATDS_sorted_list[:3000] 
#     print(len(ATDS_filtered_list))
#     for tup in ATDS_filtered_list:
#         num_group.append(tup[0])
    
#     finaldf = []
#     for num in num_group:
#         finaldf.append(wavfile_dict[num])

#     return finaldf

# def format_wav_list(data_list):
    
#     formatted_rows = []
    
#     for line in data_list:
#         # Extract filenames and numbers using string operations
#         if 'path' in line and 'num_frames' in line:
#             # Split the line and extract relevant parts
#             parts = line.split()
#             for i, part in enumerate(parts):
#                 if '.wav' in part:
#                     filename = part.strip(',"')
#                     num_frames = parts[i + 1].strip()
#                     formatted_row = f"{filename}\t{num_frames}"
#                     if formatted_row not in formatted_rows:  # Avoid duplicates
#                         formatted_rows.append(formatted_row)
#                     break

    
#     # Join all rows with newlines
#     return '\n'.join(formatted_rows)


# if __name__ == "__main__":
#     #ランダムにするならsort_atds(True)
#     data_list = sort_atds(False)
#     formatted_wavfiles = format_wav_list(data_list)
#     print(formatted_wavfiles)
#     with open('/work/data/manifests/pretrain/hindi_train_18sec6000_filterto3000_random.tsv', 'w', encoding='utf-8') as f:
#         f.write(formatted_wavfiles)

import pandas as pd
import random

def sort_atds(random_shuffle=False):
    # ATDSスコアの読み込み
    df = pd.read_csv("/work/result/ATDS_malayalam_21_20000_full.csv")
    d = df.to_dict()
    ATDS_dict = d["atds"]
    #token数の合計値取得
    df2 = pd.read_csv("/work/result/piece_counts_sums_malayalam_21_20000_full.csv")
    d2 = df2.to_dict()
    token_dict = d2["piece_counts_sum"]
    # 音声ファイル情報の読み込み
    wavfile_dict = pd.read_csv("/work/result/malayalam_21_20000_full.csv").to_dict()["data"]

    # ATDSの正規化
    normalized_atds = {}
    for idx, atds in ATDS_dict.items():
        if idx in token_dict and token_dict[idx] > 0:
            x = token_dict[idx]
            y = -0.0000004010*x*x + 0.00083133*x + 0.26619643
            print(token_dict[idx])
            print(y)
            normalized_atds[idx] = atds / y
            print(normalized_atds[idx])
    
    # 正規化されたATDSでソート
    ATDS_sorted_list = sorted(normalized_atds.items(), key=lambda x: x[1], reverse=True)
    print(ATDS_sorted_list)
    if random_shuffle:
        random.shuffle(ATDS_sorted_list)
    
    # 上位???件を選択
    ATDS_filtered_list = ATDS_sorted_list[:500]
    num_group = [tup[0] for tup in ATDS_filtered_list]
    
    # 選択されたファイル情報を取得
    finaldf = []
    for num in num_group:
        finaldf.append(wavfile_dict[num])
    return finaldf

def format_wav_list(data_list):
    formatted_rows = []
    
    for line in data_list:
        # DataFrameの形式で書かれたテキストを行ごとに分割
        rows = line.strip().split('\n')
        for row in rows[1:]:  # ヘッダー行をスキップ
            if row.strip():  # 空行でない場合
                # 行をスペースで分割して必要な情報を抽出
                parts = row.strip().split()
                # 最後の要素がフレーム数、その直前がパス
                filename = parts[-2].strip(',"')
                num_frames = parts[-1].strip()
                formatted_row = f"{filename}\t{num_frames}"
                if formatted_row not in formatted_rows:
                    formatted_rows.append(formatted_row)
    
    return '\n'.join(formatted_rows)

if __name__ == "__main__":
    data_list = sort_atds(False)  # True for random shuffle
    formatted_wavfiles = format_wav_list(data_list)
    with open('/work/data/manifests/pretrain/malayalam_21_20000to500_ATDS.tsv', 'w', encoding='utf-8') as f:
        f.write(formatted_wavfiles)

