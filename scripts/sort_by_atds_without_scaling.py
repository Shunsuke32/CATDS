import pandas as pd

def sort_atds(atds_csv, wav_csv, top_n):
    """
    ATDSスコアのCSVファイルと音声ファイル情報のCSVファイルを読み込み、
    ATDSスコアが高い順に上位top_n件の音声ファイル情報を抽出して返す。

    Parameters:
        atds_csv (str): ATDSスコアCSVファイルのパス（"atds"カラムを持つこと）
        wav_csv (str): 音声ファイル情報CSVファイルのパス（"data"カラムを持つこと）
        top_n (int): 上位何件を抽出するか（デフォルトは16000）

    Returns:
        list of str: 抽出された音声ファイル情報のリスト
    """
    # ATDSスコアの読み込み
    df = pd.read_csv(atds_csv)
    ATDS_dict = df["atds"].to_dict()

    # 音声ファイル情報の読み込み
    wavfile_dict = pd.read_csv(wav_csv).to_dict()["data"]

    # ATDSスコアが高い順にソート
    ATDS_sorted_list = sorted(ATDS_dict.items(), key=lambda x: x[1], reverse=True)

    # 上位top_n件を選択
    ATDS_filtered_list = ATDS_sorted_list[:top_n]
    num_group = [tup[0] for tup in ATDS_filtered_list]

    # 選択された番号に対応する音声ファイル情報を取得
    final_data = [wavfile_dict[num] for num in num_group if num in wavfile_dict]
    return final_data

def format_wav_list(data_list):
    """
    各音声ファイル情報のテキストから、ヘッダー行を除いた
    ファイルパスとフレーム数をタブ区切り形式に整形して返す。

    Parameters:
        data_list (list of str): 音声ファイル情報が格納された文字列のリスト

    Returns:
        str: 整形済みの音声ファイル情報（各行："ファイルパス<tab>フレーム数"）
    """
    formatted_rows = []
    
    for line in data_list:
        # 各テキストを行単位に分割（1行目はヘッダーと想定）
        rows = line.strip().split('\n')
        for row in rows[1:]:  # ヘッダー行をスキップ
            if row.strip():
                parts = row.strip().split()
                # ファイルパスとフレーム数が存在する場合のみ処理
                if len(parts) < 2:
                    continue
                filename = parts[-2].strip(',"')
                num_frames = parts[-1].strip()
                formatted_row = f"{filename}\t{num_frames}"
                if formatted_row not in formatted_rows:
                    formatted_rows.append(formatted_row)
    
    return "\n".join(formatted_rows)

if __name__ == "__main__":
    # 各種変数の管理（パスや上位件数など）
    atds_csv = "/work/result/ATDS_malayalam_21_20000_full.csv"
    wav_csv = "/work/result/malayalam_21_20000_full.csv"
    output_file = "/work/data/manifests/pretrain/malayalam_21_20000to12000_ATDS_wo_scaling.tsv"
    top_n = 12000

    # ATDSのスコアが高い順に音声ファイル情報を抽出
    data_list = sort_atds(atds_csv, wav_csv, top_n)
    formatted_wavfiles = format_wav_list(data_list)
    
    # 結果を出力ファイルに保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_wavfiles)
