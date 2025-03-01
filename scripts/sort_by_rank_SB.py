import pandas as pd

def sort_SB(SB_csv, rank_csv, wav_csv, top_n):
    """
    SBスコアCSVとrankスコアCSV、及び音声ファイル情報CSVを読み込み、
    rankスコアを第一キー、同一rank内ではSBスコアを第二キーとしてソートし、
    上位top_n件の音声ファイル情報を抽出して返す。

    Parameters:
        SB_csv (str): SBスコアCSVのパス（"SB"カラムを持つこと）
        rank_csv (str): rankスコアCSVのパス（"rank"カラムを持つこと）
        wav_csv (str): 音声ファイル情報CSVのパス（"data"カラムを持つこと）
        top_n (int): 上位何件を抽出するか（例: 4000）

    Returns:
        list of str: 抽出された音声ファイル情報のリスト
    """
    # SBスコアCSVとrankスコアCSVの読み込み（行番号がキーとして共通である前提）
    df_sb = pd.read_csv(SB_csv, index_col=0)
    df_rank = pd.read_csv(rank_csv, index_col=0)
    
    # 両DataFrameを横方向に結合
    df_combined = pd.concat([df_sb, df_rank], axis=1)
    if "SB" not in df_combined.columns or "rank" not in df_combined.columns:
        raise ValueError("SBまたはrankカラムが結合後のDataFrameに見つかりません。")
    
    # ソート: 第一キーは rank (昇順：数値が小さいほど良い)、第二キーは SB (降順：数値が大きいほど良い)
    df_sorted = df_combined.sort_values(by=["rank", "SB"], ascending=[True, False])
    
    # 上位top_n件のインデックスを取得
    top_indices = df_sorted.head(top_n).index.tolist()
    
    # 音声ファイル情報CSVの読み込み
    wav_df = pd.read_csv(wav_csv, index_col=0)
    if "data" not in wav_df.columns:
        raise ValueError("wav_csvに'data'カラムが見つかりません。")
    
    # 選択されたインデックスに対応する音声ファイル情報を取得
    final_data = [wav_df.loc[idx, "data"] for idx in top_indices if idx in wav_df.index]
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
    
    for text in data_list:
        # 各テキストを行単位に分割（1行目はヘッダーと想定）
        rows = text.strip().split('\n')
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
    SB_csv = "/work/result/SB_hindi.csv"            # "SB"カラムを持つCSV
    rank_csv = "/work/result/rank_hindi.csv"          # "rank"カラムを持つCSV
    wav_csv = "/work/result/hindi_21sec_20000_train.csv"   # "data"カラムを持つ音声ファイル情報CSV
    output_file = "/work/data/manifests/pretrain/hindi_21_20000to12000_rank.tsv"
    top_n = 12000

    # rankスコアを優先し、同一rank内はSBスコアでソートして上位エントリを抽出
    data_list = sort_SB(SB_csv, rank_csv, wav_csv, top_n)
    formatted_wavfiles = format_wav_list(data_list)
    
    # 結果をTSVファイルに保存
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(formatted_wavfiles)
