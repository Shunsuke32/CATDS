import pandas as pd
import matplotlib.pyplot as plt

# 言語ごとのCSVファイルのパスを指定
atds_files = {
    "hindi": "/work/result/ATDS_hindi_21sec_20000.csv",
    "bengali": "/work/result/ATDS_bengali_21_20000_full.csv",
    "malayalam": "/work/result/ATDS_malayalam_21_20000_full.csv",
}

token_files = {
    "hindi": "/work/result/piece_counts_sums_hindi_21sec_20000.csv",
    "bengali": "/work/result/piece_counts_sums_bengali_21_20000_full.csv",
    "malayalam": "/work/result/piece_counts_sums_malayalam_21_20000_full.csv",
}

# 言語ごとのデータを統合
all_data = []
for lang in atds_files.keys():
    df_atds = pd.read_csv(atds_files[lang])
    df_token = pd.read_csv(token_files[lang])
    
    # マージ前に言語ラベルを追加
    df_merged = pd.merge(df_atds, df_token, left_index=True, right_index=True, suffixes=('_atds', '_token'))
    df_merged["language"] = lang  # マージ後に言語ラベルを追加
    
    # 必要な列のみ保持
    df_merged = df_merged[["atds", "piece_counts_sum", "language"]]
    all_data.append(df_merged)

# すべてのデータを統合
full_df = pd.concat(all_data, ignore_index=True)

# 言語ごとに色を指定
colors = {"hindi": "red", "bengali": "blue", "malayalam": "green"}

# グラフを作成
plt.figure(figsize=(10, 6))
for lang, color in colors.items():
    subset = full_df[full_df["language"] == lang]
    plt.scatter(subset["piece_counts_sum"], subset["atds"], color=color, label=lang, alpha=0.6, s=0.2)

plt.xlabel("Piece Counts Sum")
plt.ylabel("ATDS Score")
plt.title("ATDS Score vs Piece Counts Sum for Different Languages")
plt.legend()
plt.grid(True)

# グラフを保存
plt.savefig('/work/result/hindi_bengali_malayalam_befor.png')
plt.close()  # メモリを解放するためにfigureを閉じる