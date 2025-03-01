import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 20 
plt.rcParams['xtick.labelsize'] = 16
plt.rcParams['ytick.labelsize'] = 16
# ファイルパスの指定（hindiの場合）
atds_file = "/work/result/ATDS_hindi_21sec_20000.csv"
token_file = "/work/result/piece_counts_sums_hindi_21sec_20000.csv"

# CSVファイルの読み込み
df_atds = pd.read_csv(atds_file)
df_token = pd.read_csv(token_file)

# 両ファイルはインデックスが対応している前提でマージする
df = pd.merge(df_atds, df_token, left_index=True, right_index=True, suffixes=('_atds', '_token'))

# 正規化ATDSの計算
def compute_normalized_atds(row):
    x = row["piece_counts_sum"]
    if x > 0:
        # 補正係数の計算
        y = -0.0000007272 * x * x + 0.00109283 * x + 0.23890562
        return row["atds"] / y
    else:
        return None  # piece_counts_sumが0以下の場合はNoneを返す

# 新しいカラムとして正規化ATDSを追加
df["normalized_atds"] = df.apply(compute_normalized_atds, axis=1)

# ----------------------------
# 散布図（正規化前 ATDS）
# ----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["piece_counts_sum"], df["atds"], color="blue", alpha=0.8, s=1)
plt.xlabel("Number of tokens")
plt.ylabel("Unscaled CATDS score")
plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.savefig('/work/result/hindi_original.png')
plt.close()

# ----------------------------
# 散布図（正規化後 ATDS）
# ----------------------------
plt.figure(figsize=(10, 6))
plt.scatter(df["piece_counts_sum"], df["normalized_atds"], color="blue", alpha=0.8, s=1)
plt.xlabel("Number of tokens")
plt.ylabel("Scaled CATDS score")
plt.grid(True, color='gray', linestyle='-', linewidth=0.5, alpha=0.5)
plt.savefig('/work/result/hindi_normalized.png')
plt.close()
