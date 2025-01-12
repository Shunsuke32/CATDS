import pandas as pd
import matplotlib.pyplot as plt
import japanize_matplotlib
def normalize_and_plot_atds():
    """
    ATDSをトークン数に基づいて正規化し、
    (トークン数 vs. 正規化後のATDS) の散布図を保存する。
    ソートや上位○件の抽出は行わない。
    """
    # ATDSスコアの読み込み
    df = pd.read_csv("/work/result/ATDS_hindi_21sec_20000.csv")
    d = df.to_dict()
    ATDS_dict = d["atds"]
    
    # token数の合計値取得
    df2 = pd.read_csv("/work/result/piece_counts_sums_hindi_21sec_20000.csv")
    d2 = df2.to_dict()
    token_dict = d2["piece_counts_sum"]

    # 散布図描画用リスト
    x_list = []  # token数
    y_list = []  # 正規化後のATDS

    # 正規化計算
    normalized_atds = {}
    for idx, atds in ATDS_dict.items():
        # token_dict が存在し、かつ 0より大きい場合のみ
        if idx in token_dict and token_dict[idx] > 0:
            x = token_dict[idx]
            # 正規化用の計算式
            # y = -7.272e-07 * x * x + 0.00109283 * x + 0.23890562
            # normalized_value = atds / y
            
            normalized_atds[idx] = atds

            # 散布図用
            x_list.append(x)
            y_list.append(atds)

    # 散布図を作成して保存
    plt.figure(figsize=(6, 4))
    plt.scatter(x_list, y_list, s=5, alpha=0.5)
    plt.xlabel("Token数")
    plt.ylabel("CATDS (補正前)")
    plt.grid(True, color='gray', linewidth=0.5, alpha=0.2)
    plt.tight_layout()
    plt.savefig("/work/result/scatter_token_vs_正規化前類似度.png")
    plt.close()

    # ここではソートや上位○件の抽出はせず、正規化結果だけを返す
    return normalized_atds

if __name__ == "__main__":
    # 関数を呼び出して正規化＋プロットのみを行う
    atds_data = normalize_and_plot_atds()
    
    # atds_data には {index: 正規化後ATDS} が格納されている
    # ソートやフィルタリングを行わない場合は、そのまま使うことが可能です。
    # 例：print(atds_data)
    # print(atds_data)
