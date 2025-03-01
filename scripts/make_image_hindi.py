import matplotlib.pyplot as plt
import numpy as np
#import japanize_matplotlib  # This enables Japanese font support

# データ数
x = [0, 4000, 8000, 12000, 16000, 20000]

# CATDS とランダム（RANDOM）の値
catds = [28.13636976,
26.74167902,
26.60018865,
25.19876027,
25.259399,
25.79840992
]
random1 = [28.13636976,
26.99097157,
26.52607465,
25.67039483,
25.51542919,
25.79840992
]

random2 = [28.13636976,
27.28742757,
26.72146611,
26.09486592,
25.9062121,
25.79840992
]
random3 = [28.13636976,
26.57997574,
27.44913084,
25.91294974,
25.26613664,
25.79840992
]

LID = [28.13636976,
25.93990028,
27.06508557,
26.11507883,
25.73777119,
25.79840992
]

# NumPy 配列にして平均・標準偏差を計算
random_arr = np.array([random1, random2, random3])
random_mean = np.mean(random_arr, axis=0)
random_std  = np.std(random_arr, axis=0)

# グラフの作成
plt.figure(figsize=(6,3))

# CATDSとLID の折れ線
plt.plot(x, catds,marker='^', label='CATDS')
plt.plot(x, LID ,marker='s', label='LID')
# random の折れ線を描画し、戻り値（ラインオブジェクト）を取得
random_line, = plt.plot(x, random_mean, marker='o', label='random')

# random の標準偏差を塗りつぶしで可視化
plt.fill_between(
    x,
    random_mean - random_std,
    random_mean + random_std,
    alpha=0.1,   # 塗りつぶしの透明度
    linewidth=0,
    color=random_line.get_color()  # ←ここでラインと同じ色を指定
)

# 軸ラベルなどの設定
plt.xlabel('Number of hindi clips')
plt.ylabel('WER (%)')
plt.xticks(x, x)

# 凡例を表示
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.savefig('/work/result/hindi_result.png', dpi=1000, bbox_inches='tight')