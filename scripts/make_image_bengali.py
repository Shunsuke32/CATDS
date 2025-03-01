import matplotlib.pyplot as plt
import numpy as np
#import japanize_matplotlib  # This enables Japanese font support

# データ数
x = [0, 4000, 8000, 12000, 16000, 20000]

# CATDS とランダム（RANDOM）の値
catds = [28.13636976,
28.02182994,
28.65516777,
28.13636976,
28.27112249,
29.0863765

]
random1 = [28.13636976,
28.40587522,
29.38957014,
28.06225576,
29.12680232,
29.0863765

]

random2 = [28.13636976,
28.98531195,
28.45977631,
28.4395634,
28.88424741,
29.0863765

]
random3 = [28.13636976,
27.91402776,
29.1604905,
28.7360194,
28.91119795,
29.0863765
]

LID = [28.13636976,
28.0689934,
28.23069667,
28.23069667,
28.28459776,
29.0863765
]
# NumPy 配列にして平均・標準偏差を計算
random_arr = np.array([random1, random2, random3])
random_mean = np.mean(random_arr, axis=0)
random_std  = np.std(random_arr, axis=0)

# グラフの作成
plt.figure(figsize=(6,3))

# CATDS の折れ線
plt.plot(x, catds,marker='^', label='CATDS')
plt.plot(x, LID, marker="s", label="LID")
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
plt.xlabel('Number of bengali clips')
plt.ylabel('WER (%)')
plt.xticks(x, x)

# 凡例を表示
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.savefig('/work/result/bengali_result.png', dpi=1000, bbox_inches='tight')