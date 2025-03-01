import matplotlib.pyplot as plt
import numpy as np
#import japanize_matplotlib  # This enables Japanese font support

# データ数
x = [0, 4000, 8000, 12000, 16000, 20000]

# CATDS とランダム（RANDOM）の値
catds = [28.13636976,
28.02856758,
28.95836141,
29.20765395,
29.52432287,
29.82751651

]
random1 = [28.13636976,
29.29524323,
28.99878723,
30.00269505,
29.51084759,
29.82751651

]

random2 = [28.13636976,
29.03247541,
29.09311414,
28.93141086,
29.77361542,
29.82751651
]
random3 = [28.13636976,
28.7360194,
28.80339577,
29.31545614,
29.78709069,
29.82751651
]

LID = [28.13636976,
28.30481067,
29.00552486,
29.64560032,
29.32219377,
29.82751651
]

# NumPy 配列にして平均・標準偏差を計算
random_arr = np.array([random1, random2, random3])
random_mean = np.mean(random_arr, axis=0)
random_std  = np.std(random_arr, axis=0)

# グラフの作成
plt.figure(figsize=(6,3))

# CATDS の折れ線
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
plt.xlabel('Number of malayalam clips')
plt.ylabel('WER (%)')
plt.xticks(x, x)

# 凡例を表示
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.savefig('/work/result/malayalam_result.png', dpi=1000, bbox_inches='tight')