import matplotlib.pyplot as plt
import japanize_matplotlib  # This enables Japanese font support

# ヒンディー語データ数
x = [0, 4000, 8000, 12000, 16000, 20000]

# CATDS とランダム（RANDOM）の値
catds = [28.13637, 26.74168, 26.60019, 25.19876, 25.2594, 25.79841]
random = [28.13637, 26.99097, 26.52607, 25.67039, 25.51543, 25.79841]

# グラフの作成
plt.figure(figsize=(6,3))

# CATDS の折れ線
plt.plot(x, catds,marker='^', label='CATDS')

# RANDOM の折れ線
plt.plot(x, random,marker='o', label='ランダム')

# 軸ラベルなどの設定
plt.xlabel('ヒンディ語データ数')
plt.ylabel('WER (%)')
plt.xticks(x, x)

# 凡例を表示
handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1])

plt.savefig('/work/result/wer_comparison.png', dpi=1000, bbox_inches='tight')