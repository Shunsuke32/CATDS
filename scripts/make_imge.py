import matplotlib.pyplot as plt
import japanize_matplotlib  # This enables Japanese font support

# ヒンディー語データ数
x = [0, 4000, 8000, 12000, 16000, 20000]

# CATDS とランダム（RANDOM）の値
catds = [28.2163505700545, 26.46836332, 26.21440536, 25.09861134, 24.99594748, 25.48495164]
random = [28.2163505700545, 26.59264062, 26.27114065, 25.27422057, 25.27422057, 25.48495164]

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

plt.savefig('/work/result/hindi_wer_comparison_all.png', dpi=1000, bbox_inches='tight')