import matplotlib.pyplot as plt

# データ
x = [1, 4, 16, 64, 128]

random_vals = [57.9, 55.2, 56.8, 22.7, 20.6]
deep_svdd_vals = [56.3, 55.5, 53.3, 20.8, 18.7]
algo1_vals = [58.5, 54.6, 37.8, 20.6, 21.1]

# グラフ作成
plt.figure(figsize=(6,4))
plt.plot(x, random_vals, marker='o', label='Random')
plt.plot(x, deep_svdd_vals, marker='o', label='Deep SVDD')
plt.plot(x, algo1_vals, marker='o', label='Algorithm 1')

# 軸ラベル・凡例など
plt.xlabel('Number of samples')
plt.ylabel('CER (%)')
plt.title('CER comparison')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存 (上書きに注意してください)
plt.savefig('/work/result/kizon1.png')
plt.close()
