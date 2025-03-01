import matplotlib.pyplot as plt

# データ
x = [1, 4, 16, 64, 128]

random_vals = [52.1, 46.8, 52.1, 16.4, 13.3]
deep_svdd_vals = [50.1, 45.4, 42.0, 14.1, 12.5]
algo1_vals = [52.4, 44.1, 29.9, 14.5, 14.3]

# グラフ作成
plt.figure(figsize=(6,4))
plt.plot(x, random_vals, marker='o', label='Random', )
plt.plot(x, deep_svdd_vals, marker='o', label='Deep SVDD', )
plt.plot(x, algo1_vals, marker='o', label='Algorithm 1', )

# 軸ラベル・凡例など
plt.xlabel('Number of samples')
plt.ylabel('CER (%)')
plt.title('CER comparison (1hr)')
plt.legend()
plt.grid(True)
plt.tight_layout()

# 保存 (上書きに注意)
plt.savefig('/work/result/kizon_amis_1h.png')
plt.close()
