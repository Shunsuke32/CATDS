import matplotlib.pyplot as plt

# データ
x = [1, 4, 16, 64, 128]

random_vals = [57.4, 54.8, 49.7, 20.7, 20.4]
deep_svdd_vals = [57.9, 53, 50.7, 21.2, 19.4]
algo1_vals = [58.4, 52.6, 50.6, 21.1, 17.7]

# グラフ作成
plt.figure(figsize=(6,4))
plt.plot(x, random_vals, marker='o', label='Random', )
plt.plot(x, deep_svdd_vals, marker='o', label='Deep SVDD', )
plt.plot(x, algo1_vals, marker='o', label='Algorithm 1', )

# 軸ラベル・凡例など
plt.xlabel('Number of samples')
plt.ylabel('CER (%)')
plt.title('CER comparison (seediq-1h)')
plt.legend()
plt.grid(True)
plt.tight_layout()


# 保存 (上書きに注意)
plt.savefig('/work/result/kizon_seediq_1h.png')
plt.close()
