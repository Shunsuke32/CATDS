import matplotlib.pyplot as plt

# データ
x = [1, 4, 16, 64, 128]

random_vals = [63.7, 60.8, 59.2, 31.1, 30.2]
deep_svdd_vals = [61.9, 58.9, 57.6, 31.4, 28.4]
algo1_vals = [62.4, 59.1, 58.4, 31.3, 27.2]

# グラフ作成
plt.figure(figsize=(6,4))
plt.plot(x, random_vals, marker='o', label='Random', )
plt.plot(x, deep_svdd_vals, marker='o', label='Deep SVDD', )
plt.plot(x, algo1_vals, marker='o', label='Algorithm 1', )

# 軸ラベル・凡例など
plt.xlabel('Number of samples')
plt.ylabel('CER (%)')
plt.title('CER comparison (seediq-10min)')
plt.legend()
plt.grid(True)
plt.tight_layout()


# 保存 (上書きに注意)
plt.savefig('/work/result/kizon_seediq_10min.png')
plt.close()
