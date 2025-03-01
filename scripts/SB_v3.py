import os
import pandas as pd
import torch
from speechbrain.inference.classifiers import EncoderClassifier

# ① 音声ファイルが保存されているディレクトリ
base_dir = "/work/data/IndicSUPERB/kb_data_clean_m4a/malayalam/train/audio"

# ② CSVファイルのパス（各行の "data" 列に複数のファイル情報が記載されている前提）
csv_path = "/work/result/malayalam_21_20000_full.csv"  # ご自身のCSVファイルのパスに変更してください

# ③ CSVを読み込む
df = pd.read_csv(csv_path)

# ④ SpeechBrainの言語識別モデルをロード
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp",
    run_opts={"device": "cuda"}
)

# ⑤ パンジャビ語のラベルのインデックスを取得
num_langs = 107  # VoxLingua107は107言語対応
all_ids = torch.arange(num_langs)
all_labels = language_id.hparams.label_encoder.decode_ndim(all_ids)
# ラベル一覧例: "pa: Panjabi" を利用
panjabi_label = "pa: Panjabi"
if panjabi_label not in all_labels:
    raise ValueError(f"{panjabi_label} がラベル一覧に見つかりません。")
idx_panjabi = all_labels.index(panjabi_label)
print("Panjabiのインデックス:", idx_panjabi)

# ⑥ 各インデックスのデータに対して、複数の音声ファイルを連結しパンジャビ語確率を算出
results = []
for idx, row in df.iterrows():
    data_str = row["data"]
    # 改行ごとに分割（ヘッダー行などwavが含まれない行はスキップ）
    lines = data_str.strip().splitlines()
    
    combined_signal = None
    for line in lines:
        if "wav" not in line:
            continue
        tokens = line.strip().split()
        if len(tokens) < 2:
            continue
        filename = tokens[1]
        file_path = os.path.join(base_dir, filename)
        try:
            signal = language_id.load_audio(file_path)
        except Exception as e:
            print(f"{file_path} の読み込みエラー: {e}")
            continue
        # 信号を連結（同じサンプルレートであることが前提）
        if combined_signal is None:
            combined_signal = signal
        else:
            combined_signal = torch.cat([combined_signal, signal], dim=-1)
    
    if combined_signal is not None:
        prediction = language_id.classify_batch(combined_signal)
        scores_tensor = prediction[0]  # shape: (1, 107)
        scores_1d = scores_tensor.squeeze(0)
        panjabi_log_prob = scores_1d[idx_panjabi]
        panjabi_prob = panjabi_log_prob.exp().item()  # 線形スケールに変換
    else:
        panjabi_prob = None
    results.append(panjabi_prob)
    print(f"インデックス {idx} の Panjabi 確率: {panjabi_prob}")

# ⑦ 結果のみの新しいDataFrameを作成（列名を "SB" とする）
df_SB = pd.DataFrame({'SB': results})

# ⑧ CSVとして保存（行番号を含める形式の場合）
output_csv = "result/SB_malayalam.csv"
df_SB.to_csv(output_csv, index=True)
print("新しいCSVファイルを保存しました:", output_csv)
