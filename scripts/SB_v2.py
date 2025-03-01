import torch
from speechbrain.inference.classifiers import EncoderClassifier

# モデルのロード
language_id = EncoderClassifier.from_hparams(
    source="speechbrain/lang-id-voxlingua107-ecapa",
    savedir="tmp"
)

# お好きな音声ファイルをロード
audio_path1 = "/work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/valid/audio/844424933598053-1073-m.wav"
audio_path2 = "/work/data/IndicSUPERB/kb_data_clean_m4a/punjabi/valid/audio/844424932932755-180-f.wav"

signal1 = language_id.load_audio(audio_path1)
signal2 = language_id.load_audio(audio_path2)

# 両方の信号が同じサンプルレートであることを確認してください
# 時間軸（最後の次元）に沿って連結する例
combined_signal = torch.cat([signal1, signal2], dim=-1)

# 連結したオーディオで推論を実施
prediction = language_id.classify_batch(combined_signal)
scores_tensor, score_top, index_top, predicted_label = prediction

# 全ラベル一覧を取得（VoxLingua107は107言語）
num_langs = 107  # または、内部の _tok2index を使う場合: len(language_id.hparams.label_encoder._tok2index)
all_ids = torch.arange(num_langs)
all_labels = language_id.hparams.label_encoder.decode_ndim(all_ids)
print("ラベル一覧:", all_labels)

# Panjabiのラベルは一覧中 "pa: Panjabi" となっているので、そのインデックスを取得
panjabi_label = "pa: Panjabi"
if panjabi_label not in all_labels:
    raise ValueError(f"{panjabi_label} がラベル一覧に見つかりません。")
idx_panjabi = all_labels.index(panjabi_label)
print("Panjabiのインデックス:", idx_panjabi)

# scores_tensor は (1, 107) の形状になっているので、1次元に変換
scores_1d = scores_tensor.squeeze(0)

# Panjabi のログ確率および線形確率を取得
panjabi_log_prob = scores_1d[idx_panjabi]
panjabi_prob = panjabi_log_prob.exp()  # exp() で線形スケールに変換

print("Panjabi log-prob:", panjabi_log_prob.item())
print("Panjabi prob:", panjabi_prob.item())
