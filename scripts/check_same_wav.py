def extract_wav_names(tsv_file):
    """
    TSVファイルから、wavファイル名（各行の最初の列）を抽出してセットで返す。
    ※ 1行目がディレクトリパスなどタブ区切りでない場合は無視します。
    """
    wav_names = set()
    with open(tsv_file, "r", encoding="utf-8") as f:
        for line in f:
            # タブ区切り行であれば、wavファイル情報が記録されているとみなす
            if "\t" in line:
                parts = line.strip().split("\t")
                if parts:
                    wav_names.add(parts[0])
    return wav_names

if __name__ == "__main__":
    tsv_file1 = "/work/data/manifests/pretrain/hindi_train_21sec20000_filterto4000_atds_token_v2.tsv"
    tsv_file2 = "/work/data/manifests/pretrain/hindi_21_20000to4000_rank.tsv"

    # それぞれのTSVファイルからwavファイル名を抽出
    wav_set1 = extract_wav_names(tsv_file1)
    wav_set2 = extract_wav_names(tsv_file2)

    # 2つのTSV間で共通しているwavファイル名の集合を求める
    common_wavs = wav_set1.intersection(wav_set2)
    print(f"共通するwavファイルの数: {len(common_wavs)}")
