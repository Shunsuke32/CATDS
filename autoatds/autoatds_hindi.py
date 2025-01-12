import subprocess
import pandas as pd
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def run_command(command):
    subprocess.run(command, shell=True, check=True)

def calculate_atds(donor_parquet, target_parquet):
    # ATDS.py スクリプトを呼び出し、その出力を取得します
    result = subprocess.run(
        [
            "python", "analyses/ATDS.py",
            "--langs_dir", "tmp",
            "--target_lang", target_parquet.split("-")[0],
            "--donor_lang", donor_parquet.split("-")[0]
        ],
        capture_output=True,
        text=True,
        check=True
    )
    
    # 出力を解析して ATDS 値のみを取得します
    output_lines = result.stdout.strip().split("\n")
    atds = float(output_lines[-2].split(":")[1].strip())  # ATDS値は最後から2行目にあると仮定
    
    return atds

def run_experiment(i):
    # 埋め込みを抽出します
    run_command(f"python scripts/extract_embeddings.py --checkpoint-path checkpoints/xlsr2_300m.pt --wav-dir data/IndicSUPERB/kb_data_clean_m4a/hindi/train/audio --num-hours 0.02 --output-parquet tmp/hindi_{i}.parquet")

    run_command(f"python scripts/infer_k-means.py tmp/k-means_punjabi.joblib tmp/hindi_{i}.parquet tmp/hindi-clustered_{i}.parquet")

    # 既存の ATDS.py スクリプトを使用して ATDS を計算します
    atds = calculate_atds(f"punjabi-clustered.parquet", f"hindi-clustered_{i}.parquet")
    
    return atds

def main():
    atds_results_hindi = []
    for i in tqdm(range(1000)):
        atds = run_experiment(i)
        atds_results_hindi.append(atds)
    
    # 結果を DataFrame に変換します
    df = pd.DataFrame(atds_results_hindi, columns=['ATDS'])
    
    # 結果を保存します
    df.to_csv('atds_results_hindi.csv', index=False)
    
    # ATDS の分布をプロットします
    plt.figure(figsize=(10, 6))
    plt.hist(df['ATDS'], bins=50)
    plt.title('Distribution of ATDS values')
    plt.xlabel('ATDS')
    plt.ylabel('Frequency')
    plt.savefig('atds_distribution_hindi.png')
    plt.close()

    # ATDS の基本統計量を表示します
    print("ATDS Statistics:")
    print(df['ATDS'].describe())

if __name__ == "__main__":
    main()