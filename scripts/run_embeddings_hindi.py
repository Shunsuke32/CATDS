import subprocess

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f"Error executing command: {command}")
        print(f"Error message: {stderr.decode('utf-8')}")
    else:
        print(f"Command executed successfully: {command}")

# 1000回の繰り返しを行います
for i in range(1000):
    # 埋め込みを抽出します
    extract_command = f"python scripts/extract_embeddings.py --checkpoint-path checkpoints/xlsr2_300m.pt --wav-dir data/IndicSUPERB/kb_data_clean_m4a/hindi/train/audio --num-hours 0.5 --output-parquet tmp/hindi{i}_.parquet"
    run_command(extract_command)

    # K-means推論を実行します
    infer_command = f"python scripts/infer_k-means.py tmp/k-means_punjabi.joblib tmp/hindi{i}_.parquet tmp/hindi-clustered{i}_.parquet"
    run_command(infer_command)

    print(f"Completed iteration {i+1} of 1000")

print("All iterations completed.")