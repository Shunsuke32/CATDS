import os
import shutil
from pathlib import Path

def copy_directory_complete(src_dir, dst_dir):
    """
    ディレクトリの内容（ファイルとフォルダ構造）を完全にコピーします。
    
    Parameters:
        src_dir (str): ソースディレクトリのパス
        dst_dir (str): 宛先ディレクトリのパス
    """
    try:
        # 宛先ディレクトリが既に存在する場合は削除
        if os.path.exists(dst_dir):
            shutil.rmtree(dst_dir)
        
        # ディレクトリ全体をコピー
        shutil.copytree(
            src_dir, 
            dst_dir, 
            symlinks=True,  # シンボリックリンクもコピー
            dirs_exist_ok=True  # 既存ディレクトリがあってもエラーにしない
        )
        return True
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        return False

# 使用例
if __name__ == '__main__':
    source_dir = "/home/mitsumori/w2v2-cpt-transfer/outputs"
    destination_dir = "/data3/mitsumori/outputs"
    
    if copy_directory_complete(source_dir, destination_dir):
        print(f"コピーが完了しました。")
        print(f"コピー元: {source_dir}")
        print(f"コピー先: {destination_dir}")
    else:
        print("コピーに失敗しました。")