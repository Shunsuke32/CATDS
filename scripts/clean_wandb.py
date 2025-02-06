import os
import shutil

def remove_wandb_dirs(start_path):
    """
    指定されたパス以下のwandbディレクトリを再帰的に削除します
    
    Args:
        start_path (str): 検索を開始するディレクトリのパス
    """
    # サブディレクトリを含む全てのディレクトリパスを取得
    for root, dirs, _ in os.walk(start_path):
        # dirsリストの中から'wandb'を探す
        if 'wandb' in dirs:
            wandb_path = os.path.join(root, 'wandb')
            try:
                # wandbディレクトリが存在することを確認
                if os.path.exists(wandb_path) and os.path.isdir(wandb_path):
                    print(f"削除を実行します: {wandb_path}")
                    shutil.rmtree(wandb_path)
                    print(f"削除完了: {wandb_path}")
            except Exception as e:
                print(f"削除エラー {wandb_path}: {e}")

if __name__ == "__main__":
    current_dir = os.getcwd()
    print(f"検索開始ディレクトリ: {current_dir}")
    # 実行前の確認
    confirm = input("wandbディレクトリを削除しますか？(y/n): ")
    if confirm.lower() == 'y':
        remove_wandb_dirs(current_dir)
    else:
        print("処理を中止しました")