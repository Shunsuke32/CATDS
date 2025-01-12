import pandas as pd
import re

def extract_numbers_and_get_elements(file_content, target_list):
    # CSVデータを文字列からDataFrameに変換
    df = pd.read_csv(StringIO(file_content))
    
    # 最初の3000行を取得
    df = df.head(3000)
    
    # donor_langから数字を抽出する関数
    def extract_number(text):
        match = re.search(r'clustered(\d+)', text)
        if match:
            return int(match.group(1))
        return None
    
    # 数字を抽出
    numbers = df['donor_lang'].apply(extract_number)
    
    # 有効な数字のみを取得（Noneを除外）
    valid_numbers = [num for num in numbers if num is not None]
    
    # target_listから対応する要素を取得
    result = []
    for num in valid_numbers:
        if 0 <= num < len(target_list):
            result.append(target_list[num])
    
    return result

# 使用例：
from io import StringIO

# サンプルのtarget_list（実際のリストに置き換えてください）
target_list = ["element_0", "element_1", "element_2", ..., "element_999"]

# ファイルの内容を渡して処理を実行
result = extract_numbers_and_get_elements(file_content, target_list)
print(f"抽出された要素の数: {len(result)}")
print("最初の10個の要素:", result[:10])