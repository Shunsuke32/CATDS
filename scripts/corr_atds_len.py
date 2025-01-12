import matplotlib
matplotlib.use('Agg')

import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from scipy import stats


def calculate_correlation():
    # ATDS データの読み込み
    df1 = pd.read_csv("/work/result/ATDS_hindi_21_20000_3.csv")
    
    # hindi18sec_6000_data.csv の読み込み
    df2 = pd.read_csv("/work/result/hindi_21sec_20000_train_3.csv")
    
    # 各グループ内のデータフレームを解析して num_frames の合計を計算
    total_frames = []
    
    for _, row in df2.iterrows():
        # 文字列データをデータフレームとして解析
        print(row["data"])
        try:
            group_df = pd.read_csv(StringIO(row['data']), delim_whitespace=True)
            # グループ内の num_frames の合計を計算
            total = group_df['num_frames'].sum()
            total_frames.append(total)
        except Exception as e:
            print(f"Error processing row: {row['index']}")
            print(f"Error: {e}")
            print(f"Data: {row['data'][:100]}...")
            raise
    
    # 結果を新しいデータフレームにまとめる
    result_df = pd.DataFrame({
        'atds': df1['atds'],
        'total_frames': total_frames
    })
    
    # 相関係数の計算
    correlation = result_df['atds'].corr(result_df['total_frames']/16000)
    
    print("\nFirst few rows of data:")
    print(result_df.head())
    
    print("\nBasic statistics:")
    print(result_df.describe())
    
    print("\nCorrelation coefficient:")
    print(correlation)

    plt.figure(figsize=(10, 6))
    
    # 散布図（x軸とy軸を入れ替え）
    plt.scatter(result_df['total_frames']/16000, result_df['atds'], alpha=0.5)
    
    # 回帰直線（x軸とy軸を入れ替え）
    slope, intercept, r_value, p_value, std_err = stats.linregress(result_df['total_frames']/16000, result_df['atds'])
    line = slope * result_df['total_frames']/16000 + intercept
    plt.plot(result_df['total_frames']/16000, line, color='red', label=f'R² = {r_value**2:.3f}')
    
    # グラフの設定（軸ラベルを入れ替え）
    plt.title('Correlation between Total length and ATDS')
    plt.xlabel('Total length(s)')
    plt.ylabel('ATDS')
    plt.grid(True)
    # 回帰直線の式を表示
    equation = f'y = {slope:.6f}x + {intercept:.4f}'
    plt.text(0.05, 0.95, equation, transform=plt.gca().transAxes, 
            bbox=dict(facecolor='white', alpha=0.8))

    # コンソールにも式を出力
    print("\nRegression line equation:")
    print(equation)
    print(f"Slope (coefficient): {slope:.4f}")
    print(f"Intercept: {intercept:.4f}")
    plt.legend()
    
    plt.savefig('/work/result/correlation_plot_hindi21sec20000_3.png')
    plt.close()

if __name__ == "__main__":
    calculate_correlation()
