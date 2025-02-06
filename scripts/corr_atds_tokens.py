import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

def analyze_correlation(atds_file, counts_file):
   # Load the data with error handling
   try:
       atds_df = pd.read_csv(atds_file, index_col=0)
       counts_df = pd.read_csv(counts_file, index_col=0)
   except FileNotFoundError as e:
       print(f"ファイルが見つかりません: {e}")
       return
   except pd.errors.EmptyDataError as e:
       print(f"ファイルが空です: {e}")
       return
   except Exception as e:
       print(f"データの読み込み中にエラーが発生しました: {e}")
       return
   
   # Merge the dataframes
   merged_df = pd.concat([atds_df, counts_df], axis=1)
   
   # Handle missing values
   if merged_df.isnull().values.any():
       print("データに欠損値が含まれています。欠損値を削除します。")
       merged_df = merged_df.dropna()
   
   # Calculate correlations - Note: Order switched for consistency with plot
   pearson_corr = stats.pearsonr(merged_df['piece_counts_sum'], merged_df['atds'])
   spearman_corr = stats.spearmanr(merged_df['piece_counts_sum'], merged_df['atds'])
   
   print("Correlation Analysis Results:")
   print(f"Pearson correlation coefficient: {pearson_corr[0]:.4f} (p-value: {pearson_corr[1]:.4f})")
   print(f"Spearman correlation coefficient: {spearman_corr[0]:.4f} (p-value: {spearman_corr[1]:.4f})")
   
   # Prepare data for regression - Switched X and y
   X = merged_df['piece_counts_sum'].values.reshape(-1, 1)
   y = merged_df['atds'].values
   
   # Linear regression
   linear_reg = LinearRegression()
   linear_reg.fit(X, y)
   linear_pred = linear_reg.predict(X)
   
   # Polynomial regression (degree=2) with debug information
   poly = PolynomialFeatures(degree=2, include_bias=False)  # include_bias=False に設定
   X_poly = poly.fit_transform(X)
   
   # デバッグ情報の表示
   print("\nPolynomialFeatures変換の確認:")
   print("X_poly shape:", X_poly.shape)
   print("最初の行のデータ:", X_poly[0])
   
   poly_reg = LinearRegression(fit_intercept=True)  # fit_intercept=True のまま
   poly_reg.fit(X_poly, y)
   poly_pred = poly_reg.predict(X_poly)
   
   print("係数:", poly_reg.coef_)
   print("切片:", poly_reg.intercept_)
   
   # Print regression equations
   print("\nRegression Equations:")
   print(f"Linear: y = {linear_reg.coef_[0]:.8f}x + {linear_reg.intercept_:.4f}")
   print(f"R² (linear): {linear_reg.score(X, y):.4f}")
   
   # 回帰係数の順序を確認
   # poly_reg.coef_ は [x, x^2] の順序であると仮定
   print(f"Polynomial: y = {poly_reg.coef_[1]:.10f}x² + {poly_reg.coef_[0]:.8f}x + {poly_reg.intercept_:.8f}")
   print(f"R² (polynomial): {poly_reg.score(X_poly, y):.4f}")
   
   # Create scatter plot with both regression lines
   plt.figure(figsize=(12, 8))
   
   # Scatter plot - X and y switched
   plt.scatter(X, y, alpha=0.5, label='Data points')
   
   # Sort X for smooth line plotting
   sort_idx = np.argsort(X.flatten())
   X_sorted = X[sort_idx]
   
   # Plot regression lines
   plt.plot(X_sorted, linear_pred[sort_idx], 'r--', label='Linear regression', alpha=0.8)
   plt.plot(X_sorted, poly_pred[sort_idx], 'g--', label='Polynomial regression', alpha=0.8)
   
   # Switched labels
   plt.ylabel('ATDS Score')
   plt.xlabel('Piece Counts Sum')
   plt.title('Piece Counts Sum vs ATDS Score with Regression Lines')
   plt.legend()
   plt.grid(True, alpha=0.3)
   plt.savefig('/work/result/punjabi_21_20000_full.png')
   plt.close()
   
   # Analyze ATDS vs tokens relationship
   token_analysis = pd.DataFrame({
       'ATDS': merged_df['atds'],
       'Tokens': merged_df['piece_counts_sum']
   }).sort_values('Tokens')  # Sort by Tokens instead of ATDS
   
   # Calculate statistics by token count quartile
   token_stats = token_analysis.groupby(pd.qcut(token_analysis['Tokens'], 4))['ATDS'].agg([
       'count',
       'mean',
       'std',
       'min',
       'max'
   ]).round(2)
   
   print("\nTokens vs ATDS Analysis by Quartile:")
   print(token_stats)
   
   # 検証用のコードを追加
   def verify_polynomial_match(x_value, coef, intercept):
       # 式から計算
       y_equation = coef[1] * x_value**2 + coef[0] * x_value + intercept
       
       # モデルから予測
       x_single = np.array([[x_value, x_value**2]])
       y_model = poly_reg.predict(x_single)
       
       print(f"X値: {x_value:.6f}")
       print(f"式からの計算値: {y_equation:.6f}")
       print(f"モデルからの予測値: {y_model[0]:.6f}")
       print(f"差分: {abs(y_equation - y_model[0]):.10f}")
       print("-" * 50)
   
   print("\n回帰式とプロットの一致検証:")
   # いくつかの点でテスト
   test_points = [float(X.min()), float(X.mean()), float(X.max())]
   for x in test_points:
       verify_polynomial_match(x, poly_reg.coef_, poly_reg.intercept_)
   
   return merged_df, token_stats

# Run the analysis
atds_file = '/work/result/ATDS_odia_21_19000_full.csv'
counts_file = '/work/result/piece_counts_sums_odia_21_19000_full.csv'
result_df, token_stats = analyze_correlation(atds_file, counts_file)

# Display full correlation matrix
print("\nFull correlation matrix:")
print(result_df.corr())
