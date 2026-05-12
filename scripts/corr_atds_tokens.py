#!/usr/bin/env python3
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fit quadratic bias curve between piece_counts_sum and raw CATDS score."
    )
    parser.add_argument("--atds-csv", required=True, help="CSV from atds_token.py (must include atds + group_id)")
    parser.add_argument("--counts-csv", required=True, help="CSV from atds_token.py (must include piece_counts_sum + group_id)")
    parser.add_argument("--output-plot", default=None, help="Optional path to save scatter + regression plot")
    parser.add_argument("--output-coef-csv", default=None, help="Optional path to save fitted coefficients")
    return parser.parse_args()


def analyze_correlation(atds_file, counts_file):
    atds_df = pd.read_csv(atds_file)
    counts_df = pd.read_csv(counts_file)
    if "group_id" not in atds_df.columns or "group_id" not in counts_df.columns:
        raise ValueError("Both input CSVs must contain group_id column.")

    merged_df = atds_df.merge(counts_df, on="group_id", how="inner").dropna(
        subset=["atds", "piece_counts_sum"]
    )
    if merged_df.empty:
        raise ValueError("No valid rows after merge/dropna.")

    pearson_corr = stats.pearsonr(merged_df["piece_counts_sum"], merged_df["atds"])
    spearman_corr = stats.spearmanr(merged_df["piece_counts_sum"], merged_df["atds"])

    X = merged_df["piece_counts_sum"].values.reshape(-1, 1)
    y = merged_df["atds"].values

    linear_reg = LinearRegression()
    linear_reg.fit(X, y)
    linear_pred = linear_reg.predict(X)

    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_poly = poly.fit_transform(X)
    poly_reg = LinearRegression(fit_intercept=True)
    poly_reg.fit(X_poly, y)
    poly_pred = poly_reg.predict(X_poly)

    a = float(poly_reg.coef_[1])
    b = float(poly_reg.coef_[0])
    c = float(poly_reg.intercept_)

    metrics = {
        "pearson_r": float(pearson_corr[0]),
        "pearson_p": float(pearson_corr[1]),
        "spearman_r": float(spearman_corr[0]),
        "spearman_p": float(spearman_corr[1]),
        "linear_r2": float(linear_reg.score(X, y)),
        "poly_r2": float(poly_reg.score(X_poly, y)),
        "coef_a": a,
        "coef_b": b,
        "coef_c": c,
    }
    return merged_df, linear_pred, poly_pred, metrics


def save_plot(merged_df, linear_pred, poly_pred, output_plot):
    X = merged_df["piece_counts_sum"].values.reshape(-1, 1)
    y = merged_df["atds"].values
    sort_idx = np.argsort(X.flatten())
    X_sorted = X[sort_idx]

    plt.figure(figsize=(12, 8))
    plt.scatter(X, y, alpha=0.5, label="Data points")
    plt.plot(X_sorted, linear_pred[sort_idx], "r--", label="Linear regression", alpha=0.8)
    plt.plot(X_sorted, poly_pred[sort_idx], "g--", label="Polynomial regression", alpha=0.8)
    plt.ylabel("Raw CATDS score")
    plt.xlabel("Piece Counts Sum")
    plt.title("Piece Counts Sum vs Raw CATDS Score")
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(output_plot).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_plot)
    plt.close()


def main():
    args = parse_args()
    merged_df, linear_pred, poly_pred, metrics = analyze_correlation(
        args.atds_csv, args.counts_csv
    )

    print("Correlation Analysis Results:")
    print(f"Pearson r={metrics['pearson_r']:.4f} (p={metrics['pearson_p']:.4g})")
    print(f"Spearman r={metrics['spearman_r']:.4f} (p={metrics['spearman_p']:.4g})")
    print(f"Linear R^2={metrics['linear_r2']:.4f}")
    print(f"Polynomial R^2={metrics['poly_r2']:.4f}")
    print(
        "Quadratic equation for correction denominator:\n"
        f"y = {metrics['coef_a']:.12f} * x^2 + {metrics['coef_b']:.12f} * x + {metrics['coef_c']:.12f}"
    )
    print(
        "Use these values in sort_by_atds_token.py as:\n"
        f"--coef-a {metrics['coef_a']} --coef-b {metrics['coef_b']} --coef-c {metrics['coef_c']}"
    )

    if args.output_plot:
        save_plot(merged_df, linear_pred, poly_pred, args.output_plot)
        print(f"Saved plot: {args.output_plot}")

    if args.output_coef_csv:
        Path(args.output_coef_csv).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame([metrics]).to_csv(args.output_coef_csv, index=False)
        print(f"Saved coefficients: {args.output_coef_csv}")


if __name__ == "__main__":
    main()
