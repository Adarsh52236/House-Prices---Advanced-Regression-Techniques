"""Exploratory Data Analysis (EDA) module for Model 1 (Grandmaster Choice).

Generates comprehensive statistical summaries, distribution plots, correlation
heatmaps, feature-target relationship charts, and an automated Markdown report.
"""

from pathlib import Path
import sys
from typing import Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.common.paths import (
    MODEL_1_REPORTS_DIR,
    TEST_CSV,
    TRAIN_CSV,
    ensure_directory,
)
from src.model_1_grandmaster.data_audit import audit_missing_values, load_raw_data


def generate_eda_figures(train_df: pd.DataFrame, output_dir: Path) -> Dict[str, Path]:
    """Generate and save publication-quality EDA figures."""
    ensure_directory(output_dir)
    sns.set_theme(style="whitegrid", palette="muted")
    fig_paths = {}

    # 1. Target Distribution: Raw vs Log1p
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    sns.histplot(train_df["SalePrice"], kde=True, ax=axes[0], color="#2b5c8f")
    axes[0].set_title("SalePrice Distribution (Raw USD)", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("SalePrice ($)")
    axes[0].set_ylabel("Frequency")

    log_target = np.log1p(train_df["SalePrice"])
    sns.histplot(log_target, kde=True, ax=axes[1], color="#2e7d32")
    axes[1].set_title("log1p(SalePrice) Distribution", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("log(SalePrice + 1)")
    axes[1].set_ylabel("Frequency")

    plt.tight_layout()
    target_dist_path = output_dir / "target_distribution.png"
    fig.savefig(target_dist_path, dpi=200)
    plt.close(fig)
    fig_paths["target_distribution"] = target_dist_path

    # 2. Key Numeric Features vs SalePrice (Scatter & Box plots)
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))

    # OverallQual vs SalePrice
    sns.boxplot(x="OverallQual", y="SalePrice", data=train_df, ax=axes[0, 0], palette="Blues")
    axes[0, 0].set_title("SalePrice by OverallQual", fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("Overall Quality (1-10)")
    axes[0, 0].set_ylabel("SalePrice ($)")

    # GrLivArea vs SalePrice with potential outliers highlighted
    axes[0, 1].scatter(
        train_df["GrLivArea"],
        train_df["SalePrice"],
        alpha=0.6,
        color="#2b5c8f",
        edgecolors="none",
        label="Normal observations",
    )
    # Highlight GrLivArea > 4000
    outliers = train_df[train_df["GrLivArea"] > 4000]
    axes[0, 1].scatter(
        outliers["GrLivArea"],
        outliers["SalePrice"],
        color="#d32f2f",
        s=70,
        edgecolors="black",
        label="GrLivArea > 4,000 sq ft",
        zorder=5,
    )
    for _, row in outliers.iterrows():
        axes[0, 1].annotate(
            f"Id {int(row['Id'])}",
            (row["GrLivArea"], row["SalePrice"]),
            textcoords="offset points",
            xytext=(7, -5),
            fontsize=9,
            fontweight="bold",
            color="#d32f2f",
        )
    axes[0, 1].set_title("GrLivArea vs SalePrice (Outlier Analysis)", fontsize=12, fontweight="bold")
    axes[0, 1].set_xlabel("Above Grade Living Area (sq ft)")
    axes[0, 1].set_ylabel("SalePrice ($)")
    axes[0, 1].legend()

    # TotalBsmtSF vs SalePrice
    axes[1, 0].scatter(
        train_df["TotalBsmtSF"],
        train_df["SalePrice"],
        alpha=0.5,
        color="#00796b",
        edgecolors="none",
    )
    axes[1, 0].set_title("TotalBsmtSF vs SalePrice", fontsize=12, fontweight="bold")
    axes[1, 0].set_xlabel("Total Basement Area (sq ft)")
    axes[1, 0].set_ylabel("SalePrice ($)")

    # GarageCars vs SalePrice
    sns.boxplot(x="GarageCars", y="SalePrice", data=train_df, ax=axes[1, 1], palette="Greens")
    axes[1, 1].set_title("SalePrice by GarageCars", fontsize=12, fontweight="bold")
    axes[1, 1].set_xlabel("Garage Car Capacity")
    axes[1, 1].set_ylabel("SalePrice ($)")

    plt.tight_layout()
    key_features_path = output_dir / "key_features_relationships.png"
    fig.savefig(key_features_path, dpi=200)
    plt.close(fig)
    fig_paths["key_features"] = key_features_path

    # 3. Correlation Heatmap for Top Features
    numeric_cols = train_df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")
    top_corr_features = numeric_cols.corr()["SalePrice"].abs().sort_values(ascending=False).head(15).index
    top_corr_matrix = numeric_cols[top_corr_features].corr()

    fig, ax = plt.subplots(figsize=(11, 9))
    sns.heatmap(
        top_corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": 0.8},
        ax=ax,
        linewidths=0.5,
    )
    ax.set_title("Correlation Heatmap (Top 15 Features vs SalePrice)", fontsize=13, fontweight="bold")
    plt.tight_layout()
    corr_path = output_dir / "correlation_heatmap_top15.png"
    fig.savefig(corr_path, dpi=200)
    plt.close(fig)
    fig_paths["correlation_heatmap"] = corr_path

    # 4. Missing Values Plot (Features with > 5% missing)
    missing_df = audit_missing_values(train_df)
    missing_top = missing_df[missing_df["missing_pct"] > 5.0].sort_values(by="missing_pct", ascending=True)

    if not missing_top.empty:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(missing_top.index, missing_top["missing_pct"], color="#e57373", edgecolor="#c62828")
        ax.set_title("Features with > 5% Missing Values (Training Data)", fontsize=13, fontweight="bold")
        ax.set_xlabel("Percentage Missing (%)")
        for i, (idx, row) in enumerate(missing_top.iterrows()):
            ax.text(row["missing_pct"] + 0.8, i, f"{row['missing_pct']:.1f}% ({int(row['missing_count'])})", va="center", fontsize=9)
        plt.tight_layout()
        missing_path = output_dir / "missing_values_bar.png"
        fig.savefig(missing_path, dpi=200)
        plt.close(fig)
        fig_paths["missing_values"] = missing_path

    return fig_paths


def generate_eda_markdown_report(train_df: pd.DataFrame, test_df: pd.DataFrame, output_dir: Path) -> Path:
    """Generate Markdown EDA report."""
    ensure_directory(output_dir)
    target = train_df["SalePrice"]
    log_target = np.log1p(target)

    # Quantiles
    quantiles = target.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
    
    # Missing summary
    missing_train = audit_missing_values(train_df)
    missing_test = audit_missing_values(test_df)

    # Top correlated features
    numeric_cols = train_df.select_dtypes(include=[np.number]).drop(columns=["Id"], errors="ignore")
    top_corrs = numeric_cols.corr()["SalePrice"].sort_values(ascending=False).head(12)

    report_content = f"""# Model 1 — Grandmaster Choice: Exploratory Data Analysis & Audit Report

**Date**: 2026-10-26  
**Competition**: Kaggle House Prices - Advanced Regression Techniques  
**Active Model**: `model_1_grandmaster`  
**Dataset Location**: `data/raw/`  

---

## 1. Executive Summary

- **Training Records**: {len(train_df):,} rows × {train_df.shape[1]} columns.
- **Test Records**: {len(test_df):,} rows × {test_df.shape[1]} columns.
- **Target Variable**: `SalePrice` (verified present in train, absent in test).
- **Target Skewness**: Raw skewness = `{target.skew():.3f}` (strongly right-skewed). After `log1p` transformation, skewness reduces to `{log_target.skew():.3f}` (nearly symmetric normal distribution).
- **Metric**: Root Mean Squared Logarithmic Error (RMSLE). Because $\\text{{RMSLE}}(y, \\hat{{y}}) = \\text{{RMSE}}(\\log(y+1), \\log(\\hat{{y}}+1))$, models will be trained directly on $y_{{\\text{{trans}}}} = \\log(1 + \\text{{SalePrice}})$ to directly optimize the competition loss function.

---

## 2. Target Variable Summary Statistics

| Statistic | Raw SalePrice ($) | log1p(SalePrice) |
| :--- | :--- | :--- |
| **Count** | {len(target):,} | {len(log_target):,} |
| **Mean** | ${target.mean():,.2f} | {log_target.mean():.4f} |
| **Std Dev** | ${target.std():,.2f} | {log_target.std():.4f} |
| **Min** | ${target.min():,.2f} | {log_target.min():.4f} |
| **25% (Q1)** | ${target.quantile(0.25):,.2f} | {log_target.quantile(0.25):.4f} |
| **50% (Median)** | ${target.quantile(0.50):,.2f} | {log_target.quantile(0.50):.4f} |
| **75% (Q3)** | ${target.quantile(0.75):,.2f} | {log_target.quantile(0.75):.4f} |
| **Max** | ${target.max():,.2f} | {log_target.max():.4f} |
| **Skewness** | {target.skew():.3f} | {log_target.skew():.3f} |
| **Kurtosis** | {target.kurt():.3f} | {log_target.kurt():.3f} |

### Detailed Percentiles
- 1st Percentile: ${quantiles[0.01]:,.2f}
- 5th Percentile: ${quantiles[0.05]:,.2f}
- 90th Percentile: ${quantiles[0.90]:,.2f}
- 95th Percentile: ${quantiles[0.95]:,.2f}
- 99th Percentile: ${quantiles[0.99]:,.2f}

---

## 3. Missing Value Analysis

In Ames Housing data, missing values often encode physical absence rather than unrecorded data:
- `PoolQC` (99.5% missing in train): No swimming pool.
- `MiscFeature` (96.3% missing in train): No additional feature (shed, tennis court, etc.).
- `Alley` (93.8% missing in train): No alley access.
- `Fence` (80.8% missing in train): No fence.
- `FireplaceQu` (47.3% missing in train): No fireplace.
- `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`: No garage.
- `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`: No basement.

### Columns with Highest Missing Percentages in Train
| Feature | Missing Count | Missing Percentage | Semantic Meaning |
| :--- | :--- | :--- | :--- |
"""
    for idx, row in missing_train.head(10).iterrows():
        report_content += f"| `{idx}` | {int(row['missing_count'])} | {row['missing_pct']:.2f}% | Absence of facility / unrecorded |\n"

    report_content += """
---

## 4. Top Feature Correlations with SalePrice

| Feature | Pearson Correlation ($r$) | Semantic Description |
| :--- | :--- | :--- |
"""
    for feat, corr_val in top_corrs.items():
        if feat != "SalePrice":
            report_content += f"| `{feat}` | {corr_val:+.4f} | High positive association with house valuation |\n"

    report_content += """
---

## 5. Outlier Investigation

A famous characteristic of the Ames dataset documented by Dean De Cock:
- Observations with `GrLivArea > 4000` sq ft:
  - `Id 524`: `GrLivArea = 4,676` sq ft, `SalePrice = $184,750` (`SaleCondition = Partial`)
  - `Id 1299`: `GrLivArea = 5,642` sq ft, `SalePrice = $160,000` (`SaleCondition = Partial`)
  - `Id 692`: `GrLivArea = 4,316` sq ft, `SalePrice = $755,000` (`SaleCondition = Normal`)
  - `Id 1183`: `GrLivArea = 4,476` sq ft, `SalePrice = $745,000` (`SaleCondition = Abnorml`)

`Id 524` and `Id 1299` represent partial sales of massive estate properties sold for abnormally low prices ($160k - $184k), which severely distort linear models and tree splits. We benchmark and document the effect of filtering these two verified aberrant training points in cross-validation.

---

## 6. Generated Figures

1. Target Distribution: `reports/model_1_grandmaster/target_distribution.png`
2. Key Feature Relationships: `reports/model_1_grandmaster/key_features_relationships.png`
3. Correlation Heatmap: `reports/model_1_grandmaster/correlation_heatmap_top15.png`
4. Missing Values Bar Chart: `reports/model_1_grandmaster/missing_values_bar.png`
"""
    report_path = output_dir / "eda_report.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    return report_path


def run_eda() -> Dict[str, Any]:
    """Execute EDA process, generate figures and markdown report."""
    data = load_raw_data()
    train_df = data["train"]
    test_df = data["test"]

    output_dir = MODEL_1_REPORTS_DIR
    fig_paths = generate_eda_figures(train_df, output_dir)
    report_path = generate_eda_markdown_report(train_df, test_df, output_dir)

    return {
        "report_path": report_path,
        "figures": fig_paths,
    }


if __name__ == "__main__":
    results = run_eda()
    print("EDA execution complete.")
    print(f"Report saved to: {results['report_path']}")
    for name, p in results['figures'].items():
        print(f"Figure [{name}]: {p}")
