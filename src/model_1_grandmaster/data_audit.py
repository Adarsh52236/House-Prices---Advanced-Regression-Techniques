"""Data audit module for Model 1 (Grandmaster Choice).

Performs comprehensive integrity, missingness, schema consistency, and
distribution checks on the Kaggle House Prices dataset.
"""

from pathlib import Path
import sys
from typing import Any, Dict, List

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.common.paths import (
    DATA_DESCRIPTION_TXT,
    MODEL_1_REPORTS_DIR,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TRAIN_CSV,
    ensure_directory,
)


def load_raw_data() -> Dict[str, pd.DataFrame]:
    """Load the raw datasets using verified project paths."""
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    sample_sub_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    return {
        "train": train_df,
        "test": test_df,
        "sample_submission": sample_sub_df,
    }


def audit_identifiers_and_target(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Validate Id uniqueness, missingness, and target properties."""
    train_id_missing = int(train_df["Id"].isnull().sum())
    test_id_missing = int(test_df["Id"].isnull().sum())
    train_id_unique = bool(train_df["Id"].nunique() == len(train_df))
    test_id_unique = bool(test_df["Id"].nunique() == len(test_df))

    has_target = "SalePrice" in train_df.columns
    target_in_test = "SalePrice" in test_df.columns
    target_missing = int(train_df["SalePrice"].isnull().sum()) if has_target else -1
    target_all_positive = bool((train_df["SalePrice"] > 0).all()) if has_target else False

    return {
        "train_id_missing": train_id_missing,
        "test_id_missing": test_id_missing,
        "train_id_unique": train_id_unique,
        "test_id_unique": test_id_unique,
        "target_in_train": has_target,
        "target_in_test": target_in_test,
        "target_missing": target_missing,
        "target_all_positive": target_all_positive,
        "target_min": float(train_df["SalePrice"].min()) if has_target else None,
        "target_max": float(train_df["SalePrice"].max()) if has_target else None,
        "target_median": float(train_df["SalePrice"].median()) if has_target else None,
        "target_mean": float(train_df["SalePrice"].mean()) if has_target else None,
        "target_std": float(train_df["SalePrice"].std()) if has_target else None,
    }


def audit_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing count and percentage for all columns with missing values."""
    missing_count = df.isnull().sum()
    missing_pct = (missing_count / len(df)) * 100.0
    missing_df = pd.DataFrame({
        "missing_count": missing_count,
        "missing_pct": missing_pct,
        "dtype": df.dtypes,
    })
    return missing_df[missing_df["missing_count"] > 0].sort_values(
        by="missing_count", ascending=False
    )


def audit_schema_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, Any]:
    """Check feature consistency between train and test datasets."""
    train_features = [col for col in train_df.columns if col not in ["Id", "SalePrice"]]
    test_features = [col for col in test_df.columns if col not in ["Id"]]

    same_features = sorted(train_features) == sorted(test_features)
    missing_in_test = list(set(train_features) - set(test_features))
    extra_in_test = list(set(test_features) - set(train_features))

    # Type comparison for common features
    type_mismatches = {}
    for col in train_features:
        if col in test_features:
            t_train = str(train_df[col].dtype)
            t_test = str(test_df[col].dtype)
            if t_train != t_test:
                type_mismatches[col] = {"train_dtype": t_train, "test_dtype": t_test}

    # Duplicates check (excluding Id)
    train_duplicates = int(train_df[train_features].duplicated().sum())
    test_duplicates = int(test_df[test_features].duplicated().sum())

    return {
        "feature_count_train": len(train_features),
        "feature_count_test": len(test_features),
        "feature_sets_match": same_features,
        "missing_in_test": missing_in_test,
        "extra_in_test": extra_in_test,
        "type_mismatches": type_mismatches,
        "train_duplicate_rows": train_duplicates,
        "test_duplicate_rows": test_duplicates,
    }


def audit_unseen_categories(train_df: pd.DataFrame, test_df: pd.DataFrame) -> Dict[str, List[Any]]:
    """Detect category levels occurring in test but absent from train."""
    categorical_cols = train_df.select_dtypes(include=["object"]).columns.tolist()
    unseen_by_col = {}

    for col in categorical_cols:
        if col in test_df.columns:
            train_levels = set(train_df[col].dropna().unique())
            test_levels = set(test_df[col].dropna().unique())
            unseen = list(test_levels - train_levels)
            if unseen:
                unseen_by_col[col] = sorted(unseen)

    return unseen_by_col


def audit_high_correlations(train_df: pd.DataFrame, threshold: float = 0.8) -> List[Dict[str, Any]]:
    """Identify pairs of numeric features with absolute Pearson correlation >= threshold."""
    numeric_df = train_df.select_dtypes(include=[np.number]).drop(
        columns=["Id"], errors="ignore"
    )
    corr_matrix = numeric_df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    high_corrs = []
    for col1 in upper.columns:
        for col2 in upper.index:
            val = upper.loc[col2, col1]
            if pd.notnull(val) and val >= threshold:
                high_corrs.append({
                    "feature_1": col2,
                    "feature_2": col1,
                    "correlation": float(val),
                })
    return sorted(high_corrs, key=lambda x: x["correlation"], reverse=True)


def run_data_audit() -> Dict[str, Any]:
    """Execute complete data audit and return structured results."""
    data = load_raw_data()
    train_df = data["train"]
    test_df = data["test"]

    id_target_audit = audit_identifiers_and_target(train_df, test_df)
    train_missing = audit_missing_values(train_df)
    test_missing = audit_missing_values(test_df)
    schema_audit = audit_schema_consistency(train_df, test_df)
    unseen_cats = audit_unseen_categories(train_df, test_df)
    high_corrs = audit_high_correlations(train_df, threshold=0.75)

    return {
        "dimensions": {
            "train": train_df.shape,
            "test": test_df.shape,
            "sample_submission": data["sample_submission"].shape,
        },
        "id_target": id_target_audit,
        "train_missing_cols_count": len(train_missing),
        "train_missing_summary": train_missing.to_dict(orient="index"),
        "test_missing_cols_count": len(test_missing),
        "test_missing_summary": test_missing.to_dict(orient="index"),
        "schema": schema_audit,
        "unseen_categories": unseen_cats,
        "high_correlations": high_corrs,
    }


if __name__ == "__main__":
    audit_results = run_data_audit()
    print("=== DATA AUDIT SUMMARY ===")
    print(f"Train Shape: {audit_results['dimensions']['train']}")
    print(f"Test Shape:  {audit_results['dimensions']['test']}")
    print(f"Id Unique: Train={audit_results['id_target']['train_id_unique']}, Test={audit_results['id_target']['test_id_unique']}")
    print(f"Target Present & Positive: {audit_results['id_target']['target_all_positive']}")
    print(f"Columns with Missing Values: Train={audit_results['train_missing_cols_count']}, Test={audit_results['test_missing_cols_count']}")
    print(f"Unseen Categories in Test ({len(audit_results['unseen_categories'])} columns): {list(audit_results['unseen_categories'].keys())}")
    print(f"Highly Correlated Pairs (>=0.75): {len(audit_results['high_correlations'])}")
    for pair in audit_results['high_correlations'][:5]:
        print(f"  {pair['feature_1']} <-> {pair['feature_2']}: {pair['correlation']:.3f}")
