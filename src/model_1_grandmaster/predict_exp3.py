"""Inference and candidate submission generator for Model 1 Experiment 3 (Variant 3: Layout Density)."""

from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.common.paths import (
    RAW_DATA_DIR,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    ensure_directory,
)

EXP3_MODELS_DIR = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_3"
EXP3_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_3"


def generate_experiment_3_candidate_submission() -> Path:
    """Load Experiment 3 model artifact, predict on test data, and generate verified candidate submission."""
    print("=" * 80)
    print("MODEL 1 EXPERIMENT 3: GENERATING CANDIDATE SUBMISSION (VARIANT 3)")
    print("=" * 80)

    artifact_path = EXP3_MODELS_DIR / "tree_ensemble_pipeline_exp3.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Model artifact not found at: {artifact_path}")

    art = joblib.load(artifact_path)
    best_variant = art.get("best_variant")
    variant_name = art.get("variant_name")
    print(f"Loaded artifact:")
    print(f"  Artifact Path:   {artifact_path}")
    print(f"  Best Variant:    {best_variant}")
    print(f"  Variant Name:    {variant_name}")
    print(f"  Tree Models:     {art.get('tree_model_names')}")

    if best_variant != "layout_density":
        raise ValueError(f"Expected best_variant == 'layout_density', found '{best_variant}'")

    preprocessor = art["preprocessor"]
    tree_models = art["tree_models"]
    tree_names = art["tree_model_names"]

    # Verify that preprocessor has the layout_density features
    assert hasattr(preprocessor, "feature_engineer"), "Preprocessor missing feature_engineer"
    assert preprocessor.feature_engineer.variant == "layout_density", (
        f"Feature engineer variant mismatch: expected 'layout_density', got '{preprocessor.feature_engineer.variant}'"
    )
    assert "BathToBedRatio" in preprocessor.numeric_cols_, "BathToBedRatio missing from numeric_cols_"
    assert "SF_Per_Bedroom" in preprocessor.numeric_cols_, "SF_Per_Bedroom missing from numeric_cols_"
    assert "IsRemodeled" not in preprocessor.numeric_cols_, "IsRemodeled should not be in Variant 3"
    assert "BsmtFinRatio" not in preprocessor.numeric_cols_, "BsmtFinRatio should not be in Variant 3"

    # Load test data and sample submission
    test_df = pd.read_csv(TEST_CSV)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    n_test = len(test_df)
    assert n_test == 1459, f"Expected 1,459 test records, got {n_test}"

    # Transform test features
    X_test_proc = preprocessor.transform(test_df)
    assert X_test_proc.shape[0] == 1459, f"Expected 1459 rows, got {X_test_proc.shape[0]}"
    assert np.isfinite(X_test_proc).all(), "Transformed test matrix contains non-finite values!"

    # Predict with each tree model in log-space
    model_preds_log = []
    for m_name in tree_names:
        m = tree_models[m_name]
        p_log = m.predict(X_test_proc)
        model_preds_log.append(p_log)
        p_usd = np.expm1(p_log)
        print(f"  {m_name:10s} | Min: ${np.min(p_usd):9,.2f} | Median: ${np.median(p_usd):9,.2f} | Mean: ${np.mean(p_usd):9,.2f} | Max: ${np.max(p_usd):9,.2f}")

    # Simple average in log space
    avg_log = np.mean(model_preds_log, axis=0)
    preds_raw = np.clip(np.expm1(avg_log), 1.0, None)
    preds_rounded = np.round(preds_raw, 2)

    # Build submission DataFrame
    ensure_directory(EXP3_SUBMISSIONS_DIR)
    sub_path = EXP3_SUBMISSIONS_DIR / "submission_candidate_variant_3_layout_density.csv"

    sub_df = pd.DataFrame({
        "Id": test_df["Id"].values.astype(int),
        "SalePrice": preds_rounded,
    })

    sub_df.to_csv(sub_path, index=False)
    print(f"\n[SAVED] Candidate submission written to: {sub_path}")

    # Rigorous validation checks
    assert len(sub_df) == 1459, f"Expected 1,459 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["Id", "SalePrice"], f"Expected ['Id', 'SalePrice'], got {list(sub_df.columns)}"
    assert (sub_df["Id"].values == sample_sub["Id"].values).all(), "ID sequence does not match sample_submission.csv!"
    assert sub_df["Id"].nunique() == 1459, "Duplicate IDs found!"
    assert not sub_df.isnull().any().any(), "Missing values found in submission!"
    assert (sub_df["SalePrice"].values > 0).all(), "Non-positive sale prices found!"
    assert np.isfinite(sub_df["SalePrice"].values).all(), "Non-finite sale prices found!"

    # Check rounding
    diff_round = np.max(np.abs(sub_df["SalePrice"].values - np.round(sub_df["SalePrice"].values, 2)))
    assert diff_round == 0.0, "Prices are not rounded to 2 decimal places!"

    # Summary Statistics
    print("\nCandidate Prediction Summary Statistics:")
    print(f"  Row Count:        {len(sub_df):,}")
    print(f"  Min Price:        ${sub_df['SalePrice'].min():10,.2f}")
    print(f"  25th Percentile:  ${sub_df['SalePrice'].quantile(0.25):10,.2f}")
    print(f"  Median Price:     ${sub_df['SalePrice'].median():10,.2f}")
    print(f"  Mean Price:       ${sub_df['SalePrice'].mean():10,.2f}")
    print(f"  75th Percentile:  ${sub_df['SalePrice'].quantile(0.75):10,.2f}")
    print(f"  Max Price:        ${sub_df['SalePrice'].max():10,.2f}")

    # Key diagnostic records
    idx_2550 = test_df[test_df["Id"] == 2550].index[0]
    p_2550 = sub_df.loc[idx_2550, "SalePrice"]
    print(f"\nKey Diagnostic Record:")
    print(f"  ID 2550 (Edwards massive luxury): ${p_2550:10,.2f} (bounded safely)")

    return sub_path


if __name__ == "__main__":
    generate_experiment_3_candidate_submission()
