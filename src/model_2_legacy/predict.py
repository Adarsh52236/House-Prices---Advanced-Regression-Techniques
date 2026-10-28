"""Inference and candidate submission generation for Model 2 (Legacy Statistician)."""

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

MODEL_2_MODELS_DIR = PROJECT_ROOT / "models" / "model_2_legacy"
MODEL_2_SUBMISSIONS_DIR = PROJECT_ROOT / "submissions" / "model_2_legacy"


def generate_model_2_submission() -> Path:
    print("=" * 80)
    print("MODEL 2 — GENERATING INFERENCE & CANDIDATE SUBMISSION")
    print("=" * 80)

    artifact_path = MODEL_2_MODELS_DIR / "legacy_statistician_pipeline.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(f"Missing production artifact at {artifact_path}. Run train.py first.")

    art = joblib.load(artifact_path)
    preprocessor = art["preprocessor"]
    ensemble = art["ensemble"]
    best_candidate = art.get("best_candidate", "lasso_stack")

    test_df = pd.read_csv(TEST_CSV)
    sample_sub = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    assert len(test_df) == 1459, f"Expected 1,459 test rows, found {len(test_df)}"

    # 1. Transform features with saved preprocessor
    X_test_proc = preprocessor.transform(test_df)
    assert X_test_proc.shape[0] == 1459
    assert np.isfinite(X_test_proc).all(), "Non-finite values found in transformed test matrix!"

    # 2. Predict with stacking ensemble
    if best_candidate == "lasso_stack":
        preds_log = ensemble.predict(X_test_proc)
    elif best_candidate == "simple_average":
        base_preds = ensemble.predict_base_models(X_test_proc)
        preds_matrix = np.column_stack([base_preds["lasso"], base_preds["ridge"], base_preds["huber"]])
        preds_log = np.mean(preds_matrix, axis=1)
    else:
        base_preds = ensemble.predict_base_models(X_test_proc)
        preds_log = base_preds[best_candidate]

    # 3. Invert log1p and clip
    preds_raw = np.clip(np.expm1(preds_log), 1.0, None)
    preds_rounded = np.round(preds_raw, 2)

    # 4. Create submission dataframe
    sub_df = pd.DataFrame({
        "Id": test_df["Id"].values.astype(int),
        "SalePrice": preds_rounded,
    })

    # 5. Save submission
    ensure_directory(MODEL_2_SUBMISSIONS_DIR)
    sub_path = MODEL_2_SUBMISSIONS_DIR / "submission_candidate_model_2_legacy.csv"
    sub_df.to_csv(sub_path, index=False)
    print(f"[SAVED] Model 2 candidate submission to: {sub_path}")

    # 6. Verification
    assert len(sub_df) == 1459, f"Expected 1459 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["Id", "SalePrice"], f"Expected ['Id', 'SalePrice'], got {list(sub_df.columns)}"
    assert (sub_df["Id"].values == sample_sub["Id"].values).all(), "ID sequence mismatch with sample_submission.csv!"
    assert sub_df["Id"].nunique() == 1459, "Duplicate IDs found!"
    assert not sub_df.isnull().any().any(), "Null values found in submission!"
    assert (sub_df["SalePrice"].values > 0).all(), "Non-positive prices found!"
    assert np.isfinite(sub_df["SalePrice"].values).all(), "Infinite prices found!"
    diff_round = np.max(np.abs(sub_df["SalePrice"].values - np.round(sub_df["SalePrice"].values, 2)))
    assert diff_round == 0.0, "Prices not rounded to 2 decimal places!"

    print(f"Validation successful: Exactly 1,459 positive, rounded predictions generated.")
    print(f"  Min prediction:    ${sub_df['SalePrice'].min():10,.2f}")
    print(f"  Median prediction: ${sub_df['SalePrice'].median():10,.2f}")
    print(f"  Mean prediction:   ${sub_df['SalePrice'].mean():10,.2f}")
    print(f"  Max prediction:    ${sub_df['SalePrice'].max():10,.2f}")

    return sub_path


if __name__ == "__main__":
    generate_model_2_submission()
