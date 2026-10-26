"""Inference and submission generation pipeline for Model 1 (Grandmaster Choice).

Loads the serialized production pipeline, predicts prices on test.csv,
verifies output schema and constraints, and produces competition-ready submission CSVs.
"""

from datetime import datetime
from pathlib import Path
import shutil
import sys
from typing import Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

from src.common.paths import (
    MODEL_1_MODELS_DIR,
    MODEL_1_SUBMISSIONS_DIR,
    TEST_CSV,
    ensure_directory,
)


def load_model_artifact(artifact_path: Optional[Path] = None) -> dict:
    """Load serialized Grandmaster pipeline artifact."""
    target_path = artifact_path or (MODEL_1_MODELS_DIR / "grandmaster_stack_pipeline.joblib")
    if not target_path.exists():
        raise FileNotFoundError(
            f"Trained pipeline artifact not found at {target_path}. Please run train.py first."
        )
    return joblib.load(target_path)


def generate_predictions(
    artifact_path: Optional[Path] = None,
    test_csv_path: Optional[Path] = None,
) -> Tuple[pd.DataFrame, Path, Path]:
    """Generate and validate submission predictions on test set.
    
    Returns:
        (submission_df, timestamped_submission_path, latest_submission_path)
    """
    csv_path = test_csv_path or TEST_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Test data file not found at {csv_path}")

    test_df = pd.read_csv(csv_path)
    test_ids = test_df["Id"].copy()
    expected_rows = len(test_df)

    print(f"[*] Loading Model 1 pipeline artifact...")
    artifact = load_model_artifact(artifact_path)
    preprocessor = artifact["preprocessor"]
    ensemble = artifact["ensemble"]

    print(f"[*] Transforming {expected_rows} test records through GrandmasterPreprocessor...")
    X_test_proc = preprocessor.transform(test_df)

    print(f"[*] Predicting target values with GrandmasterStackingEnsemble...")
    pred_log = ensemble.predict(X_test_proc)
    pred_raw = np.expm1(pred_log)

    # 1. Validation: Finite check
    if not np.isfinite(pred_raw).all():
        raise ValueError("Non-finite predictions detected (NaN or Inf in predicted prices).")

    # 2. Validation: Strict positivity check
    if (pred_raw <= 0).any():
        num_nonpos = int((pred_raw <= 0).sum())
        raise ValueError(f"Found {num_nonpos} nonpositive prediction(s). All prices must be > 0.")

    # 3. Validation: Row count and ID alignment check
    if len(pred_raw) != expected_rows:
        raise ValueError(
            f"Prediction count mismatch: expected {expected_rows}, got {len(pred_raw)}."
        )

    submission_df = pd.DataFrame({
        "Id": test_ids,
        "SalePrice": pred_raw,
    })

    # Ensure exact Kaggle schema
    if list(submission_df.columns) != ["Id", "SalePrice"]:
        raise ValueError(f"Invalid submission schema: {list(submission_df.columns)}")

    # 4. Save submission files
    submissions_dir = ensure_directory(MODEL_1_SUBMISSIONS_DIR)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    timestamped_filename = f"submission_model_1_grandmaster_{timestamp_str}.csv"
    timestamped_path = submissions_dir / timestamped_filename
    latest_path = submissions_dir / "submission_model_1_grandmaster_latest.csv"

    submission_df.to_csv(timestamped_path, index=False)
    shutil.copy2(timestamped_path, latest_path)

    print(f"\n{'='*70}")
    print(f"[SUCCESS] Submission generated and verified successfully!")
    print(f"  - Rows: {len(submission_df):,} | Columns: {list(submission_df.columns)}")
    print(f"  - Minimum predicted price: ${submission_df['SalePrice'].min():,.2f}")
    print(f"  - Median predicted price:  ${submission_df['SalePrice'].median():,.2f}")
    print(f"  - Mean predicted price:    ${submission_df['SalePrice'].mean():,.2f}")
    print(f"  - Maximum predicted price: ${submission_df['SalePrice'].max():,.2f}")
    print(f"  - Timestamped submission:  {timestamped_path}")
    print(f"  - Latest submission:       {latest_path}")
    print(f"{'='*70}\n")

    return submission_df, timestamped_path, latest_path


def main():
    generate_predictions()


if __name__ == "__main__":
    main()
