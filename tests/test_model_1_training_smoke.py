"""Smoke tests for Model 1 (Grandmaster Choice) training, stacking ensemble, and submissions."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import (
    MODEL_1_SUBMISSIONS_DIR,
    TEST_CSV,
)
from src.model_1_grandmaster.models import (
    GrandmasterStackingEnsemble,
    get_base_models,
    get_meta_model,
)


def test_stacking_ensemble_smoke():
    # Synthetic small dataset
    np.random.seed(42)
    n_samples = 40
    n_features = 10
    X = np.random.randn(n_samples, n_features)
    y = 10.0 + 2.0 * X[:, 0] - 1.5 * X[:, 1] + 0.5 * np.random.randn(n_samples)

    # Use fast 5-iteration models
    base_models = get_base_models(n_estimators_override=5)
    meta_model = get_meta_model()

    ensemble = GrandmasterStackingEnsemble(
        base_models=base_models,
        meta_model=meta_model,
        inner_cv_folds=3,
        random_state=42,
    )

    ensemble.fit(X, y)
    preds = ensemble.predict(X)

    assert len(preds) == n_samples
    assert np.isfinite(preds).all()

    meta_coefs = ensemble.get_meta_coefficients()
    assert isinstance(meta_coefs, dict)
    assert set(meta_coefs.keys()) == {"xgboost", "lightgbm", "catboost", "elasticnet"}


def test_submission_schema_and_id_coverage():
    """Verify generated submission format against official test set."""
    latest_sub_path = MODEL_1_SUBMISSIONS_DIR / "submission_model_1_grandmaster_latest.csv"
    assert latest_sub_path.exists(), f"Submission file {latest_sub_path} does not exist."

    sub_df = pd.read_csv(latest_sub_path)
    test_df = pd.read_csv(TEST_CSV)

    assert sub_df.shape == (1459, 2), f"Expected (1459, 2), got {sub_df.shape}"
    assert list(sub_df.columns) == ["Id", "SalePrice"]
    assert (sub_df["Id"] == test_df["Id"]).all(), "Submission Ids do not match test.csv Ids exactly."
    assert not sub_df["SalePrice"].isnull().any(), "Submission contains null SalePrice predictions."
    assert np.isfinite(sub_df["SalePrice"]).all(), "Submission contains non-finite predictions."
    assert (sub_df["SalePrice"] > 0).all(), "Submission contains non-positive predictions."


def test_model_2_still_unimplemented():
    """Verify strict constraint: Model 2 must remain untouched and empty."""
    import src.model_2_legacy as m2
    exported = [attr for attr in dir(m2) if not attr.startswith("__")]
    assert len(exported) == 0, f"Model 2 must remain unimplemented, but found: {exported}"
