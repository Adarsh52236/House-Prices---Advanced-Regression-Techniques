"""Unit and integration tests for Model 2 (Legacy Statistician)."""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_2_legacy.models import LegacyStackingEnsemble, get_base_models, get_meta_model
from src.model_2_legacy.preprocessing import LegacyStatisticianPreprocessor


def test_model_2_preprocessor_finite_and_leak_free():
    """Verify preprocessor produces strictly finite outputs with domain-aware category handling."""
    train_sample = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "MSZoning": ["RL", "RL", "RL", "RM", "FV"],
        "Functional": ["Typ", "Typ", "Typ", "Min1", "Typ"],
        "Alley": [np.nan, "Grvl", np.nan, "Pave", np.nan],
        "LotFrontage": [60.0, 70.0, 80.0, 90.0, 100.0],
        "TotalBsmtSF": [800, 900, 1000, 1100, 1200],
        "1stFlrSF": [800, 900, 1000, 1100, 1200],
        "SalePrice": [150000, 170000, 200000, 220000, 250000],
    })

    test_sample = pd.DataFrame({
        "Id": [101, 102],
        "MSZoning": [np.nan, "RL"],
        "Functional": ["Typ", np.nan],
        "Alley": [np.nan, np.nan],
        "LotFrontage": [65.0, 75.0],
        "TotalBsmtSF": [850, 950],
        "1stFlrSF": [850, 950],
    })

    prep = LegacyStatisticianPreprocessor()
    X_tr = prep.fit_transform(train_sample)
    X_te = prep.transform(test_sample)

    assert X_tr.shape[1] == X_te.shape[1]
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()

    # Verify MSZoning dummy is active on test row with null MSZoning
    feat_names = prep.get_feature_names_out()
    mszoning_cols = [i for i, name in enumerate(feat_names) if "MSZoning" in name]
    assert len(mszoning_cols) > 0
    assert X_te[0, mszoning_cols].sum() == 1.0


def test_model_2_stacking_ensemble_smoke():
    """Verify stacking ensemble fit, inner CV, and predict on synthetic data."""
    np.random.seed(42)
    X = np.random.randn(30, 8)
    y = 12.0 + 0.5 * X[:, 0] - 0.3 * X[:, 1] + 0.1 * np.random.randn(30)

    ensemble = LegacyStackingEnsemble(
        base_models=get_base_models(random_state=42),
        meta_model=get_meta_model(random_state=42, positive=True),
        inner_cv_folds=3,
        random_state=42,
    )
    ensemble.fit(X, y)
    preds = ensemble.predict(X)

    assert len(preds) == 30
    assert np.isfinite(preds).all()

    coefs = ensemble.get_meta_coefficients()
    assert set(coefs.keys()) == {"lasso", "ridge", "huber"}
    for v in coefs.values():
        assert v >= 0.0, f"Positive constraint violated: {coefs}"


def test_model_2_artifacts_exist():
    """Verify all Model 2 reports, metrics, and models exist and are valid."""
    model_art = PROJECT_ROOT / "models" / "model_2_legacy" / "legacy_statistician_pipeline.joblib"
    oof_csv = PROJECT_ROOT / "reports" / "model_2_legacy" / "oof_predictions.csv"
    metrics_json = PROJECT_ROOT / "reports" / "model_2_legacy" / "model_2_metrics.json"
    report_md = PROJECT_ROOT / "reports" / "model_2_legacy" / "model_2_report.md"

    assert model_art.exists(), "Model 2 joblib artifact must exist"
    assert oof_csv.exists(), "Model 2 OOF predictions CSV must exist"
    assert metrics_json.exists(), "Model 2 metrics JSON must exist"
    assert report_md.exists(), "Model 2 markdown report must exist"

    df_oof = pd.read_csv(oof_csv)
    assert len(df_oof) == 1460
    for col in ["oof_lasso", "oof_ridge", "oof_huber", "oof_simple_average", "oof_lasso_stack"]:
        assert col in df_oof.columns


def test_model_2_candidate_submission_validity():
    """Verify Model 2 candidate submission schema and formatting."""
    sub_path = PROJECT_ROOT / "submissions" / "model_2_legacy" / "submission_candidate_model_2_legacy.csv"
    assert sub_path.exists(), "Candidate submission must exist"

    sample_sub = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "sample_submission.csv")
    sub = pd.read_csv(sub_path)

    assert len(sub) == 1459
    assert list(sub.columns) == ["Id", "SalePrice"]
    assert (sub["Id"].values == sample_sub["Id"].values).all()
    assert sub["Id"].nunique() == 1459
    assert not sub.isnull().any().any()
    assert (sub["SalePrice"].values > 0).all()
    assert np.isfinite(sub["SalePrice"].values).all()

    # Check 2 decimal rounding
    diff_round = np.max(np.abs(sub["SalePrice"].values - np.round(sub["SalePrice"].values, 2)))
    assert diff_round == 0.0


def test_model_2_artifact_reproduces_candidate_predictions():
    """Verify saved inference artifact reproduces candidate predictions exactly (diff = 0.00)."""
    model_art = PROJECT_ROOT / "models" / "model_2_legacy" / "legacy_statistician_pipeline.joblib"
    sub_path = PROJECT_ROOT / "submissions" / "model_2_legacy" / "submission_candidate_model_2_legacy.csv"
    test_path = PROJECT_ROOT / "data" / "raw" / "test.csv"

    art = joblib.load(model_art)
    prep = art["preprocessor"]
    ensemble = art["ensemble"]
    best_candidate = art.get("best_candidate", "lasso_stack")

    test_df = pd.read_csv(test_path)
    X_test_proc = prep.transform(test_df)

    if best_candidate == "lasso_stack":
        preds_log = ensemble.predict(X_test_proc)
    elif best_candidate == "simple_average":
        base_preds = ensemble.predict_base_models(X_test_proc)
        preds_matrix = np.column_stack([base_preds["lasso"], base_preds["ridge"], base_preds["huber"]])
        preds_log = np.mean(preds_matrix, axis=1)
    else:
        base_preds = ensemble.predict_base_models(X_test_proc)
        preds_log = base_preds[best_candidate]

    regen_preds = np.round(np.clip(np.expm1(preds_log), 1.0, None), 2)
    saved_sub = pd.read_csv(sub_path)

    max_diff = float(np.max(np.abs(saved_sub["SalePrice"].values - regen_preds)))
    assert max_diff == 0.0, f"Expected 0.0 diff, got {max_diff}"
