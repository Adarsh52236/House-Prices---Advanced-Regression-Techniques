"""Tests for Experiment 2: Missing-category diagnostics and domain-aware preprocessor."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_1_grandmaster.preprocessing import GrandmasterPreprocessor


def test_domain_aware_imputation_no_all_zero_test_dummies():
    """Verify test properties with missing MSZoning or Functional receive valid mode categories."""
    train_sample = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "MSZoning": ["RL", "RL", "RL", "RM", "FV"],
        "Functional": ["Typ", "Typ", "Typ", "Min1", "Typ"],
        "Alley": [np.nan, "Grvl", np.nan, "Pave", np.nan],  # Absence feature
        "LotFrontage": [60.0, 70.0, 80.0, 90.0, 100.0],
        "TotalBsmtSF": [800, 900, 1000, 1100, 1200],
        "1stFlrSF": [800, 900, 1000, 1100, 1200],
        "SalePrice": [150000, 170000, 200000, 220000, 250000],
    })

    test_sample = pd.DataFrame({
        "Id": [101, 102],
        "MSZoning": [np.nan, "RL"],  # Id 101 has null MSZoning
        "Functional": ["Typ", np.nan],  # Id 102 has null Functional
        "Alley": [np.nan, np.nan],  # Both no alley
        "LotFrontage": [65.0, 75.0],
        "TotalBsmtSF": [850, 950],
        "1stFlrSF": [850, 950],
    })

    prep = GrandmasterPreprocessor(domain_imputation=True)
    X_tr = prep.fit_transform(train_sample)
    X_te = prep.transform(test_sample)

    assert X_tr.shape[1] == X_te.shape[1], "Feature matrix dimensions must match exactly"
    assert not np.isnan(X_tr).any()
    assert not np.isnan(X_te).any()

    # Find MSZoning dummy columns in feature names
    feature_names = prep.get_feature_names_out()
    mszoning_cols = [i for i, name in enumerate(feature_names) if "MSZoning" in name]
    assert len(mszoning_cols) > 0

    # Id 101 (row 0 in test) should NOT have all zeros across MSZoning dummies
    # It should have received the training mode ('RL')
    id_101_zoning_sum = X_te[0, mszoning_cols].sum()
    assert id_101_zoning_sum == 1.0, f"Expected active dummy for imputed mode, got sum={id_101_zoning_sum}"


def test_absence_category_imputed_as_none():
    """Verify that Alley NA becomes 'None' dummy rather than unknown dropped category."""
    train_sample = pd.DataFrame({
        "Id": [1, 2],
        "Alley": [np.nan, "Grvl"],
        "LotFrontage": [60.0, 70.0],
        "TotalBsmtSF": [800, 900],
        "1stFlrSF": [800, 900],
    })

    test_sample = pd.DataFrame({
        "Id": [101],
        "Alley": [np.nan],
        "LotFrontage": [65.0],
        "TotalBsmtSF": [850],
        "1stFlrSF": [850],
    })

    prep = GrandmasterPreprocessor(domain_imputation=True)
    prep.fit(train_sample)
    X_te = prep.transform(test_sample)

    feature_names = prep.get_feature_names_out()
    alley_none_cols = [i for i, name in enumerate(feature_names) if "Alley_None" in name]
    assert len(alley_none_cols) == 1
    assert X_te[0, alley_none_cols[0]] == 1.0


def test_experiment_2_artifacts_exist_and_consistent():
    """Verify Experiment 2 artifacts and metrics are properly saved and non-empty."""
    oof_csv = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_2" / "oof_predictions.csv"
    metrics_json = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_2" / "experiment_2_metrics.json"
    model_art = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_2" / "tree_ensemble_pipeline_exp2.joblib"

    assert oof_csv.exists(), "oof_predictions.csv must exist in experiment_2 reports"
    assert metrics_json.exists(), "experiment_2_metrics.json must exist in experiment_2 reports"
    assert model_art.exists(), "tree_ensemble_pipeline_exp2.joblib must exist in experiment_2 models"

    df_oof = pd.read_csv(oof_csv)
    assert len(df_oof) == 1460, f"Expected 1460 OOF predictions, got {len(df_oof)}"
    assert not df_oof.isnull().any().any(), "OOF predictions must have no null values"
    for col in ["oof_xgboost", "oof_lightgbm", "oof_catboost", "oof_tree_simple_average", "oof_tree_constrained_blend"]:
        assert col in df_oof.columns, f"Missing candidate column {col}"


def test_experiment_2_large_house_predictions_bounded():
    """Verify that tree ensemble pipeline predicts bounded, realistic prices on large houses."""
    import joblib
    model_art_path = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_2" / "tree_ensemble_pipeline_exp2.joblib"
    if not model_art_path.exists():
        pytest.skip("Experiment 2 model artifact not generated yet")

    art = joblib.load(model_art_path)
    prep = art["preprocessor"]
    trees = art["tree_models"]
    weights = art["blend_weights_mean"]

    # Test ID 2550
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "test.csv")
    row_2550 = test_df[test_df["Id"] == 2550]
    assert len(row_2550) == 1

    x_proc = prep.transform(row_2550)
    preds = [float(np.expm1(trees[m].predict(x_proc)[0])) for m in ["xgboost", "lightgbm", "catboost"]]
    blend_pred = float(np.sum(np.array(preds) * weights))

    # Must be bounded reasonably below $700k (unlike ElasticNet's $1.85M and Ridge Stack's $1.265M)
    assert 200000 <= blend_pred <= 700000, f"Expected bounded prediction between $200k and $700k, got ${blend_pred:,.2f}"


def test_experiment_2_candidate_submission_validity():
    """Verify that candidate submission matches competition format and sample_submission."""
    sub_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_2" / "submission_candidate_tree_simple_average.csv"
    assert sub_path.exists(), "Candidate submission file must exist"

    sample_sub_path = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
    sample_sub = pd.read_csv(sample_sub_path)
    cand_sub = pd.read_csv(sub_path)

    assert len(cand_sub) == 1459, f"Expected 1459 rows, got {len(cand_sub)}"
    assert list(cand_sub.columns) == ["Id", "SalePrice"], f"Expected columns ['Id', 'SalePrice'], got {list(cand_sub.columns)}"
    assert (cand_sub["Id"].values == sample_sub["Id"].values).all(), "Id sequence must match sample_submission exactly"
    assert not cand_sub.isnull().any().any(), "Candidate submission must contain no nulls"
    assert (cand_sub["SalePrice"].values > 0).all(), "All predicted prices must be strictly positive"
    assert not np.isinf(cand_sub["SalePrice"].values).any(), "No infinite prices allowed"


def test_experiment_2_final_submission_file_validity():
    """Verify that the final submission file exists and satisfies all competition requirements."""
    sub_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_2" / "submission_model_1_experiment_2_tree_simple_average.csv"
    assert sub_path.exists(), "Final Experiment 2 submission file must exist"

    sample_sub_path = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
    sample_sub = pd.read_csv(sample_sub_path)
    sub = pd.read_csv(sub_path)

    assert len(sub) == 1459, f"Expected 1459 rows, got {len(sub)}"
    assert list(sub.columns) == ["Id", "SalePrice"], f"Expected ['Id', 'SalePrice'], got {list(sub.columns)}"
    assert (sub["Id"].values == sample_sub["Id"].values).all(), "Id sequence must match sample_submission exactly"
    assert sub["Id"].nunique() == 1459, "Duplicate IDs found in final submission"
    assert not sub.isnull().any().any(), "Final submission contains null values"
    assert (sub["SalePrice"].values > 0).all(), "All predicted prices must be positive"
    assert not np.isinf(sub["SalePrice"].values).any(), "No infinite prices allowed"
    # Rounding check
    diff_round = np.max(np.abs(sub["SalePrice"].values - np.round(sub["SalePrice"].values, 2)))
    assert diff_round == 0.0, "Prices must be rounded to 2 decimal places"


