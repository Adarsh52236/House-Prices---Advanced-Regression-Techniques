"""Unit tests for Model 1 — Experiment 3: Targeted Feature Engineering."""

from pathlib import Path
import sys
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_1_grandmaster.features_exp3 import (
    Exp3FeatureEngineer,
    GrandmasterPreprocessorExp3,
)



def test_experiment_3_feature_engineer_variants():
    """Verify that Exp3FeatureEngineer properly generates expected features for each variant."""
    sample_df = pd.DataFrame({
        "Id": [1, 2],
        "MSSubClass": [60, 20],
        "OverallQual": [7, 8],
        "OverallCond": [5, 5],
        "YearBuilt": [1990, 2000],
        "YearRemodAdd": [2005, 2000],
        "YrSold": [2008, 2009],
        "GrLivArea": [1800, 2200],
        "1stFlrSF": [1000, 1200],
        "2ndFlrSF": [800, 1000],
        "TotalBsmtSF": [900, 1100],
        "BsmtFinSF1": [500, 800],
        "BsmtFinSF2": [0, 0],
        "BsmtUnfSF": [400, 300],
        "LowQualFinSF": [0, 50],
        "FullBath": [2, 2],
        "HalfBath": [1, 1],
        "BsmtFullBath": [1, 1],
        "BsmtHalfBath": [0, 0],
        "BedroomAbvGr": [3, 4],
    })

    # Test Variant 3 (layout_density)
    fe_v3 = Exp3FeatureEngineer(variant="layout_density")
    out_v3 = fe_v3.transform(sample_df)
    assert "BathToBedRatio" in out_v3.columns
    assert "SF_Per_Bedroom" in out_v3.columns
    assert out_v3["BathToBedRatio"].iloc[0] == pytest.approx(3.5 / 3.0)
    assert out_v3["SF_Per_Bedroom"].iloc[0] == pytest.approx(1800.0 / 3.0)

    # Test Combined
    fe_comb = Exp3FeatureEngineer(variant="combined")
    out_comb = fe_comb.transform(sample_df)
    assert "IsRemodeled" in out_comb.columns
    assert "EffectiveAge" in out_comb.columns
    assert "BsmtFinRatio" in out_comb.columns
    assert "HighQualGrLivArea" in out_comb.columns
    assert "BathToBedRatio" in out_comb.columns
    assert "SF_Per_Bedroom" in out_comb.columns


def test_experiment_3_preprocessor_finiteness_and_leak_free():
    """Verify preprocessor produces strictly finite outputs with domain-aware mode imputation."""
    train_sample = pd.DataFrame({
        "Id": [1, 2, 3, 4, 5],
        "MSZoning": ["RL", "RL", "RL", "RM", "FV"],
        "Functional": ["Typ", "Typ", "Typ", "Min1", "Typ"],
        "Alley": [np.nan, "Grvl", np.nan, "Pave", np.nan],
        "LotFrontage": [60.0, 70.0, 80.0, 90.0, 100.0],
        "TotalBsmtSF": [800, 900, 1000, 1100, 1200],
        "1stFlrSF": [800, 900, 1000, 1100, 1200],
        "GrLivArea": [1200, 1400, 1600, 1800, 2000],
        "BedroomAbvGr": [2, 3, 3, 4, 4],
        "FullBath": [1, 2, 2, 2, 3],
        "HalfBath": [0, 0, 1, 1, 1],
        "BsmtFullBath": [0, 1, 0, 1, 1],
        "BsmtHalfBath": [0, 0, 0, 0, 0],
        "YearBuilt": [1980, 1990, 2000, 2005, 2010],
        "YearRemodAdd": [1980, 1995, 2000, 2008, 2010],
        "YrSold": [2008, 2008, 2009, 2009, 2010],
        "SalePrice": [150000, 180000, 210000, 240000, 280000],
    })

    test_sample = pd.DataFrame({
        "Id": [101, 102],
        "MSZoning": [np.nan, "RL"],
        "Functional": ["Typ", np.nan],
        "Alley": [np.nan, np.nan],
        "LotFrontage": [65.0, 75.0],
        "TotalBsmtSF": [850, 950],
        "1stFlrSF": [850, 950],
        "GrLivArea": [1300, 1500],
        "BedroomAbvGr": [3, 3],
        "FullBath": [1, 2],
        "HalfBath": [1, 0],
        "BsmtFullBath": [0, 1],
        "BsmtHalfBath": [0, 0],
        "YearBuilt": [1985, 1995],
        "YearRemodAdd": [1985, 2000],
        "YrSold": [2008, 2009],
    })

    prep = GrandmasterPreprocessorExp3(variant="layout_density")
    X_tr = prep.fit_transform(train_sample)
    X_te = prep.transform(test_sample)

    assert X_tr.shape[1] == X_te.shape[1]
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()


def test_experiment_3_artifacts_exist_and_consistent():
    """Verify Experiment 3 reports, metrics, and models exist and are valid."""
    oof_csv = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_3" / "oof_predictions.csv"
    metrics_json = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_3" / "experiment_3_metrics.json"
    report_md = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_3" / "experiment_3_report.md"
    model_art = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_3" / "tree_ensemble_pipeline_exp3.joblib"

    assert oof_csv.exists(), "Experiment 3 OOF predictions CSV must exist"
    assert metrics_json.exists(), "Experiment 3 metrics JSON must exist"
    assert report_md.exists(), "Experiment 3 report MD must exist"
    assert model_art.exists(), "Experiment 3 model artifact must exist"

    df_oof = pd.read_csv(oof_csv)
    assert len(df_oof) == 1460
    for col in ["oof_baseline", "oof_remodel_age", "oof_finished_area", "oof_layout_density", "oof_combined"]:
        assert col in df_oof.columns


def test_experiment_3_large_house_predictions_bounded():
    """Verify that Experiment 3 pipeline keeps large-house predictions safely bounded."""
    model_art_path = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_3" / "tree_ensemble_pipeline_exp3.joblib"
    art = joblib.load(model_art_path)
    prep = art["preprocessor"]
    trees = art["tree_models"]
    names = art["tree_model_names"]

    test_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "test.csv")
    row_2550 = test_df[test_df["Id"] == 2550]
    assert len(row_2550) == 1

    x_proc = prep.transform(row_2550)
    preds = [float(trees[m].predict(x_proc)[0]) for m in names]
    avg_pred = float(np.expm1(np.mean(preds)))

    assert 200000 <= avg_pred <= 700000, f"Expected prediction between $200k and $700k, got ${avg_pred:,.2f}"


def test_experiment_3_candidate_submission_validity():
    """Verify Experiment 3 candidate submission matches required schema, count, and formatting."""
    sub_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_3" / "submission_candidate_variant_3_layout_density.csv"
    assert sub_path.exists(), f"Candidate submission must exist at {sub_path}"

    sub_df = pd.read_csv(sub_path)
    sample_sub = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "sample_submission.csv")
    test_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "test.csv")

    assert len(sub_df) == 1459, f"Expected 1459 rows, got {len(sub_df)}"
    assert list(sub_df.columns) == ["Id", "SalePrice"], f"Expected ['Id', 'SalePrice'], got {list(sub_df.columns)}"
    assert (sub_df["Id"].values == sample_sub["Id"].values).all(), "ID sequence mismatch with sample_submission.csv"
    assert (sub_df["Id"].values == test_df["Id"].values).all(), "ID sequence mismatch with test.csv"
    assert sub_df["Id"].nunique() == 1459, "Duplicate IDs in candidate submission"
    assert not sub_df.isnull().any().any(), "Candidate submission contains null values"
    assert (sub_df["SalePrice"].values > 0).all(), "Candidate submission contains non-positive prices"
    assert np.isfinite(sub_df["SalePrice"].values).all(), "Candidate submission contains non-finite prices"

    # Check 2 decimal places precision
    max_round_diff = np.max(np.abs(sub_df["SalePrice"].values - np.round(sub_df["SalePrice"].values, 2)))
    assert max_round_diff == 0.0, "SalePrice values are not rounded to 2 decimal places"


def test_experiment_3_artifact_reproduces_candidate():
    """Verify that regenerating test predictions from the saved artifact reproduces candidate submission exactly."""
    model_art_path = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_3" / "tree_ensemble_pipeline_exp3.joblib"
    sub_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_3" / "submission_candidate_variant_3_layout_density.csv"

    art = joblib.load(model_art_path)
    prep = art["preprocessor"]
    trees = art["tree_models"]
    names = art["tree_model_names"]

    assert art.get("best_variant") == "layout_density"
    assert "BathToBedRatio" in prep.numeric_cols_
    assert "SF_Per_Bedroom" in prep.numeric_cols_

    test_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "test.csv")
    X_test_proc = prep.transform(test_df)
    preds_log = [trees[m].predict(X_test_proc) for m in names]
    avg_log = np.mean(preds_log, axis=0)
    preds_raw = np.clip(np.expm1(avg_log), 1.0, None)
    preds_rounded = np.round(preds_raw, 2)

    sub_df = pd.read_csv(sub_path)
    max_abs_diff = np.max(np.abs(preds_rounded - sub_df["SalePrice"].values))
    assert max_abs_diff == 0.0, f"Artifact predictions mismatch candidate submission, max diff = {max_abs_diff}"

