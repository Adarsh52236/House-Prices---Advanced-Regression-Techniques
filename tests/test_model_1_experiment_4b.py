"""Unit tests for Model 1 Experiment 4B (Targeted Incomplete-Construction Feature Engineering)."""

import json
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_1_grandmaster.evaluation import calculate_metrics
from src.model_1_grandmaster.features_exp4b import (
    Exp4bFeatureEngineer,
    GrandmasterPreprocessorExp4b,
)


def test_experiment_4b_feature_logic():
    """Verify that Exp4bFeatureEngineer creates IsPartialNew and PartialNewAreaInteraction accurately."""
    sample_df = pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "SaleCondition": ["Partial", "Partial", "Normal", "Abnorml"],
        "SaleType": ["New", "WD", "WD", "New"],
        "GrLivArea": [2000, 1500, 1800, 2200],
        "FullBath": [2, 1, 2, 2],
        "HalfBath": [1, 0, 1, 1],
        "BsmtFullBath": [1, 0, 0, 1],
        "BsmtHalfBath": [0, 0, 0, 0],
        "BedroomAbvGr": [3, 2, 3, 4],
        "TotalBsmtSF": [1000, 800, 900, 1100],
        "1stFlrSF": [1000, 800, 900, 1100],
        "2ndFlrSF": [1000, 700, 900, 1100],
        "YearBuilt": [2006, 2005, 1990, 2007],
        "YearRemodAdd": [2007, 2005, 1990, 2007],
        "YrSold": [2008, 2008, 2009, 2009],
    })

    # Candidate 1: IsPartialNew
    fe1 = Exp4bFeatureEngineer(candidate="candidate_1")
    out1 = fe1.transform(sample_df)
    assert "IsPartialNew" in out1.columns
    assert "PartialNewAreaInteraction" not in out1.columns
    # Row 1: Partial & New -> 1
    assert out1["IsPartialNew"].iloc[0] == 1
    # Row 2: Partial & WD -> 0
    assert out1["IsPartialNew"].iloc[1] == 0
    # Row 3: Normal & WD -> 0
    assert out1["IsPartialNew"].iloc[2] == 0
    # Row 4: Abnorml & New -> 0
    assert out1["IsPartialNew"].iloc[3] == 0

    # Candidate 2: Interaction
    fe2 = Exp4bFeatureEngineer(candidate="candidate_2")
    out2 = fe2.transform(sample_df)
    assert "IsPartialNew" in out2.columns
    assert "PartialNewAreaInteraction" in out2.columns
    assert out2["PartialNewAreaInteraction"].iloc[0] == 1 * 2000
    assert out2["PartialNewAreaInteraction"].iloc[1] == 0 * 1500
    assert out2["PartialNewAreaInteraction"].iloc[2] == 0 * 1800
    assert out2["PartialNewAreaInteraction"].iloc[3] == 0 * 2200


def test_experiment_4b_preprocessor_finiteness_and_leak_free():
    """Verify that GrandmasterPreprocessorExp4b transforms without leakage and outputs finite matrices."""
    train_sample = pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "SaleCondition": ["Partial", "Normal", "Normal", "Partial"],
        "SaleType": ["New", "WD", "WD", "New"],
        "GrLivArea": [2000, 1500, 1800, 2500],
        "BedroomAbvGr": [3, 2, 3, 4],
        "FullBath": [2, 1, 2, 3],
        "HalfBath": [1, 0, 1, 1],
        "BsmtFullBath": [1, 0, 0, 1],
        "BsmtHalfBath": [0, 0, 0, 0],
        "TotalBsmtSF": [1000, 800, 900, 1200],
        "1stFlrSF": [1000, 800, 900, 1200],
        "2ndFlrSF": [1000, 700, 900, 1300],
        "YearBuilt": [2006, 1990, 2000, 2008],
        "YearRemodAdd": [2007, 1990, 2005, 2008],
        "YrSold": [2008, 2008, 2009, 2009],
        "MSZoning": ["RL", "RL", "RM", "FV"],
        "Functional": ["Typ", "Typ", "Min1", "Typ"],
        "SalePrice": [250000, 160000, 190000, 310000],
    })

    test_sample = pd.DataFrame({
        "Id": [101, 102],
        "SaleCondition": ["Partial", "Normal"],
        "SaleType": [np.nan, "WD"],
        "GrLivArea": [2100, 1400],
        "BedroomAbvGr": [3, 2],
        "FullBath": [2, 1],
        "HalfBath": [0, 0],
        "BsmtFullBath": [0, 0],
        "BsmtHalfBath": [0, 0],
        "TotalBsmtSF": [1050, 750],
        "1stFlrSF": [1050, 750],
        "2ndFlrSF": [1050, 650],
        "YearBuilt": [2007, 1985],
        "YearRemodAdd": [2007, 1985],
        "YrSold": [2009, 2010],
        "MSZoning": [np.nan, "RL"],
        "Functional": ["Typ", np.nan],
    })

    prep = GrandmasterPreprocessorExp4b(candidate="candidate_2")
    X_tr = prep.fit_transform(train_sample)
    X_te = prep.transform(test_sample)

    assert X_tr.shape[1] == X_te.shape[1]
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()


def test_experiment_4b_artifacts_exist_and_consistent():
    """Verify that Experiment 4B reports, metrics, and OOF predictions exist and are valid."""
    exp4b_dir = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4b"
    oof_csv = exp4b_dir / "oof_predictions.csv"
    metrics_json = exp4b_dir / "experiment_4b_metrics.json"
    report_md = exp4b_dir / "experiment_4b_report.md"

    assert oof_csv.exists(), f"Missing {oof_csv}"
    assert metrics_json.exists(), f"Missing {metrics_json}"
    assert report_md.exists(), f"Missing {report_md}"

    df = pd.read_csv(oof_csv)
    assert len(df) == 1460
    for col in ["oof_baseline_v3", "oof_candidate_1", "oof_candidate_2"]:
        assert col in df.columns
        assert not df[col].isnull().any()
        assert np.isfinite(df[col]).all()
        assert (df[col] > 0).all()

    with open(metrics_json, "r") as f:
        meta = json.load(f)

    assert meta["recommendation"]["retain_candidate"] is False
    assert meta["recommendation"]["retained_model"] == "Experiment 3 Variant 3 (0.12546 RMSLE)"
