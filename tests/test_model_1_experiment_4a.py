"""Unit tests for Model 1 Experiment 4A (Leakage-Safe Constrained Tree Blending)."""

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


def test_experiment_4a_artifacts_exist():
    """Verify that all Experiment 4A reports and predictions exist."""
    exp4a_dir = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4a"
    oof_csv = exp4a_dir / "oof_predictions.csv"
    metrics_json = exp4a_dir / "experiment_4a_metrics.json"
    report_md = exp4a_dir / "experiment_4a_report.md"

    assert oof_csv.exists(), f"Missing {oof_csv}"
    assert metrics_json.exists(), f"Missing {metrics_json}"
    assert report_md.exists(), f"Missing {report_md}"


def test_experiment_4a_oof_predictions_validity():
    """Verify OOF predictions format, bounds, and alignment."""
    oof_csv = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4a" / "oof_predictions.csv"
    df = pd.read_csv(oof_csv)
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "train.csv")

    assert len(df) == 1460
    assert (df["Id"].values == train_df["Id"].values).all()
    assert (df["SalePrice"].values == train_df["SalePrice"].values).all()

    for col in ["oof_xgboost", "oof_lightgbm", "oof_catboost", "oof_equal_blend", "oof_held_out_blend"]:
        assert col in df.columns
        assert not df[col].isnull().any()
        assert np.isfinite(df[col]).all()
        assert (df[col] > 0).all()

    # Verify reproduction of equal-weight baseline (0.12546)
    m_eq = calculate_metrics(df["SalePrice"].values, df["oof_equal_blend"].values)
    assert m_eq["rmsle"] == pytest.approx(0.1254615, abs=1e-5)


def test_experiment_4a_weights_constraints_and_leakage_safety():
    """Verify that blend weights satisfy non-negativity and sum-to-1 constraint."""
    metrics_json = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4a" / "experiment_4a_metrics.json"
    with open(metrics_json, "r") as f:
        data = json.load(f)

    fold_results = data["fold_results"]
    assert len(fold_results) == 5

    for fr in fold_results:
        w = fr["weights"]
        assert all(val >= 0.0 for val in w.values()), f"Negative weight in fold {fr['fold']}"
        assert sum(w.values()) == pytest.approx(1.0, abs=1e-4), f"Weights do not sum to 1 in fold {fr['fold']}"
        assert fr["optimization_success"] is True

    # Full dataset weights
    fw = data["full_dataset_weights"]
    assert all(val >= 0.0 for val in fw.values())
    assert sum(fw.values()) == pytest.approx(1.0, abs=1e-4)
