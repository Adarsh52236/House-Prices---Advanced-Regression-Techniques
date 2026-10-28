"""Unit tests for Model 1 Experiment 4C (Controlled Tree Regularization)."""

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


def test_experiment_4c_artifacts_exist():
    """Verify that all Experiment 4C reports, metrics, and predictions exist."""
    exp4c_dir = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4c"
    oof_csv = exp4c_dir / "oof_predictions.csv"
    metrics_json = exp4c_dir / "experiment_4c_metrics.json"
    report_md = exp4c_dir / "experiment_4c_report.md"

    assert oof_csv.exists(), f"Missing {oof_csv}"
    assert metrics_json.exists(), f"Missing {metrics_json}"
    assert report_md.exists(), f"Missing {report_md}"


def test_experiment_4c_oof_predictions_validity():
    """Verify OOF predictions format, bounds, alignment, and baseline reproduction."""
    oof_csv = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4c" / "oof_predictions.csv"
    df = pd.read_csv(oof_csv)
    train_df = pd.read_csv(PROJECT_ROOT / "data" / "raw" / "train.csv")

    assert len(df) == 1460
    assert (df["Id"].values == train_df["Id"].values).all()
    assert (df["SalePrice"].values == train_df["SalePrice"].values).all()

    for col in ["oof_baseline_config_a", "oof_modest_reg_config_b", "oof_strong_reg_config_c"]:
        assert col in df.columns
        assert not df[col].isnull().any()
        assert np.isfinite(df[col]).all()
        assert (df[col] > 0).all()

    # Verify Config A baseline reproduces 0.12546 RMSLE
    m_base = calculate_metrics(df["SalePrice"].values, df["oof_baseline_config_a"].values)
    assert m_base["rmsle"] == pytest.approx(0.1254615, abs=1e-5)


def test_experiment_4c_retention_rejection():
    """Verify that both regularized configurations were properly evaluated and rejected."""
    metrics_json = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_4c" / "experiment_4c_metrics.json"
    with open(metrics_json, "r") as f:
        data = json.load(f)

    cands = data["candidates"]
    assert cands["config_b"]["delta_vs_baseline"] > 0.0, "Config B should have regressed"
    assert cands["config_c"]["delta_vs_baseline"] > cands["config_b"]["delta_vs_baseline"], "Config C should have regressed further"

    assert data["recommendation"]["retain_candidate"] is False
    assert "Baseline" in data["recommendation"]["retained_model"]
