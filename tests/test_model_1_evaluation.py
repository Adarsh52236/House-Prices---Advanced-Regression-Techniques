"""Unit tests for Model 1 (Grandmaster Choice) evaluation metrics."""

import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_1_grandmaster.evaluation import calculate_metrics, calculate_rmsle


def test_rmsle_exact_match():
    # If predictions equal ground truth, RMSLE must be exactly 0
    y_true = np.array([100000.0, 200000.0, 350000.0])
    y_pred = np.array([100000.0, 200000.0, 350000.0])
    score = calculate_rmsle(y_true, y_pred)
    assert np.isclose(score, 0.0)


def test_rmsle_known_calculation():
    # Manual check:
    # y_true = [e - 1], y_pred = [e^2 - 1]
    # log1p(y_true) = 1, log1p(y_pred) = 2 -> diff = -1 -> RMSLE = 1.0
    y_true = np.array([np.e - 1.0])
    y_pred = np.array([np.e ** 2 - 1.0])
    score = calculate_rmsle(y_true, y_pred)
    assert np.isclose(score, 1.0)


def test_rmsle_handles_near_zero_predictions():
    y_true = np.array([100.0, 200.0])
    y_pred = np.array([0.0, -10.0])  # Nonpositive should be clipped, not raise unhandled error
    score = calculate_rmsle(y_true, y_pred, clip_min=1.0)
    assert np.isfinite(score)
    assert score > 0


def test_calculate_metrics_keys():
    y_true = np.array([150000.0, 250000.0])
    y_pred = np.array([160000.0, 240000.0])
    metrics = calculate_metrics(y_true, y_pred)
    assert "rmsle" in metrics
    assert "rmse" in metrics
    assert "mae" in metrics
    assert metrics["rmsle"] >= 0
    assert metrics["rmse"] >= 0
    assert metrics["mae"] >= 0


def test_empty_arrays_raise():
    with pytest.raises(ValueError):
        calculate_rmsle([], [])
