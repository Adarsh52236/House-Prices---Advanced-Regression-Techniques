"""Evaluation metrics and diagnostic utilities for Model 1 (Grandmaster Choice).

Implements RMSLE, RMSE, and MAE calculations, along with residual analysis
and validation reporting.
"""

from pathlib import Path
import sys
from typing import Dict, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error


def calculate_rmsle(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
    clip_min: float = 1.0,
) -> float:
    """Calculate Root Mean Squared Logarithmic Error (RMSLE).
    
    Formula:
        RMSLE = sqrt(mean((log1p(y_true) - log1p(y_pred))^2))
        
    Args:
        y_true: Ground truth target values (original scale, e.g. USD).
        y_pred: Predicted values (original scale).
        clip_min: Minimum value to clip predictions to prevent log of nonpositive numbers.
        
    Returns:
        Computed RMSLE as a float.
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    if len(y_t) == 0 or len(y_p) == 0:
        raise ValueError("Cannot calculate RMSLE on empty arrays.")
    if len(y_t) != len(y_p):
        raise ValueError(f"Array length mismatch: y_true has {len(y_t)}, y_pred has {len(y_p)}.")

    if (y_t < 0).any():
        raise ValueError("Ground truth y_true contains negative values, invalid for RMSLE.")

    # Guard against nonpositive predictions
    if (y_p <= 0).any():
        n_invalid = int((y_p <= 0).sum())
        print(f"[WARNING] {n_invalid} prediction(s) <= 0 detected. Clipping to {clip_min}.")
        y_p = np.clip(y_p, a_min=clip_min, a_max=None)

    log_true = np.log1p(y_t)
    log_pred = np.log1p(y_p)

    return float(np.sqrt(np.mean((log_true - log_pred) ** 2)))


def calculate_metrics(
    y_true: Union[np.ndarray, pd.Series],
    y_pred: Union[np.ndarray, pd.Series],
) -> Dict[str, float]:
    """Calculate primary and auxiliary regression evaluation metrics.
    
    Returns:
        Dict with 'rmsle', 'rmse' (original USD), and 'mae' (original USD).
    """
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    rmsle_val = calculate_rmsle(y_t, y_p)
    rmse_val = float(np.sqrt(mean_squared_error(y_t, y_p)))
    mae_val = float(mean_absolute_error(y_t, y_p))

    return {
        "rmsle": rmsle_val,
        "rmse": rmse_val,
        "mae": mae_val,
    }


def format_metrics_summary(model_name: str, metrics: Dict[str, float]) -> str:
    """Format metrics dictionary into a readable summary string."""
    return (
        f"[{model_name}] RMSLE: {metrics['rmsle']:.4f} | "
        f"RMSE: ${metrics['rmse']:,.2f} | "
        f"MAE: ${metrics['mae']:,.2f}"
    )
