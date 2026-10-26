"""Unit tests for Model 1 (Grandmaster Choice) preprocessing pipeline."""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.model_1_grandmaster.preprocessing import GrandmasterPreprocessor


@pytest.fixture
def sample_train_data():
    return pd.DataFrame({
        "Id": [1, 2, 3, 4],
        "MSSubClass": [20, 60, 20, 120],
        "MSZoning": ["RL", "RM", "RL", "FV"],
        "LotFrontage": [65.0, np.nan, 80.0, 70.0],
        "TotalBsmtSF": [856, 926, 920, 750],
        "1stFlrSF": [856, 926, 920, 750],
        "2ndFlrSF": [854, 0, 866, 0],
        "FullBath": [2, 1, 2, 1],
        "HalfBath": [1, 0, 1, 0],
        "GarageArea": [548, 460, 608, 0],
        "SalePrice": [208500, 181500, 223500, 140000],
    })


@pytest.fixture
def sample_test_data():
    return pd.DataFrame({
        "Id": [101, 102],
        "MSSubClass": [150, 60],  # 150 is an unseen level
        "MSZoning": ["RL", "UNKNOWN_ZONE"],  # UNKNOWN_ZONE is unseen
        "LotFrontage": [np.nan, 75.0],  # NaN in test
        "TotalBsmtSF": [800, 1000],
        "1stFlrSF": [800, 1000],
        "2ndFlrSF": [0, 800],
        "FullBath": [1, 2],
        "HalfBath": [0, 1],
        "GarageArea": [np.nan, 500],  # NaN garage area
    })


def test_id_not_used_as_feature(sample_train_data):
    prep = GrandmasterPreprocessor(drop_id=True)
    X_proc = prep.fit_transform(sample_train_data)
    feature_names = prep.get_feature_names_out()

    for feat in feature_names:
        assert "Id" not in feat, f"Id was leaked into features: {feat}"
        assert "SalePrice" not in feat, f"SalePrice was leaked into features: {feat}"


def test_missing_values_handled(sample_train_data, sample_test_data):
    prep = GrandmasterPreprocessor()
    X_tr = prep.fit_transform(sample_train_data)
    X_te = prep.transform(sample_test_data)

    assert not np.isnan(X_tr).any(), "NaN values detected in processed training array"
    assert not np.isnan(X_te).any(), "NaN values detected in processed test array"


def test_unknown_categorical_levels_supported(sample_train_data, sample_test_data):
    prep = GrandmasterPreprocessor()
    X_tr = prep.fit_transform(sample_train_data)
    
    # Should transform without raising errors despite unseen categories
    X_te = prep.transform(sample_test_data)
    assert X_te.shape[1] == X_tr.shape[1], "Transformed test column count does not match train."


def test_feature_alignment(sample_train_data, sample_test_data):
    prep = GrandmasterPreprocessor()
    prep.fit(sample_train_data)

    X_tr = prep.transform(sample_train_data)
    X_te = prep.transform(sample_test_data)

    assert X_tr.shape[1] == X_te.shape[1]
    assert np.isfinite(X_tr).all()
    assert np.isfinite(X_te).all()
