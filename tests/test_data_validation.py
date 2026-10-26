"""Automated tests for official Kaggle competition raw dataset."""

import sys
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import (
    DATA_DESCRIPTION_TXT,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TRAIN_CSV,
)


def test_raw_files_exist():
    assert TRAIN_CSV.exists() and TRAIN_CSV.stat().st_size > 0
    assert TEST_CSV.exists() and TEST_CSV.stat().st_size > 0
    assert SAMPLE_SUBMISSION_CSV.exists() and SAMPLE_SUBMISSION_CSV.stat().st_size > 0
    assert DATA_DESCRIPTION_TXT.exists() and DATA_DESCRIPTION_TXT.stat().st_size > 0


def test_train_csv_integrity():
    df = pd.read_csv(TRAIN_CSV)
    assert df.shape == (1460, 81)
    assert "SalePrice" in df.columns
    assert "Id" in df.columns
    assert not df["SalePrice"].isnull().any()


def test_test_csv_integrity():
    df = pd.read_csv(TEST_CSV)
    assert df.shape == (1459, 80)
    assert "Id" in df.columns
    assert "SalePrice" not in df.columns


def test_sample_submission_integrity():
    df = pd.read_csv(SAMPLE_SUBMISSION_CSV)
    assert df.shape == (1459, 2)
    assert list(df.columns) == ["Id", "SalePrice"]
