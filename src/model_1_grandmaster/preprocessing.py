"""Leakage-safe preprocessing pipelines for Model 1 (Grandmaster Choice).

Implements deterministic feature engineering, median/constant imputation,
and robust one-hot encoding with handle_unknown='ignore'.
All transformers fit exclusively on training data to prevent target or feature leakage.
"""

from pathlib import Path
import sys
from typing import List, Optional, Tuple, Union

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Categorical columns where NA semantically represents "None" / absence of the feature
NONE_CATEGORICAL_COLS = [
    "Alley",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "PoolQC",
    "Fence",
    "MiscFeature",
]

# Numerical columns where NA semantically represents 0
ZERO_NUMERICAL_COLS = [
    "GarageCars",
    "GarageArea",
    "BsmtFinSF1",
    "BsmtFinSF2",
    "BsmtUnfSF",
    "TotalBsmtSF",
    "BsmtFullBath",
    "BsmtHalfBath",
    "MasVnrArea",
]


class HousePricesFeatureEngineer(BaseEstimator, TransformerMixin):
    """Deterministic row-wise feature engineer for Ames housing dataset.
    
    Creates domain features without calculating cross-row statistics, ensuring
    strict absence of data leakage across cross-validation folds.
    """

    def __init__(self, convert_mssubclass_to_str: bool = True):
        self.convert_mssubclass_to_str = convert_mssubclass_to_str

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # MSSubClass is nominally encoded as integer, convert to string category
        if self.convert_mssubclass_to_str and "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)

        # Total Square Footage
        bsmt_sf = X["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in X.columns else 0
        first_sf = X["1stFlrSF"].fillna(0) if "1stFlrSF" in X.columns else 0
        second_sf = X["2ndFlrSF"].fillna(0) if "2ndFlrSF" in X.columns else 0
        X["TotalSF"] = bsmt_sf + first_sf + second_sf

        # Total Bathrooms
        full_bath = X["FullBath"].fillna(0) if "FullBath" in X.columns else 0
        half_bath = X["HalfBath"].fillna(0) if "HalfBath" in X.columns else 0
        bsmt_full = X["BsmtFullBath"].fillna(0) if "BsmtFullBath" in X.columns else 0
        bsmt_half = X["BsmtHalfBath"].fillna(0) if "BsmtHalfBath" in X.columns else 0
        X["TotalBathrooms"] = full_bath + (0.5 * half_bath) + bsmt_full + (0.5 * bsmt_half)

        # Total Porch / Deck Area
        wood_deck = X["WoodDeckSF"].fillna(0) if "WoodDeckSF" in X.columns else 0
        open_porch = X["OpenPorchSF"].fillna(0) if "OpenPorchSF" in X.columns else 0
        encl_porch = X["EnclosedPorch"].fillna(0) if "EnclosedPorch" in X.columns else 0
        ssn_porch = X["3SsnPorch"].fillna(0) if "3SsnPorch" in X.columns else 0
        screen_porch = X["ScreenPorch"].fillna(0) if "ScreenPorch" in X.columns else 0
        X["TotalPorchSF"] = wood_deck + open_porch + encl_porch + ssn_porch + screen_porch

        # Age features
        yr_sold = X["YrSold"].fillna(2010) if "YrSold" in X.columns else 2010
        if "YearBuilt" in X.columns:
            X["HouseAge"] = (yr_sold - X["YearBuilt"]).clip(lower=0)
        if "YearRemodAdd" in X.columns:
            X["RemodAge"] = (yr_sold - X["YearRemodAdd"]).clip(lower=0)

        # Indicator flags for presence of amenities
        if "GarageArea" in X.columns:
            X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
        if "TotalBsmtSF" in X.columns:
            X["HasBasement"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
        if "Fireplaces" in X.columns:
            X["HasFireplace"] = (X["Fireplaces"].fillna(0) > 0).astype(int)
        if "PoolArea" in X.columns:
            X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)

        return X


def build_preprocessor_pipeline(
    scale_numeric: bool = False,
) -> Tuple[Pipeline, List[str], List[str]]:
    """Build a scikit-learn ColumnTransformer pipeline for tabular features.
    
    Args:
        scale_numeric: If True, applies StandardScaler to numeric features (for ElasticNetCV).
    """
    pass  # Implemented in GrandmasterPreprocessor class below for full control and metadata tracking


class GrandmasterPreprocessor(BaseEstimator, TransformerMixin):
    """Complete leakage-safe preprocessor for Model 1 (Grandmaster Choice).
    
    Separates Id and SalePrice, applies deterministic feature engineering,
    imputes missing numerical values with median, imputes categorical features
    with semantic defaults / 'Missing', and encodes categories with OneHotEncoder(handle_unknown='ignore').
    """

    def __init__(self, scale_numeric: bool = False, drop_id: bool = True):
        self.scale_numeric = scale_numeric
        self.drop_id = drop_id
        self.feature_engineer = HousePricesFeatureEngineer()
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[np.ndarray] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop Id if present, preserving other features."""
        df_copy = df.copy()
        if self.drop_id and "Id" in df_copy.columns:
            df_copy = df_copy.drop(columns=["Id"])
        if "SalePrice" in df_copy.columns:
            df_copy = df_copy.drop(columns=["SalePrice"])
        return df_copy

    def fit(self, X: pd.DataFrame, y=None):
        X_clean = self._prepare_dataframe(X)
        X_engineered = self.feature_engineer.fit_transform(X_clean)

        self.numeric_cols_ = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()

        # Numeric sub-pipeline
        numeric_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            numeric_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(numeric_steps)

        # Categorical sub-pipeline
        categorical_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", numeric_pipeline, self.numeric_cols_),
                ("cat", categorical_pipeline, self.categorical_cols_),
            ],
            remainder="drop",
        )

        self.column_transformer.fit(X_engineered)
        self.feature_names_out_ = self.column_transformer.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.column_transformer is None:
            raise RuntimeError("GrandmasterPreprocessor must be fitted before calling transform().")
        X_clean = self._prepare_dataframe(X)
        X_engineered = self.feature_engineer.transform(X_clean)
        return self.column_transformer.transform(X_engineered)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.feature_names_out_


def get_preprocessor(scale_numeric: bool = False) -> GrandmasterPreprocessor:
    """Factory function returning a new GrandmasterPreprocessor instance."""
    return GrandmasterPreprocessor(scale_numeric=scale_numeric)
