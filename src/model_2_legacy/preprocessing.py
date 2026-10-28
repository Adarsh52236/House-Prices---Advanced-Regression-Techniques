"""Leakage-safe preprocessing pipelines for Model 2 (Legacy Statistician).

Implements deterministic feature engineering, median/constant imputation,
standard scaling for numeric features (required by linear/Huber estimators),
and robust one-hot encoding with handle_unknown='ignore'.
All transformers fit exclusively on training data to prevent leakage.
"""

from pathlib import Path
import sys
from typing import List, Optional

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

# Categorical columns where NA semantically represents "None" (absence of the feature)
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


class LegacyFeatureEngineer(BaseEstimator, TransformerMixin):
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


class LegacyStatisticianPreprocessor(BaseEstimator, TransformerMixin):
    """Complete leakage-safe preprocessor tailored for linear and robust estimators in Model 2.
    
    1. Removes Id and SalePrice.
    2. Generates domain features.
    3. Imputes missing numerical values with training fold median.
    4. Imputes physical absence categorical values with 'None'.
    5. Imputes unrecorded categorical values with training fold mode.
    6. Standardizes numeric features with StandardScaler (essential for Lasso/Ridge/Huber).
    7. Encodes categories with OneHotEncoder(handle_unknown='ignore').
    8. Validates that output is strictly finite (0 NaNs, 0 Infs).
    """

    def __init__(self, drop_id: bool = True):
        self.drop_id = drop_id
        self.feature_engineer = LegacyFeatureEngineer()
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[np.ndarray] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
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

        # Numeric sub-pipeline: Median imputation followed by StandardScaler
        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ])

        # Categorical absence sub-pipeline (Alley, BsmtQual, GarageType, etc.)
        absence_cols = [c for c in NONE_CATEGORICAL_COLS if c in self.categorical_cols_]
        absence_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        # Categorical mode sub-pipeline (MSZoning, KitchenQual, Functional, etc.)
        mode_cols = [c for c in self.categorical_cols_ if c not in absence_cols]
        mode_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = [
            ("num", numeric_pipeline, self.numeric_cols_),
        ]
        if absence_cols:
            transformers.append(("cat_absence", absence_pipeline, absence_cols))
        if mode_cols:
            transformers.append(("cat_mode", mode_pipeline, mode_cols))

        self.column_transformer = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
        )

        self.column_transformer.fit(X_engineered)
        self.feature_names_out_ = self.column_transformer.get_feature_names_out()
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self.column_transformer is None:
            raise RuntimeError("LegacyStatisticianPreprocessor must be fitted before calling transform().")
        X_clean = self._prepare_dataframe(X)
        X_engineered = self.feature_engineer.transform(X_clean)
        arr = self.column_transformer.transform(X_engineered)

        # Enforce finiteness guarantee
        if np.isnan(arr).any():
            raise ValueError("Transformed array contains NaN values.")
        if np.isinf(arr).any():
            raise ValueError("Transformed array contains infinite values.")
        return arr

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.feature_names_out_
