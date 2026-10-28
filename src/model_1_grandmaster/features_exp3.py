"""Feature engineering and preprocessing classes for Model 1 Experiment 3."""

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
from sklearn.preprocessing import OneHotEncoder

from src.model_1_grandmaster.preprocessing import NONE_CATEGORICAL_COLS


class Exp3FeatureEngineer(BaseEstimator, TransformerMixin):
    """Modular feature engineer for Experiment 3 hypothesis testing."""

    def __init__(self, variant: str = "baseline"):
        self.variant = variant

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Convert MSSubClass to string category
        if "MSSubClass" in X.columns:
            X["MSSubClass"] = X["MSSubClass"].astype(str)

        # Baseline features
        bsmt_sf = X["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in X.columns else 0
        first_sf = X["1stFlrSF"].fillna(0) if "1stFlrSF" in X.columns else 0
        second_sf = X["2ndFlrSF"].fillna(0) if "2ndFlrSF" in X.columns else 0
        X["TotalSF"] = bsmt_sf + first_sf + second_sf

        full_bath = X["FullBath"].fillna(0) if "FullBath" in X.columns else 0
        half_bath = X["HalfBath"].fillna(0) if "HalfBath" in X.columns else 0
        bsmt_full = X["BsmtFullBath"].fillna(0) if "BsmtFullBath" in X.columns else 0
        bsmt_half = X["BsmtHalfBath"].fillna(0) if "BsmtHalfBath" in X.columns else 0
        total_baths = full_bath + (0.5 * half_bath) + bsmt_full + (0.5 * bsmt_half)
        X["TotalBathrooms"] = total_baths

        wood_deck = X["WoodDeckSF"].fillna(0) if "WoodDeckSF" in X.columns else 0
        open_porch = X["OpenPorchSF"].fillna(0) if "OpenPorchSF" in X.columns else 0
        encl_porch = X["EnclosedPorch"].fillna(0) if "EnclosedPorch" in X.columns else 0
        ssn_porch = X["3SsnPorch"].fillna(0) if "3SsnPorch" in X.columns else 0
        screen_porch = X["ScreenPorch"].fillna(0) if "ScreenPorch" in X.columns else 0
        X["TotalPorchSF"] = wood_deck + open_porch + encl_porch + ssn_porch + screen_porch

        yr_sold = X["YrSold"].fillna(2010) if "YrSold" in X.columns else 2010
        if "YearBuilt" in X.columns:
            X["HouseAge"] = (yr_sold - X["YearBuilt"]).clip(lower=0)
        if "YearRemodAdd" in X.columns:
            X["RemodAge"] = (yr_sold - X["YearRemodAdd"]).clip(lower=0)

        if "GarageArea" in X.columns:
            X["HasGarage"] = (X["GarageArea"].fillna(0) > 0).astype(int)
        if "TotalBsmtSF" in X.columns:
            X["HasBasement"] = (X["TotalBsmtSF"].fillna(0) > 0).astype(int)
        if "Fireplaces" in X.columns:
            X["HasFireplace"] = (X["Fireplaces"].fillna(0) > 0).astype(int)
        if "PoolArea" in X.columns:
            X["HasPool"] = (X["PoolArea"].fillna(0) > 0).astype(int)

        # Variant 1: Remodel & Effective Age
        if self.variant in ["remodel_age", "combined"]:
            if "YearBuilt" in X.columns and "YearRemodAdd" in X.columns:
                X["IsRemodeled"] = (X["YearRemodAdd"] > X["YearBuilt"]).astype(int)
                remod_yr = X["YearRemodAdd"].fillna(X["YearBuilt"])
                X["EffectiveAge"] = (yr_sold - remod_yr).clip(lower=0)

        # Variant 2: Finished Basement & Area Decomposition
        if self.variant in ["finished_area", "combined"]:
            bsmt_tot = X["TotalBsmtSF"].fillna(0) if "TotalBsmtSF" in X.columns else 0
            bsmt_fin1 = X["BsmtFinSF1"].fillna(0) if "BsmtFinSF1" in X.columns else 0
            bsmt_fin2 = X["BsmtFinSF2"].fillna(0) if "BsmtFinSF2" in X.columns else 0
            bsmt_fin_tot = bsmt_fin1 + bsmt_fin2
            X["BsmtFinRatio"] = np.where(bsmt_tot > 0, bsmt_fin_tot / (bsmt_tot + 1e-5), 0.0)

            low_qual = X["LowQualFinSF"].fillna(0) if "LowQualFinSF" in X.columns else 0
            gr_liv = X["GrLivArea"].fillna(0) if "GrLivArea" in X.columns else 0
            X["HighQualGrLivArea"] = (gr_liv - low_qual).clip(lower=0)

        # Variant 3: Functional Layout Density
        if self.variant in ["layout_density", "combined"]:
            beds = X["BedroomAbvGr"].fillna(0).clip(lower=1) if "BedroomAbvGr" in X.columns else 1
            X["BathToBedRatio"] = total_baths / beds
            gr_liv = X["GrLivArea"].fillna(0) if "GrLivArea" in X.columns else 0
            X["SF_Per_Bedroom"] = gr_liv / beds

        return X


class GrandmasterPreprocessorExp3(BaseEstimator, TransformerMixin):
    """Leakage-safe preprocessor for Experiment 3."""

    def __init__(self, variant: str = "baseline"):
        self.variant = variant
        self.feature_engineer = Exp3FeatureEngineer(variant=variant)
        self.column_transformer: Optional[ColumnTransformer] = None
        self.feature_names_out_: Optional[np.ndarray] = None
        self.numeric_cols_: List[str] = []
        self.categorical_cols_: List[str] = []

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        if "Id" in df_copy.columns:
            df_copy = df_copy.drop(columns=["Id"])
        if "SalePrice" in df_copy.columns:
            df_copy = df_copy.drop(columns=["SalePrice"])
        return df_copy

    def fit(self, X: pd.DataFrame, y=None):
        X_clean = self._prepare_dataframe(X)
        X_engineered = self.feature_engineer.fit_transform(X_clean)

        self.numeric_cols_ = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        self.categorical_cols_ = X_engineered.select_dtypes(exclude=[np.number]).columns.tolist()

        numeric_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        absence_cols = [c for c in NONE_CATEGORICAL_COLS if c in self.categorical_cols_]
        mode_cols = [c for c in self.categorical_cols_ if c not in absence_cols]

        absence_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        mode_pipeline = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])

        transformers = [("num", numeric_pipeline, self.numeric_cols_)]
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
            raise RuntimeError("GrandmasterPreprocessorExp3 must be fitted first.")
        X_clean = self._prepare_dataframe(X)
        X_engineered = self.feature_engineer.transform(X_clean)
        return self.column_transformer.transform(X_engineered)

    def fit_transform(self, X: pd.DataFrame, y=None) -> np.ndarray:
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features=None) -> np.ndarray:
        if self.feature_names_out_ is None:
            raise RuntimeError("Preprocessor has not been fitted yet.")
        return self.feature_names_out_
