"""Feature engineering and preprocessing classes for Model 1 Experiment 4B."""

from pathlib import Path
import sys
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.model_1_grandmaster.features_exp3 import (
    Exp3FeatureEngineer,
    GrandmasterPreprocessorExp3,
)


class Exp4bFeatureEngineer(Exp3FeatureEngineer):
    """Feature engineer for Experiment 4B: Incomplete construction features.

    Builds on top of Variant 3 (Functional Layout Density) from Experiment 3.
    """

    def __init__(self, candidate: str = "baseline"):
        super().__init__(variant="layout_density")
        self.candidate = candidate

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = super().transform(X)

        if self.candidate in ["candidate_1", "candidate_2"]:
            # Primary feature: IsPartialNew
            sc = X["SaleCondition"] if "SaleCondition" in X.columns else pd.Series(["Normal"] * len(X), index=X.index)
            st = X["SaleType"] if "SaleType" in X.columns else pd.Series(["WD"] * len(X), index=X.index)
            is_partial_new = ((sc == "Partial") & (st == "New")).astype(int)
            X["IsPartialNew"] = is_partial_new

        if self.candidate == "candidate_2":
            # Optional interaction: PartialNewAreaInteraction
            gr_liv = X["GrLivArea"].fillna(0) if "GrLivArea" in X.columns else 0
            X["PartialNewAreaInteraction"] = X["IsPartialNew"] * gr_liv

        return X


class GrandmasterPreprocessorExp4b(GrandmasterPreprocessorExp3):
    """Leakage-safe preprocessor for Experiment 4B."""

    def __init__(self, candidate: str = "baseline"):
        super().__init__(variant="layout_density")
        self.candidate = candidate
        self.feature_engineer = Exp4bFeatureEngineer(candidate=candidate)
