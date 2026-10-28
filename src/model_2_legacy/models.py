"""Estimator factories and Legacy Stacking Ensemble for Model 2 (Legacy Statistician).

Architecture:
- Base Model A: LassoCV
- Base Model B: RidgeCV
- Base Model C: HuberRegressor (regularized robust linear estimator)
- Meta-Model: LassoCV (trained on base-model inner out-of-fold predictions)
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import HuberRegressor, LassoCV, RidgeCV
from sklearn.model_selection import KFold


def get_base_models(random_state: int = 42) -> Dict[str, BaseEstimator]:
    """Instantiate the 3 base linear/robust estimators for Model 2.
    
    Returns:
        Dict mapping model name to instantiated estimator.
    """
    # 1. Base Model A: LassoCV with broad log-spaced alpha grid
    lasso_model = LassoCV(
        alphas=np.logspace(-4, 0, 50),
        cv=5,
        max_iter=10000,
        random_state=random_state,
        n_jobs=1,
    )

    # 2. Base Model B: RidgeCV with broad log-spaced alpha grid
    ridge_model = RidgeCV(
        alphas=np.logspace(-1, 3, 50),
        cv=5,
    )

    # 3. Base Model C: HuberRegressor with regularized Huber loss
    huber_model = HuberRegressor(
        alpha=10.0,
        epsilon=1.35,
        max_iter=3000,
    )

    return {
        "lasso": lasso_model,
        "ridge": ridge_model,
        "huber": huber_model,
    }


def get_meta_model(random_state: int = 42, positive: bool = True) -> LassoCV:
    """Instantiate the Lasso stacking meta-model.
    
    Args:
        random_state: Seed for internal CV folds.
        positive: If True, constrains meta-coefficients to be non-negative,
                  preventing catastrophic subtraction of collinear base models.
    """
    return LassoCV(
        alphas=np.logspace(-4, 0, 50),
        cv=5,
        max_iter=5000,
        positive=positive,
        random_state=random_state,
        n_jobs=1,
    )


class LegacyStackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking ensemble combining LassoCV, RidgeCV, and HuberRegressor with a Lasso meta-model.
    
    Generates strict out-of-fold meta-features using inner K-fold CV during training.
    """

    def __init__(
        self,
        base_models: Optional[Dict[str, BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        inner_cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.base_models = base_models or get_base_models(random_state=random_state)
        self.meta_model = meta_model or get_meta_model(random_state=random_state)
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state

        self.fitted_base_models_: Dict[str, BaseEstimator] = {}
        self.fitted_meta_model_: Optional[BaseEstimator] = None
        self.meta_feature_names_: List[str] = list(self.base_models.keys())
        self.oof_predictions_: Optional[pd.DataFrame] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble using inner out-of-fold meta-features.
        
        Args:
            X: Standardized preprocessed training features.
            y: Target values (log1p(SalePrice)).
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples = len(X_arr)

        kf = KFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.random_state)
        oof_matrix = np.zeros((n_samples, len(self.base_models)))
        model_names = list(self.base_models.keys())

        # Step 1: Generate inner Out-Of-Fold predictions
        for in_fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr, y_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va = X_arr[val_idx]

            for m_idx, m_name in enumerate(model_names):
                fold_model = clone(self.base_models[m_name])
                fold_model.fit(X_tr, y_tr)
                oof_matrix[val_idx, m_idx] = fold_model.predict(X_va)

        self.oof_predictions_ = pd.DataFrame(oof_matrix, columns=model_names)

        # Step 2: Fit Lasso meta-model on inner OOF predictions
        self.fitted_meta_model_ = clone(self.meta_model)
        self.fitted_meta_model_.fit(oof_matrix, y_arr)

        # Step 3: Refit all base models on the entire outer training set
        self.fitted_base_models_ = {}
        for m_name, model in self.base_models.items():
            full_model = clone(model)
            full_model.fit(X_arr, y_arr)
            self.fitted_base_models_[m_name] = full_model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using the fitted stacking ensemble."""
        if not self.fitted_base_models_ or self.fitted_meta_model_ is None:
            raise RuntimeError("LegacyStackingEnsemble must be fitted before predict() is called.")

        X_arr = np.asarray(X)
        model_names = list(self.base_models.keys())
        meta_features = np.zeros((len(X_arr), len(model_names)))

        for m_idx, m_name in enumerate(model_names):
            meta_features[:, m_idx] = self.fitted_base_models_[m_name].predict(X_arr)

        return self.fitted_meta_model_.predict(meta_features)

    def predict_base_models(self, X: np.ndarray) -> Dict[str, np.ndarray]:
        """Return predictions from each individual fitted base model."""
        if not self.fitted_base_models_:
            raise RuntimeError("Stacking ensemble must be fitted first.")
        X_arr = np.asarray(X)
        return {
            m_name: model.predict(X_arr)
            for m_name, model in self.fitted_base_models_.items()
        }

    def get_meta_coefficients(self) -> Dict[str, float]:
        """Return named meta-model weights for diagnostic interpretation."""
        if self.fitted_meta_model_ is None:
            raise RuntimeError("Model has not been fitted.")

        if hasattr(self.fitted_meta_model_, "coef_"):
            coefs = self.fitted_meta_model_.coef_
            return {
                name: float(coefs[i])
                for i, name in enumerate(self.meta_feature_names_)
            }
        return {}
