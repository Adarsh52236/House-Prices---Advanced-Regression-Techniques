"""Estimator factories and Grandmaster Stacking Ensemble for Model 1.

Includes:
- XGBoost Regressor
- LightGBM Regressor
- CatBoost Regressor
- ElasticNetCV (with internal StandardScaler)
- RidgeCV (meta-model)
"""

from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import catboost as cb
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import ElasticNetCV, RidgeCV
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


def get_base_models(
    config_hyperparams: Optional[Dict[str, Any]] = None,
    n_estimators_override: Optional[int] = None,
) -> Dict[str, BaseEstimator]:
    """Instantiate the 4 base regressors with configurable hyperparameters.
    
    Args:
        config_hyperparams: Dictionary of hyperparameters from project_config.yaml.
        n_estimators_override: If provided, overrides iteration/estimator count (for fast testing).
        
    Returns:
        Dict mapping model name to instantiated estimator.
    """
    hp = config_hyperparams or {}
    xgb_hp = hp.get("xgboost", {})
    lgb_hp = hp.get("lightgbm", {})
    cb_hp = hp.get("catboost", {})
    enet_hp = hp.get("elasticnet", {})

    xgb_estimators = n_estimators_override or xgb_hp.get("n_estimators", 3000)
    lgb_estimators = n_estimators_override or lgb_hp.get("n_estimators", 3000)
    cb_iterations = n_estimators_override or cb_hp.get("iterations", 3000)

    # 1. XGBoost Regressor
    xgb_model = xgb.XGBRegressor(
        n_estimators=xgb_estimators,
        learning_rate=xgb_hp.get("learning_rate", 0.01),
        max_depth=xgb_hp.get("max_depth", 4),
        subsample=xgb_hp.get("subsample", 0.7),
        colsample_bytree=xgb_hp.get("colsample_bytree", 0.7),
        random_state=xgb_hp.get("random_state", 42),
        n_jobs=xgb_hp.get("n_jobs", 4),
        tree_method="hist",
    )

    # 2. LightGBM Regressor
    lgb_model = lgb.LGBMRegressor(
        n_estimators=lgb_estimators,
        learning_rate=lgb_hp.get("learning_rate", 0.01),
        max_depth=lgb_hp.get("max_depth", 4),
        num_leaves=lgb_hp.get("num_leaves", 15),
        subsample=lgb_hp.get("subsample", 0.7),
        random_state=lgb_hp.get("random_state", 42),
        n_jobs=lgb_hp.get("n_jobs", 4),
        verbose=-1,
    )

    # 3. CatBoost Regressor
    cb_model = cb.CatBoostRegressor(
        iterations=cb_iterations,
        learning_rate=cb_hp.get("learning_rate", 0.01),
        depth=cb_hp.get("depth", 4),
        verbose=0,
        random_seed=cb_hp.get("random_seed", 42),
        thread_count=cb_hp.get("thread_count", 4),
    )

    # 4. ElasticNetCV (standardized inside pipeline to ensure zero scale leakage)
    elastic_net = ElasticNetCV(
        l1_ratio=enet_hp.get("l1_ratio", [0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0]),
        alphas=enet_hp.get("alphas", [0.0001, 0.0005, 0.001, 0.01]),
        cv=enet_hp.get("cv", 5),
        max_iter=enet_hp.get("max_iter", 3000),
        tol=enet_hp.get("tol", 1e-3),
        selection="random",
        random_state=enet_hp.get("random_state", 42),
        n_jobs=1,  # Keep inner CV single-threaded to prevent nested thrashing
    )
    elastic_net_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("elasticnet", elastic_net),
    ])

    return {
        "xgboost": xgb_model,
        "lightgbm": lgb_model,
        "catboost": cb_model,
        "elasticnet": elastic_net_pipeline,
    }


def get_meta_model(config_hyperparams: Optional[Dict[str, Any]] = None) -> RidgeCV:
    """Instantiate the RidgeCV stacking meta-model."""
    hp = config_hyperparams or {}
    ridge_hp = hp.get("ridge_meta", {})
    alphas = ridge_hp.get("alphas", [0.1, 1.0, 10.0])
    cv = ridge_hp.get("cv", 5)
    return RidgeCV(alphas=alphas, cv=cv)


class GrandmasterStackingEnsemble(BaseEstimator, RegressorMixin):
    """Stacking ensemble combining XGBoost, LightGBM, CatBoost, and ElasticNetCV with RidgeCV.
    
    Generates strict out-of-fold meta-features using inner K-fold CV during training.
    """

    def __init__(
        self,
        base_models: Optional[Dict[str, BaseEstimator]] = None,
        meta_model: Optional[BaseEstimator] = None,
        inner_cv_folds: int = 5,
        random_state: int = 42,
    ):
        self.base_models = base_models or get_base_models()
        self.meta_model = meta_model or get_meta_model()
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state

        self.fitted_base_models_: Dict[str, BaseEstimator] = {}
        self.fitted_meta_model_: Optional[BaseEstimator] = None
        self.meta_feature_names_: List[str] = list(self.base_models.keys())
        self.oof_predictions_: Optional[pd.DataFrame] = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        """Fit stacking ensemble on training data using out-of-fold meta-features.
        
        Args:
            X: Training features (e.g. preprocessed numpy array).
            y: Target values (e.g. log1p(SalePrice)).
        """
        X_arr = np.asarray(X)
        y_arr = np.asarray(y)
        n_samples = len(X_arr)

        kf = KFold(n_splits=self.inner_cv_folds, shuffle=True, random_state=self.random_state)
        oof_matrix = np.zeros((n_samples, len(self.base_models)))
        model_names = list(self.base_models.keys())

        # Step 1: Generate Out-Of-Fold base predictions
        for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_arr, y_arr)):
            X_tr, y_tr = X_arr[train_idx], y_arr[train_idx]
            X_va = X_arr[val_idx]

            for m_idx, m_name in enumerate(model_names):
                fold_model = clone(self.base_models[m_name])
                fold_model.fit(X_tr, y_tr)
                oof_matrix[val_idx, m_idx] = fold_model.predict(X_va)

        self.oof_predictions_ = pd.DataFrame(oof_matrix, columns=model_names)

        # Step 2: Fit meta-model on OOF predictions
        self.fitted_meta_model_ = clone(self.meta_model)
        self.fitted_meta_model_.fit(oof_matrix, y_arr)

        # Step 3: Refit all base models on the entire training set
        self.fitted_base_models_ = {}
        for m_name, model in self.base_models.items():
            full_model = clone(model)
            full_model.fit(X_arr, y_arr)
            self.fitted_base_models_[m_name] = full_model

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values using the fitted stacking ensemble."""
        if not self.fitted_base_models_ or self.fitted_meta_model_ is None:
            raise RuntimeError("Stacking ensemble must be fitted before predict() is called.")

        X_arr = np.asarray(X)
        model_names = list(self.base_models.keys())
        meta_features = np.zeros((len(X_arr), len(model_names)))

        for m_idx, m_name in enumerate(model_names):
            meta_features[:, m_idx] = self.fitted_base_models_[m_name].predict(X_arr)

        return self.fitted_meta_model_.predict(meta_features)

    def get_meta_coefficients(self) -> Dict[str, float]:
        """Return named meta-model weights for diagnostic interpretation."""
        if self.fitted_meta_model_ is None:
            raise RuntimeError("Model has not been fitted.")

        # If meta_model is RidgeCV or similar linear estimator
        if hasattr(self.fitted_meta_model_, "coef_"):
            coefs = self.fitted_meta_model_.coef_
            return {
                name: float(coefs[i])
                for i, name in enumerate(self.meta_feature_names_)
            }
        return {}
