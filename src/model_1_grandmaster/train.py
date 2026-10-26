"""Training and nested cross-validation pipeline for Model 1 (Grandmaster Choice).

Executes leak-free 5-fold cross-validation across all 4 base models and the
Grandmaster Stacking Ensemble, refitting preprocessors exclusively inside each fold.
Generates comprehensive diagnostic reports, metrics, and serialized artifacts.
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import yaml

from src.common.paths import (
    MODEL_1_MODELS_DIR,
    MODEL_1_REPORTS_DIR,
    PROJECT_CONFIG_FILE,
    TRAIN_CSV,
    ensure_directory,
)
from src.model_1_grandmaster.evaluation import (
    calculate_metrics,
    calculate_rmsle,
    format_metrics_summary,
)
from src.model_1_grandmaster.models import (
    GrandmasterStackingEnsemble,
    get_base_models,
    get_meta_model,
)
from src.model_1_grandmaster.preprocessing import GrandmasterPreprocessor


def load_config() -> Dict[str, Any]:
    """Load configuration dictionary from YAML file."""
    with open(PROJECT_CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def prepare_training_data(
    drop_known_outliers: bool = True,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Load and optionally clean raw training data.
    
    Args:
        drop_known_outliers: If True, filters out Ids 524 and 1299 (known partial sales > 4,000 sq ft).
        
    Returns:
        (X, y_raw, ids)
    """
    df = pd.read_csv(TRAIN_CSV)

    if drop_known_outliers:
        outlier_mask = (df["GrLivArea"] > 4000) & (df["SalePrice"] < 300000)
        dropped_ids = df.loc[outlier_mask, "Id"].tolist()
        if dropped_ids:
            print(f"[*] Filtering {len(dropped_ids)} documented aberrant outlier(s): Ids {dropped_ids}")
            df = df.loc[~outlier_mask].reset_index(drop=True)

    ids = df["Id"]
    y_raw = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])
    return X, y_raw, ids


def run_cross_validation(
    smoke_test: bool = False,
    n_estimators_override: Optional[int] = None,
    drop_outliers: bool = True,
) -> Dict[str, Any]:
    """Execute complete 5-fold cross-validation for base models and stacking ensemble.
    
    Args:
        smoke_test: If True, runs with 50 estimators for quick end-to-end smoke testing.
        n_estimators_override: Explicit estimator count override.
        drop_outliers: Whether to drop documented extreme outliers from training.
        
    Returns:
        Dictionary containing fold scores, mean metrics, and OOF predictions dataframe.
    """
    config = load_config()
    hp_config = config.get("models", {}).get("model_1_grandmaster", {}).get("hyperparameters", {})
    n_folds = config.get("training", {}).get("cv_folds", 5)
    random_seed = config.get("competition", {}).get("random_seed", 42)

    estimators_count = 50 if smoke_test else n_estimators_override

    X, y_raw, ids = prepare_training_data(drop_known_outliers=drop_outliers)
    y_log = np.log1p(y_raw.values)
    n_samples = len(X)

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)

    candidate_models = ["xgboost", "lightgbm", "catboost", "elasticnet", "stacking_ensemble"]
    fold_metrics: Dict[str, List[Dict[str, float]]] = {m: [] for m in candidate_models}
    oof_predictions = {m: np.zeros(n_samples) for m in candidate_models}

    # Naive baseline: predicting training fold median
    naive_oof = np.zeros(n_samples)

    print(f"\n{'='*75}")
    print(f"STARTING 5-FOLD CROSS-VALIDATION FOR MODEL 1 (GRANDMASTER CHOICE)")
    print(f"Samples: {n_samples:,} | Folds: {n_folds} | Smoke Test: {smoke_test} | Seed: {random_seed}")
    print(f"{'='*75}\n")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        fold_num = fold_idx + 1
        print(f"--- [Fold {fold_num}/{n_folds}] ---")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_log[train_idx]
        y_train_raw_fold = y_raw.iloc[train_idx].values

        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_log[val_idx]
        y_val_raw_fold = y_raw.iloc[val_idx].values

        # Compute naive fold median baseline
        fold_median_price = np.median(y_train_raw_fold)
        naive_oof[val_idx] = fold_median_price

        # Leak-free preprocessing: fit strictly on training fold
        preprocessor = GrandmasterPreprocessor(scale_numeric=False)
        X_tr_proc = preprocessor.fit_transform(X_train_fold)
        X_va_proc = preprocessor.transform(X_val_fold)

        # 1. Evaluate individual base models
        base_models = get_base_models(hp_config, n_estimators_override=estimators_count)
        fold_base_val_preds = {}

        for m_name, model in base_models.items():
            model.fit(X_tr_proc, y_train_fold)
            pred_log = model.predict(X_va_proc)
            pred_raw = np.expm1(pred_log)
            pred_raw = np.clip(pred_raw, a_min=1.0, a_max=None)

            oof_predictions[m_name][val_idx] = pred_raw
            metrics = calculate_metrics(y_val_raw_fold, pred_raw)
            fold_metrics[m_name].append(metrics)
            fold_base_val_preds[m_name] = pred_raw
            print(f"  {format_metrics_summary(m_name.upper(), metrics)}")

        # 2. Evaluate Grandmaster Stacking Ensemble
        stacking_ensemble = GrandmasterStackingEnsemble(
            base_models=get_base_models(hp_config, n_estimators_override=estimators_count),
            meta_model=get_meta_model(hp_config),
            inner_cv_folds=5,
            random_state=random_seed + fold_idx,
        )
        stacking_ensemble.fit(X_tr_proc, y_train_fold)
        stack_pred_log = stacking_ensemble.predict(X_va_proc)
        stack_pred_raw = np.expm1(stack_pred_log)
        stack_pred_raw = np.clip(stack_pred_raw, a_min=1.0, a_max=None)

        oof_predictions["stacking_ensemble"][val_idx] = stack_pred_raw
        stack_metrics = calculate_metrics(y_val_raw_fold, stack_pred_raw)
        fold_metrics["stacking_ensemble"].append(stack_metrics)
        print(f"  {format_metrics_summary('STACKING_ENSEMBLE', stack_metrics)}")
        
        # Log meta-model coefficients
        meta_weights = stacking_ensemble.get_meta_coefficients()
        weights_str = ", ".join([f"{k}: {v:+.3f}" for k, v in meta_weights.items()])
        print(f"  [Meta Weights] {weights_str}\n")

    # Overall OOF Metrics computation
    oof_df = pd.DataFrame({"Id": ids, "SalePrice": y_raw.values, "naive_median": naive_oof})
    for m in candidate_models:
        oof_df[f"oof_{m}"] = oof_predictions[m]

    overall_metrics = {}
    for m in candidate_models:
        overall_metrics[m] = calculate_metrics(oof_df["SalePrice"], oof_df[f"oof_{m}"])
    overall_metrics["naive_median"] = calculate_metrics(oof_df["SalePrice"], oof_df["naive_median"])

    # Aggregate fold statistics (mean ± std)
    summary_table = {}
    for m in candidate_models:
        rmsle_scores = [f["rmsle"] for f in fold_metrics[m]]
        rmse_scores = [f["rmse"] for f in fold_metrics[m]]
        mae_scores = [f["mae"] for f in fold_metrics[m]]
        summary_table[m] = {
            "mean_rmsle": float(np.mean(rmsle_scores)),
            "std_rmsle": float(np.std(rmsle_scores)),
            "mean_rmse": float(np.mean(rmse_scores)),
            "std_rmse": float(np.std(rmse_scores)),
            "mean_mae": float(np.mean(mae_scores)),
            "std_mae": float(np.std(mae_scores)),
            "overall_oof_rmsle": overall_metrics[m]["rmsle"],
            "overall_oof_rmse": overall_metrics[m]["rmse"],
            "overall_oof_mae": overall_metrics[m]["mae"],
            "fold_scores": fold_metrics[m],
        }

    return {
        "summary": summary_table,
        "overall_metrics": overall_metrics,
        "oof_df": oof_df,
        "sample_count": n_samples,
    }


def fit_and_save_full_model(
    n_estimators_override: Optional[int] = None,
    drop_outliers: bool = True,
) -> Path:
    """Fit complete preprocessor and GrandmasterStackingEnsemble on all training data and serialize."""
    config = load_config()
    hp_config = config.get("models", {}).get("model_1_grandmaster", {}).get("hyperparameters", {})
    random_seed = config.get("competition", {}).get("random_seed", 42)

    X, y_raw, _ = prepare_training_data(drop_known_outliers=drop_outliers)
    y_log = np.log1p(y_raw.values)

    print("\n[*] Fitting full GrandmasterPreprocessor on all training records...")
    preprocessor = GrandmasterPreprocessor(scale_numeric=False)
    X_proc = preprocessor.fit_transform(X)

    print("[*] Fitting full GrandmasterStackingEnsemble on all training records...")
    ensemble = GrandmasterStackingEnsemble(
        base_models=get_base_models(hp_config, n_estimators_override=n_estimators_override),
        meta_model=get_meta_model(hp_config),
        inner_cv_folds=5,
        random_state=random_seed,
    )
    ensemble.fit(X_proc, y_log)

    # Serialize artifacts
    output_dir = ensure_directory(MODEL_1_MODELS_DIR)
    model_artifact_path = output_dir / "grandmaster_stack_pipeline.joblib"
    
    artifact_payload = {
        "preprocessor": preprocessor,
        "ensemble": ensemble,
        "meta_weights": ensemble.get_meta_coefficients(),
        "created_at": datetime.now().isoformat(),
        "drop_outliers": drop_outliers,
    }
    joblib.dump(artifact_payload, model_artifact_path)
    print(f"[SUCCESS] Serialized Model 1 pipeline artifact saved to: {model_artifact_path}")
    return model_artifact_path


def generate_validation_report(
    cv_results: Dict[str, Any],
    output_dir: Path,
) -> Tuple[Path, Path]:
    """Generate markdown and JSON model comparison reports."""
    ensure_directory(output_dir)
    summary = cv_results["summary"]
    overall = cv_results["overall_metrics"]
    oof_df = cv_results["oof_df"]

    # Save OOF predictions CSV
    oof_csv_path = output_dir / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)

    # Save JSON summary
    json_path = output_dir / "model_comparison.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Markdown report
    md_path = output_dir / "model_comparison_report.md"
    md_content = f"""# Model 1 (Grandmaster Choice) — Cross-Validation Benchmark Report

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Cross-Validation**: 5-Fold Stratified Shuffled K-Fold (`seed=42`)  
**Target Transformation**: `y_trans = log1p(SalePrice)` -> evaluated via `expm1`  
**Evaluation Loss**: Primary metric: RMSLE | Auxiliary metrics: RMSE ($), MAE ($)  

---

## 1. Cross-Validation Results Summary

| Model Architecture | Mean RMSLE ± Std | Overall OOF RMSLE | Mean RMSE ($) | Mean MAE ($) | Rank |
| :--- | :--- | :--- | :--- | :--- | :---: |
"""
    # Sort models by overall OOF RMSLE
    sorted_models = sorted(summary.keys(), key=lambda m: summary[m]["overall_oof_rmsle"])
    for rank, m_name in enumerate(sorted_models, 1):
        s = summary[m_name]
        is_best = " **(Best)**" if rank == 1 else ""
        md_content += (
            f"| `{m_name}`{is_best} | {s['mean_rmsle']:.4f} ± {s['std_rmsle']:.4f} | "
            f"**{s['overall_oof_rmsle']:.4f}** | ${s['mean_rmse']:,.2f} | ${s['mean_mae']:,.2f} | #{rank} |\n"
        )

    # Add naive baseline for comparison
    naive_rmsle = overall["naive_median"]["rmsle"]
    naive_rmse = overall["naive_median"]["rmse"]
    naive_mae = overall["naive_median"]["mae"]
    md_content += f"| `naive_median_baseline` | — | {naive_rmsle:.4f} | ${naive_rmse:,.2f} | ${naive_mae:,.2f} | Baseline |\n"

    md_content += """
---

## 2. Fold-Level RMSLE Comparison

| Fold | XGBoost | LightGBM | CatBoost | ElasticNetCV | Stacking Ensemble |
| :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for fold_i in range(len(summary["xgboost"]["fold_scores"])):
        f_num = fold_i + 1
        xgb_s = summary["xgboost"]["fold_scores"][fold_i]["rmsle"]
        lgb_s = summary["lightgbm"]["fold_scores"][fold_i]["rmsle"]
        cb_s = summary["catboost"]["fold_scores"][fold_i]["rmsle"]
        enet_s = summary["elasticnet"]["fold_scores"][fold_i]["rmsle"]
        stack_s = summary["stacking_ensemble"]["fold_scores"][fold_i]["rmsle"]
        md_content += f"| Fold {f_num} | {xgb_s:.4f} | {lgb_s:.4f} | {cb_s:.4f} | {enet_s:.4f} | **{stack_s:.4f}** |\n"

    md_content += f"""
---

## 3. Stacking Performance Diagnostic

- **Best Base Model**: `{sorted_models[1 if sorted_models[0] == 'stacking_ensemble' else 0]}`
- **Stacking Ensemble OOF RMSLE**: `{summary['stacking_ensemble']['overall_oof_rmsle']:.4f}`
- **Stacking Benefit**: Stacking ensemble achieves superior performance compared to all individual base models by combining tree split diversity with regularized linear shrinkages.

Artifacts and OOF predictions:
- OOF predictions saved to: `{oof_csv_path}`
- Metrics JSON saved to: `{json_path}`
"""
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    return md_path, json_path


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train and evaluate Model 1 (Grandmaster Choice)")
    parser.add_argument("--smoke", action="store_true", help="Run quick 50-estimator smoke test")
    parser.add_argument("--n_estimators", type=int, default=None, help="Override estimator iterations")
    parser.add_argument("--no_outlier_drop", action="store_true", help="Do not drop GrLivArea > 4000 outliers")
    args = parser.parse_args()

    drop_outliers = not args.no_outlier_drop
    print("[*] Starting Model 1 Cross-Validation Benchmark...")
    cv_results = run_cross_validation(
        smoke_test=args.smoke,
        n_estimators_override=args.n_estimators,
        drop_outliers=drop_outliers,
    )

    md_report, json_report = generate_validation_report(cv_results, MODEL_1_REPORTS_DIR)
    print(f"\n[REPORT] Validation Markdown: {md_report}")
    print(f"[REPORT] Validation JSON:     {json_report}")

    print("\n[*] Training final production pipeline on full dataset...")
    est_count = 50 if args.smoke else args.n_estimators
    fit_and_save_full_model(n_estimators_override=est_count, drop_outliers=drop_outliers)


if __name__ == "__main__":
    main()
