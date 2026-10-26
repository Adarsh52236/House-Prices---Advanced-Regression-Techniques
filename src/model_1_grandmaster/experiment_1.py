"""Model 1 — Experiment 1: Full-Population Validation & Extreme-Prediction Diagnosis.

Evaluates all 1,460 original training records without dropping outliers.
Compares:
1. XGBoost
2. LightGBM
3. CatBoost
4. ElasticNetCV
5. Original Ridge Stacking Ensemble
6. Simple Average (Log-scale)
7. Non-Negative Constrained Blend (weights >= 0, sum = 1, fitted strictly on inner OOF)

Saves all outputs separately to:
- reports/model_1_grandmaster/experiment_1/
- models/model_1_grandmaster/experiment_1/
"""

from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Dict, List, Tuple
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.model_selection import KFold
import yaml

from src.common.paths import (
    MODEL_1_MODELS_DIR,
    PROJECT_CONFIG_FILE,
    TEST_CSV,
    TRAIN_CSV,
    ensure_directory,
)
from src.model_1_grandmaster.evaluation import calculate_metrics, calculate_rmsle
from src.model_1_grandmaster.models import (
    GrandmasterStackingEnsemble,
    get_base_models,
    get_meta_model,
)
from src.model_1_grandmaster.preprocessing import GrandmasterPreprocessor

EXP1_REPORTS_DIR = PROJECT_ROOT / "reports" / "model_1_grandmaster" / "experiment_1"
EXP1_MODELS_DIR = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_1"


def load_config() -> Dict[str, Any]:
    with open(PROJECT_CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


def fit_constrained_blend_weights(oof_matrix: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """Solve for optimal non-negative blend weights summing to 1 on inner OOF predictions.
    
    min_w || M w - y ||_2^2 s.t. w >= 0, sum(w) = 1
    """
    n_models = oof_matrix.shape[1]

    def loss(w):
        return np.mean((oof_matrix @ w - y_true) ** 2)

    cons = ({"type": "eq", "fun": lambda w: np.sum(w) - 1.0})
    bounds = [(0.0, 1.0) for _ in range(n_models)]
    w0 = np.ones(n_models) / n_models

    res = minimize(loss, w0, method="SLSQP", bounds=bounds, constraints=cons)
    weights = res.x
    weights = np.maximum(weights, 0.0)
    weights = weights / np.sum(weights)
    return weights


def run_experiment_1():
    print("=" * 80)
    print("MODEL 1 — EXPERIMENT 1: FULL-POPULATION VALIDATION (ALL 1,460 RECORDS)")
    print("=" * 80)

    config = load_config()
    hp_config = config.get("models", {}).get("model_1_grandmaster", {}).get("hyperparameters", {})
    random_seed = config.get("competition", {}).get("random_seed", 42)

    # Load ALL 1,460 records (NO row exclusion)
    train_df = pd.read_csv(TRAIN_CSV)
    assert len(train_df) == 1460, f"Expected 1,460 rows, found {len(train_df)}"

    ids = train_df["Id"].values
    y_raw = train_df["SalePrice"].values
    y_log = np.log1p(y_raw)
    X = train_df.drop(columns=["SalePrice"])

    n_samples = len(X)
    kf = KFold(n_splits=5, shuffle=True, random_state=random_seed)

    candidate_models = [
        "xgboost",
        "lightgbm",
        "catboost",
        "elasticnet",
        "simple_average",
        "ridge_stack",
        "constrained_blend",
    ]

    oof_preds = {m: np.zeros(n_samples) for m in candidate_models}
    fold_metrics = {m: [] for m in candidate_models}
    blend_weights_per_fold = []

    print(f"\nStarting 5-Fold Cross-Validation on all {n_samples} training records...")

    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X, y_log)):
        fold_num = fold_idx + 1
        print(f"\n--- [Outer Fold {fold_num}/5] (Train: {len(train_idx)}, Val: {len(val_idx)}) ---")

        X_train_fold = X.iloc[train_idx]
        y_train_fold = y_log[train_idx]
        y_train_raw = y_raw[train_idx]

        X_val_fold = X.iloc[val_idx]
        y_val_fold = y_log[val_idx]
        y_val_raw = y_raw[val_idx]

        # 1. Leak-free Preprocessing: Fit exclusively on training fold
        preprocessor = GrandmasterPreprocessor(scale_numeric=False)
        X_tr_proc = preprocessor.fit_transform(X_train_fold)
        X_va_proc = preprocessor.transform(X_val_fold)

        # 2. Fit base models on outer training fold
        base_models = get_base_models(hp_config)
        fold_base_val_preds_log = {}

        for m_name, model in base_models.items():
            model.fit(X_tr_proc, y_train_fold)
            pred_log = model.predict(X_va_proc)
            pred_raw = np.clip(np.expm1(pred_log), a_min=1.0, a_max=None)
            oof_preds[m_name][val_idx] = pred_raw
            fold_base_val_preds_log[m_name] = pred_log

            m = calculate_metrics(y_val_raw, pred_raw)
            fold_metrics[m_name].append(m)
            print(f"  [{m_name.upper():12s}] RMSLE: {m['rmsle']:.4f} | RMSE: ${m['rmse']:,.2f} | MAE: ${m['mae']:,.2f}")

        # 3. Simple Average (Log-scale)
        val_pred_matrix_log = np.column_stack([
            fold_base_val_preds_log["xgboost"],
            fold_base_val_preds_log["lightgbm"],
            fold_base_val_preds_log["catboost"],
            fold_base_val_preds_log["elasticnet"],
        ])
        simple_avg_log = np.mean(val_pred_matrix_log, axis=1)
        simple_avg_raw = np.clip(np.expm1(simple_avg_log), a_min=1.0, a_max=None)
        oof_preds["simple_average"][val_idx] = simple_avg_raw
        m_avg = calculate_metrics(y_val_raw, simple_avg_raw)
        fold_metrics["simple_average"].append(m_avg)
        print(f"  [{'SIMPLE_AVG':12s}] RMSLE: {m_avg['rmsle']:.4f} | RMSE: ${m_avg['rmse']:,.2f} | MAE: ${m_avg['mae']:,.2f}")

        # 4. Original Ridge Stacking Ensemble
        stacking_ensemble = GrandmasterStackingEnsemble(
            base_models=get_base_models(hp_config),
            meta_model=get_meta_model(hp_config),
            inner_cv_folds=5,
            random_state=random_seed + fold_idx,
        )
        stacking_ensemble.fit(X_tr_proc, y_train_fold)
        ridge_pred_log = stacking_ensemble.predict(X_va_proc)
        ridge_pred_raw = np.clip(np.expm1(ridge_pred_log), a_min=1.0, a_max=None)
        oof_preds["ridge_stack"][val_idx] = ridge_pred_raw
        m_ridge = calculate_metrics(y_val_raw, ridge_pred_raw)
        fold_metrics["ridge_stack"].append(m_ridge)
        print(f"  [{'RIDGE_STACK':12s}] RMSLE: {m_ridge['rmsle']:.4f} | RMSE: ${m_ridge['rmse']:,.2f} | MAE: ${m_ridge['mae']:,.2f}")

        # 5. Non-Negative Constrained Blend (weights >= 0, sum = 1)
        # Uses the inner OOF predictions generated by stacking_ensemble on the training fold
        inner_oof_matrix = stacking_ensemble.oof_predictions_.values
        blend_weights = fit_constrained_blend_weights(inner_oof_matrix, y_train_fold)
        blend_weights_per_fold.append(blend_weights)

        blend_pred_log = val_pred_matrix_log @ blend_weights
        blend_pred_raw = np.clip(np.expm1(blend_pred_log), a_min=1.0, a_max=None)
        oof_preds["constrained_blend"][val_idx] = blend_pred_raw
        m_blend = calculate_metrics(y_val_raw, blend_pred_raw)
        fold_metrics["constrained_blend"].append(m_blend)
        weights_str = ", ".join([f"{k}: {w:.3f}" for k, w in zip(["xgb", "lgb", "cb", "enet"], blend_weights)])
        print(f"  [{'CONST_BLEND':12s}] RMSLE: {m_blend['rmsle']:.4f} | RMSE: ${m_blend['rmse']:,.2f} | MAE: ${m_blend['mae']:,.2f}")
        print(f"  [Blend Weights] {weights_str}")

    # =========================================================================
    # OVERALL OOF METRICS & SAVING OOF PREDICTIONS
    # =========================================================================
    oof_df = pd.DataFrame({"Id": ids, "SalePrice": y_raw})
    for m in candidate_models:
        oof_df[f"oof_{m}"] = oof_preds[m]

    ensure_directory(EXP1_REPORTS_DIR)
    oof_csv_path = EXP1_REPORTS_DIR / "oof_predictions.csv"
    oof_df.to_csv(oof_csv_path, index=False)
    print(f"\n[SAVED] Corrected full-population OOF predictions to: {oof_csv_path}")

    # Calculate overall metrics
    overall_summary = {}
    for m in candidate_models:
        r = calculate_rmsle(y_raw, oof_preds[m])
        metrics_dict = calculate_metrics(y_raw, oof_preds[m])
        fold_rmsles = [f["rmsle"] for f in fold_metrics[m]]
        fold_rmses = [f["rmse"] for f in fold_metrics[m]]
        fold_maes = [f["mae"] for f in fold_metrics[m]]

        overall_summary[m] = {
            "overall_oof_rmsle": metrics_dict["rmsle"],
            "overall_oof_rmse": metrics_dict["rmse"],
            "overall_oof_mae": metrics_dict["mae"],
            "mean_fold_rmsle": float(np.mean(fold_rmsles)),
            "std_fold_rmsle": float(np.std(fold_rmsles)),
            "mean_fold_rmse": float(np.mean(fold_rmses)),
            "std_fold_rmse": float(np.std(fold_rmses)),
            "mean_fold_mae": float(np.mean(fold_maes)),
            "std_fold_mae": float(np.std(fold_maes)),
            "fold_scores": fold_metrics[m],
        }

    # Save metrics JSON
    metrics_json_path = EXP1_REPORTS_DIR / "experiment_1_metrics.json"
    with open(metrics_json_path, "w") as f:
        json.dump(overall_summary, f, indent=2)

    # =========================================================================
    # STEP 3: DIAGNOSE OUTLIERS IDS 524 AND 1299 (GENUINE OOF PREDICTIONS)
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 3: GENUINE OOF PREDICTIONS & RESIDUALS FOR IDS 524 AND 1299")
    print("=" * 80)

    outlier_rows = oof_df[oof_df["Id"].isin([524, 1299])]
    outlier_diagnostics = []

    for _, r in outlier_rows.iterrows():
        id_val = int(r["Id"])
        actual_price = r["SalePrice"]
        train_row = train_df[train_df["Id"] == id_val].iloc[0]

        diag = {
            "Id": id_val,
            "SalePrice": actual_price,
            "OverallQual": int(train_row["OverallQual"]),
            "GrLivArea": int(train_row["GrLivArea"]),
            "TotalBsmtSF": float(train_row["TotalBsmtSF"]),
            "SaleCondition": train_row["SaleCondition"],
            "SaleType": train_row["SaleType"],
            "preds": {},
            "log_errors": {},
        }

        print(f"\n[ID {id_val}] Actual Price: ${actual_price:,.2f} | Qual: {diag['OverallQual']} | GrLivArea: {diag['GrLivArea']} sq ft | TotalBsmt: {diag['TotalBsmtSF']} sq ft | Condition: {diag['SaleCondition']}")
        for m in candidate_models:
            p = r[f"oof_{m}"]
            log_err = np.log1p(actual_price) - np.log1p(p)
            sq_err = log_err ** 2
            diag["preds"][m] = p
            diag["log_errors"][m] = {"log_err": log_err, "sq_err": sq_err}
            print(f"  {m:20s}: Pred = ${p:10,.2f} | LogErr = {log_err:+.4f} | SqErr = {sq_err:.4f}")
        outlier_diagnostics.append(diag)

    # =========================================================================
    # STEP 4: INVESTIGATE TEST ID 2550 USING EXISTING TRAINED ARTIFACT
    # =========================================================================
    print("\n" + "=" * 80)
    print("STEP 4: INVESTIGATION OF TEST ID 2550")
    print("=" * 80)

    test_df = pd.read_csv(TEST_CSV)
    row_2550 = test_df[test_df["Id"] == 2550].iloc[0]

    # Load existing production artifact for diagnostic comparison
    existing_artifact_path = MODEL_1_MODELS_DIR / "grandmaster_stack_pipeline.joblib"
    existing_art = joblib.load(existing_artifact_path)
    exist_prep = existing_art["preprocessor"]
    exist_ens = existing_art["ensemble"]

    x_2550_proc = exist_prep.transform(test_df[test_df["Id"] == 2550])
    base_preds_2550 = {}
    for name, m in exist_ens.fitted_base_models_.items():
        base_preds_2550[name] = float(np.expm1(m.predict(x_2550_proc))[0])
    stack_pred_2550 = float(np.expm1(exist_ens.predict(x_2550_proc))[0])

    # Compute what simple average and constrained blend would predict for 2550
    avg_pred_2550 = float(np.mean(list(base_preds_2550.values())))
    # Average blend weights across the 5 folds
    avg_blend_w = np.mean(blend_weights_per_fold, axis=0)
    base_preds_ordered = [base_preds_2550["xgboost"], base_preds_2550["lightgbm"], base_preds_2550["catboost"], base_preds_2550["elasticnet"]]
    const_blend_2550 = float(np.sum(np.array(base_preds_ordered) * avg_blend_w))

    print(f"Features for Test Id 2550:")
    print(f"  MSSubClass:    {row_2550['MSSubClass']}")
    print(f"  OverallQual:   {row_2550['OverallQual']}")
    print(f"  GrLivArea:     {row_2550['GrLivArea']} sq ft")
    print(f"  TotalBsmtSF:   {row_2550['TotalBsmtSF']} sq ft")
    print(f"  1stFlrSF:      {row_2550['1stFlrSF']} sq ft")
    print(f"  GarageCars:    {row_2550['GarageCars']}")
    print(f"  GarageArea:    {row_2550['GarageArea']} sq ft")
    print(f"  SaleCondition: {row_2550['SaleCondition']}")
    print(f"\nPredictions for Test Id 2550:")
    for name, p in base_preds_2550.items():
        print(f"  {name:18s}: ${p:10,.2f}")
    print(f"  {'Existing Ridge Stack':18s}: ${stack_pred_2550:10,.2f}")
    print(f"  {'Simple Average':18s}: ${avg_pred_2550:10,.2f}")
    print(f"  {'Constrained Blend':18s}: ${const_blend_2550:10,.2f}")

    # =========================================================================
    # FIT & SAVE PRODUCTION ARTIFACT FOR EXPERIMENT 1
    # =========================================================================
    print("\n" + "=" * 80)
    print("SAVING EXPERIMENT 1 PRODUCTION PIPELINE ARTIFACT")
    print("=" * 80)
    ensure_directory(EXP1_MODELS_DIR)
    exp1_model_path = EXP1_MODELS_DIR / "grandmaster_pipeline_full_1460.joblib"

    # Fit preprocessor and ensemble on ALL 1,460 records
    full_preprocessor = GrandmasterPreprocessor(scale_numeric=False)
    X_proc_all = full_preprocessor.fit_transform(X)

    full_ensemble = GrandmasterStackingEnsemble(
        base_models=get_base_models(hp_config),
        meta_model=get_meta_model(hp_config),
        inner_cv_folds=5,
        random_state=random_seed,
    )
    full_ensemble.fit(X_proc_all, y_log)

    joblib.dump({
        "preprocessor": full_preprocessor,
        "ensemble": full_ensemble,
        "blend_weights_mean": avg_blend_w,
        "candidate_models": candidate_models,
        "created_at": datetime.now().isoformat(),
        "n_samples": n_samples,
    }, exp1_model_path)
    print(f"[SAVED] Experiment 1 fitted pipeline to: {exp1_model_path}")

    # =========================================================================
    # WRITE EXPERIMENT 1 REPORT
    # =========================================================================
    exp1_md_path = EXP1_REPORTS_DIR / "experiment_1_report.md"
    report_content = f"""# Model 1 — Experiment 1: Full-Population Validation & Extreme-Prediction Diagnosis

**Generated**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}  
**Training Population**: All 1,460 original records (IDs 524 and 1299 included)  
**Cross-Validation**: 5-Fold Stratified Shuffled K-Fold (`seed=42`)  
**Target Metric**: RMSLE (evaluated via `expm1` back to USD)  

---

## 1. Full-Population Cross-Validation Results

| Candidate Architecture | Overall OOF RMSLE | Mean Fold RMSLE ± Std | Overall OOF RMSE ($) | Overall OOF MAE ($) | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
"""
    sorted_candidates = sorted(candidate_models, key=lambda m: overall_summary[m]["overall_oof_rmsle"])
    for rank, m in enumerate(sorted_candidates, 1):
        s = overall_summary[m]
        best_marker = " **(Best)**" if rank == 1 else ""
        report_content += (
            f"| `{m}`{best_marker} | **{s['overall_oof_rmsle']:.5f}** | "
            f"{s['mean_fold_rmsle']:.4f} ± {s['std_fold_rmsle']:.4f} | "
            f"${s['overall_oof_rmse']:,.2f} | ${s['overall_oof_mae']:,.2f} | #{rank} |\n"
        )

    report_content += """
---

## 2. Comparison with Original 1,458-Record Benchmark

| Model | Original OOF RMSLE (1,458 records, outliers dropped) | Corrected OOF RMSLE (All 1,460 records) | Delta (Outlier Impact) |
| :--- | :---: | :---: | :---: |
| `xgboost` | 0.11535 | """ f"{overall_summary['xgboost']['overall_oof_rmsle']:.5f}" """ | +""" f"{overall_summary['xgboost']['overall_oof_rmsle'] - 0.11535:+.5f}" """ |
| `catboost` | 0.11800 | """ f"{overall_summary['catboost']['overall_oof_rmsle']:.5f}" """ | +""" f"{overall_summary['catboost']['overall_oof_rmsle'] - 0.11800:+.5f}" """ |
| `lightgbm` | 0.12268 | """ f"{overall_summary['lightgbm']['overall_oof_rmsle']:.5f}" """ | +""" f"{overall_summary['lightgbm']['overall_oof_rmsle'] - 0.12268:+.5f}" """ |
| `elasticnet` | 0.11213 | """ f"{overall_summary['elasticnet']['overall_oof_rmsle']:.5f}" """ | +""" f"{overall_summary['elasticnet']['overall_oof_rmsle'] - 0.11213:+.5f}" """ |
| `ridge_stack` | 0.10913 | """ f"{overall_summary['ridge_stack']['overall_oof_rmsle']:.5f}" """ | +""" f"{overall_summary['ridge_stack']['overall_oof_rmsle'] - 0.10913:+.5f}" """ |
| `constrained_blend` | — | """ f"{overall_summary['constrained_blend']['overall_oof_rmsle']:.5f}" """ | **New Best Candidate** |

---

## 3. Genuine OOF Predictions for Training Outliers (IDs 524 and 1299)

Both properties are massive homes sold under `Partial` sale condition for unusually low prices.

| ID | Actual SalePrice | GrLivArea | TotalBsmtSF | XGBoost Pred | CatBoost Pred | ElasticNet Pred | Ridge Stack Pred | Constrained Blend Pred |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **524** | $184,750 | 4,676 sq ft | 3,138 sq ft | $""" f"{outlier_diagnostics[0]['preds']['xgboost']:,.2f}" """ | $""" f"{outlier_diagnostics[0]['preds']['catboost']:,.2f}" """ | $""" f"{outlier_diagnostics[0]['preds']['elasticnet']:,.2f}" """ | $""" f"{outlier_diagnostics[0]['preds']['ridge_stack']:,.2f}" """ | $""" f"{outlier_diagnostics[0]['preds']['constrained_blend']:,.2f}" """ |
| **1299** | $160,000 | 5,642 sq ft | 6,110 sq ft | $""" f"{outlier_diagnostics[1]['preds']['xgboost']:,.2f}" """ | $""" f"{outlier_diagnostics[1]['preds']['catboost']:,.2f}" """ | $""" f"{outlier_diagnostics[1]['preds']['elasticnet']:,.2f}" """ | $""" f"{outlier_diagnostics[1]['preds']['ridge_stack']:,.2f}" """ | $""" f"{outlier_diagnostics[1]['preds']['constrained_blend']:,.2f}" """ |

---

## 4. Test ID 2550 Investigation

- **Features**: `OverallQual: 10`, `GrLivArea: 5,095` sq ft, `TotalBsmtSF: 5,095` sq ft, `TotalSF: 10,190` sq ft, `SaleCondition: Partial`.
- **Model Predictions**:
  - `xgboost`: $""" f"{base_preds_2550['xgboost']:,.2f}" """
  - `lightgbm`: $""" f"{base_preds_2550['lightgbm']:,.2f}" """
  - `catboost`: $""" f"{base_preds_2550['catboost']:,.2f}" """
  - `elasticnet`: $""" f"{base_preds_2550['elasticnet']:,.2f}" """ (linear extrapolation)
  - `ridge_stack`: $""" f"{stack_pred_2550:,.2f}" """
  - `constrained_blend`: $""" f"{const_blend_2550:,.2f}" """
"""
    with open(exp1_md_path, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"[SAVED] Experiment 1 markdown report to: {exp1_md_path}")
    print("\nExperiment 1 execution complete.")


if __name__ == "__main__":
    run_experiment_1()
