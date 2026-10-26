# Model 1 (Grandmaster Choice) — Cross-Validation Benchmark Report

**Generated**: 2026-10-26 14:07:33  
**Cross-Validation**: 5-Fold Stratified Shuffled K-Fold (`seed=42`)  
**Target Transformation**: `y_trans = log1p(SalePrice)` -> evaluated via `expm1`  
**Evaluation Loss**: Primary metric: RMSLE | Auxiliary metrics: RMSE ($), MAE ($)  

---

## 1. Cross-Validation Results Summary

| Model Architecture | Mean RMSLE ± Std | Overall OOF RMSLE | Mean RMSE ($) | Mean MAE ($) | Rank |
| :--- | :--- | :--- | :--- | :--- | :---: |
| `stacking_ensemble` **(Best)** | 0.1089 ± 0.0070 | **0.1091** | $19,376.97 | $12,864.63 | #1 |
| `elasticnet` | 0.1119 ± 0.0069 | **0.1121** | $20,241.01 | $13,368.82 | #2 |
| `xgboost` | 0.1151 ± 0.0073 | **0.1154** | $22,089.09 | $13,871.40 | #3 |
| `catboost` | 0.1177 ± 0.0082 | **0.1180** | $23,700.72 | $14,235.70 | #4 |
| `lightgbm` | 0.1225 ± 0.0064 | **0.1227** | $23,749.04 | $14,794.19 | #5 |
| `naive_median_baseline` | — | 0.4003 | $81,464.79 | $55,629.42 | Baseline |

---

## 2. Fold-Level RMSLE Comparison

| Fold | XGBoost | LightGBM | CatBoost | ElasticNetCV | Stacking Ensemble |
| :---: | :---: | :---: | :---: | :---: | :---: |
| Fold 1 | 0.1192 | 0.1264 | 0.1184 | 0.1144 | **0.1121** |
| Fold 2 | 0.1067 | 0.1131 | 0.1115 | 0.1101 | **0.1050** |
| Fold 3 | 0.1192 | 0.1265 | 0.1243 | 0.1199 | **0.1172** |
| Fold 4 | 0.1242 | 0.1298 | 0.1285 | 0.1156 | **0.1128** |
| Fold 5 | 0.1063 | 0.1167 | 0.1059 | 0.0996 | **0.0974** |

---

## 3. Stacking Performance Diagnostic

- **Best Base Model**: `elasticnet`
- **Stacking Ensemble OOF RMSLE**: `0.1091`
- **Stacking Benefit**: Stacking ensemble achieves superior performance compared to all individual base models by combining tree split diversity with regularized linear shrinkages.

Artifacts and OOF predictions:
- OOF predictions saved to: `C:\Users\adars\OneDrive\Documents\PROJECTS\House Prices - Advanced Regression Techniques\reports\model_1_grandmaster\oof_predictions.csv`
- Metrics JSON saved to: `C:\Users\adars\OneDrive\Documents\PROJECTS\House Prices - Advanced Regression Techniques\reports\model_1_grandmaster\model_comparison.json`
