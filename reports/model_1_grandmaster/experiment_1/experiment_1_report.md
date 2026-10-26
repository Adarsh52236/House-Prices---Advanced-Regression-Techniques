# Model 1 — Experiment 1: Full-Population Validation & Extreme-Prediction Diagnosis

**Generated**: 2026-10-26 15:36:35  
**Training Population**: All 1,460 original records (IDs 524 and 1299 included)  
**Cross-Validation**: 5-Fold Stratified Shuffled K-Fold (`seed=42`)  
**Target Metric**: RMSLE (evaluated via `expm1` back to USD)  

---

## 1. Full-Population Cross-Validation Results

| Candidate Architecture | Overall OOF RMSLE | Mean Fold RMSLE ± Std | Overall OOF RMSE ($) | Overall OOF MAE ($) | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `xgboost` **(Best)** | **0.12516** | 0.1239 ± 0.0175 | $28,366.96 | $14,590.16 | #1 |
| `simple_average` | **0.12672** | 0.1246 ± 0.0232 | $31,442.28 | $14,497.41 | #2 |
| `catboost` | **0.12778** | 0.1268 ± 0.0155 | $29,416.73 | $15,169.72 | #3 |
| `lightgbm` | **0.13115** | 0.1299 ± 0.0180 | $28,910.15 | $15,499.87 | #4 |
| `constrained_blend` | **0.13564** | 0.1317 ± 0.0324 | $45,636.33 | $15,236.31 | #5 |
| `ridge_stack` | **0.13611** | 0.1317 ± 0.0344 | $47,904.49 | $15,233.30 | #6 |
| `elasticnet` | **0.14865** | 0.1419 ± 0.0442 | $76,801.52 | $16,186.74 | #7 |

---

## 2. Comparison with Original 1,458-Record Benchmark

| Model | Original OOF RMSLE (1,458 records, outliers dropped) | Corrected OOF RMSLE (All 1,460 records) | Delta (Outlier Impact) |
| :--- | :---: | :---: | :---: |
| `xgboost` | 0.11535 | 0.12516 | ++0.00981 |
| `catboost` | 0.11800 | 0.12778 | ++0.00978 |
| `lightgbm` | 0.12268 | 0.13115 | ++0.00847 |
| `elasticnet` | 0.11213 | 0.14865 | ++0.03652 |
| `ridge_stack` | 0.10913 | 0.13611 | ++0.02698 |
| `constrained_blend` | — | 0.13564 | **New Best Candidate** |

---

## 3. Genuine OOF Predictions for Training Outliers (IDs 524 and 1299)

Both properties are massive homes sold under `Partial` sale condition for unusually low prices.

| ID | Actual SalePrice | GrLivArea | TotalBsmtSF | XGBoost Pred | CatBoost Pred | ElasticNet Pred | Ridge Stack Pred | Constrained Blend Pred |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **524** | $184,750 | 4,676 sq ft | 3,138 sq ft | $638,120.75 | $608,709.66 | $1,048,486.43 | $918,325.54 | $858,856.93 |
| **1299** | $160,000 | 5,642 sq ft | 6,110 sq ft | $517,753.31 | $536,233.47 | $2,835,420.44 | $1,577,810.79 | $1,479,719.50 |

---

## 4. Test ID 2550 Investigation

- **Features**: `OverallQual: 10`, `GrLivArea: 5,095` sq ft, `TotalBsmtSF: 5,095` sq ft, `TotalSF: 10,190` sq ft, `SaleCondition: Partial`.
- **Model Predictions**:
  - `xgboost`: $642,157.25
  - `lightgbm`: $625,118.39
  - `catboost`: $609,176.16
  - `elasticnet`: $1,727,070.96 (linear extrapolation)
  - `ridge_stack`: $1,265,505.27
  - `constrained_blend`: $796,699.54
