# Model 2 — Legacy Statistician: Full Evaluation Report

**Generated**: 2026-10-26 21:32:05  
**Training Population**: All 1,460 original records (IDs 524 and 1299 included)  
**Cross-Validation**: 5-Fold Stratified Shuffled K-Fold (`seed=42`)  
**Target Metric**: RMSLE (evaluated via `expm1` back to USD)  

---

## 1. Executive Summary & Core Results

Model 2 implements the **Legacy Statistician** architecture, combining classical regularized and robust linear estimators:
- **Base Model A**: `LassoCV` (L1 sparsity)
- **Base Model B**: `RidgeCV` (L2 shrinkage)
- **Base Model C**: `HuberRegressor` (robust M-estimator for linear outlier resilience)
- **Meta-Model**: `LassoCV` (trained on inner 5-fold OOF predictions)

| Candidate Architecture | Overall OOF RMSLE | Mean Fold RMSLE ± Std | Overall OOF RMSE ($) | Overall OOF MAE ($) | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `huber` **(Best Model 2)** | **0.14386** | 0.1378 ± 0.0413 | $68,258.13 | $15,640.57 | #1 |
| `simple_average` | **0.14486** | 0.1385 ± 0.0425 | $73,745.45 | $16,119.89 | #2 |
| `lasso_stack` | **0.14769** | 0.1415 ± 0.0424 | $76,734.21 | $16,474.45 | #3 |
| `lasso` | **0.14950** | 0.1435 ± 0.0421 | $78,676.19 | $16,782.28 | #4 |
| `ridge` | **0.14976** | 0.1441 ± 0.0406 | $75,439.66 | $17,292.60 | #5 |

---

## 2. Comparison: Model 2 (Legacy Statistician) vs. Model 1 (Grandmaster Choice)

| Metric | Model 1 — Grandmaster Choice (Tree Simple Avg) | Model 2 — Legacy Statistician (`simple_average`) | Difference / Verdict |
| :--- | :---: | :---: | :--- |
| **Overall OOF RMSLE** (All 1,460) | **0.12563** | **0.14486** | **Model 1 outperforms by +0.01923 RMSLE** |
| **Mean Fold RMSLE ± Std** | 0.1244 ± 0.0173 | 0.1385 ± 0.0425 | Model 1 has lower cross-fold variance |
| **Overall OOF MAE** ($) | **$14,694.26** | **$16,119.89** | Model 1 lower MAE |
| **Best Public Leaderboard Score** | **0.12603** *(Reference)* | *Pending Submission* | Model 1 public score recorded separately |

> [!IMPORTANT]
> **Strict Metric Separation**:
> - Model 1's full-population 5-fold OOF RMSLE is **0.12563**.
> - Model 1's best Kaggle public leaderboard score is **0.12603**.
> - These two metrics are derived from different sample distributions and are never treated as interchangeable.

---

## 3. Large-House Diagnostics (IDs 524 and 1299)

Both properties are massive Edwards estates sold under `Partial` condition at steep discounts.

| ID | Actual SalePrice | Living Area | Total Bsmt | Lasso Pred | Ridge Pred | Huber Pred | Simple Avg | Lasso Stack | Model 1 Tree Avg |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **524** | $184,750 | 4,676 sq ft | 3,138 sq ft | $1,082,745.27 | $1,071,517.01 | $978,826.34 | $1,043,302.72 | $1,060,810.43 | $604,782.69 |
| **1299** | $160,000 | 5,642 sq ft | 6,110 sq ft | $2,891,983.98 | $2,750,376.32 | $2,519,634.50 | $2,716,284.05 | $2,819,206.42 | $526,720.99 |

**Insight on Linear Extrapolation vs. Huber Loss**:
- In standard linear models (Lasso/Ridge), square footage coefficients cause high-dimensional linear extrapolation when evaluating extreme properties.
- `HuberRegressor` provides superior resistance: its M-estimator loss transitions from quadratic to linear error for large residuals, reducing the leverage of extreme points and producing more restrained predictions than Ridge or Lasso alone.

---

## 4. Systematic Price Tier Bias Analysis

| Price Tier | Observation Count | Mean Actual Price | Mean Predicted Price | Mean Log Error | Mean Dollar Error | % Underpredicted |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `< $120k` | 269 | $98,659 | $104,406 | `-0.0625` | `$-5,747` | 34.6% |
| `$120k - $180k` | 627 | $149,347 | $153,671 | `-0.0011` | `$-4,324` | 52.0% |
| `$180k - $250k` | 347 | $210,487 | $211,722 | `+0.0025` | `$-1,235` | 53.0% |
| `$250k - $350k` | 163 | $290,581 | $283,468 | `+0.0291` | `$+7,113` | 62.6% |
| `> $350k` | 54 | $436,325 | $395,228 | `+0.1020` | `$+41,098` | 85.2% |

---

## 5. Top 20 Largest Absolute Log Errors

| Rank | Id | Actual Price | Predicted Price | Log Error | Dollar Error ($) | SaleCondition | OverallQual | GrLivArea | Neighborhood |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| #1 | **1299** | $160,000 | $2,716,284 | `-2.8318` | $2,556,284 | `Partial` | 10/10 | 5642 sqft | Edwards |
| #2 | **524** | $184,750 | $1,043,303 | `-1.7311` | $858,553 | `Partial` | 10/10 | 4676 sqft | Edwards |
| #3 | **31** | $40,000 | $89,607 | `-0.8065` | $49,607 | `Normal` | 4/10 | 1317 sqft | IDOTRR |
| #4 | **633** | $82,500 | $176,650 | `-0.7614` | $94,150 | `Family` | 7/10 | 1411 sqft | NWAmes |
| #5 | **463** | $62,383 | $124,758 | `-0.6931` | $62,375 | `Normal` | 5/10 | 864 sqft | Sawyer |
| #6 | **969** | $37,900 | $73,805 | `-0.6665` | $35,905 | `Abnorml` | 3/10 | 968 sqft | OldTown |
| #7 | **1325** | $147,000 | $281,063 | `-0.6481` | $134,063 | `Partial` | 8/10 | 1795 sqft | Somerst |
| #8 | **917** | $35,311 | $66,281 | `-0.6297` | $30,970 | `Abnorml` | 2/10 | 480 sqft | IDOTRR |
| #9 | **496** | $34,900 | $63,907 | `-0.6049` | $29,007 | `Abnorml` | 4/10 | 720 sqft | IDOTRR |
| #10 | **813** | $55,993 | $102,435 | `-0.6040` | $46,442 | `Alloca` | 5/10 | 1044 sqft | IDOTRR |
| #11 | **89** | $85,000 | $48,709 | `+0.5568` | $36,291 | `Abnorml` | 3/10 | 1526 sqft | IDOTRR |
| #12 | **971** | $135,000 | $83,629 | `+0.4789` | $51,371 | `Abnorml` | 4/10 | 1192 sqft | NAmes |
| #13 | **411** | $60,000 | $93,518 | `-0.4438` | $33,518 | `Abnorml` | 5/10 | 1276 sqft | Edwards |
| #14 | **1433** | $64,500 | $100,268 | `-0.4412` | $35,768 | `Normal` | 4/10 | 968 sqft | OldTown |
| #15 | **711** | $52,000 | $80,790 | `-0.4406` | $28,790 | `Normal` | 3/10 | 729 sqft | BrkSide |
| #16 | **589** | $143,000 | $220,824 | `-0.4345` | $77,824 | `Partial` | 5/10 | 1473 sqft | ClearCr |
| #17 | **1454** | $84,500 | $127,320 | `-0.4099` | $42,820 | `Abnorml` | 5/10 | 1140 sqft | Mitchel |
| #18 | **689** | $392,000 | $264,274 | `+0.3943` | $127,726 | `Partial` | 8/10 | 1419 sqft | StoneBr |
| #19 | **682** | $159,434 | $107,516 | `+0.3940` | $51,918 | `Abnorml` | 5/10 | 1520 sqft | SWISU |
| #20 | **826** | $385,000 | $260,692 | `+0.3899` | $124,308 | `Partial` | 10/10 | 2084 sqft | NridgHt |

---

## 6. Integrity and Constraint Verifications

1. **Model 1 Untouched**: All Model 1 files, Experiment 1 outputs, Experiment 2 outputs, and submissions remain unchanged.
2. **Leakage Protection**: All transformers and estimators fitted strictly on training partitions; meta-model trained strictly on inner 5-fold OOF predictions.
3. **Finite Guaranteed**: All output matrices verified 100% finite (0 NaNs, 0 Infs).
