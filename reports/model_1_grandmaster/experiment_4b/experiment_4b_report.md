# Model 1 — Experiment 4B: Targeted Incomplete-Construction Feature Engineering Report

**Generated**: 2026-10-27 00:51:49  
**Objective**: Test whether targeted construction-status features improve upon the current best Model 1 Grandmaster pipeline (0.12546 OOF RMSLE).  
**Validation Population**: All 1,460 original records (5-Fold Stratified Shuffled K-Fold, `seed=42`)  
**Base Models**: XGBoost, LightGBM, CatBoost (equal-weight log-space average)  

---

## 1. Executive Summary & Benchmark Comparison

| Candidate Configuration | Total Features | Overall OOF RMSLE | $\Delta$ vs Baseline | Mean Fold RMSLE $\pm$ Std | OOF RMSE ($) | OOF MAE ($) | Retention Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (Exp 3 Variant 3)** | 311 | **0.12546** | 0.00000 | 0.12429 $\pm$ 0.01708 | $ 28,271.27 | $ 14,701.94 | **Retained Champion** |
| **Candidate 1 (+IsPartialNew)** | 312 | **0.12545** | **-0.00001** | 0.12421 $\pm$ 0.01761 | $ 28,281.92 | $ 14,614.73 | Rejected (< 0.00030) |
| **Candidate 2 (+Interaction)** | 313 | **0.12542** | **-0.00004** | 0.12421 $\pm$ 0.01738 | $ 28,016.83 | $ 14,612.40 | Rejected (< 0.00030) |

### Key Decision
- **Recommendation**: **Neither candidate met the retention criterion of >= 0.00030 RMSLE improvement. Candidate 1 (IsPartialNew) improved RMSLE by only -0.00001 (0.12546 -> 0.12545). Candidate 2 (IsPartialNew + PartialNewAreaInteraction) improved RMSLE by only -0.00004 (0.12546 -> 0.12542). Both candidates actually regressed on Partial sales (RMSLE 0.20483 -> 0.20598 / 0.20570) and made the extreme Edwards outliers IDs 524 and 1299 slightly worse. Because SaleType_New and SaleCondition_Partial already exist in the one-hot matrix with 100% mutual coverage for new sales, the added features provided redundant signal rather than genuine construction discount mechanics.**

---

## 2. Phase 1: Data Audit & Hypothesis Verification

The hypothesis posited that homes with `SaleCondition == 'Partial'` and `SaleType == 'New'` sold at substantial discounts because they were unfinished shells when assessed, and that explicit indicator and area interaction features would enable the tree ensemble to discount them appropriately.

### Empirical Findings on Training (N=1,460) and Test (N=1,459) Data:
1. **Exact Subgroup Counts**:
   - `SaleCondition == 'Partial'`: **125** in train (8.56%), **120** in test (8.22%).
   - `SaleType == 'New'`: **122** in train (8.36%), **117** in test (8.02%).
   - **Both (`Partial` AND `New`)**: **122** in train (8.36%), **117** in test (8.02%).
   - In the training set, **100% of `New` sales are also `Partial`** (122 / 122). Only 3 `Partial` sales have a different sale type (2 WD, 1 Con).
2. **Official Documentation Fact vs. Assumption**:
   - `data_description.txt` states: `Partial: Home was not completed when last assessed (associated with New Homes)`.
   - However, partial assessment does **not** mean low price: the median sale price for `Partial & New` homes is **$247,453** (mean **$274,945**, max **$611,657**), substantially higher than the market median of **$163,000**.
   - Most partial new homes sold for high prices because buyers purchased them from developers at near-completion or contract value.
   - The extreme errors on IDs 524 ($184,750) and 1299 ($160,000) reflect distressed developer liquidation/land sales for massive 4,600+ sq ft estates, which are unrepresentative outliers even within the `Partial & New` cohort.

---

## 3. Phase 3: Detailed Comparative Evaluation

### Per-Fold RMSLE Breakdown (5 Shuffled Folds, Seed 42)
| Fold | Baseline RMSLE | Candidate 1 RMSLE | $\Delta$ vs Base | Candidate 2 RMSLE | $\Delta$ vs Base |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.12902 | 0.12799 | -0.00103 | 0.12836 | -0.00067 |
| **Fold 2** | 0.10873 | 0.10800 | -0.00073 | 0.10847 | -0.00026 |
| **Fold 3** | 0.15474 | 0.15569 | +0.00095 | 0.15527 | +0.00053 |
| **Fold 4** | 0.12061 | 0.12173 | +0.00112 | 0.12127 | +0.00067 |
| **Fold 5** | 0.10836 | 0.10762 | -0.00074 | 0.10769 | -0.00067 |
| **Overall OOF** | **0.12546** | **0.12545** | **-0.00001** | **0.12542** | **-0.00004** |
| **Fold Std** | **0.01708** | 0.01761 | +0.00052 | 0.01738 | +0.00030 |

### Subgroup Performance Comparison
| Subgroup | Sample Count | Baseline RMSLE | Candidate 1 RMSLE | Candidate 2 RMSLE | Baseline MAE | Candidate 1 MAE | Candidate 2 MAE |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal Sales** | 1198 | 0.09818 | 0.09791 | 0.09809 | $ 12,258.16 | $ 12,177.42 | $ 12,163.19 |
| **Partial Sales** | 125 | 0.20483 | 0.20598 | 0.20570 | $ 31,437.57 | $ 31,262.86 | $ 31,495.18 |
| **Partial & New Sales** | 122 | 0.20307 | 0.20468 | 0.20398 | $ 31,434.19 | $ 31,278.55 | $ 31,490.19 |

### Four High-Error IDs Diagnostics
| Id | Actual Price | Baseline Pred | Candidate 1 Pred | Candidate 2 Pred | Baseline LogErr | Cand 1 LogErr | Cand 2 LogErr |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **524** | $184,750 | $601,341.00 | $611,977.69 | $607,266.83 | -1.1802 | -1.1977 | -1.1900 |
| **1299** | $160,000 | $530,494.47 | $540,702.26 | $540,902.79 | -1.1986 | -1.2177 | -1.2181 |
| **1325** | $147,000 | $290,668.43 | $291,238.77 | $290,543.95 | -0.6817 | -0.6837 | -0.6813 |
| **689** | $392,000 | $226,248.04 | $225,675.38 | $225,912.78 | +0.5496 | +0.5522 | +0.5511 |

---

## 4. Phase 4: Retention Assessment & Scientific Insights

1. **Threshold Assessment**:
   - Required improvement: $\Delta \le -0.00030$ RMSLE.
   - Candidate 1 achieved: $\Delta = -0.00001$ (NOT qualified).
   - Candidate 2 achieved: $\Delta = -0.00004$ (NOT qualified).
2. **Why the Feature Failed to Provide Signal**:
   - The one-hot encoded matrix already contains `SaleType_New` and `SaleCondition_Partial`.
   - Because `SaleType_New` has a 100% correlation with `SaleCondition_Partial` in new sales, `IsPartialNew` is effectively identical to `SaleType_New`.
   - Adding an explicit interaction `PartialNewAreaInteraction = IsPartialNew * GrLivArea` merely gave the trees another collinear column to subsample (`colsample_bytree=0.7`), causing slight fold variance instability ($0.01708 \to 0.01738$).
   - On the targeted Partial sales cohort, RMSLE actually **degraded** from **0.20483 to 0.20570**.
3. **Conclusion**:
   - Neither candidate meets the retention threshold.
   - **Experiment 3 Variant 3 remains the active champion** (5-Fold OOF RMSLE: **0.12546**, Kaggle Public RMSLE: **0.12539**).