# Model 1 — Experiment 4 Preparation: Comprehensive Error Audit Report

**Date**: 2026-10-27  
**Auditor**: Senior Tabular ML Engineer (Model 1 Grandmaster Track)  
**Objective**: Audit the current best Model 1 Grandmaster candidate before selecting and implementing Experiment 4.  
**Target Milestone**: Achieve public Kaggle RMSLE below 0.12.  

---

## 1. Verification of Current Best Artifacts

We verified the local integrity, reproducibility, and alignment of all Model 1 Experiment 2 and Experiment 3 files:

### Benchmark Status & Score Clarification
- **Observed Kaggle Public Leaderboard RMSLE**: **0.12539**  
  *(Note: This metric was observed externally on the official Kaggle submission evaluation for `submission_candidate_variant_3_layout_density.csv`. It cannot be computed locally from test data because Kaggle ground-truth labels are withheld.)*
- **Previous Kaggle Public Leaderboard RMSLE (Exp 2)**: **0.12603**
- **Experiment 3 Variant 3 5-Fold OOF RMSLE**: **0.12546**
- **Experiment 2 Baseline 5-Fold OOF RMSLE**: **0.12563**
- **Validation-to-Leaderboard Gap**: $0.12546 \text{ (OOF)} \longleftrightarrow 0.12539 \text{ (Public)}$, showing tight alignment ($\Delta = -0.00007$).

### Artifact Inventory & Integrity
| Item | File Path | Status | Verification Summary |
| :--- | :--- | :---: | :--- |
| **Feature Engineering Code** | `src/model_1_grandmaster/features_exp3.py` | **Verified** | Implements `Exp3FeatureEngineer(variant='layout_density')`. |
| **Model Pipeline Artifact** | `models/model_1_grandmaster/experiment_3/tree_ensemble_pipeline_exp3.joblib` | **Verified** | Contains fitted preprocessor with 46 numeric & 44 categorical columns, and 3 tree models (`xgboost`, `lightgbm`, `catboost`). |
| **OOF Predictions** | `reports/model_1_grandmaster/experiment_3/oof_predictions.csv` | **Verified** | Exactly 1,460 rows, zero missing, matched 1-to-1 with `train.csv` IDs. |
| **Metrics Summary** | `reports/model_1_grandmaster/experiment_3/experiment_3_metrics.json` | **Verified** | Documents all 5 variants across 5 folds. |
| **Candidate Submission File** | `submissions/model_1_grandmaster/experiment_3/submission_candidate_variant_3_layout_density.csv` | **Verified** | Exactly 1,459 rows, columns `['Id', 'SalePrice']`, finite positive values rounded to 2 decimals, reproduces artifact predictions with max diff = 0.00. |
| **Experiment 2 Baseline Artifacts** | `models/model_1_grandmaster/experiment_2/` & `submissions/model_1_grandmaster/experiment_2/` | **Verified** | 100% untouched and preserved. |
| **Model 2 (Legacy Statistician)** | `src/model_2_legacy/` & `models/model_2_legacy/` | **Verified** | Paused and completely isolated. |

### Variant 3 Exact Feature Definitions
In [`src/model_1_grandmaster/features_exp3.py`](file:///C:/Users/adars/OneDrive/Documents/PROJECTS/House%20Prices%20-%20Advanced%20Regression%20Techniques/src/model_1_grandmaster/features_exp3.py#L92-L98):
1. **`BathToBedRatio`**:
   $$\text{BathToBedRatio} = \frac{\text{FullBath} + 0.5 \cdot \text{HalfBath} + \text{BsmtFullBath} + 0.5 \cdot \text{BsmtHalfBath}}{\max(\text{BedroomAbvGr}, 1)}$$
2. **`SF_Per_Bedroom`**:
   $$\text{SF\_Per\_Bedroom} = \frac{\text{GrLivArea}}{\max(\text{BedroomAbvGr}, 1)}$$
*(Exploratory features from Variants 1 & 2, such as `IsRemodeled` and `BsmtFinRatio`, are confirmed absent from Variant 3).*

---

## 2. Comprehensive Out-of-Fold Error Audit

All metrics are evaluated across the full validation population ($N = 1,460$), using natural log-transformed space:
$$\text{RMSLE} = \sqrt{\frac{1}{N} \sum_{i=1}^N \left( \ln(y_i + 1) - \ln(\hat{y}_i + 1) \right)^2 }$$

### Overall Model Performance
| Model Configuration | Overall OOF RMSLE | Overall OOF RMSE ($) | Overall OOF MAE ($) | $\Delta$ vs Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Experiment 2 Baseline (Tree Simple Avg)** | 0.12563 | $28,208.37 | $14,694.26 | 0.00000 |
| **Experiment 3 Variant 3 (Layout Density)** | **0.12546** | $28,271.27 | $14,701.94 | **-0.00017** |

---

### Fold-by-Fold Breakdown (5 Folds, Seed 42)
| Fold | N | Baseline RMSLE | Variant 3 RMSLE | $\Delta$ RMSLE | Baseline MAE ($) | Variant 3 MAE ($) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 292 | 0.12971 | **0.12902** | -0.00068 | $15,407.60 | $15,150.17 |
| **Fold 2** | 292 | **0.10830** | 0.10873 | +0.00043 | $13,573.01 | $13,785.40 |
| **Fold 3** | 292 | 0.15483 | **0.15474** | -0.00009 | $17,131.89 | $17,080.38 |
| **Fold 4** | 292 | 0.12141 | **0.12061** | -0.00080 | $14,612.23 | $14,697.29 |
| **Fold 5** | 292 | **0.10797** | 0.10836 | +0.00039 | $12,746.56 | $12,796.43 |
| **Mean $\pm$ Std** | — | **0.12444 $\pm$ 0.01727** | **0.12429 $\pm$ 0.01708** | **-0.00015** | $14,694.26 | $14,701.94 |

*Observation*: Variant 3 improved RMSLE on 3 out of 5 folds (Folds 1, 3, and 4) and lowered cross-fold standard deviation ($0.01727 \to 0.01708$). Fold 3 has a significantly higher RMSLE (~0.1547) because it contains the extreme Edwards partial liquidation homes (IDs 524 and 1299).

---

### Breakdown by SalePrice Bands
| Price Band | Count | % of Data | Baseline RMSLE | Variant 3 RMSLE | $\Delta$ RMSLE | Variant 3 MAE ($) | Variant 3 RMSE ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **< $100k** | 123 | 8.42% | 0.23392 | **0.23295** | -0.00097 | $13,481.92 | $19,833.27 |
| **$100k–$150k** | 496 | 33.97% | 0.10201 | **0.10200** | -0.00001 | $9,000.84 | $14,162.83 |
| **$150k–$200k** | 414 | 28.36% | 0.11675 | **0.11646** | -0.00029 | $12,301.24 | $30,927.51 |
| **$200k–$250k** | 210 | 14.38% | **0.08689** | 0.08706 | +0.00017 | $15,540.04 | $19,258.01 |
| **$250k–$350k** | 163 | 11.16% | 0.10968 | **0.10936** | -0.00032 | $23,251.96 | $30,018.30 |
| **> $350k** | 54 | 3.70% | **0.19071** | 0.19197 | +0.00126 | $59,184.13 | $85,882.68 |

*Critical Diagnostic Insights*:
1. **Core Market Excellence**: For standard homes between $100,000 and $350,000 (representing **87.87% of all sales**), the model performs with exceptional precision: RMSLE ranges from **0.087 to 0.116**.
2. **Lower-Tail Inaccuracy (< $100k)**: Sub-$100k properties show severe relative error (**0.23295 RMSLE**). Tree models suffer from a conservative lower bound; cheap distressed homes (actual prices $34k–$65k) are consistently overpredicted by $30k–$55k due to tree leaf shrinkage.
3. **Upper-Tail Luxury (> $350k)**: High-end luxury properties exhibit **0.19197 RMSLE**, where tree models struggle to extrapolate custom, rare architectural premiums.

---

### Breakdown by SaleCondition
| SaleCondition | Count | % of Data | Baseline RMSLE | Variant 3 RMSLE | $\Delta$ RMSLE | Variant 3 MAE ($) | Variant 3 RMSE ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Normal** | 1,198 | 82.05% | 0.09819 | **0.09818** | -0.00001 | $12,258.16 | $19,171.80 |
| **Partial** | 125 | 8.56% | **0.20393** | 0.20483 | +0.00090 | $31,437.57 | $65,380.21 |
| **Abnorml** | 101 | 6.92% | 0.21124 | **0.21003** | -0.00121 | $20,479.18 | $39,130.96 |
| **Family** | 20 | 1.37% | 0.24132 | **0.23732** | -0.00400 | $23,518.55 | $34,216.92 |
| **Alloca** | 12 | 0.82% | 0.22402 | **0.22074** | -0.00328 | $23,942.43 | $34,064.84 |
| **AdjLand** | 4 | 0.27% | **0.07313** | 0.07354 | +0.00041 | $5,944.82 | $8,292.84 |

*Key Findings*:
- On **Normal arm's-length sales** (82.05% of rows), the model achieves **0.09818 RMSLE** — well below the overall target of 0.12.
- In contrast, non-arm's length or incomplete transactions (**Partial**, **Abnorml**, **Family**, **Alloca**) average an RMSLE of **0.207**, more than double the error of Normal transactions.

---

### Breakdown by SaleType
| SaleType | Description | Count | % of Data | Baseline RMSLE | Variant 3 RMSLE | $\Delta$ RMSLE | Variant 3 MAE ($) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **WD** | Warranty Deed - Conventional | 1,267 | 86.78% | 0.11464 | **0.11445** | -0.00019 | $13,133.32 |
| **New** | Home just constructed | 122 | 8.36% | 0.20240 | **0.20307** | +0.00067 | $31,434.19 |
| **COD** | Court Officer Deed / Estate | 43 | 2.95% | 0.14198 | **0.13918** | -0.00280 | $12,524.34 |
| **ConLD** | Contract Low Down | 9 | 0.62% | 0.19376 | **0.19277** | -0.00099 | $15,960.16 |
| **ConLI** | Contract Low Interest | 5 | 0.34% | 0.10825 | **0.10275** | -0.00550 | $17,887.49 |
| **ConLw** | Contract Low Regular | 5 | 0.34% | 0.03253 | 0.03664 | +0.00411 | $4,292.96 |
| **CWD** | Cash Warranty Deed | 4 | 0.27% | 0.13097 | **0.12622** | -0.00475 | $25,793.08 |
| **Oth** | Other | 3 | 0.21% | 0.11578 | 0.11672 | +0.00095 | $13,154.69 |
| **Con** | Contract 15% Down | 2 | 0.14% | 0.09502 | 0.10529 | +0.01027 | $27,103.90 |

---

### Expensive Homes Diagnostics
| Subgroup | Count | % of Data | Baseline RMSLE | Variant 3 RMSLE | $\Delta$ RMSLE | Variant 3 MAE ($) | Variant 3 RMSE ($) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Homes > $350k** | 54 | 3.70% | **0.19071** | 0.19197 | +0.00126 | $59,184.13 | $85,882.68 |
| **Homes > $400k** | 28 | 1.92% | **0.20552** | 0.20806 | +0.00254 | $75,450.46 | $106,257.04 |

---

### Top 20 Largest Absolute Log Errors (Variant 3)
The table below lists the 20 properties with the highest prediction error in the entire dataset:

| Rank | Id | Actual Price | Predicted Price | Fold | SaleCondition | SaleType | Neighborhood | Qual | GrLivArea | Abs Log Error | Signed Log Error | Error Dollar Gap |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1** | 1299 | $160,000 | $530,494.47 | 3 | **Partial** | **New** | Edwards | 10 | 5,642 | **1.1986** | -1.1986 | +$370,494.47 |
| **2** | 524 | $184,750 | $601,341.00 | 3 | **Partial** | **New** | Edwards | 10 | 4,676 | **1.1802** | -1.1802 | +$416,591.00 |
| **3** | 31 | $40,000 | $94,449.88 | 1 | Normal | WD | IDOTRR | 4 | 1,317 | **0.8592** | -0.8592 | +$54,449.88 |
| **4** | 633 | $82,500 | $185,662.36 | 5 | **Family** | WD | NWAmes | 7 | 1,411 | **0.8111** | -0.8111 | +$103,162.36 |
| **5** | 496 | $34,900 | $71,728.57 | 4 | **Abnorml** | WD | IDOTRR | 4 | 720 | **0.7204** | -0.7204 | +$36,828.57 |
| **6** | 1325 | $147,000 | $290,668.43 | 3 | **Partial** | **New** | Somerst | 8 | 1,795 | **0.6817** | -0.6817 | +$143,668.43 |
| **7** | 463 | $62,383 | $122,310.42 | 2 | Normal | WD | Sawyer | 5 | 864 | **0.6733** | -0.6733 | +$59,927.42 |
| **8** | 917 | $35,311 | $66,150.44 | 1 | **Abnorml** | WD | IDOTRR | 2 | 480 | **0.6277** | -0.6277 | +$30,839.44 |
| **9** | 971 | $135,000 | $73,804.06 | 3 | **Abnorml** | WD | NAmes | 4 | 1,192 | **0.6039** | +0.6039 | -$61,195.94 |
| **10** | 969 | $37,900 | $67,476.56 | 4 | **Abnorml** | WD | OldTown | 3 | 968 | **0.5768** | -0.5768 | +$29,576.56 |
| **11** | 1183 | $745,000 | $423,963.28 | 2 | **Abnorml** | WD | NoRidge | 10 | 4,476 | **0.5637** | +0.5637 | -$321,036.72 |
| **12** | 411 | $60,000 | $104,450.57 | 2 | **Abnorml** | **COD** | Edwards | 5 | 1,276 | **0.5544** | -0.5544 | +$44,450.57 |
| **13** | 689 | $392,000 | $226,248.04 | 4 | **Partial** | **New** | StoneBr | 8 | 1,419 | **0.5496** | +0.5496 | -$165,751.96 |
| **14** | 1433 | $64,500 | $107,093.81 | 1 | Normal | WD | OldTown | 4 | 968 | **0.5070** | -0.5070 | +$42,593.81 |
| **15** | 813 | $55,993 | $90,473.54 | 1 | **Alloca** | WD | IDOTRR | 5 | 1,044 | **0.4798** | -0.4798 | +$34,480.54 |
| **16** | 1454 | $84,500 | $130,792.17 | 3 | **Abnorml** | WD | Mitchel | 5 | 1,140 | **0.4369** | -0.4369 | +$46,292.17 |
| **17** | 589 | $143,000 | $220,511.07 | 1 | **Partial** | WD | ClearCr | 5 | 1,473 | **0.4331** | -0.4331 | +$77,511.07 |
| **18** | 667 | $129,000 | $194,954.98 | 3 | **Abnorml** | WD | NAmes | 6 | 2,380 | **0.4130** | -0.4130 | +$65,954.98 |
| **19** | 1329 | $256,000 | $170,259.91 | 5 | Normal | WD | OldTown | 6 | 2,792 | **0.4078** | +0.4078 | -$85,740.09 |
| **20** | 67 | $180,000 | $270,405.14 | 3 | Normal | WD | NAmes | 7 | 2,207 | **0.4070** | -0.4070 | +$90,405.14 |

---

## 3. Data Leakage and Consistency Audit

We executed strict integrity checks against the training and evaluation pipelines:

1. **Strict Out-of-Fold Independence**:
   - Each OOF prediction in `reports/model_1_grandmaster/experiment_3/oof_predictions.csv` was generated from an estimator trained strictly on the other 4 folds. Verified: training fold indices and validation fold indices share 0 intersection.
2. **Leakage-Safe Feature Transformations**:
   - In [`GrandmasterPreprocessorExp3`](file:///C:/Users/adars/OneDrive/Documents/PROJECTS/House%20Prices%20-%20Advanced%20Regression%20Techniques/src/model_1_grandmaster/features_exp3.py#L102-L172), all learned scalers and imputers (`SimpleImputer`, `OneHotEncoder`) are fit strictly on `X_train_fold`.
   - Feature engineering steps (`BathToBedRatio`, `SF_Per_Bedroom`) use strictly unsupervised row-level arithmetic ($X \to X$), relying on zero validation labels or global dataset statistics.
3. **Partition Seed & Alignment**:
   - 5-Fold KFold configuration (`n_splits=5`, `shuffle=True`, `random_state=42`) matches Experiment 2 exactly. Every single one of the 1,460 rows is predicted once.
4. **Target Isolation**:
   - Both `SalePrice` and `Id` are stripped in `_prepare_dataframe` prior to feeding estimators, preventing accidental leakage.
5. **Diagnostic Integrity**:
   - The breakdowns by `SaleCondition`, `SaleType`, and `price_band` are performed purely ex-post on generated OOF arrays for analytical diagnostics; they do not alter training labels or create conditional validation leakage.

---

## 4. Recommended Next Experiments for Experiment 4

To bridge the gap from **0.12539 down below 0.12**, we must address the root causes identified in this error audit:

### Candidate 1: Targeted Condition & Incomplete Construction Interaction Engineering
- **Hypothesis**: The model currently assumes full finished square footage valuation for all properties regardless of transaction status. Incomplete partial construction (`SaleCondition == 'Partial'` and `SaleType == 'New'`) represents uncompleted homes sold at land/shell discount (IDs 524, 1299, 1325, 689). Explicitly engineering interaction features such as:
  $$\text{IsPartialNew} = \mathbf{1}(\text{SaleCondition} == \text{'Partial'} \land \text{SaleType} == \text{'New'})$$
  $$\text{Partial\_SF\_Discount} = \text{IsPartialNew} \times \text{TotalSF}$$
  $$\text{Qual\_Cond\_Ratio} = \frac{\text{OverallQual}}{\text{OverallCond} + 1e-5}$$
  allows trees to split on discount multipliers without distorting standard finished luxury valuations.
- **Audit Justification**: Partial sales have an RMSLE of **0.2048** (vs. 0.0982 for Normal), and IDs 524 & 1299 alone account for 12.26% of all squared log error.
- **Expected Benefit & Risk**: High potential RMSLE reduction (-0.002 to -0.005); risk is potential overfitting if partial signals in the test set behave differently from the training set.
- **Evaluation**: Evaluated using the exact same 5 folds (seed 42).
- **Retention Threshold**: OOF RMSLE improvement $\ge 0.00030$ while maintaining Normal sales RMSLE $\le 0.0982$.

---

### Candidate 2: Optimized Constrained Blending (Non-Negative Weighted Ensemble)
- **Hypothesis**: The current ensemble relies on an unweighted simple average ($\frac{1}{3}\text{XGB} + \frac{1}{3}\text{LGBM} + \frac{1}{3}\text{Cat}$). In Experiment 2/3, individual tree models performed disparately: XGBoost achieved OOF RMSLE of **0.12516**, while LightGBM had **0.13115** and CatBoost had **0.12778**. Forcing equal weights to LightGBM pulls down the overall ensemble. Optimizing non-negative weights $w = [w_{\text{XGB}}, w_{\text{LGBM}}, w_{\text{Cat}}]$ constrained to $\sum w_i = 1$ via bounded optimization (e.g., Nelder-Mead or SLSQP directly on OOF log errors) will prioritize the superior calibration of XGBoost while retaining multi-algorithm diversity.
- **Audit Justification**: Equal-weighted averaging leaves known performance on the table when one base estimator has a 0.006 RMSLE advantage over another.
- **Expected Benefit & Risk**: Low risk of regression; guaranteed to achieve $\le 0.12546$ OOF RMSLE; minimal code complexity.
- **Evaluation**: Nested fold weight optimization or out-of-fold constrained fitting on inner CV folds.
- **Retention Threshold**: OOF RMSLE $\le 0.12480$ with stable weights across folds.

---

### Candidate 3: Lower-Tail Tree Depth & Leaf Regularization Tuning
- **Hypothesis**: The model's large error on sub-$100k homes (RMSLE **0.23295**, 9 of the top 20 errors) is caused by uniform tree hyperparameters (`max_depth=4`, conservative `min_child_weight`) that clamp leaf values away from extreme distressed prices ($34k–$65k). Fine-tuning tree depth, leaf sample minimums, and regularizers (`reg_alpha`, `reg_lambda`, `l2_leaf_reg`) will improve leaf granularity at the price tails.
- **Audit Justification**: Sub-$100k homes show systematic overprediction bias of $30k–$55k per home.
- **Expected Benefit & Risk**: Improved tail sensitivity; moderate risk of overfitting if leaf minimums are set too low.
- **Evaluation**: Evaluated inside the same 5-fold CV using a grid/randomized search.
- **Retention Threshold**: Sub-$100k RMSLE drops below 0.220 without degrading overall score.

---

## 5. Single Best Next Experiment Recommendation

### Recommendation: **Candidate 2 (Optimized Constrained Blending) combined with Candidate 1 (Partial Construction Interactions)**

1. **Why this combination**:
   - Candidate 2 provides an immediate, risk-free architectural improvement by re-weighting the tree ensemble away from LightGBM's tail noise and toward XGBoost's proven tabular accuracy.
   - Candidate 1 directly attacks the single largest known vulnerability in the dataset (Partial / New liquidation sales that dominate 12% of the error metric).
2. **Execution Strategy**:
   - Test Candidate 2 first as Experiment 4A (Constrained Blending Benchmark).
   - Test Candidate 1 + Candidate 2 as Experiment 4B (Interaction-Augmented Blend).
   - Promote only if outer 5-fold OOF RMSLE reliably decreases below **0.12450**.
