# Model 1 — Experiment 4A: Leakage-Safe Constrained Tree Blending Report

**Generated**: 2026-10-27 00:43:52  
**Objective**: Evaluate whether constrained non-negative linear blending improves upon the equal-weight 3-tree ensemble (0.12546 RMSLE) without validation leakage.  
**Validation Population**: All 1,460 original records (5-Fold Stratified Shuffled K-Fold, `seed=42`)  
**Base Models**: XGBoost, LightGBM, CatBoost (trained on Variant 3: Functional Layout Density features)  

---

## 1. Executive Summary & Benchmark Comparison

| Model Architecture | OOF RMSLE | $\Delta$ vs Equal-Weight | OOF RMSE ($) | OOF MAE ($) |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost Standalone** | 0.12603 | +0.00057 | $ 28,479.07 | $ 14,705.15 |
| **CatBoost Standalone** | 0.12773 | +0.00227 | $ 29,487.07 | $ 15,281.43 |
| **LightGBM Standalone** | 0.13060 | +0.00514 | $ 28,509.10 | $ 15,431.78 |
| **Equal-Weight Blend Baseline** | **0.12546** | 0.00000 | **$ 28,271.27** | $ 14,701.94 |
| **Leakage-Safe Held-Out Constrained Blend** | **0.12542** | **-0.00004** | $ 28,576.65 | **$ 14,696.61** |

### Key Decision
- **Recommendation**: **DO NOT PROMOTE CONSTRAINED BLENDING.** Retain the **Equal-Weight Simple Average**.
- **Rationale**: The true leakage-safe held-out improvement is virtually nonexistent (**-0.00004 RMSLE**, -0.03%), while RMSE actually degraded by **+$305.38**. On Fold 2, the optimizer hit the zero boundary for LightGBM (w_LGB = 0.0000), causing a severe fold regression of **+0.00219 RMSLE**. Equal weighting acts as effective ensemble regularization against fold-level noise.

---

## 2. Phase 1 & 2: Verification of Baseline and Individual Base Predictions

1. **Deterministic Reconstruction**:
   - The original Experiment 3 `oof_predictions.csv` preserved the composite variant predictions but did not store individual base tree outputs.
   - We executed an exact, deterministic re-evaluation of the 5 folds (`seed=42`) with fixed hyperparameters from `configs/project_config.yaml`.
   - The arithmetic average of base model log predictions reproduced the saved Experiment 3 Variant 3 OOF prediction column with a maximum absolute discrepancy of **0.00000000**, confirming exact alignment.
2. **Prediction Space**:
   - Base models predict in **natural log space** ln(SalePrice + 1).
   - Blending is performed strictly as a linear combination of log prices:
     $$\hat{y}_{\text{blend, log}} = w_{\text{XGB}} \hat{y}_{\text{XGB, log}} + w_{\text{LGB}} \hat{y}_{\text{LGB, log}} + w_{\text{CAT}} \hat{y}_{\text{CAT, log}}$$
   - Final valuation is inverted via clip(expm1(\hat{y}_{\text{blend, log}}), 1.0, inf).

---

## 3. Phase 3: Leakage-Safe Held-Out Blending Results

In each outer fold k in {1, 2, 3, 4, 5}, weights were estimated strictly on the **other 4 folds** (N = 1,168) using bounded SLSQP optimization subject to w_i >= 0 and sum(w_i) = 1. The learned weights were then evaluated on the untouched held-out fold k (N = 292).

| Fold | Learned w_XGB | Learned w_LGB | Learned w_CAT | Equal Blend RMSLE | Constrained Blend RMSLE | $\Delta$ RMSLE | Boundary Solution? |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Fold 1** | 0.5283 | 0.1348 | 0.3369 | 0.12902 | **0.12803** | -0.00099 | No |
| **Fold 2** | 0.6753 | 0.0000 | 0.3247 | 0.10873 | **0.11093** | +0.00219 | Yes (w_LGB = 0) |
| **Fold 3** | 0.6234 | 0.0497 | 0.3269 | 0.15474 | **0.15464** | -0.00010 | No |
| **Fold 4** | 0.4556 | 0.0885 | 0.4559 | 0.12061 | **0.12044** | -0.00017 | No |
| **Fold 5** | 0.4576 | 0.1277 | 0.4147 | 0.10836 | **0.10743** | -0.00093 | No |
| **Overall** | — | — | — | **0.12546** | **0.12542** | **-0.00004** | — |

### Diagnosis of the Fold 2 Failure
In Fold 2, training without Fold 2 caused the optimizer to completely drop LightGBM (w_LGB = 0.0000). However, LightGBM had learned useful variance reduction for several homes unique to Fold 2. As a consequence, omitting LightGBM caused Fold 2's RMSLE to spike from **0.10873 to 0.11093** (+0.00219). This illustrates the classic danger of post-hoc weight optimization on small tabular datasets: **weights that minimize training residual variance can overfit the residual correlations of the training folds.**

---

## 4. Phase 4: Full Dataset Weights & Prospective Test Inference

When fitted across all 1,460 records, the optimal constrained weight vector is:
- **w_XGB**: **0.5654** (56.5%)
- **w_CAT**: **0.3728** (37.3%)
- **w_LGB**: **0.0618** (6.2%)

While this weight distribution aligns with standalone model rankings (XGBoost > CatBoost > LightGBM), the leakage-safe outer cross-validation proves that forcing LightGBM down to ~6% yields a negligible net gain (-0.00004 RMSLE) at the cost of higher RMSE and tail instability.

---

## 5. Decision & Next Steps

1. **Preserve Current Production Baseline**: Keep [`tree_ensemble_pipeline_exp3.joblib`](file:///c:/Users/adars/OneDrive/Documents/PROJECTS/House%20Prices%20-%20Advanced%20Regression%20Techniques/models/model_1_grandmaster/experiment_3/tree_ensemble_pipeline_exp3.joblib) and [`submission_candidate_variant_3_layout_density.csv`](file:///c:/Users/adars/OneDrive/Documents/PROJECTS/House%20Prices%20-%20Advanced%20Regression%20Techniques/submissions/model_1_grandmaster/experiment_3/submission_candidate_variant_3_layout_density.csv) (Kaggle public score: **0.12539**) as our current verified champion.
2. **Do Not Submit a New Candidate**: A new candidate submission based on constrained blending would merely gamble on test set alignment without genuine structural improvement.
3. **Next Direction**: Proceed to structural feature refinements (such as Partial Construction interaction terms) or targeted tree hyperparameter regularizations that directly address the sub-$100k and >$350k error tiers identified in the error audit.