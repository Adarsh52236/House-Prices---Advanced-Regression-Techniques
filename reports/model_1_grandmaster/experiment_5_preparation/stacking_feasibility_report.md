# Experiment 5 Preparation — Stacking Feasibility Audit

## 1. Executive Summary

| Item | Details |
| :--- | :--- |
| **Current Champion** | Model 1 Experiment 3 Variant 3 (*Functional Layout Density*) |
| **Public Kaggle RMSLE** | **0.12539** (`submission_candidate_variant_3_layout_density.csv`) |
| **5-Fold Full OOF RMSLE** | **0.12546** (RMSE: $28,271.27, MAE: $14,701.94) |
| **Champion Architecture** | Equal-weight log-space blend of XGBoost, LightGBM, and CatBoost |
| **Evaluation Scope** | Technical and statistical audit of stacked ensemble feasibility |
| **Primary Outcome** | **Stacking is feasible but unlikely to justify its computational cost.** |

This audit evaluates whether replacing the simple equal-weight log-price ensemble with a trained meta-model (stacking) is technically sound, leakage-safe, and empirically justified for the House Prices project. 

While a leakage-safe nested stacking pipeline is technically feasible and can be constructed using existing or newly generated out-of-fold folds, empirical and structural evidence indicates that **stacking three gradient-boosted decision tree (GBDT) algorithms on the same feature space is statistically inefficient and yields no meaningful generalization advantage**. 

Pairwise correlations between the base tree models exceed **0.989 to 0.995**. When base predictions are virtually collinear, a meta-regressor has virtually no orthogonal residual variance to exploit. In empirical testing on held-out folds, meta-learning (e.g. Ridge regression) degraded OOF RMSLE from **0.12546 to 0.12657** (+0.00111), while requiring **75 tree model fits** (a 15-fold increase in computational overhead).

---

## 2. Base-Model Out-of-Fold Artifact Inspection & Alignment

### 2.1 Artifact Availability
To evaluate stacking without retraining base models from scratch, we audited existing experiment outputs:
1. `reports/model_1_grandmaster/experiment_3/oof_predictions.csv`:
   - Contains composite variant predictions: `oof_baseline`, `oof_remodel_age`, `oof_finished_area`, `oof_layout_density`, `oof_combined`.
   - Does not preserve the isolated per-model predictions (`oof_xgboost`, `oof_lightgbm`, `oof_catboost`).
2. `reports/model_1_grandmaster/experiment_4a/oof_predictions.csv`:
   - Contains the exact, deterministic base-model out-of-fold predictions for the winning Variant 3 pipeline.
   - Generated during the constrained blending audit across identical 5 folds (`KFold(n_splits=5, shuffle=True, random_state=42)`).

### 2.2 Alignment & Integrity Audit
The individual base-model predictions in `reports/model_1_grandmaster/experiment_4a/oof_predictions.csv` were verified against `data/raw/train.csv`:
- **Row count**: Exactly 1,460 rows (100% complete, 0 missing or NaN values).
- **ID alignment**: Sequential integer IDs 1 through 1,460 matching `train.csv` row-for-row.
- **Target alignment**: Ground truth `SalePrice` matches `train.csv` exactly for all 1,460 records.
- **Fold balance**: Exactly 292 rows per fold across all 5 folds.
- **Simple average equivalence**: $\frac{1}{3}(\log(\text{XGB}) + \log(\text{LGB}) + \log(\text{CAT}))$ exponentiated matches `oof_layout_density` from Experiment 3 to within numerical machine precision ($< 10^{-12}$).

### 2.3 Individual Base Model Performance (Variant 3 Features)
Evaluating each base model's individual out-of-fold predictions reveals their standalone capability and the benefit already captured by simple averaging:

| Model | OOF RMSLE | OOF RMSE | OOF MAE | Standalone Rank |
| :--- | :---: | :---: | :---: | :---: |
| **XGBoost** | 0.12603 | $28,549.46 | $14,945.37 | 1 |
| **CatBoost** | 0.12773 | $28,987.56 | $14,856.88 | 2 |
| **LightGBM** | 0.13060 | $29,910.16 | $15,510.63 | 3 |
| **Equal-Weight Average (Champion)** | **0.12546** | **$28,271.27** | **$14,701.94** | **Ensemble Best** |

Simple equal-weight averaging beats the strongest individual model (XGBoost) by **0.00057 RMSLE** ($278 RMSE reduction), demonstrating that variance reduction via bagging across algorithmic implementations is already functioning efficiently.

---

## 3. Why In-Sample Meta-Model Training Causes Catastrophic Leakage

A fundamental principle of ensemble learning is that **a meta-model must never be trained on the base models' in-sample training predictions**. 

### 3.1 The Algorithmic Mechanism of In-Sample Leakage
1. **Base-Model Memorization**: GBDT algorithms (XGBoost, LightGBM, CatBoost) grow deep or asymmetric decision trees with high capacity. On training folds, tree depth and gradient-boosted iterations fit idiosyncratic sample-level noise, reaching near-perfect fit (in-sample $R^2 > 0.985$, training RMSLE $\approx 0.04-0.06$).
2. **Artificial Residual Shrinkage**: When base models evaluate the training instances they were fit on, their predictions are not independent estimates of the true conditional expectation $\mathbb{E}[Y|X]$; rather, they reflect memorized residuals.
3. **Multicollinear Weight Exploitation**: Because the base models share similar features and achieve near-zero training errors, an unconstrained or lightly regularized meta-regressor (e.g., OLS, Ridge, or a shallow neural net) does not learn how to balance true out-of-sample uncertainty. Instead, it solves an ill-conditioned linear system where tiny differences between memorized predictions are magnified into massive offsetting positive and negative coefficients:
   $$\hat{y}_{\text{meta}} = 14.2 \cdot \hat{y}_{\text{XGB}} - 8.7 \cdot \hat{y}_{\text{LGB}} - 4.5 \cdot \hat{y}_{\text{CAT}}$$
4. **Generalization Collapse on Test Data**: On unseen test observations, the idiosyncratic sample noise does not exist. The large opposing coefficients multiply uncorrelated base errors, producing catastrophic out-of-bounds predictions.

### 3.2 Historical Evidence in This Repository
This exact phenomenon was observed during **Experiment 1**:
- An unconstrained Ridge stacking meta-model trained without strict out-of-fold separation assigned large conflicting weights to collinear base models.
- When applied to the Kaggle test set, it produced an extreme prediction explosion of **$1,262,492** on test ID 2550 (a modest single-family dwelling), completely ruining the submission until safe boundaries and equal-weight averaging were established.

---

## 4. Feasibility of Valid Held-Out Stacking

To be strictly leakage-safe, the meta-model must be trained **only on out-of-fold (OOF) predictions** where each training row's base predictions were generated by models that never saw that row during base training.

There are two valid architectures for evaluating stacking:

### 4.1 Architecture A: Flat Out-of-Fold Meta-Training (Single CV)
- **Concept**: Use the pre-existing 1,460 OOF predictions from the 5-fold CV to train a single meta-model on the entire training set, then evaluate on the Kaggle public test set.
- **Limitation**: While valid for producing a test submission, this architecture **cannot produce a valid local validation score for the stacked model itself**. Training the meta-model on the full 1,460 OOF set and measuring its error on that same 1,460 set is an in-sample meta-evaluation (optimistic bias).

### 4.2 Architecture B: Nested Cross-Validation (Double K-Fold)
- **Concept**: A fully leakage-safe local validation of a stacked ensemble requires a **Nested $K \times M$ Cross-Validation** scheme:
  - **Outer Loop** (e.g., 5 folds): Evaluates the generalization performance of the stacked ensemble.
  - **Inner Loop** (e.g., 4 folds): Trains base models on 3/4 of the outer training fold to generate genuine out-of-fold meta-features for the remaining 1/4. The meta-model is then trained on these inner OOF predictions and evaluated on the held-out outer fold.
- **Feasibility**: Fully feasible algorithmically and mathematically. However, as detailed below, its computational cost and marginal utility must be scrutinized.

---

## 5. Empirical Feasibility & Collinearity Audit

To determine whether stacking could theoretically improve over equal weighting, we performed an empirical collinearity and meta-regression analysis using the 1,460 verified base OOF predictions:

### 5.1 Collinearity Analysis in Log-Price Space
We computed the Pearson correlation matrix of the out-of-fold predictions ($\log(1 + \hat{y})$):

$$\begin{pmatrix}
 & \text{XGBoost} & \text{LightGBM} & \text{CatBoost} \\
\text{XGBoost} & 1.000000 & 0.995136 & 0.993854 \\
\text{LightGBM} & 0.995136 & 1.000000 & 0.989626 \\
\text{CatBoost} & 0.993854 & 0.989626 & 1.000000
\end{pmatrix}$$

- **Finding**: Pairwise correlations range between **0.9896 and 0.9951**. 
- **Statistical Implication**: The three tree ensembles have virtually identical error distributions. They make mistakes on the exact same houses (e.g., atypical layouts, extreme living areas, unmodeled zoning conditions). When base estimators share $>99\%$ common variance, a linear combination cannot exploit orthogonal error cancellation.

### 5.2 Theoretical Upper Bound vs. Held-Out Generalization
We tested the limits of a regularized linear meta-model (Ridge regression, $\alpha \in [0.01, 100.0]$) in log-price space:

| Metric | Equal Weight Baseline | In-Sample Ridge Ceiling | Held-Out 5-Fold Ridge |
| :--- | :---: | :---: | :---: |
| **RMSLE** | **0.12546** | 0.12499 | **0.12657** |
| **Delta vs Champion** | Baseline | -0.00047 (theoretical best) | **+0.00111 (degradation)** |
| **RMSE** | $28,271.27 | $28,198.54 | $29,196.22 |
| **MAE** | $14,701.94 | $14,685.12 | $14,751.33 |

1. **Theoretical In-Sample Ceiling**: Even if a Ridge meta-model is given the entire 1,460 OOF matrix to fit its weights in-sample, the maximum achievable RMSLE reduction is a mere **0.00047** (0.12499 vs 0.12546).
2. **Held-Out Generalization Degradation**: When the meta-model is evaluated honestly via 5-fold cross-validation (training on 4 folds of OOF predictions and testing on the held-out 5th fold), performance **degrades to 0.12657** (+0.00111 worse than equal-weighting).
3. **Experiment 4A Alignment**: In Experiment 4A, constrained non-negative blending with unit sum weights achieved an OOF RMSLE of **0.12542** ($\Delta = -0.00004$), which failed the required improvement threshold ($\ge 0.0005$) and was decisively rejected.

---

## 6. Specification of a Leakage-Safe Nested Stacking Architecture

If stacking were to be implemented in a future phase, it MUST adhere strictly to the following leakage-safe protocol:

```
Full Training Set (1,460 rows)
  └── Outer 5-Fold CV (Folds 1 to 5)
        For each Outer Fold k (Train: 1,168 rows, Validation: 292 rows):
          ├── Inner 4-Fold CV on Outer Train (1,168 rows)
          │     For each Inner Fold j:
          │       - Fit Base Models (XGB, LGB, CAT) on 3/4 (876 rows)
          │       - Predict on remaining 1/4 (292 rows)
          │     Assemble Inner OOF Matrix Z_inner (1,168 x 3)
          ├── Fit Meta-Model M_k on Z_inner and y_train_outer
          ├── Fit Full Outer Base Models B_k,(XGB,LGB,CAT) on all 1,168 rows
          ├── Generate Validation Meta-Features: Z_val = B_k(X_val)
          └── Evaluate Meta-Model: y_val_pred = M_k(Z_val) -> Record Outer RMSLE

Final Test Inference:
  Test Features X_test (1,459 rows)
    ├── Passed to 5 Outer Base Ensembles -> Average Base Predictions
    └── Fed into 5 Outer Meta-Models -> Average Log Predictions -> expm1
```

### Computational Complexity & Overhead
- **Inner fits**: 5 outer folds $\times$ 4 inner folds $\times$ 3 base models = **60 tree models**.
- **Outer base fits**: 5 outer folds $\times$ 3 base models = **15 tree models**.
- **Total model fits required**: **75 full gradient-boosted tree models**.
- **Estimated runtime**: 10 to 15 minutes on the local CPU environment.
- **Overhead factor**: **15$\times$** the compute of a single 5-fold CV run.

---

## 7. Official Outcome & Recommendation

### 7.1 Selected Outcome
> **Stacking is feasible but unlikely to justify its computational cost.**

### 7.2 Decision Rationale
1. **Base Predictor Homogeneity**: XGBoost, LightGBM, and CatBoost are all decision-tree ensembles trained on the exact same 83 engineered tabular features. Their out-of-fold predictions share $>0.99$ correlation.
2. **Lack of Model Family Diversity**: Literature and competitive experience demonstrate that stacking succeeds when combining radically diverse model families (e.g., GBDTs + Neural Networks + Kernel Ridge/Splines). Model 2 (Legacy Statistician - linear models) was isolated and paused after achieving 0.14386 RMSLE, meaning no competitive non-tree model is available to provide diverse residuals.
3. **Negative Held-Out Yield**: Empirical validation of regularized linear meta-models demonstrates a performance degradation (+0.00111 RMSLE) due to fold-specific weight variance, while the theoretical upper bound is negligible ($< 0.0005$).
4. **Disproportionate Compute**: Requiring 75 gradient-boosted tree fits for an expected score delta of $< 0.0001$ is an inefficient allocation of development resources.
5. **No Unsupported Claim Rule**: In accordance with strict competition safeguards, we make no unsubstantiated claims that stacking will improve the score. All available evidence demonstrates that simple equal-weight log-averaging remains the most robust, regularized, and generalizable ensemble method for this configuration.

### 7.3 Recommended Next Steps for Model 1
Instead of allocating engineering time to an expensive, multi-collinear stacking meta-model, future Model 1 improvements should focus on:
1. **Targeted Sub-Segment Feature Engineering**: Addressing remaining structural residuals identified in the Error Audit (e.g., older partial remodels, high-density residential lot configurations).
2. **Diverse Model Paradigm Exploration**: If ensembling is revisited, explore introducing a fundamentally distinct architecture (e.g., a multi-layer perceptron with entity embeddings or a regularized spline model) that provides truly uncorrelated errors before training a meta-learner.
3. **Maintain Champion Integrity**: Experiment 3 Variant 3 remains the verified champion baseline (Kaggle public RMSLE: **0.12539**, local OOF RMSLE: **0.12546**).
