# House Prices — Advanced Regression Techniques (Dual-Model Stacking Project)

A reproducible machine learning project for the Kaggle competition [House Prices - Advanced Regression Techniques](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques).

The project develops and benchmarks two independent model architectures against the competition target: predicting residential home sales prices in Ames, Iowa using 79 explanatory features.

---

## 1. Project Objectives & Architecture

We develop and compare two independent model stacks in the same repository:

### Model 1 — Grandmaster Choice (Active Target)
- **Base Models**:
  - `XGBoost Regressor`
  - `LightGBM Regressor`
  - `CatBoost Regressor`
  - `ElasticNetCV`
- **Meta-Model (Stacking)**: `RidgeCV`
- **Focus**: High-capacity gradient boosting decision trees combined with regularized linear models, optimized for tabular competition benchmarks.

### Model 2 — Legacy Statistician (Not Started)
- **Base Models**:
  - `LassoCV`
  - `RidgeCV`
  - `HuberRegressor`
- **Meta-Model (Stacking)**: `LassoCV`
- **Focus**: Pure classical regularized and robust linear statistical estimators.

> **Development Constraint**: Model 1 will be developed end-to-end, validated, and its submission prepared and tested before Model 2 is started. Model 2's components remain completely unimplemented until authorized.

---

## 2. Evaluation Metric

- **Metric**: Root Mean Squared Logarithmic Error (RMSLE) between predicted values and actual sale prices.
$$\text{RMSLE} = \sqrt{\frac{1}{n} \sum_{i=1}^n \left( \log(p_i + 1) - \log(y_i + 1) \right)^2}$$
- **Target Variable**: `SalePrice`
- **Identifier**: `Id`

---

## 3. Repository Directory Structure

```
House-Prices-Advanced-Regression/
├── README.md                           # Project documentation & operational runbook
├── requirements.txt                    # Project dependencies
├── .gitignore                          # Excludes data, model weights, secrets, caches
├── .env.example                        # Template for Kaggle API credentials
├── configs/
│   └── project_config.yaml             # Central configuration (paths, seeds, metrics, model status)
├── data/
│   ├── raw/                            # Immutable official competition files
│   ├── interim/                        # Cleaned/intermediate feature sets
│   └── processed/                      # Model-ready arrays and transformed sets
├── models/
│   ├── model_1_grandmaster/            # Model 1 saved weights and artifacts
│   └── model_2_legacy/                 # Model 2 saved weights (frozen)
├── submissions/
│   ├── model_1_grandmaster/            # Model 1 generated submission CSVs
│   └── model_2_legacy/                 # Model 2 generated submission CSVs (frozen)
├── reports/
│   ├── model_1_grandmaster/            # Model 1 validation figures and metric logs
│   └── model_2_legacy/                 # Model 2 validation figures (frozen)
├── notebooks/                          # Exploratory analysis notebooks
├── src/
│   ├── common/
│   │   ├── __init__.py
│   │   └── paths.py                    # Pathlib-based dynamic project paths
│   ├── model_1_grandmaster/            # Model 1 pipelines and estimators
│   │   └── __init__.py
│   └── model_2_legacy/                 # Model 2 namespace (empty/unimplemented)
│       └── __init__.py
├── tests/                              # Unit and pipeline tests
└── scripts/
    └── download_data.py                # Official Kagglehub data downloader & validator
```

---

## 4. Setup & Environment

### Prerequisites
- Python 3.10+ (tested on Python 3.11)

### Installation
1. Clone or navigate to the repository directory.
2. Install initial dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Kaggle Authentication Prerequisites
To download the official dataset, Kaggle authentication is required:
1. **API Token**:
   - Visit [Kaggle Account Settings](https://www.kaggle.com/settings) and click **"Create New Token"**.
   - Save the downloaded `kaggle.json` to `~/.kaggle/kaggle.json` (`%USERPROFILE%\.kaggle\kaggle.json` on Windows).
   - Alternatively, copy `.env.example` to `.env` and fill in:
     ```bash
     KAGGLE_USERNAME=your_kaggle_username
     KAGGLE_KEY=your_kaggle_api_key
     ```
2. **Competition Rules**:
   - Visit [House Prices Competition Rules](https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques/rules).
   - Click **"I Understand and Accept"** to ensure your account has permission to access the dataset.

---

## 5. Dataset Acquisition & Validation

Download the official dataset and verify integrity:

```bash
python scripts/download_data.py
```

The script performs:
1. Verification of whether raw files already exist to prevent redundant downloads.
2. Official API download via `kagglehub.competition_download("house-prices-advanced-regression-techniques")`.
3. Placement into `data/raw/` preserving original files without modification.
4. Integrity checks against expected dimensions:
   - `train.csv`: 1,460 rows, 81 columns (includes `SalePrice` target and `Id`)
   - `test.csv`: 1,459 rows, 80 columns (includes `Id`, excludes `SalePrice`)
   - `sample_submission.csv`: 1,459 rows, 2 columns (`Id`, `SalePrice`)
   - `data_description.txt`: Full feature codebook

---

## 6. Current Progress & Next Steps

- [x] Repository scaffolding and pathlib configuration
- [x] Configuration schema (`configs/project_config.yaml`)
- [x] Kaggle acquisition and validation script (`scripts/download_data.py`)
- [ ] User authentication configuration (`kaggle.json` or `.env`)
- [ ] Dataset download and validation execution
- [ ] **Next Active Phase**: Implement Model 1 (Grandmaster Choice) preprocessing, feature engineering, baseline cross-validation, and stacking.
- [ ] Model 2 (Legacy Statistician) remains frozen until Model 1 is complete.
