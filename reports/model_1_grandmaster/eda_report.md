# Model 1 — Grandmaster Choice: Exploratory Data Analysis & Audit Report

**Date**: 2026-10-26  
**Competition**: Kaggle House Prices - Advanced Regression Techniques  
**Active Model**: `model_1_grandmaster`  
**Dataset Location**: `data/raw/`  

---

## 1. Executive Summary

- **Training Records**: 1,460 rows × 81 columns.
- **Test Records**: 1,459 rows × 80 columns.
- **Target Variable**: `SalePrice` (verified present in train, absent in test).
- **Target Skewness**: Raw skewness = `1.883` (strongly right-skewed). After `log1p` transformation, skewness reduces to `0.121` (nearly symmetric normal distribution).
- **Metric**: Root Mean Squared Logarithmic Error (RMSLE). Because $\text{RMSLE}(y, \hat{y}) = \text{RMSE}(\log(y+1), \log(\hat{y}+1))$, models will be trained directly on $y_{\text{trans}} = \log(1 + \text{SalePrice})$ to directly optimize the competition loss function.

---

## 2. Target Variable Summary Statistics

| Statistic | Raw SalePrice ($) | log1p(SalePrice) |
| :--- | :--- | :--- |
| **Count** | 1,460 | 1,460 |
| **Mean** | $180,921.20 | 12.0241 |
| **Std Dev** | $79,442.50 | 0.3994 |
| **Min** | $34,900.00 | 10.4603 |
| **25% (Q1)** | $129,975.00 | 11.7751 |
| **50% (Median)** | $163,000.00 | 12.0015 |
| **75% (Q3)** | $214,000.00 | 12.2737 |
| **Max** | $755,000.00 | 13.5345 |
| **Skewness** | 1.883 | 0.121 |
| **Kurtosis** | 6.536 | 0.810 |

### Detailed Percentiles
- 1st Percentile: $61,815.97
- 5th Percentile: $88,000.00
- 90th Percentile: $278,000.00
- 95th Percentile: $326,100.00
- 99th Percentile: $442,567.01

---

## 3. Missing Value Analysis

In Ames Housing data, missing values often encode physical absence rather than unrecorded data:
- `PoolQC` (99.5% missing in train): No swimming pool.
- `MiscFeature` (96.3% missing in train): No additional feature (shed, tennis court, etc.).
- `Alley` (93.8% missing in train): No alley access.
- `Fence` (80.8% missing in train): No fence.
- `FireplaceQu` (47.3% missing in train): No fireplace.
- `GarageType`, `GarageFinish`, `GarageQual`, `GarageCond`: No garage.
- `BsmtQual`, `BsmtCond`, `BsmtExposure`, `BsmtFinType1`, `BsmtFinType2`: No basement.

### Columns with Highest Missing Percentages in Train
| Feature | Missing Count | Missing Percentage | Semantic Meaning |
| :--- | :--- | :--- | :--- |
| `PoolQC` | 1453 | 99.52% | Absence of facility / unrecorded |
| `MiscFeature` | 1406 | 96.30% | Absence of facility / unrecorded |
| `Alley` | 1369 | 93.77% | Absence of facility / unrecorded |
| `Fence` | 1179 | 80.75% | Absence of facility / unrecorded |
| `MasVnrType` | 872 | 59.73% | Absence of facility / unrecorded |
| `FireplaceQu` | 690 | 47.26% | Absence of facility / unrecorded |
| `LotFrontage` | 259 | 17.74% | Absence of facility / unrecorded |
| `GarageType` | 81 | 5.55% | Absence of facility / unrecorded |
| `GarageYrBlt` | 81 | 5.55% | Absence of facility / unrecorded |
| `GarageFinish` | 81 | 5.55% | Absence of facility / unrecorded |

---

## 4. Top Feature Correlations with SalePrice

| Feature | Pearson Correlation ($r$) | Semantic Description |
| :--- | :--- | :--- |
| `OverallQual` | +0.7910 | High positive association with house valuation |
| `GrLivArea` | +0.7086 | High positive association with house valuation |
| `GarageCars` | +0.6404 | High positive association with house valuation |
| `GarageArea` | +0.6234 | High positive association with house valuation |
| `TotalBsmtSF` | +0.6136 | High positive association with house valuation |
| `1stFlrSF` | +0.6059 | High positive association with house valuation |
| `FullBath` | +0.5607 | High positive association with house valuation |
| `TotRmsAbvGrd` | +0.5337 | High positive association with house valuation |
| `YearBuilt` | +0.5229 | High positive association with house valuation |
| `YearRemodAdd` | +0.5071 | High positive association with house valuation |
| `GarageYrBlt` | +0.4864 | High positive association with house valuation |

---

## 5. Outlier Investigation

A famous characteristic of the Ames dataset documented by Dean De Cock:
- Observations with `GrLivArea > 4000` sq ft:
  - `Id 524`: `GrLivArea = 4,676` sq ft, `SalePrice = $184,750` (`SaleCondition = Partial`)
  - `Id 1299`: `GrLivArea = 5,642` sq ft, `SalePrice = $160,000` (`SaleCondition = Partial`)
  - `Id 692`: `GrLivArea = 4,316` sq ft, `SalePrice = $755,000` (`SaleCondition = Normal`)
  - `Id 1183`: `GrLivArea = 4,476` sq ft, `SalePrice = $745,000` (`SaleCondition = Abnorml`)

`Id 524` and `Id 1299` represent partial sales of massive estate properties sold for abnormally low prices ($160k - $184k), which severely distort linear models and tree splits. We benchmark and document the effect of filtering these two verified aberrant training points in cross-validation.

---

## 6. Generated Figures

1. Target Distribution: `reports/model_1_grandmaster/target_distribution.png`
2. Key Feature Relationships: `reports/model_1_grandmaster/key_features_relationships.png`
3. Correlation Heatmap: `reports/model_1_grandmaster/correlation_heatmap_top15.png`
4. Missing Values Bar Chart: `reports/model_1_grandmaster/missing_values_bar.png`
