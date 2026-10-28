"""Pre-submission verification script for Model 1 — Experiment 2."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

model_path = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_2" / "tree_ensemble_pipeline_exp2.joblib"
test_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
sub_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_2" / "submission_candidate_tree_simple_average.csv"
sample_sub_path = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"

# 1. Load artifact
art = joblib.load(model_path)
prep = art["preprocessor"]
trees = art["tree_models"]
tree_names = art["tree_model_names"]

# 2. Regenerate predictions
test_df = pd.read_csv(test_path)
x_test_proc = prep.transform(test_df)
preds_log = [trees[m].predict(x_test_proc) for m in tree_names]
avg_log = np.mean(preds_log, axis=0)
regen_preds = np.clip(np.expm1(avg_log), 1.0, None)
regen_preds_rounded = np.round(regen_preds, 2)

# 3. Load saved candidate and sample submission
cand_sub = pd.read_csv(sub_path)
sample_sub = pd.read_csv(sample_sub_path)

# 4. Compare
diff_exact = np.abs(cand_sub["SalePrice"].values - regen_preds_rounded)
max_diff = float(np.max(diff_exact))
mean_diff = float(np.mean(diff_exact))

print("=== REGENERATION & COMPARISON RESULTS ===")
print(f"Max absolute prediction difference: {max_diff:.8f}")
print(f"Mean absolute prediction difference: {mean_diff:.8f}")
print(f"Candidate row count: {len(cand_sub)} (Sample sub: {len(sample_sub)})")
print(f"Candidate column names: {list(cand_sub.columns)}")
print(f"Sample submission column names: {list(sample_sub.columns)}")
print(f"Column match: {list(cand_sub.columns) == list(sample_sub.columns)}")
print(f"ID ordering match with test.csv: {(cand_sub['Id'].values == test_df['Id'].values).all()}")
print(f"ID ordering match with sample_submission.csv: {(cand_sub['Id'].values == sample_sub['Id'].values).all()}")
print(f"All finite: {bool(np.isfinite(cand_sub['SalePrice'].values).all())}")
print(f"All positive: {bool((cand_sub['SalePrice'].values > 0).all())}")
print(f"Null count in Id: {cand_sub['Id'].isnull().sum()}")
print(f"Null count in SalePrice: {cand_sub['SalePrice'].isnull().sum()}")
print(f"Min prediction: ${cand_sub['SalePrice'].min():,.2f}")
print(f"Median prediction: ${cand_sub['SalePrice'].median():,.2f}")
print(f"Mean prediction: ${cand_sub['SalePrice'].mean():,.2f}")
print(f"Max prediction: ${cand_sub['SalePrice'].max():,.2f}")
