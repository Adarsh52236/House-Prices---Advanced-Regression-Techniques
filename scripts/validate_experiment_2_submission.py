"""Validation script for Model 1 — Experiment 2 final submission file."""

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd

# Paths
model_path = PROJECT_ROOT / "models" / "model_1_grandmaster" / "experiment_2" / "tree_ensemble_pipeline_exp2.joblib"
test_path = PROJECT_ROOT / "data" / "raw" / "test.csv"
sample_sub_path = PROJECT_ROOT / "data" / "raw" / "sample_submission.csv"
new_sub_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_2" / "submission_model_1_experiment_2_tree_simple_average.csv"
candidate_path = PROJECT_ROOT / "submissions" / "model_1_grandmaster" / "experiment_2" / "submission_candidate_tree_simple_average.csv"

# 1. Regenerate from saved artifact
art = joblib.load(model_path)
prep = art["preprocessor"]
trees = art["tree_models"]
tree_names = art["tree_model_names"]

test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

X_test_proc = prep.transform(test_df)
preds_log = [trees[m].predict(X_test_proc) for m in tree_names]
avg_log = np.mean(preds_log, axis=0)
preds_raw = np.clip(np.expm1(avg_log), 1.0, None)
preds_rounded = np.round(preds_raw, 2)

# 2. Write / Verify new submission file
new_sub_df = pd.DataFrame({
    "Id": test_df["Id"].values.astype(int),
    "SalePrice": preds_rounded,
})
new_sub_df.to_csv(new_sub_path, index=False)
print(f"Submission written to: {new_sub_path.resolve()}")

# 3. Comprehensive Validations
sub_df = pd.read_csv(new_sub_path)

# A. Row count check
row_count = len(sub_df)
assert row_count == 1459, f"Expected 1459 rows, got {row_count}"

# B. Column names and order
col_names = list(sub_df.columns)
assert col_names == ["Id", "SalePrice"], f"Expected ['Id', 'SalePrice'], got {col_names}"

# C. ID sequence matching
id_match_test = (sub_df["Id"].values == test_df["Id"].values).all()
id_match_sample = (sub_df["Id"].values == sample_sub["Id"].values).all()
assert id_match_test, "ID sequence does not match test.csv!"
assert id_match_sample, "ID sequence does not match sample_submission.csv!"

# D. Duplicate ID check
unique_ids = sub_df["Id"].nunique()
assert unique_ids == 1459, f"Duplicate IDs found! Unique count: {unique_ids}"

# E. Missing, infinite, zero, or negative check
null_count = sub_df.isnull().sum().sum()
assert null_count == 0, f"Found {null_count} null values!"

is_finite = np.isfinite(sub_df["SalePrice"].values).all()
assert is_finite, "Found non-finite values!"

is_positive = (sub_df["SalePrice"].values > 0).all()
assert is_positive, "Found non-positive values!"

# F. Two decimal places check
diff_round = np.max(np.abs(sub_df["SalePrice"].values - np.round(sub_df["SalePrice"].values, 2)))
assert diff_round == 0.0, f"Found values not rounded to 2 decimals: max diff {diff_round}"

# G. Max absolute difference from regenerated predictions
diff_regen = float(np.max(np.abs(sub_df["SalePrice"].values - preds_rounded)))
assert diff_regen == 0.0, f"Max absolute difference from regenerated is {diff_regen}"

# H. Check match with candidate
cand_df = pd.read_csv(candidate_path)
diff_cand = float(np.max(np.abs(sub_df["SalePrice"].values - cand_df["SalePrice"].values)))
assert diff_cand == 0.0, f"Max difference from candidate file is {diff_cand}"

print("\n=== VALIDATION SUMMARY ===")
print(f"Absolute file path: {new_sub_path.resolve()}")
print(f"Total rows: {row_count}")
print(f"Columns: {col_names}")
print(f"ID match with test.csv: {id_match_test}")
print(f"ID match with sample_submission.csv: {id_match_sample}")
print(f"Duplicate IDs: 0 (Unique: {unique_ids})")
print(f"Missing/null values: {null_count}")
print(f"All finite & positive: {is_finite and is_positive}")
print(f"Rounded to 2 decimals: True (Max deviation: {diff_round:.8f})")
print(f"Max abs difference from artifact predictions: {diff_regen:.8f}")
print(f"Min predicted price: ${sub_df['SalePrice'].min():,.2f}")
print(f"Median predicted price: ${sub_df['SalePrice'].median():,.2f}")
print(f"Mean predicted price: ${sub_df['SalePrice'].mean():,.2f}")
print(f"Max predicted price: ${sub_df['SalePrice'].max():,.2f}")
print("ALL VALIDATION CHECKS PASSED.")
