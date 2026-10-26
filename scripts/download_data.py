"""Dataset acquisition and validation script for Kaggle House Prices competition.

Downloads the official dataset using kagglehub and verifies data integrity.
"""

import os
import shutil
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except ImportError:
    pass

import pandas as pd
from src.common.paths import (
    DATA_DESCRIPTION_TXT,
    RAW_DATA_DIR,
    SAMPLE_SUBMISSION_CSV,
    TEST_CSV,
    TRAIN_CSV,
    ensure_directory,
)

COMPETITION_ID = "house-prices-advanced-regression-techniques"
EXPECTED_FILES = {
    "train.csv": TRAIN_CSV,
    "test.csv": TEST_CSV,
    "sample_submission.csv": SAMPLE_SUBMISSION_CSV,
    "data_description.txt": DATA_DESCRIPTION_TXT,
}

EXPECTED_DIMENSIONS = {
    "train.csv": (1460, 81),
    "test.csv": (1459, 80),
    "sample_submission.csv": (1459, 2),
}


def are_files_present_in_raw() -> bool:
    """Check if all expected competition files already exist and are non-empty in data/raw/."""
    for filename, filepath in EXPECTED_FILES.items():
        if not filepath.exists() or filepath.stat().st_size == 0:
            return False
    return True


def format_size(num_bytes: int) -> str:
    """Format bytes into a human-readable string."""
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024.0:
            return f"{num_bytes:.2f} {unit}"
        num_bytes /= 1024.0
    return f"{num_bytes:.2f} TB"


def download_official_dataset() -> Path:
    """Download competition files via kagglehub."""
    import kagglehub

    print(f"[*] Contacting Kaggle via kagglehub for competition '{COMPETITION_ID}'...")
    try:
        download_path = kagglehub.competition_download(COMPETITION_ID)
        print("Path to competition files:", download_path)
        return Path(download_path)
    except Exception as exc:
        print("\n[!] Error during Kaggle dataset download:")
        print(f"    {exc}\n")
        print("----------------------------------------------------------------------")
        print("KAGGLE SETUP INSTRUCTIONS:")
        print("1. Kaggle Account & API Credentials:")
        print("   - Go to https://www.kaggle.com/settings -> 'Create New Token'")
        print("   - Place 'kaggle.json' in: ~/.kaggle/kaggle.json")
        print("     (Windows: %USERPROFILE%\\.kaggle\\kaggle.json)")
        print("   - OR set environment variables:")
        print("     KAGGLE_USERNAME=your_kaggle_username")
        print("     KAGGLE_KEY=your_kaggle_api_key")
        print("2. Competition Rules Acceptance:")
        print(f"   - Visit: https://www.kaggle.com/competitions/{COMPETITION_ID}/rules")
        print("   - Click 'I Understand and Accept' to accept the competition rules.")
        print("----------------------------------------------------------------------\n")
        raise


def copy_files_to_raw(download_dir: Path) -> None:
    """Locate downloaded files and copy them to data/raw/ without modification."""
    ensure_directory(RAW_DATA_DIR)
    
    # Locate files recursively within the download directory
    for filename, target_path in EXPECTED_FILES.items():
        candidates = list(download_dir.rglob(filename))
        if not candidates:
            raise FileNotFoundError(
                f"Expected file '{filename}' was not found in downloaded folder: {download_dir}"
            )
        source_file = candidates[0]
        print(f"[*] Copying {filename} from {source_file} -> {target_path}")
        shutil.copy2(source_file, target_path)


def validate_dataset() -> bool:
    """Validate all files in data/raw, checking existence, readability, and dimensions."""
    print("\n" + "=" * 70)
    print("DATASET VALIDATION REPORT")
    print("=" * 70)

    all_valid = True

    for filename, filepath in EXPECTED_FILES.items():
        if not filepath.exists():
            print(f"[FAIL] Missing file: {filepath}")
            all_valid = False
            continue

        file_size = filepath.stat().st_size
        print(f"\n[FILE] {filename}")
        print(f"  - Location: {filepath}")
        print(f"  - Size: {file_size:,} bytes ({format_size(file_size)})")

        if filename.endswith(".csv"):
            try:
                df = pd.read_csv(filepath)
                actual_shape = df.shape
                print(f"  - Shape: {actual_shape[0]:,} rows x {actual_shape[1]} columns")
                
                expected_shape = EXPECTED_DIMENSIONS.get(filename)
                if expected_shape:
                    if actual_shape == expected_shape:
                        print(f"  - Dimension check: PASS (expected {expected_shape})")
                    else:
                        print(
                            f"  - Dimension check: DISCREPANCY! "
                            f"Expected {expected_shape}, got {actual_shape}"
                        )
                        all_valid = False

                # Target and ID checks
                if filename == "train.csv":
                    has_target = "SalePrice" in df.columns
                    has_id = "Id" in df.columns
                    print(f"  - Contains 'SalePrice' (target): {has_target}")
                    print(f"  - Contains 'Id' (identifier): {has_id}")
                    if not has_target or not has_id:
                        all_valid = False
                elif filename == "test.csv":
                    has_target = "SalePrice" in df.columns
                    has_id = "Id" in df.columns
                    print(f"  - Contains 'SalePrice' (target): {has_target} (Expected: False)")
                    print(f"  - Contains 'Id' (identifier): {has_id} (Expected: True)")
                    if has_target or not has_id:
                        all_valid = False
                elif filename == "sample_submission.csv":
                    cols = list(df.columns)
                    print(f"  - Columns: {cols} (Expected: ['Id', 'SalePrice'])")
                    if cols != ["Id", "SalePrice"]:
                        all_valid = False

            except Exception as e:
                print(f"  - Error reading CSV: {e}")
                all_valid = False

        elif filename.endswith(".txt"):
            try:
                with open(filepath, "r", encoding="utf-8", errors="replace") as f:
                    lines = f.readlines()
                print(f"  - Text file lines: {len(lines):,} lines")
                print("  - Readability check: PASS")
            except Exception as e:
                print(f"  - Error reading text file: {e}")
                all_valid = False

    print("\n" + "=" * 70)
    if all_valid:
        print("[SUCCESS] All 4 competition files validated successfully!")
    else:
        print("[WARNING] Dataset validation encountered discrepancies or errors.")
    print("=" * 70 + "\n")

    return all_valid


def main():
    print(f"[*] Starting Kaggle dataset acquisition for '{COMPETITION_ID}'...")
    
    if are_files_present_in_raw():
        print("[*] Required competition files already exist in data/raw/. Skipping download.")
    else:
        download_dir = download_official_dataset()
        copy_files_to_raw(download_dir)

    success = validate_dataset()
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
