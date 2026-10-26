"""Path management module for the House Prices Advanced Regression project.

Uses pathlib to ensure all paths are absolute and independent of the
current working directory.
"""

from pathlib import Path

# Project root: 3 levels up from src/common/paths.py -> src/common -> src -> <project_root>
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Configuration
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROJECT_CONFIG_FILE = CONFIGS_DIR / "project_config.yaml"

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Expected official raw data files
TRAIN_CSV = RAW_DATA_DIR / "train.csv"
TEST_CSV = RAW_DATA_DIR / "test.csv"
SAMPLE_SUBMISSION_CSV = RAW_DATA_DIR / "sample_submission.csv"
DATA_DESCRIPTION_TXT = RAW_DATA_DIR / "data_description.txt"

# Model artifact directories
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_1_MODELS_DIR = MODELS_DIR / "model_1_grandmaster"
MODEL_2_MODELS_DIR = MODELS_DIR / "model_2_legacy"

# Submissions directories
SUBMISSIONS_DIR = PROJECT_ROOT / "submissions"
MODEL_1_SUBMISSIONS_DIR = SUBMISSIONS_DIR / "model_1_grandmaster"
MODEL_2_SUBMISSIONS_DIR = SUBMISSIONS_DIR / "model_2_legacy"

# Reports directories
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_1_REPORTS_DIR = REPORTS_DIR / "model_1_grandmaster"
MODEL_2_REPORTS_DIR = REPORTS_DIR / "model_2_legacy"

# Code & documentation directories
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
TESTS_DIR = PROJECT_ROOT / "tests"
SRC_DIR = PROJECT_ROOT / "src"


def ensure_directory(path: Path) -> Path:
    """Ensure that a directory exists, creating parents if necessary.
    
    Args:
        path: Path object representing the directory to ensure.
        
    Returns:
        The ensured Path object.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path
