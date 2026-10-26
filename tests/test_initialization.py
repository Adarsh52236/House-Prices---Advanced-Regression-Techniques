"""Initialization and configuration tests."""

import sys
from pathlib import Path
import yaml

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.common.paths import (
    CONFIGS_DIR,
    DATA_DIR,
    MODELS_DIR,
    PROJECT_CONFIG_FILE,
    PROJECT_ROOT as PATHS_PROJECT_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    SUBMISSIONS_DIR,
)


def test_paths_exist():
    assert PATHS_PROJECT_ROOT.exists()
    assert CONFIGS_DIR.exists()
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert MODELS_DIR.exists()
    assert REPORTS_DIR.exists()
    assert SUBMISSIONS_DIR.exists()
    assert PROJECT_CONFIG_FILE.exists()


def test_config_structure():
    with open(PROJECT_CONFIG_FILE, "r") as f:
        config = yaml.safe_load(f)

    assert config["competition"]["id"] == "house-prices-advanced-regression-techniques"
    assert config["competition"]["target_column"] == "SalePrice"
    assert config["competition"]["id_column"] == "Id"
    assert config["competition"]["random_seed"] == 42
    assert config["competition"]["evaluation_metric"] == "RMSLE"

    assert config["models"]["active_model"] == "model_1_grandmaster"
    assert config["models"]["model_1_grandmaster"]["status"] == "in_development"
    assert config["models"]["model_2_legacy"]["status"] == "not_started"


def test_model_2_unimplemented():
    """Verify Model 2 directory and package are empty apart from minimal markers."""
    import src.model_2_legacy as m2
    
    # Model 2 package should not have models, pipelines, or classes implemented
    exported = [attr for attr in dir(m2) if not attr.startswith("__")]
    assert len(exported) == 0, f"Model 2 must remain unimplemented, but found exports: {exported}"
