from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ML_DIR = PROJECT_ROOT / "ML"

# Canonical labeled dataset used for retraining.
RAW_PATH = ML_DIR / "raw_posts_labeled.csv"

# Reproducible pipeline outputs.
ARTIFACTS_DIR = ML_DIR / "artifacts"
PROCESSED_PATH = ARTIFACTS_DIR / "processed_posts.csv"
SPLIT_DIR = ARTIFACTS_DIR / "splits"
TRAIN_PATH = SPLIT_DIR / "train.csv"
VAL_PATH = SPLIT_DIR / "val.csv"
TEST_PATH = SPLIT_DIR / "test.csv"

RESULTS_DIR = ARTIFACTS_DIR / "results"
BASELINE_RESULTS_DIR = RESULTS_DIR / "baseline"
EXPERIMENTS_RESULTS_DIR = RESULTS_DIR / "experiments"
FINAL_RESULTS_DIR = RESULTS_DIR / "final"
MLFLOW_RUNS_DIR = ML_DIR / "mlruns"

MODELS_DIR = ARTIFACTS_DIR / "models"
LOGREG_MODELS_DIR = MODELS_DIR / "logreg_tfidf"
DEPLOY_MODELS_DIR = MODELS_DIR / "deploy_nb"

# Runtime inference export target used by service/ and bot/.
SERVICE_CONFIG_DIR = PROJECT_ROOT / "service" / "config"
SERVICE_MODELS_DIR = SERVICE_CONFIG_DIR / "models"


def ensure_artifact_dirs() -> None:
    for path in [
        ARTIFACTS_DIR,
        SPLIT_DIR,
        BASELINE_RESULTS_DIR,
        EXPERIMENTS_RESULTS_DIR,
        FINAL_RESULTS_DIR,
        MLFLOW_RUNS_DIR,
        LOGREG_MODELS_DIR,
        DEPLOY_MODELS_DIR,
        SERVICE_CONFIG_DIR,
        SERVICE_MODELS_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
