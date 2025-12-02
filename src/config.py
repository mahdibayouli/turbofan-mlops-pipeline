from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]  # src/ is one level below root

# Dataset
DATA_DIR = REPO_ROOT / "data" / "CMAPSSData"
TRAIN_FD001_PATH = DATA_DIR / "train_FD001.txt"

# Column definitions
OP_SETTINGS = [f"op_setting_{i}" for i in range(1, 4)]
SENSORS = [f"sensor_{i}" for i in range(1, 22)]
COLS = ["unit_number", "time_in_cycles"] + OP_SETTINGS + SENSORS

# Artifacts
ARTIFACTS_DIR = REPO_ROOT / "models"
SCALER_PATH = ARTIFACTS_DIR / "scaler.joblib"
ARTIFACTS_INFO_PATH = ARTIFACTS_DIR / "artifacts_info.json"
MODEL_PATH = ARTIFACTS_DIR / "detector.pth"

# Training hyperparameters
SEQUENCE_LENGTH = 30
HEALTHY_CYCLES = 40
BATCH_SIZE = 128
N_EPOCHS = 25
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 32
