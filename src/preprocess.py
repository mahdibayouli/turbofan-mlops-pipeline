from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Sequence

import joblib
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from src.config import (
    TRAIN_FD001_PATH,
    ARTIFACTS_DIR,
    SCALER_PATH,
    ARTIFACTS_INFO_PATH,
    SENSORS,
    COLS,
)

logger = logging.getLogger(__name__)


def fit_and_save_artifacts(
    data_path: Path = TRAIN_FD001_PATH,
    cols: Sequence[str] = COLS,
    artifacts_dir: Path = ARTIFACTS_DIR,
    scaler_path: Path = SCALER_PATH,
    info_path: Path = ARTIFACTS_INFO_PATH,
) -> None:
    """Fit a MinMaxScaler on training data and save preprocessing artifacts.

    The scaler is fit only on non-constant sensor columns. The function writes:

    * `scaler.joblib`: fitted MinMaxScaler
    * `artifacts_info.json`: metadata about which sensor columns to drop/scale
    
    Args:
        data_path: Path to the training data file.
        cols: List of column names for the dataset.
        artifacts_dir: Directory to save artifacts.
        scaler_path: Path for saving the fitted scaler.
        info_path: Path for saving artifacts metadata JSON.
    """
    logger.info("Starting preprocessing using data at %s", data_path)

    if not data_path.is_file():
        logger.error("Data file not found at %s", data_path)
        raise FileNotFoundError(
            f"Data file not found at {data_path}. "
            "Expected 'train_FD001.txt' under 'data/CMAPSSData/'."
        )

    df = pd.read_csv(
        data_path,
        sep=r'\s+',
        header=None,
        names=list(cols),
    )

    # Identify constant columns among sensors
    sensor_df = df.loc[:, SENSORS]
    variance = sensor_df.var()
    cols_to_drop = variance[variance == 0.0].index.to_list()
    cols_to_scale = [col for col in SENSORS if col not in cols_to_drop]

    logger.info("Found %d constant sensor columns to drop", len(cols_to_drop))
    logger.info("Found %d sensor columns to scale", len(cols_to_scale))

    # Fit scaler on non-constant sensors (training data only)
    scaler = MinMaxScaler()
    scaler.fit(df.loc[:, cols_to_scale])

    # Save artifacts
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(scaler, scaler_path)
    logger.info("Scaler saved to %s", scaler_path)

    artifacts_info = {
        "cols_to_drop": cols_to_drop,
        "cols_to_scale": cols_to_scale,
    }

    info_path.parent.mkdir(parents=True, exist_ok=True)
    with info_path.open("w", encoding="utf-8") as f:
        json.dump(artifacts_info, f, indent=4)

    logger.info("Artifacts metadata saved to %s", info_path)
    logger.info("Preprocessing complete")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    fit_and_save_artifacts()


if __name__ == "__main__":
    main()
