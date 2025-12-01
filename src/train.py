from __future__ import annotations

import json
import logging



import joblib


import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from src.config import (
    TRAIN_FD001_PATH,
    ARTIFACTS_INFO_PATH,
    SCALER_PATH,
    ARTIFACTS_DIR,
    COLS,
)
from src.data import create_sequences, TurbofanSequenceDataset
from src.model import Autoencoder

logger = logging.getLogger(__name__)

# Hyperparameters for the detector
SEQUENCE_LENGTH = 30
HEALTHY_CYCLES = 40
BATCH_SIZE = 128
N_EPOCHS = 25
LEARNING_RATE = 1e-3
EMBEDDING_DIM = 32

MODEL_PATH = ARTIFACTS_DIR / "detector.pth"
TRAIN_PLOT_PATH = ARTIFACTS_DIR / "training_loss.png"


def _load_scaled_fd001() -> tuple[pd.DataFrame, list[str]]:
    """Load FD001 data and apply the fitted scaler.

    This function expects preprocessing to have run already so that
    scaler.joblib and artifacts_info.json exist under models/.

    Returns:
        A tuple (df, cols_to_scale) where df is the scaled FD001 DataFrame
        and cols_to_scale are the sensor columns that were scaled.
    """
    if not SCALER_PATH.is_file() or not ARTIFACTS_INFO_PATH.is_file():
        msg = (
            "Missing scaler or artifacts_info.json. "
            "Please run `python -m src.preprocess` first."
        )
        logger.error(msg)
        raise FileNotFoundError(msg)

    scaler = joblib.load(SCALER_PATH)
    with ARTIFACTS_INFO_PATH.open("r", encoding="utf-8") as f:
        artifacts_info = json.load(f)

    cols_to_scale = artifacts_info["cols_to_scale"]

    df = pd.read_csv(
        TRAIN_FD001_PATH,
        sep=r'\s+',
        header=None,
        names=list(COLS),
    )

    df.loc[:, cols_to_scale] = scaler.transform(df.loc[:, cols_to_scale])

    return df, cols_to_scale


def train_model() -> None:
    """Train the LSTM autoencoder on healthy FD001 sequences and save artifacts."""
    logger.info("Starting model training process.")

    # Load scaled data
    logger.info("Loading scaled FD001 data and preprocessing artifacts.")
    df, cols_to_scale = _load_scaled_fd001()
    n_features = len(cols_to_scale)
    logger.info("Number of features for model: %d", n_features)

    # Restrict to healthy cycles for each engine (group by engine unit_number and only take first HEALTHY_CYCLES)
    df_healthy = df.groupby("unit_number", as_index=False).head(HEALTHY_CYCLES) 
    logger.info("Total data shape: %s", df.shape)
    logger.info("Healthy data shape: %s", df_healthy.shape)

    # Build sliding window sequences
    healthy_sequences = create_sequences(
        data_df=df_healthy,
        sensor_cols=cols_to_scale,
        sequence_length=SEQUENCE_LENGTH,
    )

    logger.info("Created %d healthy sequences.", healthy_sequences.shape[0])

    dataset = TurbofanSequenceDataset(healthy_sequences)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2,
    )

    # Model, loss, optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    model = Autoencoder(
        n_features=n_features,
        sequence_length=SEQUENCE_LENGTH,
        embedding_dim=EMBEDDING_DIM,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    logger.info("Starting training loop for %d epochs.", N_EPOCHS)
    model.train()
    train_losses = []

    for epoch in range(N_EPOCHS):
        epoch_loss = 0.0

        for seq_in, seq_out in dataloader:
            seq_in, seq_out = seq_in.to(device), seq_out.to(device)

            optimizer.zero_grad()
            reconstructed = model(seq_in)
            loss = criterion(reconstructed, seq_out)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / max(1, len(dataloader))
        train_losses.append(avg_epoch_loss)
        logger.info("Epoch %d/%d, loss: %.6f", epoch + 1, N_EPOCHS, avg_epoch_loss)

    # Save model weights
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    logger.info("Training complete. Model saved to %s", MODEL_PATH)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )
    train_model()


if __name__ == "__main__":
    main()
