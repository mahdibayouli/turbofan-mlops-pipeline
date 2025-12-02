from __future__ import annotations

from typing import Sequence

import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch import Tensor
from torch.utils.data import Dataset

def create_sequences(
    data_df: pd.DataFrame,
    sensor_cols: Sequence[str],
    sequence_length: int,
) -> npt.NDArray[np.float32]:
    """Convert per engine time series (cycles) into sliding window sequences ready for model training.
    
    The function groups rows by engine id (unit_number) and builds overlapping windows
    of fixed length over the selected sensor columns.
    
    Args:
        data_df: DataFrame containing engine sensor data with 'unit_number' column.
        sensor_cols: List of sensor column names to include in sequences.
        sequence_length: Number of consecutive cycles in each window.
    
    Returns:
        A NumPy array of shape (num_sequences, sequence_length, n_features).
    """
    sequences = []
    
    for engine_id in data_df["unit_number"].unique():
        engine_data = data_df.loc[data_df["unit_number"] == engine_id, sensor_cols].to_numpy()
        
        if len(engine_data) < sequence_length:
            # Engine does not have enough cycles for even a single window.
            continue
        
        for start_idx in range(len(engine_data) - sequence_length + 1):
            window = engine_data[start_idx: start_idx + sequence_length]
            sequences.append(window)
        
    if not sequences:
        # No engine had enough cycles (all engines are too short).
        return np.empty((0, sequence_length, len(sensor_cols)), dtype=np.float32)

    return np.asarray(sequences, dtype=np.float32)

class TurbofanSequenceDataset(Dataset):
    """Dataset of sliding window engine sequences for model training.
    
    Args:
        sequences: NumPy array of shape (num_sequences, sequence_length, n_features).
    """
    def __init__(self, sequences: npt.NDArray[np.float32]) -> None:
        self.sequences = torch.from_numpy(sequences)
        
    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """Get a sequence pair (input, target) for training.
        
        Returns the same sequence as both input and target for reconstruction.
        """
        seq = self.sequences[idx]
        return seq, seq 