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
    """Convert per engine times series (cycles) into sliding window sequences ready for model training.
    
    The function groups rows by engine id (unit_number) and builds overlapping windows of fixed length over the selected sensor columns.
    
    Returns: 
        A NumPy array of shape `(num_sequences, sequence_length, n_features)`

    """
    sequences = []
    
    for engine_id in data_df["unit_number"].unique():
        engine_data = data_df.loc[data_df["unit_number"] == engine_id, sensor_cols].to_numpy()
        
        if len(engine_data) < sequence_length:
            # engine does not have enough cycle for even a single window
            continue
        
        for start_idx in range(len(engine_data)-sequence_length +1):
            window = engine_data[start_idx: start_idx + sequence_length]
            sequences.append(window)
        
    if not sequences:
        # num_sequences is 0 (no engine had enough cycles)
        return np.empty((0, sequence_length, len(sensor_cols)), dtype=np.float32)

    return np.asarray(sequences, dtype=np.float32)

class TurbofanSequenceDataset(Dataset):
    """Dataset of sliding window engine sequences of shape (num_sequences, sequence_length, n_features) for model training.
    """
    def __init__(self, sequences: npt.NDArray[np.float32]) -> None:
        self.sequences = torch.from_numpy(sequences)
        
    def __len__(self) -> int:
        return self.sequences.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        seq = self.sequences[idx]
        # Each item returns the same sequence as input and target, which is standard for reconstruction based anomaly detection
        return seq, seq 