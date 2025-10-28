import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
import json
import os

# --- 1. Configuration & Column Definitions ---

# Define paths
DATA_PATH = 'data/CMAPSSData/train_FD001.txt'
ARTIFACTS_DIR = 'models/' # Dir to store scaler & col info
SCALER_PATH = os.path.join(ARTIFACTS_DIR, 'scaler.joblib')
ARTIFACTS_INFO_PATH = os.path.join(ARTIFACTS_DIR, 'artifacts_info.json')

# Define column names
op_settings = [f'op_setting_{i+1}' for i in range(3)]
sensors = [f'sensor_{i+1}' for i in range(21)]
cols = ['unit_number', 'time_in_cycles'] + op_settings + sensors

# --- 2. Windowing Function ---

def create_sequences(data_df: pd.DataFrame, sensor_cols: list, sequence_length: int) -> np.ndarray:
    """
    Transforms time-series data into sequences for the autoencoder.
    
    This function processes data per-engine.
    
    Args:
        data_df (pd.DataFrame): The input DataFrame (should be scaled).
        sensor_cols (list): List of sensor column names to use.
        sequence_length (int): The window size.
            
    Returns:
        np.ndarray: An array of sequences (samples, time_steps, features).
    """
    sequences = []
    
    # Iterate over each engine unit
    for unit_id in data_df['unit_number'].unique():
        # Get the sensor data for this specific engine
        unit_data = data_df[data_df['unit_number'] == unit_id][sensor_cols].values
        
        # Create sequences for this engine
        for i in range(len(unit_data) - sequence_length + 1):
            seq = unit_data[i:i + sequence_length]
            sequences.append(seq)
            
    return np.array(sequences)

# --- 3. Main Script to Fit and Save Scaler ---

def fit_and_save_artifacts(data_path, cols, artifacts_dir, scaler_path, info_path):
    """
    Fits a scaler on the training data and saves it.
    Also identifies constant columns and saves all artifact info.
    """
    print("Starting preprocessing...")
    
    # 1. Load data
    try:
        df = pd.read_csv(data_path, sep=r"\s+", header=None, names=cols)
    except FileNotFoundError:
        print(f"Error: Data file not found at {data_path}")
        print("Please ensure 'train_FD001.txt' is in the 'data/CMAPSSData/' directory.")
        return
        
    # 2. Identify constant columns
    sensor_df = df[sensors]
    variance = sensor_df.var()
    # These are sensors with 0 variance, providing no info
    cols_to_drop = variance[variance == 0].index.tolist()
    
    # 3. Identify columns to scale
    # These are sensor columns that are *not* constant
    cols_to_scale = [col for col in sensors if col not in cols_to_drop]
    
    print(f"Identified {len(cols_to_drop)} constant columns to drop: {cols_to_drop}")
    print(f"Identified {len(cols_to_scale)} columns to scale.")

    # 4. Fit scaler
    # We use MinMaxScaler.
    scaler = MinMaxScaler()
    
    # Fit the scaler ONLY on the 'cols_to_scale'
    # from the training dataset (train_FD001.txt).
    scaler.fit(df[cols_to_scale])
    
    # 5. Save artifacts
    os.makedirs(artifacts_dir, exist_ok=True)
    
    # Save the scaler. We use joblib, as it's more efficient 
    # for sklearn models containing large numpy arrays.
    joblib.dump(scaler, scaler_path)
    print(f"\nScaler saved to: {scaler_path}")
    
    # Save the column info. This is CRITICAL for our API later.
    # It needs to know which columns to drop and which to scale.
    artifacts_info = {
        'cols_to_drop': cols_to_drop,
        'cols_to_scale': cols_to_scale
    }
    with open(info_path, 'w') as f:
        json.dump(artifacts_info, f, indent=4)
    print(f"Artifacts info (column lists) saved to: {info_path}")
    print("Preprocessing complete.")

if __name__ == "__main__":
    fit_and_save_artifacts(
        data_path=DATA_PATH,
        cols=cols,
        artifacts_dir=ARTIFACTS_DIR,
        scaler_path=SCALER_PATH,
        info_path=ARTIFACTS_INFO_PATH
    )