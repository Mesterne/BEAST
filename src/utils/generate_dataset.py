from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np


def generate_windows_dataset(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    include_columns: List[str] = None,
) -> List[pd.DataFrame]:
    dataset = []

    if include_columns is not None:
        data = data[include_columns]

    for i in tqdm(range(0, len(data) - window_size + 1, step_size)):
        dataset.append(data.iloc[i : i + window_size])

    return dataset

def create_training_windows(df: pd.DataFrame, input_cols: list, target_col: str, window_size: int, forecast_horizon: int):
    df = df.dropna().reset_index(drop=True)
    
    num_samples = len(df) - window_size - forecast_horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough data to create windows.")

    X, y = [], []

    for i in range(num_samples):
        input_window = df.loc[i:i + window_size - 1, input_cols].values.flatten()
        
        target_window = df.loc[i + window_size:i + window_size + forecast_horizon - 1, target_col].values.flatten()
        
        X.append(input_window)
        y.append(target_window)

    return np.array(X), np.array(y)
