from typing import List
import pandas as pd
from tqdm import tqdm
import numpy as np
from src.utils.features import decomp_and_features


def generate_feature_dataframe(data, series_periodicity, dataset_size):
    decomps, features = decomp_and_features(
        data, series_periodicity=series_periodicity, dataset_size=dataset_size
    )

    ts_indices_to_names = {0: "grid1-load", 1: "grid1-loss", 2: "grid1-temp"}

    data = []
    for idx in range(features.shape[0]):
        for ts_idx in range(features.shape[1]):
            row = {
                "index": idx,
                "ts_name": ts_indices_to_names[ts_idx],
                "trend-strength": features[idx, ts_idx, 0],
                "trend-slope": features[idx, ts_idx, 1],
                "trend-linearity": features[idx, ts_idx, 2],
                "seasonal-strength": features[idx, ts_idx, 3],
            }
            data.append(row)

    df = pd.DataFrame(data)

    feature_df = df.pivot_table(
        index="index",
        columns="ts_name",
        values=[
            "trend-strength",
            "trend-slope",
            "trend-linearity",
            "seasonal-strength",
        ],
    )

    feature_df.columns = [f"{ts}_{feature}" for feature, ts in feature_df.columns]

    # Extract time series names and their features
    ts_names = df["ts_name"].unique()
    features = ["trend-strength", "trend-slope", "trend-linearity", "seasonal-strength"]

    # Create the ordered column list
    ordered_columns = [f"{ts}_{feature}" for ts in ts_names for feature in features]

    # Reorder columns based on the ordered list
    feature_df = feature_df[ordered_columns]
    return feature_df, decomps


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


def create_training_windows(
    df: pd.DataFrame,
    input_cols: list,
    target_col: str,
    window_size: int,
    forecast_horizon: int,
):
    df = df.dropna().reset_index(drop=True)

    num_samples = len(df) - window_size - forecast_horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough data to create windows.")

    X, y = [], []

    for i in range(num_samples):
        input_window = df.loc[i : i + window_size - 1, input_cols].values.flatten()

        target_window = df.loc[
            i + window_size : i + window_size + forecast_horizon - 1, target_col
        ].values.flatten()

        X.append(input_window)
        y.append(target_window)

    return np.array(X), np.array(y)
