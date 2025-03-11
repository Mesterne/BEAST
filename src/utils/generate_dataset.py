from typing import List
import pandas as pd
from pandas.core.window import Window
from tqdm import tqdm
import numpy as np
from src.utils.features import decomp_and_features
from src.utils.logging_config import logger


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
    features = [
        "trend-strength",
        "trend-slope",
        # "trend-linearity",  # We ignore trend trend-linearity
        "seasonal-strength",
    ]

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
    logger.info(f"Generating training windows for forecasting...")

    num_samples = len(df) - window_size - forecast_horizon + 1
    if num_samples <= 0:
        raise ValueError("Not enough data to create windows.")

    X, y = [], []

    for i in range(num_samples):
        # Get the window slice as before
        input_window_df = df.loc[i : i + window_size - 1, input_cols]

        # Instead of flattening row by row, reshape to organize by column first
        # This stacks all values from first column, then all from second column, etc.
        input_window = input_window_df.values.T.flatten()

        target_window = df.loc[
            i + window_size : i + window_size + forecast_horizon - 1, target_col
        ].values.flatten()

        X.append(input_window)
        y.append(target_window)

    X = np.array(X)
    y = np.array(y)

    logger.info(
        f"Created forecasting time series. X shape: {X.shape}, y shape: {y.shape}"
    )

    return X, y


def create_training_windows_from_mts(
    mts: List[List[List[float]]],
    target_col_index: int,
    window_size: int,
    forecast_horizon: int,
):
    """
    Create training windows from multivariate time series data.

    Args:
        mts: List of time series, where each time series is a list of features,
             and each feature is a list of values
        target_col_index: Index of the target column to predict
        window_size: Size of the input window
        forecast_horizon: Number of future steps to predict

    Returns:
        X: Shape (number_of_training_windows, window_size*number_of_features)
        Y: Shape (number_of_training_windows, forecast_horizon)
    """
    X, Y = [], []

    # Iterate through each time series
    for ts in mts:
        # Get the length of this time series (assuming all features have same length)
        ts_length = len(ts[0])
        num_features = len(ts)

        # Check if we have enough data to create at least one window
        if ts_length < window_size + forecast_horizon:
            logger.warning("Time series too short to create windows, skipping.")
            continue

        # Calculate how many windows we can create from this time series
        num_windows = ts_length - window_size - forecast_horizon + 1

        # Create windows
        for i in range(num_windows):
            # Create input window with all features
            x_window = []
            for feature_idx in range(num_features):
                # Add all values for this feature in the window
                x_window.extend(ts[feature_idx][i : i + window_size])

            # Create target window (only from the target feature)
            y_window = ts[target_col_index][
                i + window_size : i + window_size + forecast_horizon
            ]

            X.append(x_window)
            Y.append(y_window)

    X = np.array(X)
    Y = np.array(Y)

    logger.info(
        f"Created forecasting time series. X shape: {X.shape}, y shape: {Y.shape}"
    )

    return X, Y
