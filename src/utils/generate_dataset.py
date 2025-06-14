from typing import List, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult
from tqdm import tqdm

from src.utils.features import decomp_and_features
from src.utils.logging_config import logger


def generate_feature_dataframe(
    data: np.ndarray,  # Shape (num_mts, num_uts_in_mts, num timesteps)
    series_periodicity: int,
    num_features_per_uts: int,
) -> Tuple[np.ndarray, List[DecomposeResult]]:
    decomps, features = decomp_and_features(
        data,
        series_periodicity=series_periodicity,
        num_features_per_uts=num_features_per_uts,
    )

    return features, decomps


def generate_windows_dataset(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    include_columns: List[str] = [],
) -> List[pd.DataFrame]:
    dataset = []

    if len(include_columns) > 0:
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
        input_window_df = df.loc[i : i + window_size - 1, input_cols]

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
    mts: np.ndarray,
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

    for ts in mts:
        ts_length = len(ts[0])
        num_features = len(ts)

        if ts_length < window_size + forecast_horizon:
            logger.warning("Time series too short to create windows, skipping.")
            continue

        num_windows = ts_length - window_size - forecast_horizon + 1

        for i in range(num_windows):
            x_window = []
            for feature_idx in range(num_features):
                x_window.extend(ts[feature_idx][i : i + window_size])

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
