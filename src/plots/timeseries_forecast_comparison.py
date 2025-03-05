import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


def plot_timeseries_forecast_comparison(
    X_original,
    X_transformed,
    y_original,
    y_transformed,
    inferred_original,
    inferred_transformed,
    feature_names,
    target_name,
):
    """
    Create a plot comparing original and transformed time series with inferred values.

    Parameters:
    -----------
    X_original : numpy.ndarray
        Original feature time series data, shape (n_samples*n_features,)
        Features are interleaved: [feat1_t1, feat2_t1, ..., featN_t1, feat1_t2, ...]
    X_transformed : numpy.ndarray
        Transformed feature time series data, same shape as X_original
    y_original : numpy.ndarray
        Original target time series data (future values)
    y_transformed : numpy.ndarray
        Transformed target time series data (future values)
    inferred_original : numpy.ndarray
        Inferred values in original scale (future predictions)
    inferred_transformed : numpy.ndarray
        Inferred values in transformed scale (future predictions)
    feature_names : list
        Names of the features
    target_name : str
        Name of the target variable
    """
    # Calculate number of time steps
    n_features = len(feature_names)
    n_samples = len(X_original) // n_features
    forecast_length = len(y_original)

    # Reshape the contiguous blocks into 2D arrays where each column is a feature
    X_original_reshaped = np.zeros((n_samples, n_features))
    X_transformed_reshaped = np.zeros((n_samples, n_features))

    for i in range(n_features):
        # Extract each feature from its contiguous block
        start_idx = i * n_samples
        end_idx = (i + 1) * n_samples
        X_original_reshaped[:, i] = X_original[start_idx:end_idx]
        X_transformed_reshaped[:, i] = X_transformed[start_idx:end_idx]

    # Find the index of the target feature
    try:
        target_feature_index = feature_names.index(target_name)
    except ValueError:
        # If target_name is not in feature_names, assume it's a separate variable
        target_feature_index = None

    # Create figure with 2 columns (original and transformed) and n_features rows
    fig, axes = plt.subplots(n_features, 2, figsize=(14, 4 * n_features), sharex="col")

    # Handle case where there's only one feature (axes will be 1D)
    if n_features == 1:
        axes = axes.reshape(1, 2)

    # Set titles for columns
    axes[0, 0].set_title("Original Time Series", fontsize=14)
    axes[0, 1].set_title("Transformed Time Series", fontsize=14)

    # Create time arrays for plotting
    time_features = np.arange(n_samples)
    time_forecast = np.arange(n_samples, n_samples + forecast_length)

    # Plot features
    for i in range(n_features):
        # Original data
        ax0 = axes[i, 0]
        ax0.plot(
            time_features,
            X_original_reshaped[:, i],
            label=feature_names[i],
            color="blue",
        )

        # If this is the target feature, also plot the target and inferred values
        if i == target_feature_index:
            # Plot target as future values
            ax0.plot(
                time_forecast,
                y_original,
                label=f"{target_name} (Target)",
                color="green",
                alpha=0.7,
            )
            ax0.plot(
                time_forecast,
                inferred_original,
                label="Inferred",
                color="red",
                linestyle="--",
            )

            # Add a vertical line to separate historical data from future predictions
            ax0.axvline(x=n_samples - 1, color="gray", linestyle="--", alpha=0.8)

        ax0.set_ylabel(feature_names[i], fontsize=12)
        ax0.legend(loc="upper right")
        ax0.grid(True, linestyle="--", alpha=0.7)

        if i == n_features - 1:  # Only add x-label to bottom plot
            ax0.set_xlabel("Time", fontsize=12)

        # Transformed data
        ax1 = axes[i, 1]
        ax1.plot(
            time_features,
            X_transformed_reshaped[:, i],
            label=feature_names[i],
            color="blue",
        )

        # If this is the target feature, also plot the target and inferred values
        if i == target_feature_index:
            # Plot target as future values
            ax1.plot(
                time_forecast,
                y_transformed,
                label=f"{target_name} (Target)",
                color="green",
                alpha=0.7,
            )
            ax1.plot(
                time_forecast,
                inferred_transformed,
                label="Inferred",
                color="red",
                linestyle="--",
            )

            # Add a vertical line to separate historical data from future predictions
            ax1.axvline(x=n_samples - 1, color="gray", linestyle="--", alpha=0.8)

        ax1.set_ylabel(feature_names[i], fontsize=12)
        ax1.legend(loc="upper right")
        ax1.grid(True, linestyle="--", alpha=0.7)

        if i == n_features - 1:  # Only add x-label to bottom plot
            ax1.set_xlabel("Time", fontsize=12)

    plt.tight_layout()
    return fig
