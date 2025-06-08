###
# Some of the code in this file is partially from ChatGPT.


import os
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from src.data.constants import OUTPUT_DIR
from src.models.forecasting.forcasting_model import ForecastingModel
from src.plots.ohe_plots import \
    create_and_save_plots_of_ohe_activated_performances_forecasting_space
from src.utils.evaluation.forecaster_evaluation import (
    mae_for_all_predictions, mae_for_each_forecast, rmse_for_all_predictions,
    rmse_for_each_forecast)
from src.utils.generate_dataset import create_training_windows_from_mts


def compare_original_and_transformed_forecasting(
    window_length,
    window_value,
    forecast_length,
    worst_forecast_old,
    worst_forecast_new,
    true_value,
    mae,
):
    custom_palette = {
        "Original Timeseries": "gray",
        "Old Forecast": "green",
        "New Forecast": "red",
    }

    custom_zorder = {
        "Original Timeseries": 1,
        "Old Forecast": 2,
        "New Forecast": 3,
    }

    custom_linewidth = {
        "Original Timeseries": 1,
        "Old Forecast": 1,
        "New Forecast": 1.5,
    }
    original_timeseries = np.concatenate([window_value, true_value])

    # Create DataFrame
    data = pd.DataFrame(
        {
            "Label": ["Original Timeseries"] * (window_length + forecast_length)
            + ["Old Forecast"] * forecast_length
            + ["New Forecast"] * forecast_length,
            "Value": np.concatenate(
                [original_timeseries, worst_forecast_old, worst_forecast_new]
            ),
            "Index": list(range(-window_length, 0))
            + list(range(1, forecast_length + 1)) * 3,
        }
    )

    fig, ax = plt.subplots(figsize=(8, 5))

    for label, group_data in data.groupby("Label"):
        ax.plot(
            group_data["Index"],
            group_data["Value"],
            label=label,
            color=custom_palette[label],
            zorder=custom_zorder[label],
            linewidth=custom_linewidth[label],
        )

    ax.set_xticks([])
    ax.xaxis.set_visible(False)

    ax.set_title(f"Comparison of forecasting models (MAE Old forecast: {mae})")
    ax.set_xlabel("Forecast Stage")
    ax.set_ylabel("Value")
    ax.legend()

    return fig


def compute_metrics(y, old_pred, new_pred):
    rmse_old = rmse_for_each_forecast(y, old_pred)
    rmse_new = rmse_for_each_forecast(y, new_pred)
    rmse_total_old = rmse_for_all_predictions(y, old_pred)
    rmse_total_new = rmse_for_all_predictions(y, new_pred)
    mae_old = mae_for_each_forecast(y, old_pred)
    mae_new = mae_for_each_forecast(y, new_pred)
    mae_total_old = mae_for_all_predictions(y, old_pred)
    mae_total_new = mae_for_all_predictions(y, new_pred)

    return (
        rmse_old,
        rmse_new,
        rmse_total_old,
        rmse_total_new,
        mae_old,
        mae_new,
        mae_total_old,
        mae_total_new,
    )


def compute_deltas(train_old, train_new, val_old, val_new, test_old, test_new):
    # Bigger is better
    delta_train = train_old - train_new
    delta_val = val_old - val_new
    delta_test = test_old - test_new

    return delta_train, delta_val, delta_test


def compare_old_and_new_model(
    config: Dict[str, Any],
    train_timeseries: np.ndarray,
    validation_timeseries: np.ndarray,
    test_timeseries: np.ndarray,
    forecasting_model_wrapper_old: ForecastingModel,
    forecasting_model_wrapper_new: ForecastingModel,
    ohe: np.ndarray,
    retrain_on: str,
    model_type: str,
):
    inferred_old_train = forecasting_model_wrapper_old.forecast(
        test_timeseries=train_timeseries
    )
    inferred_new_train = forecasting_model_wrapper_new.forecast(
        test_timeseries=train_timeseries
    )

    inferred_old_val = forecasting_model_wrapper_old.forecast(
        test_timeseries=validation_timeseries
    )
    inferred_new_val = forecasting_model_wrapper_new.forecast(
        test_timeseries=validation_timeseries
    )

    inferred_old_test = forecasting_model_wrapper_old.forecast(
        test_timeseries=test_timeseries
    )
    inferred_new_test = forecasting_model_wrapper_new.forecast(
        test_timeseries=test_timeseries
    )

    X_train, y_train = create_training_windows_from_mts(
        mts=train_timeseries,
        target_col_index=1,
        window_size=config["model_args"]["forecasting_model_args"]["window_size"],
        forecast_horizon=config["model_args"]["forecasting_model_args"][
            "horizon_length"
        ],
    )
    X_val, y_val = create_training_windows_from_mts(
        mts=validation_timeseries,
        target_col_index=1,
        window_size=config["model_args"]["forecasting_model_args"]["window_size"],
        forecast_horizon=config["model_args"]["forecasting_model_args"][
            "horizon_length"
        ],
    )
    X_test, y_test = create_training_windows_from_mts(
        mts=test_timeseries,
        target_col_index=1,
        window_size=config["model_args"]["forecasting_model_args"]["window_size"],
        forecast_horizon=config["model_args"]["forecasting_model_args"][
            "horizon_length"
        ],
    )
    errors = mae_for_each_forecast(y_true=y_test, y_pred=inferred_old_test)
    sorted_indices = np.argsort(errors)

    bottom_3 = sorted_indices[:3]
    top_3 = sorted_indices[-3:]
    midpoint = len(sorted_indices) // 2
    half_window = 1
    start = max(midpoint - half_window, 0)
    end = start + 3
    median_3 = sorted_indices[start:end]

    indices = np.concatenate([top_3, bottom_3, median_3])

    for i in indices:
        worst_forecast_old = inferred_old_test[i]
        worst_forecast_new = inferred_new_test[i]
        window_value = X_test[i][168:336]
        true_value = y_test[i]

        window_length = len(window_value)
        forecast_length = len(worst_forecast_old)

        forecast_plot = compare_original_and_transformed_forecasting(
            window_length=window_length,
            window_value=window_value,
            forecast_length=forecast_length,
            worst_forecast_old=worst_forecast_old,
            worst_forecast_new=worst_forecast_new,
            true_value=true_value,
            mae=errors[i],
        )
        forecast_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecast Grids",
                model_type,
                f"{int(errors[i])}_forecast_retrain_on_{retrain_on}_idx_{i}.png",
            )
        )

    (
        train_rmse_old,
        train_rmse_new,
        train_rmse_total_old,
        train_rmse_total_new,
        train_mae_old,
        train_mae_new,
        train_mae_total_old,
        train_mae_total_new,
    ) = compute_metrics(y_train, inferred_old_train, inferred_new_train)
    (
        val_rmse_old,
        val_rmse_new,
        val_rmse_total_old,
        val_rmse_total_new,
        val_mae_old,
        val_mae_new,
        val_mae_total_old,
        val_mae_total_new,
    ) = compute_metrics(y_val, inferred_old_val, inferred_new_val)
    (
        test_rmse_old,
        test_rmse_new,
        test_rmse_total_old,
        test_rmse_total_new,
        test_mae_old,
        test_mae_new,
        test_mae_total_old,
        test_mae_total_new,
    ) = compute_metrics(y_test, inferred_old_test, inferred_new_test)

    delta_train_rmse, delta_val_rmse, delta_test_rmse = compute_deltas(
        train_old=train_rmse_old,
        train_new=train_rmse_new,
        val_old=val_rmse_old,
        val_new=val_rmse_new,
        test_old=test_rmse_old,
        test_new=test_rmse_new,
    )

    rmse_delta_plot = plot_delta_distributions(
        delta_train=delta_train_rmse,
        delta_val=delta_val_rmse,
        delta_test=delta_test_rmse,
        metric_name="RMSE",
    )

    delta_train_mae, delta_val_mae, delta_test_mae = compute_deltas(
        train_old=train_mae_old,
        train_new=train_mae_new,
        val_old=val_mae_old,
        val_new=val_mae_new,
        test_old=test_mae_old,
        test_new=test_mae_new,
    )

    mae_delta_plot = plot_delta_distributions(
        delta_train=delta_train_mae,
        delta_val=delta_val_mae,
        delta_test=delta_test_mae,
        metric_name="MAE",
    )

    rmse_delta_comparison_plot = plot_delta_comparison_plot(
        train_metrics=[train_rmse_total_old, train_rmse_total_new],
        val_metrics=[val_rmse_total_old, val_rmse_total_new],
        test_metrics=[test_rmse_total_old, test_rmse_total_new],
        metric_name="RMSE",
    )

    mae_delta_comparison_plot = plot_delta_comparison_plot(
        train_metrics=[train_mae_total_old, train_mae_total_new],
        val_metrics=[val_mae_total_old, val_mae_total_new],
        test_metrics=[test_mae_total_old, test_mae_total_new],
        metric_name="MAE",
    )

    # RMSE barplot
    rmse_plot = plot_metric_comparison_train_validation_test(
        train_metrics=[train_rmse_total_old, train_rmse_total_new],
        val_metrics=[val_rmse_total_old, val_rmse_total_new],
        test_metrics=[test_rmse_total_old, test_rmse_total_new],
        metric_name="RMSE",
    )

    # MAE barplot
    mae_plot = plot_metric_comparison_train_validation_test(
        train_metrics=[train_mae_total_old, train_mae_total_new],
        val_metrics=[val_mae_total_old, val_mae_total_new],
        test_metrics=[test_mae_total_old, test_mae_total_new],
        metric_name="MAE",
    )

    return (
        rmse_plot,
        rmse_delta_plot,
        mae_plot,
        mae_delta_plot,
        rmse_delta_comparison_plot,
        mae_delta_comparison_plot,
    )


def plot_metric_comparison_train_validation_test(
    train_metrics: np.ndarray,
    val_metrics: np.ndarray,
    test_metrics: np.ndarray,
    metric_name: str = "None",
):
    plot = plt.figure(figsize=(10, 6))
    df = pd.DataFrame(
        {
            "Dataset": ["Train", "Train", "Validation", "Validation", "Test", "Test"],
            "Model": ["Old", "New"] * 3,
            "metric": [
                train_metrics[0],
                train_metrics[1],
                val_metrics[0],
                val_metrics[1],
                test_metrics[0],
                test_metrics[1],
            ],
        }
    )
    ax = sns.barplot(x="Dataset", y="metric", hue="Model", data=df)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)

    plt.title(f"{metric_name} Comparison: Old vs New Model")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Dataset")

    y_min = min(
        train_metrics[0],
        train_metrics[1],
        val_metrics[0],
        val_metrics[1],
        test_metrics[0],
        test_metrics[1],
    )
    y_max = max(
        train_metrics[0],
        train_metrics[1],
        val_metrics[0],
        val_metrics[1],
        test_metrics[0],
        test_metrics[1],
    )

    margin = (y_max - y_min) * 0.1  # 10% margin
    plt.ylim(y_min - margin, y_max + margin)

    return plot


def plot_delta_comparison_plot(
    train_metrics: np.ndarray,
    val_metrics: np.ndarray,
    test_metrics: np.ndarray,
    metric_name: str = "None",
):
    plot = plt.figure(figsize=(10, 6))
    df = pd.DataFrame(
        {
            "Dataset": ["Train", "Validation", "Test"],
            "metric": [
                train_metrics[0] - train_metrics[1],
                val_metrics[0] - val_metrics[1],
                test_metrics[0] - test_metrics[1],
            ],
        }
    )
    ax = sns.barplot(x="Dataset", y="metric", data=df)

    for container in ax.containers:
        if isinstance(container, BarContainer):
            ax.bar_label(container, fmt="%.4f", label_type="edge", padding=3)

    plt.title(f"{metric_name} Comparison: Delta (Bigger is better)")
    plt.ylabel(f"{metric_name} Delta")
    plt.xlabel("Dataset")

    y_min = min(
        train_metrics[0] - train_metrics[1],
        val_metrics[0] - val_metrics[1],
        test_metrics[0] - test_metrics[1],
    )
    y_max = max(
        train_metrics[0] - train_metrics[1],
        val_metrics[0] - val_metrics[1],
        test_metrics[0] - test_metrics[1],
    )

    margin = (y_max - y_min) * 0.1  # 10% margin
    plt.ylim(y_min - margin, y_max + margin)

    return plot


def plot_delta_distributions(delta_train, delta_val, delta_test, metric_name):
    mse_delta_plot = plt.figure(figsize=(10, 6))
    sns.kdeplot(
        data=delta_train,
        label="Train",
        fill=True,
    )
    sns.kdeplot(
        data=delta_val,
        label="Validation",
        fill=True,
    )
    sns.kdeplot(
        data=delta_test,
        label="Test",
        fill=True,
    )
    plt.title(f"Distribution of {metric_name} Deltas (Bigger is better)")
    plt.xlabel(f"{metric_name} Delta")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    return mse_delta_plot
