from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.container import BarContainer

from src.models.forecasting.forcasting_model import ForecastingModel
from src.plots.ohe_plots import \
    create_and_save_plots_of_ohe_activated_performances_forecasting_space
from src.utils.evaluation.forecaster_evaluation import (
    mase_for_all_predictions, mase_for_each_forecast, mse_for_all_predictions,
    mse_for_each_forecast)
from src.utils.generate_dataset import create_training_windows_from_mts


def compare_original_and_transformed_forecasting(
    window_length,
    window_value,
    forecast_length,
    worst_forecast_old,
    worst_forecast_new,
    true_value,
):
    custom_palette = {
        "Original Timeseries": "gray",
        "Old Forecast": "green",
        "New Forecast": "red",
    }

    custom_zorder = {
        "Original Timeseries": 1,  # Bottom
        "Old Forecast": 2,
        "New Forecast": 3,  # Top
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

    fig, ax = plt.subplots(figsize=(8, 5))  # Create figure and axes

    # Plot each group on the same axes
    for label, group_data in data.groupby("Label"):
        ax.plot(
            group_data["Index"],
            group_data["Value"],
            label=label,
            color=custom_palette[label],
            zorder=custom_zorder[label],
            linewidth=custom_linewidth[label],
        )

    ax.set_xticks([])  # Remove x-axis ticks
    ax.xaxis.set_visible(False)  # Hide the x-axis line

    ax.set_title("Comparison of Worst Forecast: Old vs New Model with Window Value")
    ax.set_xlabel("Forecast Stage")
    ax.set_ylabel("Value")
    ax.legend()

    return fig


def compute_metrics(y, old_pred, new_pred, insample):
    mse_old = mse_for_each_forecast(y, old_pred)
    mse_new = mse_for_each_forecast(y, new_pred)
    mse_total_old = mse_for_all_predictions(y, old_pred)
    mse_total_new = mse_for_all_predictions(y, new_pred)
    mase_old = mase_for_each_forecast(y, old_pred, insample)
    mase_new = mase_for_each_forecast(y, new_pred, insample)
    mase_total_old = mase_for_all_predictions(y, old_pred, insample)
    mase_total_new = mase_for_all_predictions(y, new_pred, insample)

    return (
        mse_old,
        mse_new,
        mse_total_old,
        mse_total_new,
        mase_old,
        mase_new,
        mase_total_old,
        mase_total_new,
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
    # Plot worst forecast before and after
    errors = np.abs(y_val - inferred_old_val)
    errors_summed = np.sum(errors, axis=1)
    worst_index = np.argmax(errors_summed)
    worst_forecast_old = inferred_old_val[worst_index]
    worst_forecast_new = inferred_new_val[worst_index]
    window_value = X_val[worst_index][168:336]
    true_value = y_val[worst_index]

    window_length = len(window_value)
    forecast_length = len(worst_forecast_old)

    forecast_plot = compare_original_and_transformed_forecasting(
        window_length=window_length,
        window_value=window_value,
        forecast_length=forecast_length,
        worst_forecast_old=worst_forecast_old,
        worst_forecast_new=worst_forecast_new,
        true_value=true_value,
    )

    (
        train_mse_old,
        train_mse_new,
        train_mse_total_old,
        train_mse_total_new,
        train_mase_old,
        train_mase_new,
        train_mase_total_old,
        train_mase_total_new,
    ) = compute_metrics(y_train, inferred_old_train, inferred_new_train, X_train)
    (
        val_mse_old,
        val_mse_new,
        val_mse_total_old,
        val_mse_total_new,
        val_mase_old,
        val_mase_new,
        val_mase_total_old,
        val_mase_total_new,
    ) = compute_metrics(y_val, inferred_old_val, inferred_new_val, X_val)
    (
        test_mse_old,
        test_mse_new,
        test_mse_total_old,
        test_mse_total_new,
        test_mase_old,
        test_mase_new,
        test_mase_total_old,
        test_mase_total_new,
    ) = compute_metrics(y_test, inferred_old_test, inferred_new_test, X_test)

    delta_train_mse, delta_val_mse, delta_test_mse = compute_deltas(
        train_old=train_mse_old,
        train_new=train_mse_new,
        val_old=val_mse_old,
        val_new=val_mse_new,
        test_old=test_mse_old,
        test_new=test_mse_new,
    )

    mse_delta_plot = plot_delta_distributions(
        delta_train=delta_train_mse,
        delta_val=delta_val_mse,
        delta_test=delta_test_mse,
        metric_name="MSE",
    )

    delta_train_mase, delta_val_mase, delta_test_mase = compute_deltas(
        train_old=train_mase_old,
        train_new=train_mase_new,
        val_old=val_mase_old,
        val_new=val_mase_new,
        test_old=test_mase_old,
        test_new=test_mase_new,
    )

    mase_delta_plot = plot_delta_distributions(
        delta_train=delta_train_mase,
        delta_val=delta_val_mase,
        delta_test=delta_test_mase,
        metric_name="MASE",
    )

    mse_delta_comparison_plot = plot_delta_comparison_plot(
        train_metrics=[train_mse_total_old, train_mse_total_new],
        val_metrics=[val_mse_total_old, val_mse_total_new],
        test_metrics=[test_mse_total_old, test_mse_total_new],
        metric_name="MSE",
    )

    mase_delta_comparison_plot = plot_delta_comparison_plot(
        train_metrics=[train_mase_total_old, train_mase_total_new],
        val_metrics=[val_mase_total_old, val_mase_total_new],
        test_metrics=[test_mase_total_old, test_mase_total_new],
        metric_name="MASE",
    )

    # MSE barplot
    mse_plot = plot_metric_comparison_train_validation_test(
        train_metrics=[train_mse_total_old, train_mse_total_new],
        val_metrics=[val_mse_total_old, val_mse_total_new],
        test_metrics=[test_mse_total_old, test_mse_total_new],
        metric_name="MSE",
    )

    # MASE barplot
    mase_plot = plot_metric_comparison_train_validation_test(
        train_metrics=[train_mase_total_old, train_mase_total_new],
        val_metrics=[val_mase_total_old, val_mase_total_new],
        test_metrics=[test_mase_total_old, test_mase_total_new],
        metric_name="MASE",
    )

    create_and_save_plots_of_ohe_activated_performances_forecasting_space(
        ohe=ohe,
        train_metrics=[train_mse_old, train_mse_new],
        val_metrics=[val_mse_old, val_mse_new],
        test_metrics=[test_mse_old, test_mse_new],
        metric_name="MSE",
        retrain_on=retrain_on,
    )
    create_and_save_plots_of_ohe_activated_performances_forecasting_space(
        ohe=ohe,
        train_metrics=[train_mase_old, train_mase_new],
        val_metrics=[val_mase_old, val_mase_new],
        test_metrics=[test_mase_old, test_mase_new],
        metric_name="MASE",
        retrain_on=retrain_on,
    )

    return (
        forecast_plot,
        mse_plot,
        mse_delta_plot,
        mase_plot,
        mase_delta_plot,
        mse_delta_comparison_plot,
        mase_delta_comparison_plot,
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

    # Automatically adjust y-axis to fit the values better
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

    # Set the y-axis limits to fit the range of the data
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

    # Automatically adjust y-axis to fit the values better
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

    # Set the y-axis limits to fit the range of the data
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
