from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plots.timeseries_forecast_comparison import plot_timeseries_forecast_comparison
from src.utils.evaluation.forecaster_evaluation import (
    mase_for_all_predictions,
    mase_for_each_forecast,
    mse_for_all_predictions,
    mse_for_each_forecast,
)
from src.utils.generate_dataset import create_training_windows


def compare_original_and_transformed_forecasting(
    original_mts, transformed_mts, forecasting_model_wrapper, forecasting_model_params
):
    (
        X_mts_original,
        y_mts_original,
    ) = create_training_windows(
        df=original_mts,
        input_cols=["grid1-load", "grid1-loss", "grid1-temp"],
        target_col="grid1-loss",
        window_size=forecasting_model_params["window_size"],
        forecast_horizon=forecasting_model_params["horizon_length"],
    )
    (
        X_mts_transformed,
        y_mts_transformed,
    ) = create_training_windows(
        df=transformed_mts,
        input_cols=["grid1-load", "grid1-loss", "grid1-temp"],
        target_col="grid1-loss",
        window_size=forecasting_model_params["window_size"],
        forecast_horizon=forecasting_model_params["horizon_length"],
    )

    inferred_original = forecasting_model_wrapper.infer(X=X_mts_original)
    inferred_transformed = forecasting_model_wrapper.infer(
        X=X_mts_transformed,
    )

    forecast_plot = plot_timeseries_forecast_comparison(
        X_original=X_mts_original[0],
        X_transformed=X_mts_transformed[0],
        y_original=y_mts_original[0],
        y_transformed=y_mts_transformed[0],
        inferred_original=inferred_original[0],
        inferred_transformed=inferred_transformed[0],
        feature_names=["grid1-load", "grid1-loss", "grid1-temp"],
        target_name="grid1-loss",
    )
    return forecast_plot


def compute_metrics(y, old_pred, new_pred, insample):
    mse_old = mse_for_each_forecast(y, old_pred)
    mse_new = mse_for_each_forecast(y, new_pred)
    mse_total_old = mse_for_all_predictions(y, old_pred)
    mse_total_new = mse_for_all_predictions(y, new_pred)
    mase_old = mase_for_each_forecast(y, old_pred, insample)
    mase_new = mase_for_each_forecast(y, new_pred, insample)
    mase_total_old = mase_for_all_predictions(y, old_pred, insample)
    mase_total_new = mase_for_all_predictions(y, new_pred, insample)

    return {
        "mse_each": (mse_old, mse_new),
        "mse_total": (mse_total_old, mse_total_new),
        "mase_each": (mase_old, mase_new),
        "mase_total": (mase_total_old, mase_total_new),
    }


def compare_old_and_new_model(
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    forecasting_model_wrapper_old,
    forecasting_model_wrapper_new,
):
    inferred_old_train = forecasting_model_wrapper_old.infer(X=X_train)
    inferred_new_train = forecasting_model_wrapper_new.infer(X=X_train)

    inferred_old_val = forecasting_model_wrapper_old.infer(X=X_val)
    inferred_new_val = forecasting_model_wrapper_new.infer(X=X_val)

    inferred_old_test = forecasting_model_wrapper_old.infer(X=X_test)
    inferred_new_test = forecasting_model_wrapper_new.infer(X=X_test)

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

    # Forecast plot
    data = pd.DataFrame(
        {
            "Forecast Type": ["Window Value"] * window_length
            + ["Old Model"] * forecast_length
            + ["New Model"] * forecast_length
            + ["True Value"] * forecast_length,
            "Value": np.concatenate(
                [window_value, worst_forecast_old, worst_forecast_new, true_value]
            ),
            "Index": list(range(-window_length, 0))
            + list(range(1, forecast_length + 1)) * 3,
        }
    )

    forecast_plot = plt.figure(figsize=(8, 5))
    ax = sns.lineplot(data=data, x="Index", y="Value", hue="Forecast Type")
    plt.title("Comparison of Worst Forecast: Old vs New Model with Window Value")
    plt.xlabel("Forecast Stage")
    plt.ylabel("Value")

    # Metric computation
    train_metrics = compute_metrics(
        y_train, inferred_old_train, inferred_new_train, X_train
    )
    val_metrics = compute_metrics(y_val, inferred_old_val, inferred_new_val, X_val)
    test_metrics = compute_metrics(y_test, inferred_old_test, inferred_new_test, X_test)

    # MSE Delta KDE plot
    mse_delta_plot = plt.figure(figsize=(10, 6))
    sns.kdeplot(
        train_metrics["mse_each"][1] - train_metrics["mse_each"][0],
        label="Train",
        fill=True,
    )
    sns.kdeplot(
        val_metrics["mse_each"][1] - val_metrics["mse_each"][0],
        label="Validation",
        fill=True,
    )
    sns.kdeplot(
        test_metrics["mse_each"][1] - test_metrics["mse_each"][0],
        label="Test",
        fill=True,
    )
    plt.title("Distribution of MSE Deltas (New - Old)")
    plt.xlabel("MSE Delta")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()

    # MSE barplot
    mse_plot = plot_metric_comparison_train_validation_test(
        train_metrics=train_metrics["mse_total"],
        val_metrics=val_metrics["mse_total"],
        test_metrics=test_metrics["mse_total"],
        metric_name="MSE",
    )

    # MASE barplot
    mase_plot = plot_metric_comparison_train_validation_test(
        train_metrics=train_metrics["mase_total"],
        val_metrics=val_metrics["mase_total"],
        test_metrics=test_metrics["mase_total"],
        metric_name="mase",
    )

    return forecast_plot, mse_plot, mse_delta_plot, mase_plot


def plot_metric_comparison_train_validation_test(
    train_metrics: Tuple[float, float],
    val_metrics: Tuple[float, float],
    test_metrics: Tuple[float, float],
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
    plt.title(f"{metric_name} Comparison: Old vs New Model")
    plt.ylabel(f"{metric_name}")
    plt.xlabel("Dataset")
    for p in ax.patches:
        ax.text(
            p.get_x() + p.get_width() / 2.0,
            p.get_height() + 0.01,
            f"{p.get_height():.4f}",
            ha="center",
        )
    plt.tight_layout()
    return plot
