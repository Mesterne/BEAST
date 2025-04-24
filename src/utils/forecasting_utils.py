import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.plots.timeseries_forecast_comparison import plot_timeseries_forecast_comparison
from src.utils.evaluation.forecaster_evaluation import mse_for_all_predictions, mse_for_each_forecast
from src.utils.generate_dataset import create_training_windows
from src.utils.logging_config import logger


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

    # Plot overall improvements
    mse_old_train = mse_for_each_forecast(y_train, inferred_old_train)
    mse_old_train_total = mse_for_all_predictions(y_train, inferred_old_train)
    mse_new_train = mse_for_each_forecast(y_train, inferred_new_train)
    mse_new_train_total = mse_for_all_predictions(y_train, inferred_new_train)
    mse_delta_train = mse_new_train - mse_old_train


    mse_old_val = mse_for_each_forecast(y_val, inferred_old_val)
    mse_old_val_total = mse_for_all_predictions(y_val, inferred_old_val)
    mse_new_val = mse_for_each_forecast(y_val, inferred_new_val)
    mse_new_val_total = mse_for_all_predictions(y_val, inferred_new_val)
    mse_delta_val = mse_new_val - mse_old_val

    mse_old_test = mse_for_each_forecast(y_test, inferred_old_test)
    mse_old_test_total = mse_for_all_predictions(y_test, inferred_old_test)
    mse_new_test = mse_for_each_forecast(y_test, inferred_new_test)
    mse_new_test_total = mse_for_all_predictions(y_test, inferred_new_test)
    mse_delta_test = mse_new_test - mse_old_test

    # Plot delta
    mse_delta_plot = plt.figure(figsize=(10, 6))
    sns.kdeplot(mse_delta_train, label='Train', fill=True)
    sns.kdeplot(mse_delta_val, label='Validation', fill=True)
    sns.kdeplot(mse_delta_test, label='Test', fill=True)
    plt.title('Distribution of MSE Deltas')
    plt.xlabel('MSE Delta')
    plt.ylabel('Density')
    plt.legend()
    plt.tight_layout()

    # Plot results in seaborn, return fig
    data = {
        "Dataset": ["Train", "Train", "Validation", "Validation", "Test", "Test"],
        "Model": ["Old", "New", "Old", "New", "Old", "New"],
        "MSE": [
            mse_old_train_total,
            mse_new_train_total,
            mse_old_val_total,
            mse_new_val_total,
            mse_old_test_total,
            mse_new_test_total,
        ],
    }
    df = pd.DataFrame(data)

    # Create the plot
    mse_plot = plt.figure()
    ax = sns.barplot(x="Dataset", y="MSE", hue="Model", data=df)

    # Add labels and title
    plt.title("MSE Comparison: Old vs New Model")
    plt.ylabel("Mean Squared Error")
    plt.xlabel("Dataset")

    # Add text annotations with MSE values
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2.0, height + 0.01, f"{height:.4f}", ha="center"
        )

    plt.tight_layout()

    return forecast_plot, mse_plot, mse_delta_plot
