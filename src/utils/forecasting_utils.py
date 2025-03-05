from src.plots.timeseries_forecast_comparison import plot_timeseries_forecast_comparison
from src.utils.evaluation.forecaster_evaluation import mse_for_forecast
from src.utils.generate_dataset import create_training_windows
from src.utils.logging_config import logger

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


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

    mse_old_train = mse_for_forecast(y_train, inferred_old_train)
    mse_new_train = mse_for_forecast(y_train, inferred_new_train)

    mse_old_val = mse_for_forecast(y_val, inferred_old_val)
    mse_new_val = mse_for_forecast(y_val, inferred_new_val)

    mse_old_test = mse_for_forecast(y_test, inferred_old_test)
    mse_new_test = mse_for_forecast(y_test, inferred_new_test)

    # Plot results in seaborn, return fig
    data = {
        "Dataset": ["Train", "Train", "Validation", "Validation", "Test", "Test"],
        "Model": ["Old", "New", "Old", "New", "Old", "New"],
        "MSE": [
            mse_old_train,
            mse_new_train,
            mse_old_val,
            mse_new_val,
            mse_old_test,
            mse_new_test,
        ],
    }
    df = pd.DataFrame(data)

    # Create the plot
    plt.figure(figsize=(10, 6))
    fig = plt.figure()
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

    # Print summary of results
    print(
        f"Train MSE - Old: {mse_old_train:.4f}, New: {mse_new_train:.4f}, Diff: {mse_old_train - mse_new_train:.4f}"
    )
    print(
        f"Validation MSE - Old: {mse_old_val:.4f}, New: {mse_new_val:.4f}, Diff: {mse_old_val - mse_new_val:.4f}"
    )
    print(
        f"Test MSE - Old: {mse_old_test:.4f}, New: {mse_new_test:.4f}, Diff: {mse_old_test - mse_new_test:.4f}"
    )

    return fig
