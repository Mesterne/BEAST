from src.plots.timeseries_forecast_comparison import plot_timeseries_forecast_comparison
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
