import numpy as np


def retrain_and_evaluate_forecasting_model(
    inferred_mts: np.ndarray, forecasting_model_params: Dict[Str, any]
):

    (
        X_transformed,
        y_transformed,
    ) = create_training_windows_from_mts(
        mts=inferred_mts,
        target_col_index=1,
        window_size=forecasting_model_params["window_size"],
        forecast_horizon=forecasting_model_params["horizon_length"],
    )

    X_new_train: np.ndarray = np.vstack((X_mts_train, X_transformed))
    y_new_train: np.ndarray = np.vstack((y_mts_train, y_transformed))

    forecasting_model_new: FeedForwardForecaster = FeedForwardForecaster(
        model_params=forecasting_model_params,
    )

    forecasting_model_wrapper_new: NeuralNetworkWrapper = NeuralNetworkWrapper(
        model=forecasting_model_new, training_params=forecasting_model_training_params
    )
    forecasting_model_wrapper_new.train(
        X_train=X_new_train,
        y_train=y_new_train,
        X_val=X_new_train,
        y_val=y_new_train,
        log_to_wandb=False,
    )

    model_comparison_fig = compare_old_and_new_model(
        X_test=X_mts_test,
        y_test=y_mts_test,
        X_val=X_mts_validation,
        y_val=y_mts_validation,
        X_train=X_mts_train,
        y_train=y_mts_train,
        forecasting_model_wrapper_old=forecasting_model_wrapper,
        forecasting_model_wrapper_new=forecasting_model_wrapper_new,
    )
    model_comparison_fig.savefig(
        os.path.join(OUTPUT_DIR, "forecasting_model_comparison.png")
    )
