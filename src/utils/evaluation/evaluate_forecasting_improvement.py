import os
from typing import Any, Dict

import numpy as np

from src.data.constants import OUTPUT_DIR
from src.models.forecasting.feedforward_forecasting_model import \
    FeedForwardForecastingModel
from src.models.forecasting.forcasting_model import ForecastingModel
from src.models.forecasting.lstm_forecasting_model import LSTMForecastingModel
from src.models.forecasting.n_linear_forecasting_model import \
    NLinearForecastingModel
from src.models.forecasting.tcn_forecasting_model import TCNForecastingModel
from src.utils.forecasting_utils import compare_old_and_new_model
from src.utils.generate_dataset import \
    create_training_windows_from_mts  # noqa: E402
from src.utils.logging_config import logger


class ForecasterEvaluator:
    def __init__(
        self,
        mts_dataset,
        train_indices,
        validation_indices,
        test_indices,
        config,
        window_size,
        horizon_length,
        num_epochs,
        model_type,
    ) -> None:
        self.mts_dataset = mts_dataset
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices
        self.window_size = window_size
        self.horizon_length = horizon_length
        self.model_type = model_type
        self.num_epochs = num_epochs
        self.config = config

        self.train_mts_array, self.validation_mts_array, self.test_mts_array = (
            self._extract_train_val_test()
        )

        self.original_forecasting_model: ForecastingModel = (
            self._initialize_forecasting_model()
        )
        self.original_forecasting_model.train(
            train_timeseries=self.train_mts_array,
            validation_timeseries=self.validation_mts_array,
        )
        self.original_forecasting_model.plot_loss(model_name="NLinearOld")

    def _extract_train_val_test(self):
        return (
            self.mts_dataset[self.train_indices],
            self.mts_dataset[self.validation_indices],
            self.mts_dataset[self.test_indices],
        )

    def _create_training_windows(self, dataset):
        (X_mts, y_mts) = create_training_windows_from_mts(
            mts=dataset,
            target_col_index=1,
            window_size=self.window_size,
            forecast_horizon=self.horizon_length,
        )

        return X_mts, y_mts

    def _initialize_forecasting_model(self) -> ForecastingModel:
        if self.model_type == "n_linear":
            logger.info("Running NLinearForecastingModel for forecasting evaluations")
            return NLinearForecastingModel(
                num_epochs=self.num_epochs,
                horizon_length=self.horizon_length,
                window_size=self.window_size,
            )
        elif self.model_type == "tcn":
            logger.info("Running TCNForecastingModel for forecasting evaluations")
            return TCNForecastingModel(
                num_epochs=self.num_epochs,
                horizon_length=self.horizon_length,
                window_size=self.window_size,
                dropout=0.1,
            )
        elif self.model_type == "lstm":
            logger.info("Running TCNForecastingModel for forecasting evaluations")
            return LSTMForecastingModel(
                num_epochs=self.num_epochs,
                horizon_length=self.horizon_length,
                window_size=self.window_size,
                dropout=0.1,
            )
        else:
            logger.info("Running NLinearForecastingModel for forecasting evaluations")
            return NLinearForecastingModel(
                num_epochs=self.num_epochs,
                horizon_length=self.horizon_length,
                window_size=self.window_size,
            )

    def evaluate_on_evaluation_set(self, inferred_mts_array, ohe, type=""):
        new_train_mts_array = np.vstack([self.train_mts_array, inferred_mts_array])

        new_forecasting_model: ForecastingModel = self._initialize_forecasting_model()
        new_forecasting_model.train(
            train_timeseries=new_train_mts_array,
            validation_timeseries=self.validation_mts_array,
        )
        new_forecasting_model.plot_loss(model_name="NLinearNew")
        (
            mse_plot,
            mse_delta_plot,
            mase_plot,
            mase_delta_plot,
            mse_delta_comparison,
            mase_delta_comparison,
        ) = compare_old_and_new_model(
            config=self.config,
            train_timeseries=self.train_mts_array,
            validation_timeseries=self.validation_mts_array,
            test_timeseries=self.test_mts_array,
            forecasting_model_wrapper_old=self.original_forecasting_model,
            forecasting_model_wrapper_new=new_forecasting_model,
            ohe=ohe,
            retrain_on=type,
            model_type=self.model_type,
        )
        mse_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecasting space evaluations",
                self.model_type,
                f"FORECASTING_SPACE_retrain_on_{type}_forecasting_model_comparison_mse.png",
            )
        )

        mse_delta_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecasting space evaluations",
                self.model_type,
                f"FORECASTING_SPACE_retrain_on_{type}_forecasting_model_improvement_delta_mse.png",
            )
        )
        mase_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecasting space evaluations",
                self.model_type,
                f"FORECASTING_SPACE_retrain_on_{type}_forecasting_model_comparison_mase.png",
            )
        )
        mase_delta_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecasting space evaluations",
                self.model_type,
                f"FORECASTING_SPACE_retrain_on_{type}_forecasting_model_improvement_delta_mase.png",
            )
        )

        mse_delta_comparison.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecasting space evaluations",
                self.model_type,
                f"FORECASTING_SPACE_retrain_on_{type}_forecasting_delta_mse.png",
            )
        )
        mase_delta_comparison.savefig(
            os.path.join(
                OUTPUT_DIR,
                "Forecasting space evaluations",
                self.model_type,
                f"FORECASTING_SPACE_retrain_on_{type}_forecasting_delta_mase.png",
            )
        )
