from typing import Any, Dict

import numpy as np

from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.forecasting.forcasting_model import ForecastingModel
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.utils.generate_dataset import create_training_windows_from_mts


class FeedForwardForecastingModel(ForecastingModel):
    def __init__(
        self,
        config: Dict[str, Any],
    ) -> None:
        self.config = config

        self.model = self._initialize_forecasting_model()

    def _create_training_windows(self, partition):
        (X_mts, y_mts) = create_training_windows_from_mts(
            mts=partition,
            target_col_index=1,
            window_size=self.config["model_args"]["forecasting_model_args"][
                "window_size"
            ],
            forecast_horizon=self.config["model_args"]["forecasting_model_args"][
                "horizon_length"
            ],
        )

        return X_mts, y_mts

    def _initialize_forecasting_model(self) -> NeuralNetworkWrapper:
        forecasting_model: FeedForwardForecaster = FeedForwardForecaster(
            model_params=self.config["model_args"]["forecasting_model_args"]
        )
        forecasting_model_wrapper: NeuralNetworkWrapper = NeuralNetworkWrapper(
            model=forecasting_model,
            training_params=self.config["model_args"]["forecasting_model_args"][
                "training_args"
            ],
        )
        return forecasting_model_wrapper

    def train(self, train_timeseries: np.ndarray, validation_timeseries: np.ndarray):
        X_mts_train, y_mts_train = self._create_training_windows(
            partition=train_timeseries
        )
        X_mts_validation, y_mts_validation = self._create_training_windows(
            partition=validation_timeseries
        )
        self.model.train(
            X_train=X_mts_train,
            y_train=y_mts_train,
            X_val=X_mts_validation,
            y_val=y_mts_validation,
            plot_loss=True,
        )

    def plot_loss(self, model_name: str):
        return super().plot_loss(model_name)

    def forecast(self, test_timeseries: np.ndarray) -> np.ndarray:
        X_mts_test, _ = self._create_training_windows(partition=test_timeseries)
        return self.model.infer(X_mts_test)
