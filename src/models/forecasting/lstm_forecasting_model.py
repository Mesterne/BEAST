import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from darts.dataprocessing.transformers.scaler import Scaler
from darts.models.forecasting.rnn_model import RNNModel

from src.data.constants import OUTPUT_DIR
from src.models.forecasting.forcasting_model import ForecastingModel
from src.models.forecasting.loss_tracker import LossTracker
from src.utils.darts_utils import array_to_timeseries
from src.utils.logging_config import logger


class LSTMForecastingModel(ForecastingModel):
    def __init__(self, window_size, horizon_length, num_epochs, dropout) -> None:
        self.window_size = window_size
        self.horizon_length = horizon_length
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.loss_tracker = LossTracker()
        self.model = self._initialize_forecasting_model()
        self.scaler = Scaler()
        self.covariates_scaler = Scaler()

    def _initialize_forecasting_model(self) -> RNNModel:
        return RNNModel(
            model="LSTM",
            input_chunk_length=self.window_size,
            output_chunk_length=self.horizon_length,
            n_epochs=self.num_epochs,
            dropout=self.dropout,
            hidden_dim=25,
            random_state=0,
            pl_trainer_kwargs={
                "precision": "32-true",
                "callbacks": [self.loss_tracker],
                "enable_model_summary": False,
                "log_every_n_steps": 1,
            },
        )

    def train(
        self, train_timeseries: np.ndarray, validation_timeseries: np.ndarray
    ) -> None:
        train_targets, train_covariates = array_to_timeseries(train_timeseries)
        val_targets, val_covariates = array_to_timeseries(validation_timeseries)

        train_targets_scaled = self.scaler.fit_transform(train_targets)
        val_targets_scaled = self.scaler.transform(val_targets)

        train_covariates_scaled = self.covariates_scaler.fit_transform(train_covariates)
        val_covariates_scaled = self.covariates_scaler.transform(val_covariates)

        self.model.fit(
            series=train_targets_scaled,
            past_covariates=train_covariates_scaled,
            val_series=val_targets_scaled,
            val_past_covariates=val_covariates_scaled,
        )

    def forecast(self, test_timeseries: np.ndarray) -> np.ndarray:
        test_targets, test_covariates = array_to_timeseries(test_timeseries)
        test_targets_scaled = self.scaler.transform(test_targets)
        test_covariates_scaled = self.covariates_scaler.transform(test_covariates)

        forecast_series = self.model.predict(
            n=self.horizon_length,
            series=test_targets_scaled,
            past_covariates=test_covariates_scaled,
        )
        forecast_series = self.scaler.inverse_transform(forecast_series)

        results: List = []
        for series in forecast_series:
            results.append(series.values().squeeze())
        return np.array(results)

    def plot_loss(self, model_name: str) -> None:
        if not self.loss_tracker.train_loss:
            logger.warning(
                "No training loss recorded. Did you forget to train the model?"
            )
            return

        plt.figure(figsize=(8, 5))
        plt.plot(self.loss_tracker.train_loss, label="Train Loss", color="orange")
        if self.loss_tracker.val_loss:
            plt.plot(self.loss_tracker.val_loss, label="Validation Loss", color="red")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"Training and Validation Loss - {model_name}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"Loss_{model_name}.png"), dpi=600)
