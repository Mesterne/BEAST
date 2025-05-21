import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from darts.dataprocessing.transformers.scaler import Scaler
from darts.models.forecasting.tcn_model import TCNModel

from src.data.constants import OUTPUT_DIR
from src.models.forecasting.forcasting_model import ForecastingModel
from src.models.forecasting.loss_tracker import LossTracker
from src.utils.darts_utils import array_to_timeseries
from src.utils.logging_config import logger


class TCNForecastingModel(ForecastingModel):
    def __init__(self, window_size, horizon_length, num_epochs, dropout) -> None:
        self.window_size = window_size
        self.horizon_length = horizon_length
        self.num_epochs = num_epochs
        self.dropout = dropout
        self.loss_tracker = LossTracker()
        self.model = self._initialize_forecasting_model()
        self.scaler = Scaler()
        self.covariates_scaler = Scaler()

    def _initialize_forecasting_model(self) -> TCNModel:  # <- Changed model class
        return TCNModel(
            input_chunk_length=self.window_size,
            output_chunk_length=self.horizon_length,
            n_epochs=self.num_epochs,
            dropout=self.dropout,  # You can tune this
            kernel_size=3,
            dilation_base=2,
            num_filters=3,
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

        self.model.fit(
            series=train_targets,
            past_covariates=train_covariates,
            val_series=val_targets,
            val_past_covariates=val_covariates,
        )

    def forecast(self, test_timeseries: np.ndarray) -> np.ndarray:
        test_targets, test_covariates = array_to_timeseries(test_timeseries)

        forecast_series = self.model.predict(
            n=self.horizon_length,
            series=test_targets,
            past_covariates=test_covariates,
        )

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
