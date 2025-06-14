import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
from darts.dataprocessing.transformers import Scaler
from darts.models.forecasting.nlinear import NLinearModel
from pytorch_lightning.callbacks import EarlyStopping

from src.data.constants import OUTPUT_DIR
from src.models.forecasting.forcasting_model import ForecastingModel
from src.models.forecasting.loss_tracker import LossTracker
from src.utils.darts_utils import array_to_timeseries
from src.utils.logging_config import logger


class NLinearForecastingModel(ForecastingModel):
    def __init__(
        self, window_size, horizon_length, num_epochs, early_stopping_patience
    ) -> None:
        self.window_size = window_size
        self.horizon_length = horizon_length
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
        self.loss_tracker = LossTracker()
        self.target_scaler = Scaler()
        self.covariate_scaler = Scaler()
        self.model = self._initialize_forecasting_model()

    def _initialize_forecasting_model(self) -> NLinearModel:
        early_stopping = EarlyStopping(
            monitor="val_loss",
            patience=self.early_stopping_patience,
            min_delta=0.0001,
            mode="min",
        )

        return NLinearModel(
            input_chunk_length=self.window_size,
            output_chunk_length=self.horizon_length,
            n_epochs=self.num_epochs,
            random_state=0,
            loss_fn=torch.nn.L1Loss(),
            pl_trainer_kwargs={
                "precision": "32-true",
                "callbacks": [self.loss_tracker, early_stopping],
                "enable_model_summary": False,
                "log_every_n_steps": 1,
            },  # Done to be able to run on laptop
        )

    def train(
        self, train_timeseries: np.ndarray, validation_timeseries: np.ndarray
    ) -> None:
        train_targets, train_covariates = array_to_timeseries(train_timeseries)
        val_targets, val_covariates = array_to_timeseries(validation_timeseries)

        self.target_scaler.fit(train_targets)
        self.covariate_scaler.fit(train_covariates)

        scaled_train_targets = self.target_scaler.transform(train_targets)
        scaled_val_targets = self.target_scaler.transform(val_targets)

        scaled_train_covariates = self.covariate_scaler.transform(train_covariates)
        scaled_val_covariates = self.covariate_scaler.transform(val_covariates)

        self.model.fit(
            series=scaled_train_targets,
            past_covariates=scaled_train_covariates,
            val_series=scaled_val_targets,
            val_past_covariates=scaled_val_covariates,
        )

    def forecast(self, test_timeseries: np.ndarray) -> np.ndarray:
        test_targets, test_covariates = array_to_timeseries(test_timeseries)

        scaled_test_targets = self.target_scaler.transform(test_targets)
        scaled_test_covariates = self.covariate_scaler.transform(test_covariates)

        forecast_series = self.model.predict(
            n=self.horizon_length,
            series=scaled_test_targets,
            past_covariates=scaled_test_covariates,
        )

        forecast_series = self.target_scaler.inverse_transform(forecast_series)

        results = []
        for series in forecast_series:
            results.append(series.values().squeeze())
        return np.array(results)

    def plot_loss(self, model_name: str) -> None:
        """
        Plots the training and validation loss stored in the LossTracker.
        """
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
