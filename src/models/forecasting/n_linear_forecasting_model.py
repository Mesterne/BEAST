from typing import Any, Dict, List

import numpy as np
from darts import concatenate
from darts.dataprocessing.transformers.scaler import Scaler
from darts.models.forecasting.nlinear import NLinearModel
from darts.timeseries import TimeSeries

from src.models.forecasting.forcasting_model import ForecastingModel
from src.utils.darts_utils import array_to_timeseries


class NLinearForecastingModel(ForecastingModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = self._initialize_forecasting_model()
        self.scaler = Scaler()
        self.covariates_scaler = Scaler()

    def _initialize_forecasting_model(self) -> NLinearModel:
        return NLinearModel(
            input_chunk_length=self.config["model_args"]["forecasting_model_args"][
                "window_size"
            ],
            output_chunk_length=self.config["model_args"]["forecasting_model_args"][
                "horizon_length"
            ],
            n_epochs=self.config["model_args"]["forecasting_model_args"][
                "training_args"
            ]["num_epochs"],
            random_state=0,
            pl_trainer_kwargs={
                "precision": "32-true"
            },  # Done to be able to run on laptop
        )

    def train(
        self, train_timeseries: np.ndarray, validation_timeseries: np.ndarray
    ) -> None:
        train_targets, train_covariates = array_to_timeseries(train_timeseries)
        val_targets, val_covariates = array_to_timeseries(validation_timeseries)

        # We scale the time series. As recommended in Darts documentation
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
            n=self.config["model_args"]["forecasting_model_args"]["horizon_length"],
            series=test_targets_scaled,
            past_covariates=test_covariates_scaled,
        )
        forecast_series = self.scaler.inverse_transform(forecast_series)

        results: List = []

        for series in forecast_series:
            results.append(series.values().squeeze())

        return np.array(results)
