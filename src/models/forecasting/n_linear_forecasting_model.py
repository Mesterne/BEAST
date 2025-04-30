from typing import Any, Dict, List

import numpy as np
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
        train_series: List[TimeSeries] = array_to_timeseries(train_timeseries)
        val_series: List[TimeSeries] = array_to_timeseries(validation_timeseries)

        # We scale the time series. As recommended in Darts documentation
        train_series_scaled: List[TimeSeries] = self.scaler.fit_transform(train_series)
        val_series_scaled: List[TimeSeries] = self.scaler.transform(val_series)

        self.model.fit(series=train_series_scaled, val_series=val_series_scaled)

    def forecast(self, test_timeseries: np.ndarray) -> np.ndarray:
        test_series: List[TimeSeries] = array_to_timeseries(test_timeseries)
        test_series_scaled: List[TimeSeries] = self.scaler.transform(test_series)

        forecast_series = self.model.predict(
            n=self.config["model_args"]["forecasting_model_args"]["horizon_length"],
            series=test_series_scaled,
        )
        forecast_series: List[TimeSeries] = self.scaler.inverse_transform(
            forecast_series
        )

        results = []

        for series in forecast_series:
            results.append(series.values()[:, 1])

        return results
