from abc import ABC, abstractmethod
from typing import Any, Dict

import numpy as np


class ForecastingModel(ABC):
    @abstractmethod
    def __init__(self, config: Dict[str, Any]) -> None:
        pass

    @abstractmethod
    def train(self, train_timeseries: np.ndarray, validation_timeseries: np.ndarray):
        pass

    @abstractmethod
    def plot_loss(self, model_name: str):
        pass

    @abstractmethod
    def forecast(self, test_timeseries: np.ndarray) -> np.ndarray:
        pass
