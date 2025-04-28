from typing import Any, Dict, List, Tuple

import numpy as np

from src.models.basic_models.horizontal_flip import HorizontalFlip
from src.models.basic_models.jitter import Jitter
from src.models.basic_models.scaler import Scaler
from src.models.basic_models.vertical_flip import VerticalFlip
from src.models.timeseries_transformation_model import \
    TimeseriesTransformationModel


class BasicModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model = self._choose_underlying_model_based_on_config()

    def _choose_underlying_model_based_on_config(self):
        model_name = self.config["model_args"]["model_name"]
        if model_name == "scaler":
            return Scaler()
        elif model_name == "jitter":
            return Jitter()
        elif model_name == "vertical_flip":
            return VerticalFlip()
        elif model_name == "horizontal_flip":
            return HorizontalFlip()
        return Scaler()

    def create_training_data(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):

        return np.array([]), np.array([]), np.array([]), np.array([])

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb: bool = False,
    ) -> Tuple[List[float], List[float]]:
        return ([0], [0])

    def create_inference_data(
        self, mts_dataset: np.ndarray, evaluation_set_indices: np.ndarray
    ):
        X = mts_dataset[evaluation_set_indices[:, 0]]
        return X

    def infer(self, X: np.ndarray) -> np.ndarray:
        transformed = self.model.transform(X)
        return transformed, None
