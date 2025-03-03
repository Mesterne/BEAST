from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np


class FeatureTransformationModel(ABC):
    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb=False,
    ) -> Tuple[List[float], List[float]]:
        pass

    @abstractmethod
    def infer(self, X: np.ndarray) -> np.ndarray:
        pass
