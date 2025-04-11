from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np


class ReconstructionModel(ABC):
    # FIXME: We do not actually need all these parameters.
    #   This is done only to make it plug and play with GeneticAlgorithmWrapper
    @abstractmethod
    def __init__(self, model_params: dict, config: dict) -> None:
        pass

    @abstractmethod
    def train(
        self,
        mts_dataset,
        X,
        y,
        plot_loss: bool = False,
        model_name: str = "unnamed_reconstruction_model",
    ) -> None:
        pass

    @abstractmethod
    def transform(
        self, predicted_features: np.ndarray, original_mts_indices: np.ndarray
    ) -> Tuple[List, List]:
        pass
