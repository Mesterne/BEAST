from typing import List, Tuple, override

import numpy as np

from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import CVAEWrapper
from src.models.reconstruction.reconstruction_model import ReconstructionModel


class CVAEReconstructionModel(ReconstructionModel):
    @override
    def __init__(self, model_params: dict, config: dict) -> None:
        self.cvae = MTSCVAE(model_params=model_params)
        self.model = CVAEWrapper(
            model=self.cvae, training_params=model_params["training_args"]
        )
        pass

    @override
    def train(self, mts_dataset, X, y) -> None:
        self.model.train(X_train=X, y_train=y, X_val=X, y_val=y, plot_loss=False)

    @override
    def transform(
        self, predicted_features: np.ndarray, original_mts_indices: np.ndarray
    ) -> Tuple[List, List]:
        inferred_mts = self.model.infer(X=predicted_features)
        return (inferred_mts, predicted_features)
