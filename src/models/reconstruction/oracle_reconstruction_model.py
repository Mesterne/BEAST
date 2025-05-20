from typing import List, Tuple, override

import numpy as np

from src.models.reconstruction.reconstruction_model import ReconstructionModel


class OracleReconstructionModel(ReconstructionModel):
    @override
    def __init__(self, model_params: dict, config: dict):
        self.model_params = model_params
        self.config = config
        self.trained = False

    @override
    def train(
        self,
        mts_dataset,
        X_train,
        y_train,
        X_val,
        y_val,
        plot_loss: bool = False,
        model_name: str = "oracle reconstruction_model",
    ) -> None:
        self.mts_dataset = mts_dataset
        self.trained = True

    @override
    def transform(
        self,
        predicted_features: np.ndarray,
        original_mts_indices: np.ndarray,
    ) -> Tuple[List, List]:
        assert (
            self.trained
        ), "Reconstruction model needs to be trained before transformation"
        predicted_features = predicted_features.copy()

        all_mts_transformed = []
        all_mts_transformed_features = []

        return (
            all_mts_transformed,
            all_mts_transformed_features,
        )
