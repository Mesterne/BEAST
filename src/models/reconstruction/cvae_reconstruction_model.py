from typing import List, Tuple, override

import numpy as np

from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import CVAEWrapper
from src.models.reconstruction.reconstruction_model import ReconstructionModel
from src.utils.features import decomp_and_features


class CVAEReconstructionModel(ReconstructionModel):
    @override
    def __init__(self, model_params: dict, config: dict) -> None:
        print(model_params)
        self.cvae = MTSCVAE(model_params=model_params)
        self.model = CVAEWrapper(
            model=self.cvae, training_params=model_params["training_args"]
        )
        self.config = config
        pass

    @override
    def train(
        self,
        mts_dataset,
        X,
        y,
        plot_loss: bool = False,
        model_name: str = "CVAEReconstructionModel",
    ) -> None:
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        seasonal_period: int = self.config["stl_args"]["series_periodicity"]

        self.mts_dataset = mts_dataset
        self.mts_decomp, self.mts_features = decomp_and_features(
            mts=self.mts_dataset,
            num_features_per_uts=num_features_per_uts,
            series_periodicity=seasonal_period,
            decomps_only=False,
        )

        self.model.train(
            X_train=X,
            y_train=y,
            X_val=X,
            y_val=y,
            plot_loss=plot_loss,
            model_name=model_name,
        )

    @override
    def transform(
        self, predicted_features: np.ndarray, original_mts_indices: np.ndarray
    ) -> Tuple[List, List]:
        X_mts = self.mts_dataset[original_mts_indices]
        X_mts = X_mts.reshape(-1, X_mts.shape[1] * X_mts.shape[2])
        delta_condition = predicted_features - self.mts_features[original_mts_indices]
        X = np.concatenate([X_mts, delta_condition], axis=1)
        inferred_mts = self.model.infer(X=X)
        return (inferred_mts, predicted_features)
