from typing import List, Tuple

import numpy as np

from src.models.feature_transformation_model import FeatureTransformationModel


class PerfectFeatureModel(FeatureTransformationModel):
    def __init__(self, params):
        self.number_of_features_in_each_uts = params["number_of_features_per_uts"]
        self.number_of_uts_in_mts = params["number_of_uts_in_mts"]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        plot_loss=False,
        model_name="",
    ) -> Tuple[List[float], List[float]]:
        self.validation_targets = y_val
        return [], []

    # NOTE: This only works on validation set.
    def infer(self, X: np.ndarray) -> np.ndarray:
        if X.shape[0] != self.validation_targets.shape[0]:
            return X[
                :, : -(self.number_of_uts_in_mts + self.number_of_features_in_each_uts)
            ]
        else:
            return self.validation_targets
