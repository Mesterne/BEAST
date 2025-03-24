from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.models.feature_transformation_model import FeatureTransformationModel


class CovarianceModel(FeatureTransformationModel):
    def __init__(self, params):
        self.number_of_features_in_each_uts = params["number_of_features_per_uts"]
        self.number_of_uts_in_mts = params["number_of_uts_in_mts"]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb=False,
    ) -> Tuple[List[float], List[float]]:
        # The actual features (Omitting the delta values)
        features = X_train[
            :, : -(self.number_of_uts_in_mts + self.number_of_features_in_each_uts)
        ]

        # We then calcualte the correlation matrix based on features
        self.covariance_matrix = np.cov(features, rowvar=False)
        self.mean_vector = np.mean(features, axis=0)  # Compute feature means
        return [], []

    def infer(self, X: np.ndarray) -> np.ndarray:
        # Extract feature, delta, and one-hot encoding parts
        features = X[
            :, : -(self.number_of_uts_in_mts + self.number_of_features_in_each_uts)
        ]
        delta_values = X[
            :,
            self.number_of_features_in_each_uts
            * self.number_of_uts_in_mts : -self.number_of_uts_in_mts,
        ]
        one_hot_encodings = X[:, -self.number_of_uts_in_mts :]

        num_samples = features.shape[0]
        num_features = self.number_of_features_in_each_uts
        num_uts = self.number_of_uts_in_mts

        predicted_features = np.zeros((num_samples, num_features * num_uts))

        for row_index in tqdm(range(num_samples), total=num_samples):
            delta_vector = delta_values[row_index]
            one_hot_vector = one_hot_encodings[row_index]
            feature_vector = features[row_index]
            activated_uts_index = np.argmax(one_hot_vector)

            start_idx = activated_uts_index * num_features
            end_idx = start_idx + num_features

        return predicted_features
