from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from src.models.feature_transformation_model import FeatureTransformationModel


class CorrelationModel(FeatureTransformationModel):
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
    ) -> Tuple[List[float], List[float]]:
        # The actual features (Omitting the delta values)
        features = X_train[
            :, : -(self.number_of_uts_in_mts + self.number_of_features_in_each_uts)
        ]

        # We then calcualte the correlation matrix based on features
        self.correlation_matrix = np.corrcoef(features, rowvar=False)
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

        predicted_features = np.zeros(
            (num_samples, num_features * num_uts)
        )  # Preallocate array

        for row_index in tqdm(range(num_samples), total=num_samples):
            delta_vector = delta_values[row_index]
            one_hot_vector = one_hot_encodings[row_index]
            feature_vector = features[row_index]
            activated_uts_index = np.argmax(one_hot_vector)

            # Extract relevant rows from correlation matrix
            relevant_rows = self.correlation_matrix[
                activated_uts_index
                * num_features : (activated_uts_index * num_features)
                + num_features
                + 1,
                :,
            ]

            # Compute predictions using vectorized operations
            tmp_delta_values = np.tile(delta_vector, num_uts)
            tmp_corr_values = [
                relevant_rows[i % num_features, j]
                for j, i in enumerate(range(num_features * num_uts))
            ]

            # Perform vectorized computation
            predicted_features[row_index] = (
                tmp_corr_values * tmp_delta_values
            ) + feature_vector

        return predicted_features
