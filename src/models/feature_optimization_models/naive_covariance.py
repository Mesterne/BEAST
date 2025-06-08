from tracemalloc import start
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
        plot_loss=False,
        model_name="unnamed",
    ) -> Tuple[List[float], List[float]]:
        features = X_train[
            :, : -(self.number_of_uts_in_mts + self.number_of_features_in_each_uts)
        ]

        self.covariance_matrix = np.cov(features, rowvar=False)
        self.mean_vector = np.mean(features, axis=0)  # Compute feature means
        return [], []

    def infer(self, X: np.ndarray) -> np.ndarray:
        """
        Uses the covariance_matrix, mean vector and input X to infer new featues using gaussian updates.
        """
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
            known_features = feature_vector[start_idx:end_idx] + delta_vector

            # 1 = Unknown
            # 2 = Known
            mu_2 = self.mean_vector[start_idx:end_idx]
            mu_1 = np.concatenate(
                [self.mean_vector[:start_idx], self.mean_vector[end_idx:]]
            )

            # Extract parts of covariance matrix used for later calculations
            # Each matrix essential extracts its own parts of the covariance matrix.
            # These are then used to calculated conditional mean and covariances. See pg. 84 of Murphy et. al. for complete formula and derivations
            sigma_22 = self.covariance_matrix[start_idx:end_idx, start_idx:end_idx]
            sigma_12 = np.vstack(
                [
                    self.covariance_matrix[:start_idx, start_idx:end_idx],
                    self.covariance_matrix[end_idx:, start_idx:end_idx],
                ]
            )
            sigma_21 = np.hstack(
                [
                    self.covariance_matrix[start_idx:end_idx, :start_idx],
                    self.covariance_matrix[start_idx:end_idx, end_idx:],
                ]
            )
            sigma_11 = np.block(
                [
                    [
                        self.covariance_matrix[:start_idx, :start_idx],
                        self.covariance_matrix[:start_idx, end_idx:],
                    ],
                    [
                        self.covariance_matrix[end_idx:, :start_idx],
                        self.covariance_matrix[end_idx:, end_idx:],
                    ],
                ]
            )

            sigma_22_inv = np.linalg.pinv(sigma_22)  # Inverse of known covariance
            mean_1_given_2 = mu_1 + sigma_12 @ sigma_22_inv @ (known_features - mu_2)
            sigma_1_given_2 = sigma_11 - sigma_12 @ sigma_22_inv @ sigma_21

            # Sample from multivariate normal distribution, using the conditional mean and covariance_matrix calculated earlier
            sampled_values = np.random.multivariate_normal(
                mean_1_given_2, sigma_1_given_2
            )

            predicted_features[row_index, :start_idx] = sampled_values[:start_idx]
            predicted_features[row_index, start_idx:end_idx] = known_features
            predicted_features[row_index, end_idx:] = sampled_values[start_idx:]

        return predicted_features
