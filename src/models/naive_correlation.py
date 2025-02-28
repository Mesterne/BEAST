import pandas as pd
import numpy as np
from tqdm import tqdm
from src.models.feature_transformation_model import FeatureTransformationModel

from src.utils.logging_config import logger


class CorrelationModel(FeatureTransformationModel):
    def __init__(self, params):
        self.number_of_features_in_each_uts = params["number_of_features_in_each_uts"]
        self.number_of_uts_in_mts = params["number_of_uts_in_mts"]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb=False,
    ):
        # The actual features (Omitting the delta values)
        features = X_train[
            :, : -self.number_of_features_in_each_uts * self.number_of_uts_in_mts
        ]

        # We then calcualte the correlation matrix based on features
        self.correlation_matrix = np.corrcoef(features, rowvar=False)

    def infer(self, X: np.ndarray) -> np.ndarray:
        delta_values = X[
            :, -self.number_of_features_in_each_uts * self.number_of_uts_in_mts :
        ]
        features = X[
            :, : -self.number_of_features_in_each_uts * self.number_of_uts_in_mts
        ]

        logger.info(f"Delta values shape: {delta_values.shape}")
        logger.info(f"feature values shape: {features.shape}")
        logger.info(f"correlation matrix shape: {self.correlation_matrix.shape}")
        logger.info(f"correlation matrix: {self.correlation_matrix}")

        # Iterate over each row in the features. Selecting the largest delta value
        predicted_features = []
        for idx, row in tqdm(enumerate(features), total=features.shape[0]):
            # Find the index of the largest delta value
            delta_vector = delta_values[idx]
            max_delta_index = np.argmax(delta_vector)
            correlation_vector = self.correlation_matrix[max_delta_index]
            # Select the feature with the highest correlation
            predicted_feature = row + np.dot(correlation_vector, delta_vector)
            predicted_features.append(predicted_feature)

        return np.array(predicted_features)
