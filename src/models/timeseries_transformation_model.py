from abc import ABC, abstractmethod
from typing import Any, Dict, List, Tuple

import numpy as np


class TimeseriesTransformationModel(ABC):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.model = None

    @abstractmethod
    def create_training_data(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):
        """
        Generates training and validation data from a given multivariate time series (MTS) dataset.

        Args:
            mts_dataset (np.ndarray): A dataset containing multiple MTS samples.
                Shape: (num_mts, num_univariate_series_per_mts, num_samples_per_series).
            train_transformation_indices (np.ndarray): An array of index pairs indicating
                the source and target MTS for training transformations. Shape: (num_transformations, 2).
            validation_transformation_indices (np.ndarray): An array of index pairs indicating
                the source and target MTS for validation transformations. Shape: (num_transformations, 2).

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
                - X_train (input features for training)
                - y_train (target labels for training)
                - X_val (input features for validation)
                - y_val (target labels for validation)
        """
        pass

    @abstractmethod
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb: bool = False,
    ) -> Tuple[List[float], List[float]]:
        """
        Trains the model using the provided training data and evaluates it on validation data.

        Args:
            X_train (np.ndarray): Input features for training.
            y_train (np.ndarray): Target values for training.
            X_val (np.ndarray): Input features for validation.
            y_val (np.ndarray): Target values for validation.
            log_to_wandb (bool, optional): If True, logs training progress to Weights & Biases. Default is False.

        Returns:
            Tuple[List[float], List[float]]:
                - Training loss history (list of loss values per epoch).
                - Validation loss history (list of loss values per epoch).
        """
        pass

    @abstractmethod
    def create_inference_data(
        self,
        mts_dataset: np.ndarray,
        evaluation_set_indices: np.ndarray,
    ):
        """
        Prepares the input data for inference based on a given dataset.

        Args:
            mts_dataset (np.ndarray): A dataset containing multiple MTS samples.
                Shape: (num_mts, num_univariate_series_per_mts, num_samples_per_series).
            evaluation_set_indices (np.ndarray): An array of index pairs specifying
                the source and target MTS for inference transformations. Shape: (num_transformations, 2).

        Returns:
            np.ndarray: The input features (X) prepared for inference.
        """
        pass

    @abstractmethod
    def infer(self, X: np.ndarray) -> np.ndarray:
        """
        Performs inference using the trained model to generate predicted time series.

        Args:
            X (np.ndarray): The input data for inference.

        Returns:
            np.ndarray: The predicted time series output.
        """
        pass
