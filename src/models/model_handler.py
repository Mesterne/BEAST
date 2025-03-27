from typing import Dict, Optional, Tuple

import numpy as np

from src.models.feature_optimization_models.feature_optimization_model import (
    FeatureOptimizationModel,
)
from src.models.generative_models.generative_model import GenerativeModel
from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from src.utils.logging_config import logger


class ModelHandler:
    def __init__(self, config: Dict[str, any]) -> None:
        """
        Initializes the model handler with the correct variables.

        Args:
            config: The configuration dict for the experiment
        """
        self.config = config
        self.model: TimeseriesTransformationModel = None

    def choose_model_category(self):
        """
        Based on the config file, chooses and sets the internal model to either GenerativeModel or FeatureOptimizationModel.
        """
        if self.config["is_conditional_gen_model"]:
            self.model = GenerativeModel(self.config)
            logger.info("Running conditional generative model")
        else:
            self.model = FeatureOptimizationModel(self.config)
            logger.info("Running feature+optimization model")

    def train(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):
        """
        Takes the entire dataset, creates the correct training data for the selected model and trains the model.

        Args:
            mts_dataset: The entire dataset. Has shape (number of MTS in dataset, number of UTS in MTS, Number of samples per UTS)
            train_transformation_indices: The transformation indices for the training set. Is a list of tuples where the first
                element is the original MTS, the second element is the target MTS index.
            validation_transformation_indices: The transformation indices for the validation set. Is a list of tuples where the first
                element is the original MTS, the second element is the target MTS index.
        """
        assert self.model is not None, "The model to train must be defined"

        logger.info("Generating training and validation pairs...")
        X_train, y_train, X_val, y_val = self.model.create_training_data(
            mts_dataset=mts_dataset,
            train_transformation_indices=train_transformation_indices,
            validation_transformation_indices=validation_transformation_indices,
        )
        logger.info(
            f"Successfully generated training and validation pairs.\nShapes:\nX_train: {X_train.shape}\ny_train: {y_train.shape}\nX_validation: {X_val.shape}\ny_validation: {y_val.shape}"
        )
        logger.info("Training model...")
        self.model.train(X_train, y_train, X_val, y_val)
        logger.info("Successfully trained model...")

    def infer(
        self, mts_dataset, evaluation_transformation_indinces: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Takes the entire dataset, creates the correct inference data (X) for the selected model and runs inference.

        Args:
            mts_dataset: The entire dataset. Has shape (number of MTS in dataset, number of UTS in MTS, Number of samples per UTS)
            evaluation_transformation_indices: The transformation indices for the evaluation set. Is a list of tuples where the first
                element is the original MTS, the second element is the target MTS index.
        Returns:
            A numpy array of shape (Number of MTS inferred, Number of samples per UTS * Number of UTS in MTS) containing all inferred MTS
        """
        assert self.model is not None, "The model must be defined"

        logger.info("Generating inference data...")
        X = self.model.create_inference_data(
            mts_dataset=mts_dataset,
            evaluation_set_indices=evaluation_transformation_indinces,
        )
        logger.info(f"Successfully generated inference data with shape X: {X.shape}")

        logger.info("Running inference...")
        predicted_y, intermediate_features = self.model.infer(X)

        return predicted_y, np.array(intermediate_features)
