from typing import Dict

import numpy as np

from src.models.feature_optimization_models.feature_optimization_model import (
    FeatureOptimizationModel,
)
from src.models.generative_models.generative_model import GenerativeModel
from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from src.utils.logging_config import logger


class ModelHandler:
    def __init__(self, config: Dict[str, any]) -> None:
        self.config = config
        self.model: TimeseriesTransformationModel = None

    def choose_model_category(self):
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
        assert self.model is not None, "The model to train must be defined"

        logger.info("Generating training and validation pairs...")
        X_train, y_train, X_val, y_val = self.model.create_training_data(
            mts_dataset=mts_dataset,
            train_transformation_indices=train_transformation_indices,
            validation_transformation_indices=validation_transformation_indices,
        )
        logger.info("Successfully generated training and validation pairs")
        logger.info("Training model...")
        self.model.train(X_train, y_train, X_val, y_val)
        logger.info("Successfully trained model...")

    def infer(self, mts_dataset, evaluation_set_indinces: np.ndarray) -> np.ndarray:
        assert self.model is not None, "The model must be defined"

        logger.info("Generating inference data...")
        X = self.model.create_inference_data(
            mts_dataset=mts_dataset,
            evaluation_set_indices=evaluation_set_indinces,
        )

        print(X.shape)

        logger.info("Running inference...")
        predicted_y = self.model.infer(X)

        return predicted_y
