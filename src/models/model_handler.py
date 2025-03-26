import numpy as np

from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from wandb import config


class ModelHandler:
    def __init__(self, config: Dict[str, Any]) -> None:
        self.config = config
        self.model: TimeseriesTransformationModel = None

    def choose_model_category(self):
        if config["is_conditional_gen_model"]:
            self.model = GenerativeModel(self.config)
        else:
            self.model = FeatureOptimizationModel(self.config)

    def train(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):
        assert self.model is not None, "The model to train must be defined"

        X_train, y_train, X_val, y_val = self.model.create_training_data(
            mts_dataset=mts_dataset,
            train_transformation_indices=train_transformation_indices,
            validation_transformation_indices=validation_transformation_indices,
        )
        self.model.train(X_train, y_train, X_val, y_val)

    def infer(self, evaluation_set_indinces: np.ndarray) -> np.ndarray:
        assert self.model is not None, "The model must be defined"

        X = self.model.create_inference_data(
            mts_dataset=mts_dataset,
            evaluation_set_indices=evaluation_set_indinces,
        )

        predicted_y = self.model.infer(X)

        return predicted_y
