from typing import Dict, override

import numpy as np

from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import (
    CVAEWrapper,
    create_conditioned_dataset_for_inference,
    create_conditioned_dataset_for_training,
)
from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from src.utils.generate_dataset import generate_feature_dataframe
from src.utils.logging_config import logger


class GenerativeModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, any]) -> None:
        self.config = config
        self.model = self._choose_underlying_model_based_on_config(config)

    def _choose_underlying_model_based_on_config(self, config: Dict[str, any]):
        model_params: Dict[str, any] = config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]
        training_params: Dict[str, any] = config["model_args"]["feature_model_args"][
            "training_args"
        ]
        cvae = MTSCVAE(model_params=model_params)
        model = CVAEWrapper(cvae, training_params=training_params)
        return model

    @override
    def create_training_data(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):

        logger.info("Preparing data set for conditional generative model...")
        condition_type: str = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["condition_type"]
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        use_one_hot_encoding: int = self.config["dataset_args"]["use_one_hot_encoding"]

        mts_features_array, _ = generate_feature_dataframe(
            data=mts_dataset,
            series_periodicity=self.config["stl_args"]["series_periodicity"],
            num_features_per_uts=self.config["dataset_args"]["num_features_per_uts"],
        )

        X_train, y_train = create_conditioned_dataset_for_training(
            mts_array=mts_dataset,
            mts_features=mts_features_array,
            condition_type=condition_type,
            transformation_indices=train_transformation_indices,
            number_of_uts_in_mts=num_uts_in_mts,
            number_of_features_in_mts=num_features_per_uts,
        )
        X_validation, y_validation = create_conditioned_dataset_for_training(
            mts_array=mts_dataset,
            mts_features=mts_features_array,
            condition_type=condition_type,
            transformation_indices=validation_transformation_indices,
            number_of_uts_in_mts=num_uts_in_mts,
            number_of_features_in_mts=num_features_per_uts,
        )

        return X_train, y_train, X_validation, y_validation

    @override
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb=False,
    ):
        self.model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            log_to_wandb=False,
        )

    @override
    def create_inference_data(
        self, mts_dataset: np.ndarray, evaluation_set_indices: np.ndarray
    ):
        logger.info("Preparing data set for conditional generative model...")
        condition_type: str = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["condition_type"]
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        use_one_hot_encoding: int = self.config["dataset_args"]["use_one_hot_encoding"]

        mts_features_array, _ = generate_feature_dataframe(
            data=mts_dataset,
            series_periodicity=self.config["stl_args"]["series_periodicity"],
            num_features_per_uts=self.config["dataset_args"]["num_features_per_uts"],
        )

        X, y = create_conditioned_dataset_for_inference(
            mts_array=mts_dataset,
            mts_features=mts_features_array,
            condition_type=condition_type,
            transformation_indices=evaluation_set_indices,
            number_of_uts_in_mts=num_uts_in_mts,
            number_of_features_in_mts=num_features_per_uts,
        )

        return X

    @override
    def infer(self, X: np.ndarray) -> np.ndarray:
        predicted_mts = self.model.infer(X)

        # Reshape to have shape (Number of MTS predicted, Number of UTS in MTS, Number of samples in UTS)
        num_samples_in_uts: int = self.config["dataset_args"]["window_size"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        predicted_mts = predicted_mts.reshape(-1, num_uts_in_mts, num_samples_in_uts)

        return predicted_mts
