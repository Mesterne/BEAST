from typing import Dict, List, Tuple, override

import numpy as np

from src.data_transformations.generation_of_supervised_pairs import (
    concat_delta_values_to_features,
)
from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import (
    CVAEWrapper,
    create_delta_conditioned_dataset,
    create_feature_conditioned_dataset,
    prepare_cgen_data,
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

        # TODO: Unsure if this function works as inteded now
        if condition_type == "feature":
            X_train, y_train = create_feature_conditioned_dataset(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                transformation_indices=train_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )
            X_validation, y_validation = create_feature_conditioned_dataset(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                transformation_indices=validation_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )
        elif condition_type == "feature_delta":
            X_train, y_train = create_delta_conditioned_dataset(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                transformation_indices=train_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )
            X_validation, y_validation = create_delta_conditioned_dataset(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                transformation_indices=validation_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )
        else:
            raise ValueError("Condition type can only be: feature or feature_delta")

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

    # TODO:
    @override
    def create_inference_data(
        self, mts_dataset: np.ndarray, evaluation_set_indices: np.ndarray
    ):
        return super().create_inference_data(evaluation_set_indices)

    # TODO:
    @override
    def infer(self, X: np.ndarray) -> np.ndarray:
        return super().infer(X)
