from typing import Dict, override

import numpy as np
from torch import nn

from data.constants import NETWORK_ARCHITECTURES
from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import (
    CVAEWrapper,
    create_conditioned_dataset_for_inference,
    create_conditioned_dataset_for_training,
    create_ohe_conditioned_dataset_for_inference,
    create_ohe_conditioned_dataset_for_training,
)
from src.models.generative_models.rnn_cvae import RNNCVAE
from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from src.utils.generate_dataset import generate_feature_dataframe


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
        model_params["mts_size"] = config["dataset_args"]["mts_size"]
        model_params["uts_size"] = config["dataset_args"]["uts_size"]
        architecture: str = model_params["architecture"]
        model_params["mts_size"] = config["dataset_args"]["mts_size"]
        model_params["uts_size"] = config["dataset_args"]["uts_size"]
        cvae = self._select_cvae_architecture(architecture, model_params)
        model = CVAEWrapper(cvae, training_params=training_params)
        return model

    def _select_cvae_architecture(
        self, architecture: str, model_params: Dict[str, any]
    ) -> nn.Module:
        """
        Return CVAE with desired encoder-decoder architecture.
        """
        if architecture not in NETWORK_ARCHITECTURES:
            raise ValueError(
                f"Unknown architecture: {architecture}. Supported architectures are: {NETWORK_ARCHITECTURES}"
            )
        if architecture == NETWORK_ARCHITECTURES[2]:
            return RNNCVAE(model_params=model_params)
        return MTSCVAE(model_params=model_params)

    @override
    def create_training_data(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):

        condition_type: str = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["condition_type"]
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        use_one_hot_encoding: int = self.config["dataset_args"]["use_one_hot_encoding"]

        input_size_without_conditions = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["input_size_without_conditions"]
        number_of_conditions = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["number_of_conditions"]
        mts_size = self.config["dataset_args"]["mts_size"]
        mts_features_array, _ = generate_feature_dataframe(
            data=mts_dataset,
            series_periodicity=self.config["stl_args"]["series_periodicity"],
            num_features_per_uts=self.config["dataset_args"]["num_features_per_uts"],
        )

        if use_one_hot_encoding:
            X_train, y_train = create_ohe_conditioned_dataset_for_training(
                mts_array=mts_dataset,
                transformation_indices=train_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                mts_features=mts_features_array,
            )
            X_validation, y_validation = create_ohe_conditioned_dataset_for_inference(
                mts_array=mts_dataset,
                transformation_indices=train_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                mts_features=mts_features_array,
            )
        else:
            X_train, y_train = create_conditioned_dataset_for_training(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                condition_type=condition_type,
                transformation_indices=train_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )
            X_validation, y_validation = create_conditioned_dataset_for_inference(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                condition_type=condition_type,
                transformation_indices=validation_transformation_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )

        X_train = X_train[:, -(input_size_without_conditions + number_of_conditions) :]
        y_train = y_train[:, -mts_size:]
        X_validation = X_validation[
            :, -(input_size_without_conditions + number_of_conditions) :
        ]
        y_validation = y_validation[:, -mts_size:]

        return X_train, y_train, X_validation, y_validation

    @override
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        plot_loss=True,
    ):
        self.model.train(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            plot_loss=plot_loss,
            model_name="CVAE_generative_model",
        )

    @override
    def create_inference_data(
        self, mts_dataset: np.ndarray, evaluation_set_indices: np.ndarray
    ):
        condition_type: str = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["condition_type"]
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        use_one_hot_encoding: int = self.config["dataset_args"]["use_one_hot_encoding"]

        input_size_without_conditions = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["input_size_without_conditions"]
        number_of_conditions = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["number_of_conditions"]

        mts_features_array, _ = generate_feature_dataframe(
            data=mts_dataset,
            series_periodicity=self.config["stl_args"]["series_periodicity"],
            num_features_per_uts=self.config["dataset_args"]["num_features_per_uts"],
        )

        if use_one_hot_encoding:
            X, y = create_ohe_conditioned_dataset_for_inference(
                mts_array=mts_dataset,
                transformation_indices=evaluation_set_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                mts_features=mts_features_array,
            )
        else:
            X, y = create_conditioned_dataset_for_inference(
                mts_array=mts_dataset,
                mts_features=mts_features_array,
                condition_type=condition_type,
                transformation_indices=evaluation_set_indices,
                number_of_uts_in_mts=num_uts_in_mts,
                number_of_features_in_mts=num_features_per_uts,
            )
        X = X[:, -(input_size_without_conditions + number_of_conditions) :]

        return X

    @override
    def infer(self, X: np.ndarray) -> np.ndarray:
        predicted_mts = self.model.infer(X)

        # Reshape to have shape (Number of MTS predicted, Number of UTS in MTS, Number of samples in UTS)
        num_samples_in_uts: int = self.config["dataset_args"]["window_size"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        predicted_mts = predicted_mts.reshape(-1, num_uts_in_mts, num_samples_in_uts)

        return predicted_mts, None
