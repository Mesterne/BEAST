from typing import Dict, List, Tuple, override

import numpy as np

from src.data_transformations.generation_of_supervised_pairs import (
    concat_delta_values_to_features,
    concat_delta_values_to_features_for_inference,
)
from src.models.feature_optimization_models.feedforward import FeedForwardFeatureModel
from src.models.feature_optimization_models.naive_correlation import CorrelationModel
from src.models.feature_optimization_models.naive_covariance import CovarianceModel
from src.models.feature_optimization_models.perfect_feature_model import (
    PerfectFeatureModel,
)
from src.models.feature_transformation_model import FeatureTransformationModel
from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import CVAEWrapper
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.models.reconstruction.cvae_reconstruction_model import CVAEReconstructionModel
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.models.reconstruction.reconstruction_model import ReconstructionModel
from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from src.utils.ga_utils import generate_new_time_series
from src.utils.generate_dataset import generate_feature_dataframe
from src.utils.logging_config import logger


class FeatureOptimizationModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, any]) -> None:
        self.config = config
        self.feature_model = self._choose_underlying_feature_model_based_on_config()
        self.reconstruction_model = (
            self._choose_underlying_reconstruction_model_based_on_config()
        )

    def _choose_underlying_feature_model_based_on_config(
        self,
    ) -> FeatureTransformationModel:
        feature_model_params: Dict[str, Any] = self.config["model_args"][
            "feature_model_args"
        ]
        training_params: Dict[str, Any] = feature_model_params["training_args"]
        model_type: str = feature_model_params["model_name"]
        if model_type == "correlation_model":
            return CorrelationModel(feature_model_params)
        elif model_type == "perfect_feature_model":
            return PerfectFeatureModel(params=feature_model_params)
        elif model_type == "covariance_model":
            return CovarianceModel(params=feature_model_params)
        elif model_type == "feedforward_neural_network":
            nn = FeedForwardFeatureModel(feature_model_params)
            model = NeuralNetworkWrapper(nn, training_params=training_params)
            return model
        elif model_type == "feature_cvae":
            cvae_params = feature_model_params["conditional_gen_model_args"]
            cvae = MTSCVAE(model_params=cvae_params)
            model = CVAEWrapper(cvae, training_params=training_params)
            return model
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def _choose_underlying_reconstruction_model_based_on_config(
        self,
    ) -> ReconstructionModel:
        reconstruction_model_params: Dict[str, Any] = self.config["model_args"][
            "reconstruction_model_args"
        ]

        model_type: str = reconstruction_model_params["model_type"]

        if model_type == "cvae":
            model = CVAEReconstructionModel(
                model_params=self.config["model_args"]["reconstruction_model_args"],
                config=self.config,
            )
            return model
        elif model_type == "ga":
            model = GeneticAlgorithmWrapper(
                model_params=self.config["model_args"]["reconstruction_model_args"],
                config=self.config,
            )
            return model

    @override
    def create_training_data(
        self,
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        use_one_hot_encoding: int = self.config["dataset_args"]["use_one_hot_encoding"]
        self.mts_dataset = mts_dataset

        mts_features_array, _ = generate_feature_dataframe(
            data=mts_dataset,
            series_periodicity=self.config["stl_args"]["series_periodicity"],
            num_features_per_uts=self.config["dataset_args"]["num_features_per_uts"],
        )
        X_train: np.ndarray = mts_features_array[train_transformation_indices[:, 0]]
        y_train: np.ndarray = mts_features_array[train_transformation_indices[:, 1]]
        X_validation: np.ndarray = mts_features_array[
            validation_transformation_indices[:, 0]
        ]
        y_validation: np.ndarray = mts_features_array[
            validation_transformation_indices[:, 1]
        ]

        delta_values = (
            mts_features_array[train_transformation_indices[:, 1]]
            - mts_features_array[train_transformation_indices[:, 0]]
        )
        X_mts = mts_dataset[train_transformation_indices[:, 0]]
        X_mts = X_mts.reshape(-1, X_mts.shape[1] * X_mts.shape[2])
        self.X_reconstruction = np.concatenate([X_mts, delta_values], axis=1)
        self.y_reconstruction = mts_dataset[train_transformation_indices[:, 1]]
        # We reshape to get shape (Number of samples, Number of samples per MTS flattened)
        self.y_reconstruction = self.y_reconstruction.reshape(
            -1, self.y_reconstruction.shape[1] * self.y_reconstruction.shape[2]
        )
        logger.info(f"X_reconstruction shape: {self.X_reconstruction.shape}")
        logger.info(f"y_reconstruction shape: {self.y_reconstruction.shape}")

        X_train, y_train = concat_delta_values_to_features(
            X_train,
            y_train,
            use_one_hot_encoding=use_one_hot_encoding,
            number_of_features_in_mts=num_features_per_uts,
            number_of_uts_in_mts=num_uts_in_mts,
        )
        X_validation, y_validation = concat_delta_values_to_features(
            X_validation,
            y_validation,
            use_one_hot_encoding=use_one_hot_encoding,
            number_of_features_in_mts=num_features_per_uts,
            number_of_uts_in_mts=num_uts_in_mts,
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
    ) -> Tuple[List[float], List[float]]:
        logger.info("Training feature model...")
        self.feature_model.train(
            X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val
        )
        logger.info("Training reconstruction model...")
        self.reconstruction_model.train(
            mts_dataset=self.mts_dataset,
            X=self.X_reconstruction,
            y=self.y_reconstruction,
        )

    @override
    def create_inference_data(
        self, mts_dataset: np.ndarray, evaluation_set_indices: np.ndarray
    ):
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        use_one_hot_encoding: int = self.config["dataset_args"]["use_one_hot_encoding"]

        self.evaluation_set_indices = evaluation_set_indices
        self.mts_dataset = mts_dataset

        mts_features_array, _ = generate_feature_dataframe(
            data=mts_dataset,
            series_periodicity=self.config["stl_args"]["series_periodicity"],
            num_features_per_uts=self.config["dataset_args"]["num_features_per_uts"],
        )
        X: np.ndarray = mts_features_array[evaluation_set_indices[:, 0]]
        y: np.ndarray = mts_features_array[evaluation_set_indices[:, 1]]

        X, y = concat_delta_values_to_features_for_inference(
            X,
            y,
            use_one_hot_encoding=use_one_hot_encoding,
            number_of_features_in_mts=num_features_per_uts,
            number_of_uts_in_mts=num_uts_in_mts,
        )
        return X

    @override
    def infer(self, X: np.ndarray) -> np.ndarray:
        predicted_features = self.feature_model.infer(X)

        inferred_mts, intermediate_features = generate_new_time_series(
            original_indices=self.evaluation_set_indices[:, 0],
            predicted_features=predicted_features,
            reconstruction_model=self.reconstruction_model,
        )
        return inferred_mts, intermediate_features
