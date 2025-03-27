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
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.models.timeseries_transformation_model import TimeseriesTransformationModel
from src.utils.features import decomp_and_features
from src.utils.ga_utils import generate_new_time_series
from src.utils.generate_dataset import generate_feature_dataframe


class FeatureOptimizationModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, any]) -> None:
        self.config = config
        self.model = self._choose_underlying_model_based_on_config()

    def _choose_underlying_model_based_on_config(self) -> FeatureTransformationModel:
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
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def initialize_ga_model(self, mts_dataset: np.ndarray) -> GeneticAlgorithmWrapper:
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(self.config["dataset_args"]["timeseries_to_use"])
        seasonal_period: int = self.config["stl_args"]["series_periodicity"]
        mts_decomps, _ = decomp_and_features(
            mts=mts_dataset,
            num_features_per_uts=num_features_per_uts,
            series_periodicity=seasonal_period,
            decomps_only=True,
        )

        return GeneticAlgorithmWrapper(
            ga_params=self.config["model_args"]["genetic_algorithm_args"],
            mts_dataset=mts_dataset,
            mts_decomp=mts_decomps,
            num_uts_in_mts=num_uts_in_mts,
            num_features_per_uts=num_features_per_uts,
        )

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
        self.model.train(X_train=X_train, y_train=y_train, X_val=X_val, y_val=y_val)

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
        ga = self.initialize_ga_model(self.mts_dataset)
        predicted_features = self.model.infer(X)

        inferred_mts, _ = generate_new_time_series(
            original_indices=self.evaluation_set_indices[:, 0],
            predicted_features=predicted_features,
            ga=ga,
        )
        return inferred_mts
