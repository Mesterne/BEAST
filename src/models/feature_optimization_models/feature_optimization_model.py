from typing import override

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


class FeatureOptimizationModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.model = self._choose_underlying_model_based_on_config()
        self.config = config

    def _choose_underlying_model_based_on_config(self) -> FeatureTransformationModel:
        feature_model_params: Dict[str, Any] = self.config["model_args"][
            "feature_model_args"
        ]
        training_params: Dict[str, Any] = feature_model_params["training_args"]
        model_type: str = feature_model_params["model_name"]
        if model_type == "correlation_model":
            return CorrelationModel(model_params)
        elif model_type == "perfect_feature_model":
            return PerfectFeatureModel(params=model_params)
        elif model_type == "covariance_model":
            return CovarianceModel(params=model_params)
        elif model_type == "feedforward_neural_network":
            nn = FeedForwardFeatureModel(model_params)
            model = NeuralNetworkWrapper(nn, training_params=training_params)
            return model
        else:
            raise ValueError(f"Model type {model_type} not supported")

    def initialize_ga_model(mts_dataset: np.ndarray) -> GeneticAlgorithmWrapper:
        num_features_per_uts: int = self.config["dataset_args"]["num_features_per_uts"]
        num_uts_in_mts: int = len(["dataset_args"]["timeseries_to_use"])
        seasonal_period: int = self.config["stl_args"]["series_periodicity"]
        mts_decomps = decomp_and_features(
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

    # TODO:
    @override
    def create_training_data(
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):
        return super().create_training_data(
            train_transformation_indices, validation_transformation_indices
        )

    # TODO:
    @override
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb=False,
    ) -> Tuple[List[float], List[float]]:
        return super().train(X_train, y_train, X_val, y_val, log_to_wandb)

    # TODO:
    @override
    def create_inference_data(
        mts_dataset: np.ndarray, evaluation_set_indices: np.ndarray
    ):
        return super().create_inference_data(evaluation_set_indices)

    # TODO:
    @override
    def infer(self, X: np.ndarray) -> np.ndarray:
        return super().infer(X)
