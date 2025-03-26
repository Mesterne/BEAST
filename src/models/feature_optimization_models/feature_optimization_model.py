from typing import override

from src.models.timeseries_transformation_model import TimeseriesTransformationModel


class FeatureOptimizationModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.model = self._choose_underlying_model_based_on_config(config)

    def _choose_underlying_model_based_on_config(self, config: Dict[str, Any]):
        pass

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
