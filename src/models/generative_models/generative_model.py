from typing import Dict, List, Tuple, override

import numpy as np

from src.models.generative_models.cvae import MTSCVAE
from src.models.generative_models.cvae_wrapper import CVAEWrapper, prepare_cgen_data
from src.models.timeseries_transformation_model import TimeseriesTransformationModel


class GenerativeModel(TimeseriesTransformationModel):
    def __init__(self, config: Dict[str, any]) -> None:
        super().__init__()
        self.model = self._choose_underlying_model_based_on_config(config)

    def _choose_underlying_model_based_on_config(self, config: Dict[str, any]):
        model_params: Dict[str, any] = config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]
        training_params: Dict[str, any] = model_params["training_params"]
        cvae = MTSCVAE(model_params=model_params)
        model = CVAEWrapper(cvae, training_params=training_params)
        return model

    # TODO:
    @override
    def create_training_data(
        mts_dataset: np.ndarray,
        train_transformation_indices: np.ndarray,
        validation_transformation_indices: np.ndarray,
    ):

        logger.info("Preparing data set for conditional generative model...")
        condition_type: str = self.config["model_args"]["feature_model_args"][
            "conditional_gen_model_args"
        ]["condition_type"]
        (
            X_y_pairs_cgen_train,
            X_y_pairs_cgen_validation,
            X_y_pairs_cgen_test,
        ) = prepare_cgen_data(
            condition_type,
            mts_dataset,
            train_features_supervised_dataset,
            validation_features_supervised_dataset,
            test_features_supervised_dataset,
        )
        logger.info("Successfully prepared data for conditional generative model")
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
