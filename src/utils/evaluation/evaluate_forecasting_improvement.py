import os
from typing import Dict

import numpy as np

from src.data.constants import OUTPUT_DIR
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.utils.forecasting_utils import compare_old_and_new_model
from src.utils.generate_dataset import \
    create_training_windows_from_mts  # noqa: E402


class ForecasterEvaluator:
    def __init__(
        self,
        config: Dict[str, any],
        mts_dataset,
        train_indices,
        validation_indices,
        test_indices,
    ) -> None:
        self.config = config
        self.mts_dataset = mts_dataset
        self.train_indices = train_indices
        self.validation_indices = validation_indices
        self.test_indices = test_indices

        self.train_mts_array, self.validation_mts_array, self.test_mts_array = (
            self._extract_train_val_test()
        )
        self.X_mts_train, self.y_mts_train = self._create_training_windows(
            dataset=self.train_mts_array
        )
        self.X_mts_validation, self.y_mts_validation = self._create_training_windows(
            dataset=self.validation_mts_array
        )
        self.X_mts_test, self.y_mts_test = self._create_training_windows(
            dataset=self.test_mts_array
        )

        self.original_forecasting_model = self._initialize_forecasting_model()
        self.original_forecasting_model.train(
            X_train=self.X_mts_train,
            y_train=self.y_mts_train,
            X_val=self.X_mts_train,
            y_val=self.y_mts_train,
            plot_loss=False,
        )

    def _extract_train_val_test(self):
        return (
            self.mts_dataset[self.train_indices],
            self.mts_dataset[self.validation_indices],
            self.mts_dataset[self.test_indices],
        )

    def _create_training_windows(self, dataset):
        (X_mts, y_mts) = create_training_windows_from_mts(
            mts=dataset,
            target_col_index=1,
            window_size=self.config["model_args"]["forecasting_model_args"][
                "window_size"
            ],
            forecast_horizon=self.config["model_args"]["forecasting_model_args"][
                "horizon_length"
            ],
        )

        return X_mts, y_mts

    def _initialize_forecasting_model(self):
        forecasting_model: FeedForwardForecaster = FeedForwardForecaster(
            model_params=self.config["model_args"]["forecasting_model_args"]
        )
        forecasting_model_wrapper: NeuralNetworkWrapper = NeuralNetworkWrapper(
            model=forecasting_model,
            training_params=self.config["model_args"]["forecasting_model_args"][
                "training_args"
            ],
        )
        return forecasting_model_wrapper

    def evaluate_on_evaluation_set(self, inferred_mts_array, type=""):
        X_inferred, y_inferred = self._create_training_windows(inferred_mts_array)

        X_new_train: np.ndarray = np.vstack((self.X_mts_train, X_inferred))
        y_new_train: np.ndarray = np.vstack((self.y_mts_train, y_inferred))

        new_forecasting_model: NeuralNetworkWrapper = (
            self._initialize_forecasting_model()
        )
        new_forecasting_model.train(
            X_train=X_new_train,
            y_train=y_new_train,
            X_val=X_new_train,
            y_val=y_new_train,
            plot_loss=False,
        )
        forecast_plot, mse_plot, mse_delta_plot, mase_plot, mase_delta_plot = (
            compare_old_and_new_model(
                X_test=self.X_mts_test,
                y_test=self.y_mts_test,
                X_val=self.X_mts_validation,
                y_val=self.y_mts_validation,
                X_train=self.X_mts_train,
                y_train=self.y_mts_train,
                forecasting_model_wrapper_old=self.original_forecasting_model,
                forecasting_model_wrapper_new=new_forecasting_model,
            )
        )
        forecast_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"retrain_on_{type}_forecasting_model_comparison_forecast.png",
            )
        )
        mse_plot.savefig(
            os.path.join(
                OUTPUT_DIR, f"retrain_on_{type}_forecasting_model_comparison_mse.png"
            )
        )

        mse_delta_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"retrain_on_{type}_forecasting_model_improvement_delta_mse.png",
            )
        )
        mase_plot.savefig(
            os.path.join(
                OUTPUT_DIR, f"retrain_on_{type}_forecasting_model_comparison_mase.png"
            )
        )
        mase_delta_plot.savefig(
            os.path.join(
                OUTPUT_DIR,
                f"retrain_on_{type}_forecasting_model_improvement_delta_mase.png",
            )
        )
