from ast import List
from typing import Optional

import numpy as np

from src.plots.generated_vs_target_comparison import (
    create_and_save_plots_of_model_performances,
)
from src.utils.evaluation.feature_space_evaluation import (
    calculate_mse_for_each_feature,
    calculate_total_mse_for_each_mts,
)
from src.utils.features import decomp_and_features


def reshape_and_calculate_features(
    mts: np.ndarray,  # Shape: (Number of MTS, Number of samples per UTS * Number of UTS)
    num_features_per_uts: int,
    num_samples_per_uts: int,
    num_uts_in_mts: int,
    series_periodicity: int,
) -> np.ndarray:
    reshaped_mts = mts.reshape(-1, num_uts_in_mts, num_samples_per_uts)
    _, features = decomp_and_features(
        mts=reshaped_mts,
        num_features_per_uts=num_features_per_uts,
        series_periodicity=series_periodicity,
    )
    return features


# TODO: Remove hardcoded values
def evaluate(
    mts_array: np.ndarray,  # Shape (Number of MTS, Number of UTS in MTS, Number of samples in UTS)
    train_transformation_indices: np.ndarray,  # Shape (number of transformation in train set, 2) Entry 1 is the original index, Entry 2 is target
    validation_transformation_indices: np.ndarray,  # Shape (number of transformation in validation set, 2) Entry 1 is the original index, Entry 2 is target
    test_transformation_indices: np.ndarray,  # Shape (number of transformation in test set, 2) Entry 1 is the original index, Entry 2 is target
    inferred_mts_validation: np.ndarray,  # Shape: (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
    inferred_mts_test: np.ndarray,  # Shape: (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
    inferred_intermediate_features_validation: Optional[np.ndarray] = None,
    inferred_intermediate_features_test: Optional[np.ndarray] = None,
):
    mts_dataset_features = reshape_and_calculate_features(
        mts=mts_array,
        num_features_per_uts=4,
        num_samples_per_uts=192,
        num_uts_in_mts=3,
        series_periodicity=24,
    )
    inferred_features_validation = reshape_and_calculate_features(
        mts=inferred_mts_validation,
        num_features_per_uts=4,
        num_samples_per_uts=192,
        num_uts_in_mts=3,
        series_periodicity=24,
    )
    inferred_features_test = reshape_and_calculate_features(
        mts=inferred_mts_test,
        num_features_per_uts=4,
        num_samples_per_uts=192,
        num_uts_in_mts=3,
        series_periodicity=24,
    )

    y_features_validation = mts_dataset_features[
        validation_transformation_indices[:, 1]
    ]
    y_features_test = mts_dataset_features[test_transformation_indices[:, 1]]

    # Calcuate MSE
    mse_values_for_each_feature_validation = calculate_mse_for_each_feature(
        predicted_features=inferred_features_validation,
        target_features=y_features_validation,
    )
    mse_values_for_each_feature_test = calculate_mse_for_each_feature(
        predicted_features=inferred_features_test,
        target_features=y_features_test,
    )
    total_mse_for_each_mts_validation = calculate_total_mse_for_each_mts(
        mse_per_feature=mse_values_for_each_feature_validation
    )
    total_mse_for_each_mts_test = calculate_total_mse_for_each_mts(
        mse_per_feature=mse_values_for_each_feature_test
    )

    print(f"mts_dataset_features shape: {mts_dataset_features.shape}")
    print(f"inferred_features_validation shape: {mts_dataset_features.shape}")

    # Plotting for validation
    create_and_save_plots_of_model_performances(
        total_mse_for_each_mts=total_mse_for_each_mts_validation,
        mse_per_feature=mse_values_for_each_feature_validation,
        mts_dataset_array=mts_array,
        mts_dataset_features=mts_dataset_features,
        transformation_indices=validation_transformation_indices,
        inferred_mts_array=inferred_mts_validation,
        inferred_mts_features_before_ga=inferred_intermediate_features_validation,
        inferred_mts_features=inferred_features_validation,
    )
