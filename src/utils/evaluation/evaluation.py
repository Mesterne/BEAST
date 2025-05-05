import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import OUTPUT_DIR
from src.plots.generated_vs_target_comparison import \
    create_and_save_plots_of_model_performances
from src.plots.ohe_plots import \
    create_and_save_plots_of_ohe_activated_performances
from src.plots.pca_total_generation import plot_pca_for_all_generated_mts
from src.plots.plot_feature_evaluation_bar_plot import \
    plot_metric_for_each_feature_bar_plot
from src.plots.plot_feature_evaluation_distribution import (
    plot_feature_evaluation, plot_feature_mse_distribution)
from src.utils.evaluation.feature_space_evaluation import (
    calculate_mse_for_each_feature, calculate_total_mse_for_each_mts)
from src.utils.features import decomp_and_features
from src.utils.logging_config import logger


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


def evaluate(
    config: Dict[str, Any],
    mts_array: np.ndarray,  # Shape (Number of MTS, Number of UTS in MTS, Number of samples in UTS)
    train_transformation_indices: np.ndarray,  # Shape (number of transformation in train set, 2) Entry 1 is the original index, Entry 2 is target
    validation_transformation_indices: np.ndarray,  # Shape (number of transformation in validation set, 2) Entry 1 is the original index, Entry 2 is target
    test_transformation_indices: np.ndarray,  # Shape (number of transformation in test set, 2) Entry 1 is the original index, Entry 2 is target
    inferred_mts_validation: np.ndarray,  # Shape: (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
    inferred_mts_test: np.ndarray,  # Shape: (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
    inferred_intermediate_features_validation: Optional[np.ndarray],
    inferred_intermediate_features_test: Optional[
        np.ndarray
    ],  # Shape: (Number of transformations, Number of UTS in MTS, Number of features in UTS)
    ohe_val: np.ndarray,
    ohe_test: np.ndarray,
):
    """
    Takes the dataset, defined transformation indices and inferred MTS for the
    different datasets to evaluate the inferred MTS with the true target values.

    Args:
        config: Dict, The config file loaded for the experiment
        mts_array: np.ndarray,  Entire dataset of MTS with shape = (Number of MTS, Number of UTS in MTS, Number of samples in UTS)
        train_transformation_indices: np.ndarray, Indices for transformations in training set with Shape = (number of transformation in train set, 2)
            Entry 1 is the original index, Entry 2 is target
        validation_transformation_indices: np.ndarray, Indices for transformations in validation set with Shape = (number of transformation in train set, 2)
            Entry 1 is the original index, Entry 2 is target
        test_transformation_indices: np.ndarray, Indices for transformations in test set with Shape = (number of transformation in train set, 2)
            Entry 1 is the original index, Entry 2 is target
        inferred_mts_validation: np.ndarray,  The inferred MTS of validation set with shape = (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
        inferred_mts_test: np.ndarray,  The inferred MTS of test set with shape = (number of transformations in test set, Number of UTS in MTS * Number of samples in UTS)
        inferred_intermediate_features_validation: Optional input with intermediate feature values calculated during inference. Shape = (Number of transformations in validation set, Number of UTS in MTS, Number of features in UTS),
        inferred_intermediate_features_test: Optional input with intermediate feature values calculated during inference. Shape = (Number of transformations in test set, Number of UTS in MTS, Number of features in UTS),

    """

    logger.info(
        "Calculating features for original dataset, inferred validation timeseries and inferred test timeseries"
    )
    mts_dataset_features = reshape_and_calculate_features(
        mts=mts_array,
        num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
        num_samples_per_uts=config["dataset_args"]["window_size"],
        num_uts_in_mts=len(config["dataset_args"]["timeseries_to_use"]),
        series_periodicity=config["stl_args"]["series_periodicity"],
    )
    inferred_features_validation = reshape_and_calculate_features(
        mts=inferred_mts_validation,
        num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
        num_samples_per_uts=config["dataset_args"]["window_size"],
        num_uts_in_mts=len(config["dataset_args"]["timeseries_to_use"]),
        series_periodicity=config["stl_args"]["series_periodicity"],
    )
    inferred_features_test = reshape_and_calculate_features(
        mts=inferred_mts_test,
        num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
        num_samples_per_uts=config["dataset_args"]["window_size"],
        num_uts_in_mts=len(config["dataset_args"]["timeseries_to_use"]),
        series_periodicity=config["stl_args"]["series_periodicity"],
    )
    if inferred_intermediate_features_validation is None:
        inferred_intermediate_features_validation = inferred_features_validation
    if inferred_intermediate_features_test is None:
        inferred_intermediate_features_test = inferred_features_test
    inferred_intermediate_features_validation = (
        inferred_intermediate_features_validation.reshape(
            -1,
            len(config["dataset_args"]["timeseries_to_use"])
            * config["dataset_args"]["num_features_per_uts"],
        )
    )
    inferred_intermediate_features_test = inferred_intermediate_features_test.reshape(
        -1,
        len(config["dataset_args"]["timeseries_to_use"])
        * config["dataset_args"]["num_features_per_uts"],
    )
    y_features_validation = mts_dataset_features[
        validation_transformation_indices[:, 1]
    ]
    y_features_test = mts_dataset_features[test_transformation_indices[:, 1]]

    # Calcuate MSE
    logger.info("Calculating MSE values...")
    final_mse_values_for_each_feature_validation = calculate_mse_for_each_feature(
        predicted_features=inferred_features_validation,
        target_features=y_features_validation,
    )
    final_mse_values_for_each_feature_test = calculate_mse_for_each_feature(
        predicted_features=inferred_features_test,
        target_features=y_features_test,
    )
    final_total_mse_for_each_mts_validation = calculate_total_mse_for_each_mts(
        mse_per_feature=final_mse_values_for_each_feature_validation
    )
    final_total_mse_for_each_mts_test = calculate_total_mse_for_each_mts(
        mse_per_feature=final_mse_values_for_each_feature_test
    )
    final_total_mse_features_validation = np.mean(
        final_total_mse_for_each_mts_validation
    )
    final_total_mse_features_test = np.mean(final_total_mse_for_each_mts_test)
    intermediate_mse_values_for_each_feature_validation = (
        calculate_mse_for_each_feature(
            predicted_features=inferred_intermediate_features_validation,
            target_features=y_features_validation,
        )
    )
    intermediate_mse_values_for_each_feature_test = calculate_mse_for_each_feature(
        predicted_features=inferred_intermediate_features_test,
        target_features=y_features_test,
    )
    intermediate_total_mse_for_each_mts_validation = calculate_total_mse_for_each_mts(
        mse_per_feature=intermediate_mse_values_for_each_feature_validation
    )
    intermediate_total_mse_for_each_mts_test = calculate_total_mse_for_each_mts(
        mse_per_feature=intermediate_mse_values_for_each_feature_test
    )
    intermediate_total_mse_features_validation = np.mean(
        intermediate_total_mse_for_each_mts_validation
    )
    intermediate_total_mse_features_test = np.mean(
        intermediate_total_mse_for_each_mts_test
    )

    logger.info("Creating feature evaluation plots...")
    mse_distribution_feature_space = plot_feature_mse_distribution(
        feature_space_mse_validation=final_total_mse_for_each_mts_validation,
        feature_space_mse_test=final_total_mse_for_each_mts_test,
    )

    mse_for_each_feature_validation = plot_metric_for_each_feature_bar_plot(
        intermediate_evaluation_for_each_feature=intermediate_mse_values_for_each_feature_validation,
        final_evaluation_for_each_feature=final_mse_values_for_each_feature_validation,
        metric="MSE",
        dataset="Validation",
    )
    mse_for_each_feature_validation.savefig(
        os.path.join(OUTPUT_DIR, "mse_for_each_feature_validation.png")
    )
    mse_for_each_feature_test = plot_metric_for_each_feature_bar_plot(
        intermediate_evaluation_for_each_feature=intermediate_mse_values_for_each_feature_test,
        final_evaluation_for_each_feature=final_mse_values_for_each_feature_test,
        metric="MASE",
        dataset="Test",
    )
    mse_for_each_feature_test.savefig(
        os.path.join(OUTPUT_DIR, "mse_for_each_feature_test.png")
    )

    mse_distribution_feature_space.savefig(
        os.path.join(OUTPUT_DIR, "mse_feature_space_distribution.png")
    )

    mse_feature_space = plot_feature_evaluation(
        inferred_feature_space_mse_validation=intermediate_total_mse_features_validation,
        final_feature_space_mse_validation=final_total_mse_features_validation,
        inferred_feature_space_mse_test=intermediate_total_mse_features_test,
        final_feature_space_mse_test=final_total_mse_features_test,
        metric_name="MSE",
    )
    mse_feature_space.savefig(os.path.join(OUTPUT_DIR, "mse_feature_space.png"))

    pca_plot_intermediate_validation_features = plot_pca_for_all_generated_mts(
        mts_dataset_features=mts_dataset_features,
        mts_generated_features=inferred_intermediate_features_validation,
        evaluation_set_indices=validation_transformation_indices,
    )
    pca_plot_intermediate_validation_features.savefig(
        os.path.join(OUTPUT_DIR, "total_generation_pca_intermediate_validation.png")
    )
    pca_plot_intermediate_test_features = plot_pca_for_all_generated_mts(
        mts_dataset_features=mts_dataset_features,
        mts_generated_features=inferred_intermediate_features_test,
        evaluation_set_indices=test_transformation_indices,
    )
    pca_plot_intermediate_test_features.savefig(
        os.path.join(OUTPUT_DIR, "total_generation_pca_intermediate_test.png")
    )

    create_and_save_plots_of_ohe_activated_performances(
        ohe=ohe_val,
        evaluation=final_mse_values_for_each_feature_validation,
        metric_name="MSE",
        dataset="Validation",
    )

    logger.info("Creating plots for validation...")
    create_and_save_plots_of_model_performances(
        total_mse_for_each_mts=final_total_mse_for_each_mts_validation,
        mse_per_feature=final_mse_values_for_each_feature_validation,
        mts_dataset_array=mts_array,
        mts_dataset_features=mts_dataset_features,
        transformation_indices=validation_transformation_indices,
        inferred_mts_array=inferred_mts_validation,
        inferred_mts_features_before_ga=inferred_intermediate_features_validation,
        inferred_mts_features=inferred_features_validation,
    )
