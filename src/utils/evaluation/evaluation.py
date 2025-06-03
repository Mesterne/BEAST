import os
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import OUTPUT_DIR
from src.plots.generated_vs_target_comparison import \
    create_grid_plot_of_worst_median_best
from src.plots.ohe_plots import \
    create_and_save_plots_of_ohe_activated_performances_feature_space
from src.plots.overlapping_mts_plot import plot_overlapping_mts
from src.plots.pca_total_generation import (
    plot_pca_for_all_generated_mts,
    plot_pca_for_all_generated_mts_for_each_uts)
from src.plots.plot_feature_evaluation_bar_plot import \
    plot_metric_for_each_feature_bar_plot
from src.plots.plot_feature_evaluation_distribution import \
    plot_feature_evaluation
from src.utils.evaluation.feature_space_evaluation import \
    calculate_evaluation_for_each_feature
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


def plot_evaluation_summaries_over_feature_space(
    intermediate_features_validation: np.ndarray,
    intermediate_features_test: np.ndarray,
    final_features_validation: np.ndarray,
    final_features_test: np.ndarray,
    target_features_validation: np.ndarray,
    target_features_test: np.ndarray,
    ohe_val: np.ndarray,
    ohe_test: np.ndarray,
    metric: str,
):
    final_evaluation_values_for_each_feature_validation = (
        calculate_evaluation_for_each_feature(
            predicted_features=final_features_validation,
            target_features=target_features_validation,
            metric=metric,
        )
    )
    final_evaluation_values_for_each_feature_test = (
        calculate_evaluation_for_each_feature(
            predicted_features=final_features_test,
            target_features=target_features_test,
            metric=metric,
        )
    )

    intermediate_evaluation_values_for_each_feature_validation = (
        calculate_evaluation_for_each_feature(
            predicted_features=intermediate_features_validation,
            target_features=target_features_validation,
            metric=metric,
        )
    )
    intermediate_evaluation_values_for_each_feature_test = (
        calculate_evaluation_for_each_feature(
            predicted_features=intermediate_features_test,
            target_features=target_features_test,
            metric=metric,
        )
    )

    evaluation_for_each_feature_validation = plot_metric_for_each_feature_bar_plot(
        intermediate_evaluation_for_each_feature=intermediate_evaluation_values_for_each_feature_validation,
        final_evaluation_for_each_feature=final_evaluation_values_for_each_feature_validation,
        metric=metric,
        dataset="Validation",
    )
    evaluation_for_each_feature_validation.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            f"FEATURE_SPACE_{metric}_for_each_feature_validation.png",
        )
    )
    plt.close(evaluation_for_each_feature_validation)
    evaluation_for_each_feature_test = plot_metric_for_each_feature_bar_plot(
        intermediate_evaluation_for_each_feature=intermediate_evaluation_values_for_each_feature_test,
        final_evaluation_for_each_feature=final_evaluation_values_for_each_feature_test,
        metric=metric,
        dataset="Test",
    )
    evaluation_for_each_feature_test.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            f"FEATURE_SPACE_{metric}_for_each_feature_test.png",
        )
    )
    plt.close(evaluation_for_each_feature_test)

    evaluation_feature_space = plot_feature_evaluation(
        intermediate_mse_values_for_each_feature_validation=intermediate_evaluation_values_for_each_feature_validation,
        intermediate_mse_values_for_each_feature_test=intermediate_evaluation_values_for_each_feature_test,
        final_mse_values_for_each_feature_validation=final_evaluation_values_for_each_feature_validation,
        final_mse_values_for_each_feature_test=final_evaluation_values_for_each_feature_test,
        metric_name=metric,
    )
    evaluation_feature_space.savefig(
        os.path.join(
            OUTPUT_DIR, "Feature space evaluations", f"{metric}_feature_space.png"
        )
    )
    plt.close(evaluation_feature_space)

    create_and_save_plots_of_ohe_activated_performances_feature_space(
        ohe=ohe_val,
        evaluation=final_evaluation_values_for_each_feature_validation,
        metric_name=metric,
        dataset="Validation",
    )
    create_and_save_plots_of_ohe_activated_performances_feature_space(
        ohe=ohe_test,
        evaluation=final_evaluation_values_for_each_feature_test,
        metric_name=metric,
        dataset="Test",
    )


def evaluate(
    config: Dict[str, Any],
    mts_array: np.ndarray,  # Shape (Number of MTS, Number of UTS in MTS, Number of samples in UTS)
    train_transformation_indices: np.ndarray,  # Shape (number of transformation in train set, 2) Entry 1 is the original index, Entry 2 is target
    validation_transformation_indices: np.ndarray,  # Shape (number of transformation in validation set, 2) Entry 1 is the original index, Entry 2 is target
    test_transformation_indices: np.ndarray,  # Shape (number of transformation in test set, 2) Entry 1 is the original index, Entry 2 is target
    inferred_mts_validation: np.ndarray,  # Shape: (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
    inferred_mts_test: np.ndarray,  # Shape: (number of transformations in validation set, Number of UTS in MTS * Number of samples in UTS)
    intermediate_features_validation: Optional[np.ndarray],
    intermediate_features_test: Optional[
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

    validation_overlapping_mts_plot = plot_overlapping_mts(
        set_of_mts=inferred_mts_validation
    )
    validation_overlapping_mts_plot.savefig(
        os.path.join(OUTPUT_DIR, "Generated MTS", "validation_overlapping.png")
    )
    test_overlapping_mts_plot = plot_overlapping_mts(set_of_mts=inferred_mts_test)
    test_overlapping_mts_plot.savefig(
        os.path.join(OUTPUT_DIR, "Generated MTS", "test_overlapping.png")
    )

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
    if intermediate_features_validation is None:
        intermediate_features_validation = inferred_features_validation
    if intermediate_features_test is None:
        intermediate_features_test = inferred_features_test
    intermediate_features_validation = intermediate_features_validation.reshape(
        -1,
        len(config["dataset_args"]["timeseries_to_use"])
        * config["dataset_args"]["num_features_per_uts"],
    )
    intermediate_features_test = intermediate_features_test.reshape(
        -1,
        len(config["dataset_args"]["timeseries_to_use"])
        * config["dataset_args"]["num_features_per_uts"],
    )
    y_features_validation = mts_dataset_features[
        validation_transformation_indices[:, 1]
    ]
    y_features_test = mts_dataset_features[test_transformation_indices[:, 1]]

    logger.info("Creating feature evaluation plots...")
    plot_evaluation_summaries_over_feature_space(
        intermediate_features_validation=intermediate_features_validation,
        intermediate_features_test=intermediate_features_test,
        final_features_validation=inferred_features_validation,
        final_features_test=inferred_features_test,
        target_features_validation=y_features_validation,
        target_features_test=y_features_test,
        ohe_val=ohe_val,
        ohe_test=ohe_test,
        metric="MSE",
    )
    plot_evaluation_summaries_over_feature_space(
        intermediate_features_validation=intermediate_features_validation,
        intermediate_features_test=intermediate_features_test,
        final_features_validation=inferred_features_validation,
        final_features_test=inferred_features_test,
        target_features_validation=y_features_validation,
        target_features_test=y_features_test,
        ohe_val=ohe_val,
        ohe_test=ohe_test,
        metric="MAE",
    )

    # PCA Plots
    pca_plot_intermediate_validation_features = plot_pca_for_all_generated_mts(
        mts_dataset_features=mts_dataset_features,
        mts_generated_features=intermediate_features_validation,
        train_transformations=train_transformation_indices,
        evaluation_transformations=validation_transformation_indices,
    )
    pca_plot_intermediate_validation_features.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            "FEATURE_SPACE_total_generation_pca_intermediate_validation.png",
        )
    )
    plt.close(pca_plot_intermediate_validation_features)

    pca_plot_for_all_uts_intermediate_validation_features = (
        plot_pca_for_all_generated_mts_for_each_uts(
            mts_dataset_features=mts_dataset_features,
            mts_generated_features=intermediate_features_validation,
            train_transformations=train_transformation_indices,
            evaluation_transformations=validation_transformation_indices,
        )
    )
    pca_plot_for_all_uts_intermediate_validation_features.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            "FEATURE_SPACE_total_generation_pca_intermediate_validation_all_uts.png",
        )
    )
    plt.close(pca_plot_for_all_uts_intermediate_validation_features)

    pca_plot_intermediate_test_features = plot_pca_for_all_generated_mts(
        mts_dataset_features=mts_dataset_features,
        mts_generated_features=intermediate_features_test,
        train_transformations=train_transformation_indices,
        evaluation_transformations=test_transformation_indices,
    )
    pca_plot_intermediate_test_features.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            "FEATURE_SPACE_total_generation_pca_intermediate_test.png",
        )
    )
    plt.close(pca_plot_intermediate_test_features)
    logger.info("Plotting PCA of features of all MTS")
    pca_plot_final_test_features = plot_pca_for_all_generated_mts(
        mts_dataset_features=mts_dataset_features,
        mts_generated_features=inferred_features_test,
        train_transformations=train_transformation_indices,
        evaluation_transformations=test_transformation_indices,
    )
    pca_plot_final_test_features.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            "FEATURE_SPACE_total_generation_pca_final_test.png",
        )
    )
    plt.close(pca_plot_final_test_features)
    pca_plot_for_all_uts_intermediate_test_features = (
        plot_pca_for_all_generated_mts_for_each_uts(
            mts_dataset_features=mts_dataset_features,
            mts_generated_features=intermediate_features_test,
            train_transformations=train_transformation_indices,
            evaluation_transformations=test_transformation_indices,
        )
    )
    pca_plot_for_all_uts_intermediate_test_features.savefig(
        os.path.join(
            OUTPUT_DIR,
            "Feature space evaluations",
            "FEATURE_SPACE_total_generation_pca_intermediate_test_all_uts.png",
        )
    )
    plt.close(pca_plot_for_all_uts_intermediate_test_features)

    logger.info("Creating plots for validation...")
    create_grid_plot_of_worst_median_best(
        mts_dataset_array=mts_array,
        mts_dataset_features=mts_dataset_features,
        train_transformation_indices=train_transformation_indices,
        evaluation_transformation_indices=validation_transformation_indices,
        inferred_mts_array=inferred_mts_validation,
        inferred_mts_features_before_ga=intermediate_features_validation,
        inferred_mts_features=inferred_features_validation,
    )
