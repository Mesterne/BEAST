import os
from typing import Optional

import numpy as np

from src.data.constants import OUTPUT_DIR
from src.plots.feature_wise_error import plot_distribution_of_feature_wise_error
from src.plots.full_time_series import plot_time_series_for_all_uts
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)
from src.plots.pca_total_generation import plot_pca_for_all_generated_mts
from src.plots.total_mse_distribution import plot_total_mse_distribution
from src.utils.logging_config import logger


def create_and_save_plots_of_model_performances(
    total_mse_for_each_mts: np.array,  # Shape: (number of timeseries generated,)
    mse_per_feature: np.array,  # Shape (number of timeseries generated, number of features)
    mts_dataset_array: np.array,
    mts_dataset_features: np.array,
    transformation_indices: np.ndarray,  # Shape (number of transformation in train set, 2) Entry 1 is the original index, Entry 2 is target
    inferred_mts_array: np.array,  # Shape: (number of time series generated, number of uts, number of time steps)
    inferred_mts_features: np.array,
    inferred_mts_features_before_ga: Optional[
        np.array
    ] = None,  # Shape: (number of time series generated, number of features)
):
    # For instance when using CVAE we dont have features before generation.
    # In this case, we set them to be the same as the features of inferred mts.
    # This is done to not break the program
    if inferred_mts_features_before_ga is None:
        inferred_mts_features_before_ga = inferred_mts_features

    X_features = mts_dataset_features[transformation_indices[:, 0]]
    y_features = mts_dataset_features[transformation_indices[:, 1]]

    X_mts = mts_dataset_array[transformation_indices[:, 0]]
    y_mts = mts_dataset_array[transformation_indices[:, 1]]

    # MSE Plots
    logger.info("Generating total MSE plot...")
    total_mse_plot = plot_total_mse_distribution(
        total_mse_for_each_mts=total_mse_for_each_mts
    )
    total_mse_plot.savefig(os.path.join(OUTPUT_DIR, "total_mse_distribution.png"))

    # logger.info("Generating feature wise MSE plot...")
    # print(mse_per_feature.shape)
    # feature_wise_error_plot = plot_distribution_of_feature_wise_error(
    #     mse_per_feature=mse_per_feature
    # )
    # feature_wise_error_plot.savefig(
    #     os.path.join(OUTPUT_DIR, "distribution_of_feature_wise_error.png")
    # )

    best_generated_mts_index = np.argmin(total_mse_for_each_mts)
    logger.info(f"Best genereated mts index: {best_generated_mts_index}")

    logger.info("Generating full TS plot...")
    ts_plot_of_best_generated_mts = plot_time_series_for_all_uts(
        original_mts=X_mts[best_generated_mts_index],
        target_mts=y_mts[best_generated_mts_index],
        transformed_mts=inferred_mts_array[best_generated_mts_index],
        original_mts_features=X_features[best_generated_mts_index],
        predicted_mts_features=inferred_mts_features_before_ga[
            best_generated_mts_index
        ],
        transformed_mts_features=inferred_mts_features[best_generated_mts_index],
        target_mts_features=y_features[best_generated_mts_index],
    )
    ts_plot_of_best_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "best_timeseries_generated_mts.png")
    )
    logger.info("Generating prediction PCA plot...")
    pca_plot_of_best_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_dataset_features=mts_dataset_features,
        mts_features_evaluation_set=y_features,
        original_mts_features=X_features[best_generated_mts_index],
        target_mts_features=y_features[best_generated_mts_index],
        predicted_mts_features=inferred_mts_features[best_generated_mts_index],
    )
    pca_plot_of_best_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "best_timeseries_generated_mts_pca.png")
    )

    worst_generated_mts_index = np.argmax(total_mse_for_each_mts)
    logger.info(f"Worst genereated mts index: {worst_generated_mts_index}")
    logger.info("Generating full TS plot...")
    ts_plot_of_worst_generated_mts = plot_time_series_for_all_uts(
        original_mts=X_mts[worst_generated_mts_index],
        target_mts=y_mts[worst_generated_mts_index],
        transformed_mts=inferred_mts_array[worst_generated_mts_index],
        original_mts_features=X_features[worst_generated_mts_index],
        predicted_mts_features=inferred_mts_features_before_ga[
            worst_generated_mts_index
        ],
        transformed_mts_features=inferred_mts_features[worst_generated_mts_index],
        target_mts_features=y_features[worst_generated_mts_index],
    )
    ts_plot_of_worst_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "worst_timeseries_generated_mts.png")
    )
    logger.info("Generating prediction PCA plot...")
    pca_plot_of_worst_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_dataset_features=mts_dataset_features,
        mts_features_evaluation_set=y_features,
        original_mts_features=X_features[worst_generated_mts_index],
        target_mts_features=y_features[worst_generated_mts_index],
        predicted_mts_features=inferred_mts_features[worst_generated_mts_index],
    )
    pca_plot_of_worst_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "worst_timeseries_generated_mts_pca.png")
    )

    median_mse = np.median(total_mse_for_each_mts)
    # NOTE: This way of finding the median index always returns one index. This is acceptable in this use case.
    median_mts_index = np.argmin(np.abs(total_mse_for_each_mts - median_mse))

    logger.info("Generating full TS plot...")
    ts_plot_of_median_generated_mts = plot_time_series_for_all_uts(
        original_mts=X_mts[median_mts_index],
        target_mts=y_mts[median_mts_index],
        transformed_mts=inferred_mts_array[median_mts_index],
        original_mts_features=X_features[median_mts_index],
        predicted_mts_features=inferred_mts_features_before_ga[median_mts_index],
        transformed_mts_features=inferred_mts_features[median_mts_index],
        target_mts_features=y_features[median_mts_index],
    )
    ts_plot_of_median_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "median_timeseries_generated_mts.png")
    )
    logger.info("Generating prediction PCA plot...")
    pca_plot_of_median_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_dataset_features=mts_dataset_features,
        mts_features_evaluation_set=y_features,
        original_mts_features=X_features[median_mts_index],
        target_mts_features=y_features[median_mts_index],
        predicted_mts_features=inferred_mts_features[median_mts_index],
    )
    pca_plot_of_median_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "median_timeseries_generated_mts_pca.png")
    )

    logger.info("Generating PCA plot of all predictions...")
    pca_total_plot = plot_pca_for_all_generated_mts(
        mts_dataset_features=mts_dataset_features,
        evaluation_set_indices=transformation_indices[:, 1],
        mts_generated_features=inferred_mts_features,
    )
    pca_total_plot.savefig(os.path.join(OUTPUT_DIR, "total_generation_pca.png"))
