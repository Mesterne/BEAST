import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.data.constants import OUTPUT_DIR
from src.plots.full_time_series import plot_time_series_for_all_uts
from src.plots.pca_for_each_uts_with_transformed import \
    plot_pca_for_each_uts_with_transformed
from src.plots.pca_total_generation import (
    plot_pca_for_all_generated_mts,
    plot_pca_for_all_generated_mts_for_each_uts)
from src.plots.plot_only_timeseries_generated import \
    plot_only_timeseries_generated
from src.utils.evaluation.feature_space_evaluation import \
    calculate_total_evaluation_for_each_mts
from src.utils.logging_config import logger


def create_grid_plot_of_worst_median_best(
    mts_dataset_array: np.ndarray,
    mts_dataset_features: np.ndarray,
    train_transformation_indices: np.ndarray,
    evaluation_transformation_indices: np.ndarray,  # Shape (number of transformation in train set, 2) Entry 1 is the original index, Entry 2 is target
    inferred_mts_array: np.ndarray,  # Shape: (number of time series generated, number of uts, number of time steps)
    inferred_mts_features: np.ndarray,
    inferred_mts_features_before_ga: Optional[
        np.ndarray
    ] = None,  # Shape: (number of time series generated, number of features)
    optional_indices: Optional[np.ndarray] = None,
):
    # For instance when using CVAE we dont have features before generation.
    # In this case, we set them to be the same as the features of inferred mts.
    # This is done to not break the program
    if inferred_mts_features_before_ga is None:
        inferred_mts_features_before_ga = inferred_mts_features

    X_features = mts_dataset_features[evaluation_transformation_indices[:, 0]]
    y_features = mts_dataset_features[evaluation_transformation_indices[:, 1]]

    X_mts = mts_dataset_array[evaluation_transformation_indices[:, 0]]
    y_mts = mts_dataset_array[evaluation_transformation_indices[:, 1]]

    total_rmse_for_each_mts = calculate_total_evaluation_for_each_mts(
        predicted_features=inferred_mts_features,
        target_features=y_features,
        metric="MSE",
    )

    best_generated_mts_index = np.argmin(total_rmse_for_each_mts)
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
        os.path.join(
            OUTPUT_DIR, "Generation grids", "best_timeseries_generated_mts.png"
        ),
        dpi=600,
    )
    plt.close(ts_plot_of_best_generated_mts)
    logger.info("Generating prediction PCA plot...")
    pca_plot_of_best_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_dataset_features=mts_dataset_features,
        train_transformations=train_transformation_indices,
        mts_features_evaluation_set=y_features,
        original_mts_features=X_features[best_generated_mts_index],
        target_mts_features=y_features[best_generated_mts_index],
        predicted_mts_features=inferred_mts_features[best_generated_mts_index],
    )
    pca_plot_of_best_generated_mts.savefig(
        os.path.join(
            OUTPUT_DIR, "Generation grids", "best_timeseries_generated_mts_pca.png"
        ),
        dpi=600,
    )
    plt.close(pca_plot_of_best_generated_mts)

    worst_generated_mts_index = np.argmax(total_rmse_for_each_mts)
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
        os.path.join(
            OUTPUT_DIR, "Generation grids", "worst_timeseries_generated_mts.png"
        ),
        dpi=600,
    )
    plt.close(ts_plot_of_best_generated_mts)
    logger.info("Generating prediction PCA plot...")
    pca_plot_of_worst_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_dataset_features=mts_dataset_features,
        mts_features_evaluation_set=y_features,
        original_mts_features=X_features[worst_generated_mts_index],
        train_transformations=train_transformation_indices,
        target_mts_features=y_features[worst_generated_mts_index],
        predicted_mts_features=inferred_mts_features[worst_generated_mts_index],
    )
    pca_plot_of_worst_generated_mts.savefig(
        os.path.join(
            OUTPUT_DIR, "Generation grids", "worst_timeseries_generated_mts_pca.png"
        ),
        dpi=600,
    )
    plt.close(pca_plot_of_worst_generated_mts)

    median_rmse = np.median(total_rmse_for_each_mts)
    # NOTE: This way of finding the median index always returns one index. This is acceptable in this use case.
    median_mts_index = np.argmin(np.abs(total_rmse_for_each_mts - median_rmse))

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
        os.path.join(
            OUTPUT_DIR, "Generation grids", "median_timeseries_generated_mts.png"
        ),
        dpi=600,
    )
    plt.close(ts_plot_of_median_generated_mts)
    logger.info("Generating prediction PCA plot...")
    pca_plot_of_median_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_dataset_features=mts_dataset_features,
        mts_features_evaluation_set=y_features,
        train_transformations=train_transformation_indices,
        original_mts_features=X_features[median_mts_index],
        target_mts_features=y_features[median_mts_index],
        predicted_mts_features=inferred_mts_features[median_mts_index],
    )
    pca_plot_of_median_generated_mts.savefig(
        os.path.join(
            OUTPUT_DIR, "Generation grids", "median_timeseries_generated_mts_pca.png"
        ),
        dpi=600,
    )
    plt.close(pca_plot_of_median_generated_mts)

    if optional_indices is not None:
        for i in optional_indices:
            logger.info(f"Generating grid for MTS {i}")
            pca_plot_of_optional_index = plot_time_series_for_all_uts(
                original_mts=X_mts[i],
                target_mts=y_mts[i],
                transformed_mts=inferred_mts_array[i],
                original_mts_features=X_features[i],
                predicted_mts_features=inferred_mts_features_before_ga[i],
                transformed_mts_features=inferred_mts_features[i],
                target_mts_features=y_features[i],
            )
            pca_plot_of_optional_index.savefig(
                os.path.join(
                    OUTPUT_DIR, "Generation grids", f"optional_indices_{i}.png"
                ),
                dpi=600,
            )
            plt.close(pca_plot_of_optional_index)

    logger.info("Generating PCA plot of all predictions...")
    pca_for_each_uts = plot_pca_for_all_generated_mts_for_each_uts(
        mts_dataset_features=mts_dataset_features,
        mts_generated_features=inferred_mts_features,
        train_transformations=train_transformation_indices,
        evaluation_transformations=evaluation_transformation_indices,
    )
    pca_for_each_uts.savefig(
        os.path.join(OUTPUT_DIR, "total_generation_pca_for_each_uts.png"), dpi=600
    )
    plt.close(pca_for_each_uts)

    logger.info("Plotting individual time series")
    only_timeseries_plot_best = plot_only_timeseries_generated(
        original_mts=X_mts[best_generated_mts_index],
        target_mts=y_mts[best_generated_mts_index],
        transformed_mts=inferred_mts_array[best_generated_mts_index],
    )
    only_timeseries_plot_best.savefig(
        os.path.join(OUTPUT_DIR, "Generated MTS", "best_generated.png"), dpi=600
    )
    plt.close(only_timeseries_plot_best)
    only_timeseries_plot_median = plot_only_timeseries_generated(
        original_mts=X_mts[median_mts_index],
        target_mts=y_mts[median_mts_index],
        transformed_mts=inferred_mts_array[median_mts_index],
    )
    only_timeseries_plot_median.savefig(
        os.path.join(OUTPUT_DIR, "Generated MTS", "median_generated.png"), dpi=600
    )
    plt.close(only_timeseries_plot_median)
    only_timeseries_plot_worst = plot_only_timeseries_generated(
        original_mts=X_mts[worst_generated_mts_index],
        target_mts=y_mts[worst_generated_mts_index],
        transformed_mts=inferred_mts_array[worst_generated_mts_index],
    )
    only_timeseries_plot_worst.savefig(
        os.path.join(OUTPUT_DIR, "Generated MTS", "worst_generated.png"), dpi=600
    )
    plt.close(only_timeseries_plot_worst)
