import os

import numpy as np

from src.data.constants import OUTPUT_DIR
from src.plots.feature_wise_error import plot_distribution_of_feature_wise_error
from src.plots.full_time_series import plot_time_series_for_all_uts
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)
from src.plots.pca_total_generation import plot_pca_for_all_generated_mts


def create_and_save_plots_of_model_performances(
    total_mse_for_each_mts: np.array,  # Shape: (number of timeseries generated,)
    mse_per_feature: np.array,  # Shape (number of timeseries generated, number of features)
    mts_features_train: np.array,  # Shape: (number of entries in training set, number of features)
    mts_features_validation: np.array,  # Shape: (number of entries in validation set, number of features)
    mts_features_test: np.array,  # Shape: (number of entries in test set, number of features)
    original_mts: np.array,  # Shape: (number of timeseries generated, number of uts, number of time steps)
    target_mts: np.array,  # Shape: (number of time series generated, number of uts, number of time steps)
    generated_mts: np.array,  # Shape: (number of time series generated, number of uts, number of time steps)
    original_mts_features: np.array,  # Shape: (number of  time series generated, number of features)
    transformed_mts_features: np.array,  # Shape: (number of time series generated, number of features)
    target_mts_features: np.array,  # Shape: (number of time series genereated, number of features)
):
    # Make sure that we are not plotting duplicates
    mts_features_train = np.unique(mts_features_train, axis=0)
    mts_features_validation = np.unique(mts_features_validation, axis=0)
    mts_features_test = np.unique(mts_features_test, axis=0)

    feature_wise_errror_plot = plot_distribution_of_feature_wise_error(
        mse_per_feature=mse_per_feature
    )
    feature_wise_errror_plot.savefig(
        os.path.join(OUTPUT_DIR, "distribution_of_feature_wise_error.png")
    )

    best_generated_mts_index = np.argmin(total_mse_for_each_mts)

    ts_plot_of_best_generated_mts = plot_time_series_for_all_uts(
        original_mts=original_mts[best_generated_mts_index],
        target_mts=target_mts[best_generated_mts_index],
        transformed_mts=generated_mts[best_generated_mts_index],
        original_mts_features=original_mts_features[best_generated_mts_index],
        transformed_mts_features=transformed_mts_features[best_generated_mts_index],
        target_mts_features=target_mts_features[best_generated_mts_index],
    )
    ts_plot_of_best_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "best_timeseries_generated_mts.png")
    )
    pca_plot_of_best_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_features_train=mts_features_train,
        mts_features_validation=mts_features_validation,
        mts_features_test=mts_features_test,
        original_mts_features=original_mts_features[best_generated_mts_index],
        target_mts_features=target_mts_features[best_generated_mts_index],
        predicted_mts_features=transformed_mts_features[best_generated_mts_index],
    )
    pca_plot_of_best_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "best_timeseries_generated_mts_pca.png")
    )

    worst_generated_mts_index = np.argmax(total_mse_for_each_mts)
    ts_plot_of_worst_generated_mts = plot_time_series_for_all_uts(
        original_mts=original_mts[worst_generated_mts_index],
        target_mts=target_mts[worst_generated_mts_index],
        transformed_mts=generated_mts[worst_generated_mts_index],
        original_mts_features=original_mts_features[worst_generated_mts_index],
        transformed_mts_features=transformed_mts_features[worst_generated_mts_index],
        target_mts_features=target_mts_features[worst_generated_mts_index],
    )
    ts_plot_of_worst_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "worst_timeseries_generated_mts.png")
    )
    pca_plot_of_worst_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_features_train=mts_features_train,
        mts_features_validation=mts_features_validation,
        mts_features_test=mts_features_test,
        original_mts_features=original_mts_features[worst_generated_mts_index],
        target_mts_features=target_mts_features[worst_generated_mts_index],
        predicted_mts_features=transformed_mts_features[worst_generated_mts_index],
    )
    pca_plot_of_worst_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "worst_timeseries_generated_mts_pca.png")
    )

    median_mse = np.median(total_mse_for_each_mts)
    # NOTE: This way of finding the median index always returns one index. This is acceptable in this use case.
    median_mts_index = np.argmin(np.abs(total_mse_for_each_mts - median_mse))

    ts_plot_of_median_generated_mts = plot_time_series_for_all_uts(
        original_mts=original_mts[median_mts_index],
        target_mts=target_mts[median_mts_index],
        transformed_mts=generated_mts[median_mts_index],
        original_mts_features=original_mts_features[median_mts_index],
        transformed_mts_features=transformed_mts_features[median_mts_index],
        target_mts_features=target_mts_features[median_mts_index],
    )
    ts_plot_of_median_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "median_timeseries_generated_mts.png")
    )
    pca_plot_of_median_generated_mts = plot_pca_for_each_uts_with_transformed(
        mts_features_train=mts_features_train,
        mts_features_validation=mts_features_validation,
        mts_features_test=mts_features_test,
        original_mts_features=original_mts_features[median_mts_index],
        target_mts_features=target_mts_features[median_mts_index],
        predicted_mts_features=transformed_mts_features[median_mts_index],
    )
    pca_plot_of_median_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "median_timeseries_generated_mts_pca.png")
    )

    pca_total_plot = plot_pca_for_all_generated_mts(
        mts_features_train=mts_features_train,
        mts_features_validation=mts_features_validation,
        mts_features_test=mts_features_test,
        mts_generated_features=transformed_mts_features,
    )
    pca_total_plot.savefig(os.path.join(OUTPUT_DIR, "total_generation_pca.png"))
