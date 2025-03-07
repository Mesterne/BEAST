import os
import numpy as np
from src.data.constants import OUTPUT_DIR
from src.plots.feature_wise_error import plot_distribution_of_feature_wise_error
from src.plots.full_time_series import plot_time_series_for_all_uts


def create_and_save_plots_of_model_performances(
    total_mse_for_each_uts: np.array,
    mse_per_feature: np.array,
    original_mts: np.array,
    target_mts: np.array,
    generated_mts: np.array,
    original_mts_features: np.array,
    transformed_mts_features: np.array,
    target_mts_features: np.array,
):
    feature_wise_errror_plot = plot_distribution_of_feature_wise_error(
        mse_per_feature=mse_per_feature
    )
    feature_wise_errror_plot.savefig(
        os.path.join(OUTPUT_DIR, "distribution_of_feature_wise_error.png")
    )

    best_generated_mts_index = np.argmin(total_mse_for_each_uts)
    ts_plot_of_best_generated_mts = plot_time_series_for_all_uts(
        original_mts=original_mts[best_generated_mts_index],
        target_mts=target_mts[best_generated_mts_index],
        transformed_mts=generated_mts[best_generated_mts_index],
        original_mts_features=original_mts_features[best_generated_mts_index],
        transformed_mts_features=transformed_mts_features[best_generated_mts_index],
        target_mts_features=target_mts_features[best_generated_mts_index],
    )
    ts_plot_of_best_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "best_timeseries_genereated_mts.png")
    )

    worst_generated_mts_index = np.argmax(total_mse_for_each_uts)
    ts_plot_of_worst_generated_mts = plot_time_series_for_all_uts(
        original_mts=original_mts[worst_generated_mts_index],
        target_mts=target_mts[worst_generated_mts_index],
        transformed_mts=generated_mts[worst_generated_mts_index],
        original_mts_features=original_mts_features[worst_generated_mts_index],
        transformed_mts_features=transformed_mts_features[worst_generated_mts_index],
        target_mts_features=target_mts_features[worst_generated_mts_index],
    )
    ts_plot_of_worst_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "worst_timeseries_genereated_mts.png")
    )

    random_generated_mts_index = np.random.randint(len(total_mse_for_each_uts))
    ts_plot_of_random_generated_mts = plot_time_series_for_all_uts(
        original_mts=original_mts[random_generated_mts_index],
        target_mts=target_mts[random_generated_mts_index],
        transformed_mts=generated_mts[random_generated_mts_index],
        original_mts_features=original_mts_features[random_generated_mts_index],
        transformed_mts_features=transformed_mts_features[random_generated_mts_index],
        target_mts_features=target_mts_features[random_generated_mts_index],
    )
    ts_plot_of_random_generated_mts.savefig(
        os.path.join(OUTPUT_DIR, "random_timeseries_genereated_mts.png")
    )
