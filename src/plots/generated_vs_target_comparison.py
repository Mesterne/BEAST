import os
import numpy as np
from src.data.constants import OUTPUT_DIR
from src.plots.feature_wise_error import plot_distribution_of_feature_wise_error


def create_and_save_plots_of_model_performances(
    total_mse_for_each_uts: np.array, mse_per_feature: np.array
):
    feature_wise_errror_plot = plot_distribution_of_feature_wise_error(
        mse_per_feature=mse_per_feature
    )
    feature_wise_errror_plot.savefig(
        os.path.join(OUTPUT_DIR, "distrubtion_of_feature_wise_mse.png")
    )
