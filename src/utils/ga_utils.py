import os
from typing import List, Tuple
from src.plots.full_time_series import plot_time_series_for_all_uts
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)
import pandas as pd
import numpy as np


def generate_new_time_series(
    original_indices: List[int],
    predicted_features: np.ndarray,  # Shape: (num_mts, num_uts_in_mts, num_features_per_uts)
    ga,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze a prediction, run the genetic algorithm on it, and create visualizations.

    Args:
        prediction_index: Index of the prediction to analyze
        validation_supervised_dataset: DataFrame containing supervised dataset
        validation_predicted_features: DataFrame containing model predictions
        mts_dataset: Dataset containing multivariate time series
        mts_decomps: Decompositions of the time series
        mts_feature_df: DataFrame with features of the time series
        ga: GeneticAlgorithmWrapper instance
        uts_names: Names of the univariate time series in the MTS
        output_dir: Directory to save plots
        plot_name_prefix: Prefix to add to the plot filenames (e.g., "worst_", "best_")

    Returns:
        tuple: (original_mts, target_mts, transformed_mts, transformed_features)
    """
    # Get the target row from the validation dataset

    # Run genetic algorithm transformation
    (
        transformed_mts,
        transformed_features,
        _,
        _,
    ) = ga.transform(
        predicted_features=predicted_features,
        original_mts_indices=original_indices,
    )

    return np.array(transformed_mts), transformed_features
