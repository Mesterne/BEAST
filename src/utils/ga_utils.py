from typing import List, Tuple

import numpy as np


def generate_new_time_series(
    original_indices: List[int],
    predicted_features: np.ndarray,  # Shape: (num_mts, num_uts_in_mts, num_features_per_uts)
    reconstruction_model,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Analyze a prediction, run the reconstruction model on it

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
    (
        transformed_mts,
        transformed_features,
    ) = reconstruction_model.transform(
        predicted_features=predicted_features,
        original_mts_indices=original_indices,
    )

    return np.array(transformed_mts), transformed_features
