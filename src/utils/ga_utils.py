import os
from typing import List, Tuple
from src.plots.full_time_series import plot_time_series_for_all_uts
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)
import pandas as pd
import numpy as np


def analyze_and_visualize_prediction(
    prediction_index,
    supervised_dataset,
    predicted_features,
    mts_dataset,
    mts_decomps,
    mts_feature_df,
    ga,
    uts_names,
    output_dir,
    plot_name_prefix="",
):
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
    row_in_validation = supervised_dataset.iloc[prediction_index]
    original_mts_index = int(row_in_validation["original_index"])
    target_mts_index = int(row_in_validation["target_index"])

    # Get the predicted features for this sample
    predicted_features = predicted_features[
        predicted_features["prediction_index"] == prediction_index
    ].drop(["prediction_index"], axis=1)

    # Get original and target MTS
    original_mts = mts_dataset[original_mts_index]
    original_mts_decomps = mts_decomps[original_mts_index]
    original_mts_features = mts_feature_df.iloc[original_mts_index]

    target_mts = mts_dataset[target_mts_index]
    target_mts_features = mts_feature_df.iloc[target_mts_index]

    # Run genetic algorithm transformation
    (
        transformed_mts,
        transformed_features,
        _,
        _,
    ) = ga.transform(
        predicted_features=predicted_features,
        original_mts_indices=[original_mts_index],
    )

    # Flatten and format the transformed features
    transformed_features = np.array(transformed_features).reshape(-1, 12)

    transformed_features = pd.DataFrame(
        transformed_features, columns=predicted_features.columns
    )

    # Create PCA visualization
    uts_wise_pca_fig = plot_pca_for_each_uts_with_transformed(
        mts_features_df=mts_feature_df,
        predicted_features=predicted_features,
        transformed_features=transformed_features,
        original_index=original_mts_index,
        target_index=target_mts_index,
        uts_names=uts_names,
    )
    uts_wise_pca_fig.savefig(
        os.path.join(output_dir, f"{plot_name_prefix}uts_wise_pca.png")
    )

    # Format transformed MTS
    transformed_mts = pd.DataFrame(
        # We index 0, since it only creates 1 time series in this case
        {name: transformed_mts[0][i] for i, name in enumerate(original_mts.columns)}
    )

    # Create time series visualization
    full_time_series_fig = plot_time_series_for_all_uts(
        original_mts=original_mts,
        target_mts=target_mts,
        transformed_mts=transformed_mts,
        original_mts_features=original_mts_features,
        target_mts_features=target_mts_features,
        transformed_mts_features=transformed_features,
    )
    full_time_series_fig.savefig(
        os.path.join(output_dir, f"{plot_name_prefix}full_time_series.png")
    )

    return original_mts, target_mts, transformed_mts, transformed_features


def generate_new_time_series(
    supervised_dataset,
    predicted_features,
    ga,
) -> Tuple[List, List]:
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
    original_mts_indices = supervised_dataset["original_index"].astype(int)

    # Run genetic algorithm transformation
    (
        transformed_mts,
        transformed_features,
        _,
        _,
    ) = ga.transform(
        predicted_features=predicted_features,
        original_mts_indices=original_mts_indices,
    )

    return np.array(transformed_mts), transformed_features
