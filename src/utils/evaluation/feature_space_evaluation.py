import numpy as np
import pandas as pd
from tqdm import tqdm


def calculate_evaluation_for_each_feature(
    predicted_features: np.ndarray, target_features: np.ndarray, metric: str
):
    """
    Calculate the Mean Squared Error (MSE) for each feature for each timeseries.

    The MSE is computed for each feature by comparing the predicted values against the target values
    for each timeseries, and the function returns the MSE for each feature for each timeseries.

    Args:
    predicted_features (np.array): A NumPy array of shape (num_timeseries, 12) representing the predicted feature values.
    target_features (np.array): A NumPy array of shape (num_timeseries, 12) representing the target feature values.

    Returns:
    np.array: A NumPy array of shape (num_timeseries, 12) where each entry corresponds to the MSE for a specific feature for that specific timeseries.
    """

    assert metric in [
        "MSE",
        "MAE",
    ], "Metric must be either MSE or MAE for feature space evaluations"
    if metric == "MSE":
        # Calculate the squared difference between predicted and target values
        squared_diff = (predicted_features - target_features) ** 2

        # The MSE for each feature for each timeseries is the squared difference itself
        return squared_diff

    if metric == "MAE":
        return np.abs(predicted_features - target_features)

    return (predicted_features - target_features) ** 2


def calculate_total_evaluation_for_each_mts(
    predicted_features: np.ndarray, target_features: np.ndarray, metric: str
):
    """
    Calculate the total Mean Squared Error (MSE) for each multi-timeseries (MTS).

    This function sums the MSE for all features for each timeseries to compute the total MSE for each MTS.

    Args:
    mse_per_feature (np.array): A numpy array of shape (num_timeseries, 12) representing the MSE for each feature for each timeseries.

    Returns:
    np.array: A numpy array of shape (num_timeseries,), where each entry is the total MSE for each generated MTS' features.
    """
    eval_for_each_feature = calculate_evaluation_for_each_feature(
        predicted_features=predicted_features,
        target_features=target_features,
        metric=metric,
    )

    # TODO: Diskuter dette.
    # Sum the MSE across the feature axis (axis 1)
    total_mse = np.sum(eval_for_each_feature, axis=1)

    return total_mse


def find_error_of_each_feature_for_each_sample(predictions, labelled_test_dataset):
    # Initialize an empty list to store differences
    all_differences = []

    for idx, row in tqdm(predictions.iterrows(), total=len(predictions)):
        prediction_idx = int(row["prediction_index"])

        target_row = labelled_test_dataset.loc[prediction_idx]

        prediction_columns = [
            col
            for col in predictions.columns
            if not col.startswith(("pca", "prediction_index"))
        ]
        target_columns = [col for col in target_row.index if col.startswith("target_")]

        target_row_filtered = target_row[target_columns]
        target_row_filtered.index = target_row_filtered.index.str.replace(
            "^target_", "", regex=True
        )

        differences = {
            col: abs(row[col] - target_row_filtered[col])
            for col in prediction_columns
            if col in target_row_filtered.index
        }
        differences["prediction_index"] = prediction_idx
        all_differences.append(differences)

    differences_df = pd.DataFrame(all_differences)
    return differences_df
