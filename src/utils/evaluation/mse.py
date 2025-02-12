import numpy as np
import pandas as pd
from src.utils.logging_config import logger


def get_mse_for_features_and_overall(differences_df):
    logger.info("Calculating MSE for each feature")

    # Excluding 'prediction_index' without explicit dropping
    feature_columns = differences_df.columns.difference(["prediction_index"])

    # Compute squared errors efficiently using NumPy
    squared_errors = differences_df[feature_columns].to_numpy() ** 2
    mse_values_for_each_feature = np.mean(squared_errors, axis=0)

    logger.info("Calculating overall MSE")
    overall_mse = np.mean(mse_values_for_each_feature)

    # Return as a series to maintain feature names
    mse_series = pd.Series(mse_values_for_each_feature, index=feature_columns)

    return overall_mse, mse_series
