from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.constants import COLUMN_NAMES, FEATURE_NAMES, UTS_NAMES
from src.utils.logging_config import logger


def create_train_val_test_split(
    pca_array: np.ndarray,  # Shape (number of mts, 2)
    mts_feature_array: np.ndarray,  # Shape (number of mts, number of uts * number of features in uts)
):
    """
    Generate X and y list for train, validation and testing supervised datasets.
    It does this by selecting certain regions in the PCA space which is looked at as
    out of distribution.
    Args:
        pca_df (np.ndarray): The PCA data as a numpy array, with columns for pca1 and pca2
        feature_df (pd.DataFrame): The feature dataframe, having the features of all MTSs in the dataset!
    Returns:
        X_train
        y_train
        X_validation
        y_validation
        X_test
        y_test
        These are the numpy arrays of feature/target pairs for train, validation and test
    """
    logger.info(
        f"Generating X,y pairs of feature space for train, validation and test sets..."
    )

    feature_df: pd.DataFrame = pd.DataFrame(mts_feature_array, columns=COLUMN_NAMES)
    pca_df_converted = pd.DataFrame(pca_array, columns=["pca1", "pca2"])

    # Add a sequential index column from 0 to len(pca_df)-1
    pca_df_converted["index"] = np.arange(len(pca_df_converted))

    validation_indices = pca_df_converted[
        (pca_df_converted["pca1"] > 0.1) & (pca_df_converted["pca2"] > 0)
    ]["index"].values

    # Now use the converted DataFrame for the filtering operations
    test_indices = pca_df_converted[
        (pca_df_converted["pca1"] < 0.1) & (pca_df_converted["pca2"] > 0.0)
    ]["index"].values

    train_indices = pca_df_converted["index"][
        ~(
            pca_df_converted["index"].isin(test_indices)
            | pca_df_converted["index"].isin(validation_indices)
        )
    ].values

    assert (
        len(train_indices) > 0
    ), "Train set must be larger than 0. Check your sampling techniques"
    assert (
        len(validation_indices) > 0
    ), "Validation set must be larger than 0. Check your sampling techniques"
    assert (
        len(test_indices) > 0
    ), "Test set must be larger than 0. Check your sampling techniques"

    # Add the train/validation/test indicators to the DataFrame
    pca_df_converted["isTrain"] = pca_df_converted["index"].isin(train_indices)
    pca_df_converted["isValidation"] = pca_df_converted["index"].isin(
        validation_indices
    )
    pca_df_converted["isTest"] = pca_df_converted["index"].isin(test_indices)

    # Continue with the rest of the function using the converted DataFrame
    train_features = feature_df[feature_df.index.isin(train_indices)]
    validation_features = feature_df[feature_df.index.isin(validation_indices)]
    test_features = feature_df[feature_df.index.isin(test_indices)]

    # To generate a training set, we create a matching between all MTSs in the
    # defined training feature space
    logger.info(f"Generating supervised training dataset...")
    train_supervised_dataset = (
        generate_supervised_dataset_from_original_and_target_dist(
            train_features, train_features
        )
    )

    logger.info(f"Generating supervised validation dataset...")
    validation_supervised_dataset = (
        generate_supervised_dataset_from_original_and_target_dist(
            train_features, validation_features
        )
    )

    logger.info(f"Generating supervised test dataset...")
    test_supervised_dataset = generate_supervised_dataset_from_original_and_target_dist(
        train_features, test_features
    )

    def generate_X_y_pairs_from_df(df):
        # Prefix COLUMN_NAMES with original_ and delta_
        original_features_names: List[str] = [
            f"original_{name}" for name in COLUMN_NAMES
        ]
        delta_names: List[str] = [f"delta_{name}" for name in FEATURE_NAMES]
        one_hot_encoded_names: List[str] = [
            f"{uts_name}_is_delta" for uts_name in UTS_NAMES
        ]
        target_names: List[str] = [f"target_{name}" for name in COLUMN_NAMES]

        # Extract X as both original features and delta values
        original_features = df.loc[:, original_features_names].values
        one_hot_encoding = df.loc[:, one_hot_encoded_names].values
        delta_features = df.loc[:, delta_names].values

        # Combine original features and delta features horizontally
        X = np.hstack((original_features, delta_features, one_hot_encoding))

        # Extract y (targets) as usual
        y = df.loc[:, target_names].values

        return X, y

    logger.info(f"Generating X,y pairs for training dataset...")
    X_train, y_train = generate_X_y_pairs_from_df(train_supervised_dataset)
    logger.info(f"Generating X,y pairs for validation dataset...")

    X_validation, y_validation = generate_X_y_pairs_from_df(
        validation_supervised_dataset
    )
    logger.info(f"Generating X,y pairs for test dataset...")
    X_test, y_test = generate_X_y_pairs_from_df(test_supervised_dataset)
    logger.info(
        f""" Generated X, y pairs for training, test and validation. With shapes:
            X_training: {X_train.shape}
            y_training: {y_train.shape}
            X_validation: {X_validation.shape}
            y_validation: {y_validation.shape}
            X_test: {X_test.shape}
            y_test: {y_test.shape}
    """
    )
    return (
        X_train,
        y_train,
        X_validation,
        y_validation,
        X_test,
        y_test,
        train_supervised_dataset,
        validation_supervised_dataset,
        test_supervised_dataset,
    )


def generate_supervised_dataset_from_original_and_target_dist(
    original_distribution, target_distribution
):
    """
    Generate a supervised dataset by creating all possible pairwise combinations between
    an original distribution and a target distribution, computing deltas between paired features,
    and simulating user change behavior by activating specific groups of delta columns.

    Args:
        original_distribution (pd.DataFrame): The DataFrame representing the original distribution.
            Each row corresponds to an instance, and columns represent features.
        target_distribution (pd.DataFrame): The DataFrame representing the target distribution
            with the same feature structure as `original_distribution`.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - Features from both the original and target distributions, prefixed by `original_` and `target_`.
            - Delta columns showing the difference between paired features, prefixed by `delta_`.
            - For each row, specific groups of `delta_` columns retain non-zero values while others are set to zero.

    Notes:
        - The original and target indices are reset and renamed to avoid conflicts during merging.
        - The function filters out rows where the original and target indices are identical.
        - Cross joins between distributions are used to create all possible pairs.
        - Activates grouped delta columns together based on specific prefixes.
    """
    # Copy distributions to avoid inplace alteration and add prefixes to columns
    orig_copy = original_distribution.copy().add_prefix("original_")
    target_copy = target_distribution.copy().add_prefix("target_")

    # Reset and rename indices to avoid conflicts during merging
    orig_copy.reset_index(inplace=True)
    target_copy.reset_index(inplace=True)
    orig_copy.rename(columns={"index": "original_index"}, inplace=True)
    target_copy.rename(columns={"index": "target_index"}, inplace=True)

    # Perform cross join to create all possible pairs
    dataset = pd.merge(orig_copy, target_copy, how="cross")

    # Remove pairs where the original and target indices are identical
    dataset = dataset[dataset["original_index"] != dataset["target_index"]]

    delta_feature_columns = [f"delta_{feature}" for feature in FEATURE_NAMES]

    # For each row, we create 3 supervised rows - One for each uts activated
    # For each uts row, we compute the delta of the
    expanded_rows: List = []
    for _, row in tqdm(dataset.iterrows(), total=len(dataset)):
        for uts_name in UTS_NAMES:
            new_row = row.copy()
            # For each uts activated iterate through features and calculate delta
            for delta_feature in delta_feature_columns:
                original_col = f"original_{uts_name}_{delta_feature[len('delta_') :]}"
                target_col = f"target_{uts_name}_{delta_feature[len('delta_') :]}"
                new_row[delta_feature] = row[target_col] - row[original_col]
            for uts in UTS_NAMES:
                if uts_name == uts:
                    new_row[f"{uts}_is_delta"] = 1
                else:
                    new_row[f"{uts}_is_delta"] = 0
            expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    expanded_dataset = pd.DataFrame(expanded_rows).reset_index(drop=True)

    return expanded_dataset
