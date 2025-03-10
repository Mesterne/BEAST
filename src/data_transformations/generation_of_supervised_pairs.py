import pandas as pd
import numpy as np
from random import sample
from src.data.constants import OUTPUT_DIR
from src.plots.pca_train_test_pairing import pca_plot_train_test_pairing
from src.utils.logging_config import logger
import os
from scipy.stats import zscore


def create_train_val_test_split(
    pca_df: np.ndarray, feature_df, FEATURES_NAMES, DELTA_NAMES, TARGET_NAMES, SEED
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

    # Convert the numpy array to a pandas DataFrame
    pca_df_converted = pd.DataFrame(
        pca_df, columns=["pca1", "pca2"]  # Assuming these are the columns in the array
    )

    # Add a sequential index column from 0 to len(pca_df)-1
    pca_df_converted["index"] = np.arange(len(pca_df_converted))

    # Now use the converted DataFrame for the filtering operations
    validation_indices = pca_df_converted[
        (pca_df_converted["pca1"] > 0.0)
        & (pca_df_converted["pca1"] < 0.2)
        & (pca_df_converted["pca2"] > 0.4)
    ]["index"].values

    test_indices = pca_df_converted[
        (pca_df_converted["pca1"] > 0.8) & (pca_df_converted["pca2"] > 0)
    ]["index"].values

    train_indices = pca_df_converted["index"][
        ~(
            pca_df_converted["index"].isin(test_indices)
            | pca_df_converted["index"].isin(validation_indices)
        )
    ].values

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

    dataset_row = test_supervised_dataset.sample(n=1, random_state=SEED).reset_index(
        drop=True
    )

    # Pass the converted DataFrame to the plotting function
    fig = pca_plot_train_test_pairing(pca_df_converted, dataset_row)
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_train_test_pairing.png"))
    logger.info("Generated PCA plot with target/test pairing")

    def generate_X_y_pairs_from_df(df):
        # Extract X as both original features and delta values
        original_features = df.loc[:, FEATURES_NAMES].values
        delta_features = df.loc[:, DELTA_NAMES].values

        # Combine original features and delta features horizontally
        X = np.hstack((original_features, delta_features))

        # Extract y (targets) as usual
        y = df.loc[:, TARGET_NAMES].values

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


def create_train_val_test_split_outliers(
    pca_df, feature_df, FEATURES_NAMES, DELTA_NAMES, TARGET_NAMES, SEED
):
    """
    Generate X and y lists for train, validation, and test supervised datasets.
    It detects outliers in PCA space using Z-scores and assigns them to validation and test sets.
    """
    logger.info("Generating X, y pairs for train, validation, and test sets...")

    # Compute Z-scores for each PCA component
    pca_df["z_pca1"] = zscore(pca_df["pca1"])
    pca_df["z_pca2"] = zscore(pca_df["pca2"])

    # Compute overall outlier score (e.g., Euclidean distance in Z-score space)
    pca_df["outlier_score"] = np.sqrt(pca_df["z_pca1"] ** 2 + pca_df["z_pca2"] ** 2)

    # Define outliers as points with a Z-score distance > threshold (e.g., 3 std devs)
    outlier_threshold = 2  # Adjustable threshold
    outliers = pca_df[pca_df["outlier_score"] > outlier_threshold].copy()
    logger.info(f"Found {len(outliers)} outliers")

    # Randomly split outliers into validation and test sets
    outlier_indices = outliers.index.values
    np.random.shuffle(outlier_indices)

    split_idx = len(outlier_indices) // 2
    validation_indices = outlier_indices[:split_idx]
    test_indices = outlier_indices[split_idx:]

    # Remaining points go to training set
    train_indices = pca_df.index[~pca_df.index.isin(outlier_indices)].values

    pca_df["isTrain"] = pca_df.index.isin(train_indices)
    pca_df["isValidation"] = pca_df.index.isin(validation_indices)
    pca_df["isTest"] = pca_df.index.isin(test_indices)

    train_features = feature_df.loc[train_indices]
    validation_features = feature_df.loc[validation_indices]
    test_features = feature_df.loc[test_indices]

    logger.info("Generating supervised training dataset...")
    train_supervised_dataset = (
        generate_supervised_dataset_from_original_and_target_dist(
            train_features, train_features
        )
    )
    logger.info("Generating supervised validation dataset...")
    validation_supervised_dataset = (
        generate_supervised_dataset_from_original_and_target_dist(
            train_features, validation_features
        )
    )
    logger.info("Generating supervised test dataset...")
    test_supervised_dataset = generate_supervised_dataset_from_original_and_target_dist(
        train_features, test_features
    )

    dataset_row = test_supervised_dataset.sample(n=1, random_state=SEED).reset_index(
        drop=True
    )
    fig = pca_plot_train_test_pairing(pca_df, dataset_row)
    fig.savefig(os.path.join(OUTPUT_DIR, "pca_train_test_pairing.png"))
    logger.info("Generated PCA plot with target/test pairing")

    def generate_X_y_pairs_from_df(df):
        # Extract X as both original features and delta values
        original_features = df.loc[:, FEATURES_NAMES].values
        delta_features = df.loc[:, DELTA_NAMES].values

        # Combine original features and delta features horizontally
        X = np.hstack((original_features, delta_features))

        # Extract y (targets) as usual
        y = df.loc[:, TARGET_NAMES].values

        return X, y

    logger.info("Generating X, y pairs for training dataset...")
    X_train, y_train = generate_X_y_pairs_from_df(train_supervised_dataset)
    logger.info("Generating X, y pairs for validation dataset...")
    X_validation, y_validation = generate_X_y_pairs_from_df(
        validation_supervised_dataset
    )
    logger.info("Generating X, y pairs for test dataset...")
    X_test, y_test = generate_X_y_pairs_from_df(test_supervised_dataset)

    logger.info(
        f"Generated X, y pairs with shapes:\n"
        f"X_train: {X_train.shape}, y_train: {y_train.shape}\n"
        f"X_validation: {X_validation.shape}, y_validation: {y_validation.shape}\n"
        f"X_test: {X_test.shape}, y_test: {y_test.shape}"
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

    # Compute delta columns
    for col in orig_copy.columns:
        if col.startswith("original_"):
            target_col = "target_" + col[len("original_") :]
            delta_col = "delta_" + col[len("original_") :]
            dataset[delta_col] = dataset[target_col] - dataset[col]

    delta_columns = [col for col in dataset.columns if col.startswith("delta_")]

    # Define groups of delta columns
    load_columns = [col for col in delta_columns if col.startswith("delta_grid1-load")]
    loss_columns = [col for col in delta_columns if col.startswith("delta_grid1-loss")]
    temp_columns = [col for col in delta_columns if col.startswith("delta_grid1-temp")]

    grouped_columns = [load_columns, loss_columns, temp_columns]

    # Create rows by activating each group separately
    expanded_rows = []
    for index, row in dataset.iterrows():
        for group in grouped_columns:
            new_row = row.copy()
            # Set all delta columns to 0
            for col in delta_columns:
                new_row[col] = 0
            # Activate all columns in the current group
            for col in group:
                new_row[col] = row[col]
            expanded_rows.append(new_row)

    # Create a new DataFrame from the expanded rows
    expanded_dataset = pd.DataFrame(expanded_rows).reset_index(drop=True)

    return expanded_dataset


def get_col_names_original_target_delta():
    ORIGNIAL_COLS = [
        "original_index",
        "original_grid1-load_trend-strength",
        "original_grid1-load_trend-slope",
        "original_grid1-load_trend-linearity",
        "original_grid1-load_seasonal-strength",
        "original_grid1-loss_trend-strength",
        "original_grid1-loss_trend-slope",
        "original_grid1-loss_trend-linearity",
        "original_grid1-loss_seasonal-strength",
        "original_grid1-temp_trend-strength",
        "original_grid1-temp_trend-slope",
        "original_grid1-temp_trend-linearity",
        "original_grid1-temp_seasonal-strength",
    ]

    TARGET_COLS = [
        "target_index",
        "target_grid1-load_trend-strength",
        "target_grid1-load_trend-slope",
        "target_grid1-load_trend-linearity",
        "target_grid1-load_seasonal-strength",
        "target_grid1-loss_trend-strength",
        "target_grid1-loss_trend-slope",
        "target_grid1-loss_trend-linearity",
        "target_grid1-loss_seasonal-strength",
        "target_grid1-temp_trend-strength",
        "target_grid1-temp_trend-slope",
        "target_grid1-temp_trend-linearity",
        "target_grid1-temp_seasonal-strength",
    ]

    DELTA_COLS = [
        "delta_index",
        "delta_grid1-load_trend-strength",
        "delta_grid1-load_trend-slope",
        "delta_grid1-load_trend-linearity",
        "delta_grid1-load_seasonal-strength",
        "delta_grid1-loss_trend-strength",
        "delta_grid1-loss_trend-slope",
        "delta_grid1-loss_trend-linearity",
        "delta_grid1-loss_seasonal-strength",
        "delta_grid1-temp_trend-strength",
        "delta_grid1-temp_trend-slope",
        "delta_grid1-temp_trend-linearity",
        "delta_grid1-temp_seasonal-strength",
    ]

    return [*ORIGNIAL_COLS, *TARGET_COLS, *DELTA_COLS]


def get_col_names_original_target():
    ORIGINAL_NAMES = [
        "original_grid1-load_trend-strength",
        "original_grid1-load_trend-slope",
        "original_grid1-load_trend-linearity",
        "original_grid1-load_seasonal-strength",
        "original_grid1-loss_trend-strength",
        "original_grid1-loss_trend-slope",
        "original_grid1-loss_trend-linearity",
        "original_grid1-loss_seasonal-strength",
        "original_grid1-temp_trend-strength",
        "original_grid1-temp_trend-slope",
        "original_grid1-temp_trend-linearity",
        "original_grid1-temp_seasonal-strength",
    ]

    DELTA_NAMES = [
        "delta_grid1-load_trend-strength",
        "delta_grid1-load_trend-slope",
        "delta_grid1-load_trend-linearity",
        "delta_grid1-load_seasonal-strength",
        "delta_grid1-loss_trend-strength",
        "delta_grid1-loss_trend-slope",
        "delta_grid1-loss_trend-linearity",
        "delta_grid1-loss_seasonal-strength",
        "delta_grid1-temp_trend-strength",
        "delta_grid1-temp_trend-slope",
        "delta_grid1-temp_trend-linearity",
        "delta_grid1-temp_seasonal-strength",
    ]

    TARGET_NAMES = [
        "target_grid1-load_trend-strength",
        "target_grid1-load_trend-slope",
        "target_grid1-load_trend-linearity",
        "target_grid1-load_seasonal-strength",
        "target_grid1-loss_trend-strength",
        "target_grid1-loss_trend-slope",
        "target_grid1-loss_trend-linearity",
        "target_grid1-loss_seasonal-strength",
        "target_grid1-temp_trend-strength",
        "target_grid1-temp_trend-slope",
        "target_grid1-temp_trend-linearity",
        "target_grid1-temp_seasonal-strength",
    ]

    return ORIGINAL_NAMES, DELTA_NAMES, TARGET_NAMES
