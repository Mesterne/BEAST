import logging
import pandas as pd
import numpy as np
from src.plots.pca_train_test_pairing import pca_plot_train_test_pairing

def create_train_val_test_split(pca_df, feature_df, FEATURES_NAMES, TARGET_NAMES, SEED):
    """
    Generate X and y list for train, validation and testing supervised datasets.
    It does this by selecting certain regions in the PCA space which is looked at as
    out of distribution.
    Args:
        feature_df (pd.DataFrame): The feature dataframe, haing the features of all MTSs in the dataset!
    Returns:
        X_train
        y_train
        X_validation
        y_validation
        X_test
        y_test
        These are the numpy arrays of feature/target pairs for train, validation and test
    """
    logging.info(f'Generating X,y pairs of feature space for train, validation and test sets...')

    validation_indices = pca_df[
        (pca_df["pca1"] > 0.0)
        & (pca_df["pca1"] < 0.2)
        & (pca_df["pca2"] > 0.4)
    ]["index"].values
    test_indices = pca_df[(pca_df["pca1"] > 0.8) & (pca_df["pca2"] > 0)][
        "index"
    ].values
    train_indices = pca_df["index"][
        ~(
            pca_df["index"].isin(test_indices)
            & pca_df["index"].isin(validation_indices)
        )
    ].values

    pca_df["isTrain"] = pca_df["index"].isin(train_indices)
    pca_df["isValidation"] = pca_df["index"][
        pca_df["index"].isin(validation_indices)
    ]
    pca_df["isTest"] = pca_df["index"][pca_df["index"].isin(test_indices)]

    train_features = feature_df[feature_df.index.isin(train_indices)]
    validation_features = feature_df[feature_df.index.isin(validation_indices)]
    test_features = feature_df[feature_df.index.isin(test_indices)]


    # To generate a training set, we create a matching between all MTSs in the
    # defined training feature space
    logging.info(f'Generating supervised training dataset...')
    train_supervised_dataset = (
        generate_supervised_dataset_from_original_and_target_dist(
            train_features, train_features
        )
    )
    logging.info(f'Generating supervised validation dataset...')
    validation_supervised_dataset = (
        generate_supervised_dataset_from_original_and_target_dist(
            train_features, validation_features
        )
    )
    logging.info(f'Generating supervised test dataset...')
    test_supervised_dataset = generate_supervised_dataset_from_original_and_target_dist(
        train_features, test_features
    )

    dataset_row = test_supervised_dataset.sample(n=1, random_state=SEED).reset_index(
        drop=True
    )
    fig = pca_plot_train_test_pairing(pca_df, dataset_row)
    fig.write_html('pca_train_test_pairing.html')
    logging.info("Generated PCA plot with target/test pairing")

    def generate_X_y_pairs_from_df(df):
        # Extract X and y as NumPy arrays (if needed by the model)
        X = df.loc[:, FEATURES_NAMES].values
        y = df.loc[:, TARGET_NAMES].values
 
        return X, y

    logging.info(f'Generating X,y pairs for training dataset...')
    X_train, y_train = generate_X_y_pairs_from_df(train_supervised_dataset)
    logging.info(f'Generating X,y pairs for validation dataset...')
    X_validation, y_validation = generate_X_y_pairs_from_df(
        validation_supervised_dataset
    )
    logging.info(f'Generating X,y pairs for test dataset...')
    X_test, y_test = generate_X_y_pairs_from_df(test_supervised_dataset)
    logging.info(
        f""" Generated X, y pairs for training, test and validation. With shapes:
            X_training: {X_train.shape}         
            y_training: {y_train.shape}         
            \nX_validation: {X_validation.shape}         
            y_validation: {y_validation.shape}         
            \nX_test: {X_test.shape}         
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
    and simulating user change behavior by selecting one random delta column for each pair.

    Args:
        original_distribution (pd.DataFrame): The DataFrame representing the original distribution.
            Each row corresponds to an instance, and columns represent features.
        target_distribution (pd.DataFrame): The DataFrame representing the target distribution
            with the same feature structure as `original_distribution`.

    Returns:
        pd.DataFrame: A DataFrame containing:
            - Features from both the original and target distributions, prefixed by `original_` and `target_`.
            - Delta columns showing the difference between paired features, prefixed by `delta_`.
            - For each row, only one `delta_` column retains a non-zero value to simulate user changes.

    Notes:
        - The original and target indices are reset and renamed to avoid conflicts during merging.
        - The function filters out rows where the original and target indices are identical.
        - Cross joins between distributions are used to create all possible pairs.
        - Randomly selects one delta column per row and sets other deltas to zero.
    """
    # We copy the distributions, to avoid inplace alteration. Also add prefix
    # to each column
    orig_copy = original_distribution.copy().add_prefix("original_")
    target_copy = target_distribution.copy().add_prefix("target_")

    # Fix index of the original and target distribution
    orig_copy.reset_index(inplace=True)
    target_copy.reset_index(inplace=True)
    orig_copy.rename(columns={"index": "original_index"}, inplace=True)
    target_copy.rename(columns={"index": "target_index"}, inplace=True)

    # This is where we match original distribution with target distribution using
    # cross join.
    dataset = pd.merge(orig_copy, target_copy, how="cross")

    dataset = dataset[dataset["original_index"] != dataset["target_index"]]

    for col in orig_copy.columns:
        if col.startswith("original_"):
            target_col = "target_" + col[len("original_") :]
            delta_col = "delta_" + col[len("original_") :]
            dataset[delta_col] = dataset[target_col] - dataset[col]

    delta_columns = [col for col in dataset.columns if col.startswith("delta_")]

    # Finally we choose a random column to simulate the 'user changes on'
    for index, _ in dataset.iterrows():
        # Randomly choose one delta column to keep
        chosen_delta_column = np.random.choice(delta_columns)

        # Set all other delta columns to 0
        for delta_col in delta_columns:
            if delta_col != chosen_delta_column:
                dataset.at[index, delta_col] = 0

    return dataset.drop(columns=["delta_index"])
