import pandas as pd
import numpy as np


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

    # To avoid the target and original MTS being the same, we filter on this
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
