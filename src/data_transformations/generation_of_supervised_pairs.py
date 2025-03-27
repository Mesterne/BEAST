import random
from typing import Dict, List

import numpy as np
from tqdm import tqdm

from src.plots.feature_distribution import plot_feature_distribution
from src.utils.generate_dataset import generate_feature_dataframe
from src.utils.logging_config import logger
from src.utils.pca import PCAWrapper


def create_train_val_test_split(
    mts_dataset_array: np.ndarray,
    config: Dict[str, any],
):
    """ """
    number_of_transformations_in_test_set = config["dataset_args"][
        "test_set_sample_size"
    ]
    logger.info(f"Generating transformation index pairs for training...")
    mts_features_array, mts_decomps = generate_feature_dataframe(
        data=mts_dataset_array,
        series_periodicity=config["stl_args"]["series_periodicity"],
        num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
    )

    dist_of_features = plot_feature_distribution(mts_features_array)
    dist_of_features.savefig("distribution_of_features.png")

    mts_pca_array: np.ndarray = PCAWrapper().fit_transform(mts_features_array)
    logger.info("Successfully generated MTS PCA space")

    logger.info("Splitting PCA space into train, validation and test indices...")
    pca1, pca2 = mts_pca_array[:, 0], mts_pca_array[:, 1]
    indices = np.arange(len(mts_pca_array))

    validation_indices = indices[(pca1 > 0.1) & (pca2 > 0)]
    test_indices = indices[(pca1 < 0.1) & (pca2 > 0)]
    train_indices = np.setdiff1d(
        indices, np.concatenate([validation_indices, test_indices])
    )

    logger.info("Pairing training transformations")
    train_transformation_indices: List[Tuple[int, int]] = []
    for i in tqdm(train_indices):
        for j in train_indices:
            train_transformation_indices.append((i, j))
    logger.info("Pairing validation transformations")
    validation_transformation_indices: List[Tuple[int, int]] = []
    for i in tqdm(train_indices):
        for j in validation_indices:
            validation_transformation_indices.append((i, j))
    logger.info("Pairing test transformations")
    test_transformation_indices: List[Tuple[int, int]] = []
    for i in tqdm(train_indices):
        for j in test_indices:
            test_transformation_indices.append((i, j))

    assert len(train_transformation_indices) > 0, "Training set must have elements"
    assert (
        len(validation_transformation_indices) > number_of_transformations_in_test_set
    ), "Validation set must have more than number of defined samples in evaluation set entries."

    assert (
        len(test_transformation_indices) > number_of_transformations_in_test_set
    ), "Test set must have more than number of defined samples in evaluation set entries."
    validation_transformation_indices = random.sample(
        validation_transformation_indices, number_of_transformations_in_test_set
    )
    test_transformation_indices = random.sample(
        test_transformation_indices, number_of_transformations_in_test_set
    )
    return (
        np.array(train_transformation_indices),
        np.array(validation_transformation_indices),
        np.array(test_transformation_indices),
    )


def concat_delta_values_to_features(
    X_features: np.ndarray,
    y_features: np.ndarray,
    use_one_hot_encoding: bool,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
):
    expanded_x_rows: List = []
    expanded_y_rows: List = []
    for row_index, _ in tqdm(enumerate(X_features), total=len(X_features)):
        X_row = X_features[row_index]
        y_row = y_features[row_index]
        for uts_index in range(0, number_of_uts_in_mts):
            new_row = X_row.copy()
            start_index = uts_index * number_of_uts_in_mts
            end_index = start_index + number_of_features_in_mts
            if not use_one_hot_encoding:
                delta = [0] * len(y_row)
                delta[start_index:end_index] = (
                    y_row[start_index:end_index] - X_row[start_index:end_index]
                )
                new_row = np.concatenate((new_row, delta))
            else:
                delta = y_row[start_index:end_index] - X_row[start_index:end_index]
                new_row = np.concatenate((new_row, delta))

            if use_one_hot_encoding:
                one_hot_encoding = [0] * number_of_uts_in_mts
                one_hot_encoding[uts_index] = 1
                new_row = np.concatenate((new_row, one_hot_encoding))
            expanded_x_rows.append(new_row)
            expanded_y_rows.append(y_row)

    return np.array(expanded_x_rows), np.array(expanded_y_rows)


# Pick random UTS to deactivate. Done to prevent explosion in validation, test set
def concat_delta_values_to_features_for_inference(
    X_features: np.ndarray,
    y_features: np.ndarray,
    use_one_hot_encoding: bool,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
):
    expanded_x_rows: List = []
    expanded_y_rows: List = []
    for row_index, _ in tqdm(enumerate(X_features), total=len(X_features)):
        X_row = X_features[row_index]
        y_row = y_features[row_index]
        uts_index = np.random.choice(range(0, number_of_uts_in_mts))
        new_row = X_row.copy()
        start_index = uts_index * number_of_uts_in_mts
        end_index = start_index + number_of_features_in_mts
        if not use_one_hot_encoding:
            delta = [0] * len(y_row)
            delta[start_index:end_index] = (
                y_row[start_index:end_index] - X_row[start_index:end_index]
            )
            new_row = np.concatenate((new_row, delta))
        else:
            delta = y_row[start_index:end_index] - X_row[start_index:end_index]
            new_row = np.concatenate((new_row, delta))

        if use_one_hot_encoding:
            one_hot_encoding = [0] * number_of_uts_in_mts
            one_hot_encoding[uts_index] = 1
            new_row = np.concatenate((new_row, one_hot_encoding))
        expanded_x_rows.append(new_row)
        expanded_y_rows.append(y_row)

    return np.array(expanded_x_rows), np.array(expanded_y_rows)
