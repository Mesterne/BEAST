import os
import random
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgba_array
from tqdm import tqdm

from src.data.constants import OUTPUT_DIR
from src.plots.feature_distribution import plot_feature_distribution
from src.plots.plot_train_val_split import plot_train_val_test_split
from src.plots.plot_transformation_directions import \
    plot_transformation_directions
from src.utils.generate_dataset import generate_feature_dataframe
from src.utils.logging_config import logger
from src.utils.pca import PCAWrapper

EVALUATION_FRACTION = 0.1
MIN_TRANSFORMATION_DISTANCE = 0.75


def euclidean_distance_between_arrays(array1: np.ndarray, array2: np.ndarray):
    """
    Calcuates the euclidean_distance_between_arrays. Inspired from
    https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
    """
    return np.linalg.norm(array1 - array2)


def create_train_val_test_split(
    mts_dataset_array: np.ndarray,
    test_size: int,
    config: Dict[str, Any],
):
    """ """
    logger.info(f"Generating transformation index pairs for training...")

    mts_features_array, _ = generate_feature_dataframe(
        data=mts_dataset_array,
        series_periodicity=config["stl_args"]["series_periodicity"],
        num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
    )

    dist_of_features = plot_feature_distribution(mts_features_array)
    dist_of_features.savefig(os.path.join(OUTPUT_DIR, "distribution_of_features.png"))
    plt.close(dist_of_features)

    indices = np.arange(len(mts_features_array))
    test_indices = indices[test_size:]
    indices_used_for_training = indices[:test_size]

    validation_size = int(EVALUATION_FRACTION * len(indices_used_for_training))
    np.random.shuffle(indices_used_for_training)

    validation_indices = indices_used_for_training[:validation_size]
    train_indices = indices_used_for_training[validation_size:]

    pca = PCAWrapper()
    pca.fit_transform(mts_features_array[train_indices])
    mts_pca_array: np.ndarray = pca.transform(mts_features_array)
    logger.info("Successfully generated MTS PCA space")

    logger.info("Splitting PCA space into train, validation and test indices...")

    train_transformation_indices: List[Tuple[int, int]] = []
    validation_transformation_indices: List[Tuple[int, int]] = []
    test_transformation_indices: List[Tuple[int, int]] = []

    if config["dataset_args"]["use_identity_mapping"] == True:
        logger.info("Using identity mapping")
        logger.info("Pairing training transformations")
        for i in tqdm(train_indices):
            train_transformation_indices.append((i, i))
    else:
        logger.info("Using transformation mapping")
        logger.info("Pairing training transformations")
        for i in tqdm(train_indices):
            for j in train_indices:
                if (
                    euclidean_distance_between_arrays(
                        mts_features_array[i], mts_features_array[j]
                    )
                    > MIN_TRANSFORMATION_DISTANCE
                ):
                    train_transformation_indices.append((i, j))
    logger.info("Pairing validation transformations")
    for i in tqdm(train_indices):
        for j in validation_indices:
            if (
                euclidean_distance_between_arrays(
                    mts_features_array[i], mts_features_array[j]
                )
                > MIN_TRANSFORMATION_DISTANCE
            ):
                validation_transformation_indices.append((i, j))
    logger.info("Pairing test transformations")
    for i in tqdm(train_indices):
        for j in test_indices:
            if (
                euclidean_distance_between_arrays(
                    mts_features_array[i], mts_features_array[j]
                )
                > MIN_TRANSFORMATION_DISTANCE
            ):
                test_transformation_indices.append((i, j))

    assert len(train_transformation_indices) > 0, "Training set must have elements"

    # If test_set_sample_size is in the config, we sample validation and test
    test_set_sample_size = config.get("dataset_args", {}).get(
        "test_set_sample_size", None
    )
    if test_set_sample_size is not None:
        logger.info(
            f"Sampling validation and test sets to size of {test_set_sample_size}"
        )
        number_of_transformations_in_test_set = min(
            config["dataset_args"]["test_set_sample_size"],
            max(
                len(validation_transformation_indices), len(test_transformation_indices)
            ),
        )
        assert (
            len(validation_transformation_indices)
            > number_of_transformations_in_test_set
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

    train_transformation_indices = np.array(train_transformation_indices)
    validation_transformation_indices = np.array(validation_transformation_indices)
    test_transformation_indices = np.array(test_transformation_indices)
    print(
        f"SE HER Plot time size of test indices: {test_transformation_indices[:, 1].shape}"
    )
    plot_train_val_test_split(
        mts_dataset_pca=mts_pca_array,
        train_indices=train_transformation_indices[:, 0],
        validation_indices=validation_transformation_indices[:, 1],
        test_indices=test_transformation_indices[:, 1],
    )

    arrow_plot_train = plot_transformation_directions(
        mts_dataset_pca=mts_pca_array,
        transformation_indices=train_transformation_indices,
    )
    arrow_plot_train.savefig(os.path.join(OUTPUT_DIR, "train_arrows.png"))
    arrow_plot_validation = plot_transformation_directions(
        mts_dataset_pca=mts_pca_array,
        transformation_indices=validation_transformation_indices,
    )
    arrow_plot_validation.savefig(os.path.join(OUTPUT_DIR, "validation_arrows.png"))
    arrow_plot_test = plot_transformation_directions(
        mts_dataset_pca=mts_pca_array,
        transformation_indices=test_transformation_indices,
    )
    arrow_plot_test.savefig(os.path.join(OUTPUT_DIR, "test_arrows.png"))
    return (
        train_transformation_indices,
        validation_transformation_indices,
        test_transformation_indices,
    )


def concat_delta_values_to_features(
    X_features: np.ndarray,  # Shape (Number of input vector, Number of features in UTS * Number of UTS in MTS)
    y_features: np.ndarray,  # Shape (Number of output vector, Number of features in UTS * Number of UTS in MTS)
    use_one_hot_encoding: bool,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
):
    """
    Augments the input feature vectors (X) by computing and appending delta values
    based on the difference between y and X. Optionally applies one-hot encoding.
    """
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
    X_features: np.ndarray,  # Shape (Number of input vector, Number of features in UTS * Number of UTS in MTS)
    y_features: np.ndarray,  # Shape (Number of output vector, Number of features in UTS * Number of UTS in MTS)
    use_one_hot_encoding: bool,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
):
    """
    Augments the input feature vectors (X) by computing and appending delta values
    based on the difference between y and X. Optionally applies one-hot encoding.
    This function is for inference data. This is done so that we dont add more
    test data than designed.
    """
    expanded_x_rows: List = []
    expanded_y_rows: List = []
    for row_index, _ in tqdm(enumerate(X_features), total=len(X_features)):
        X_row = X_features[row_index]
        y_row = y_features[row_index]
        uts_index = np.random.choice(range(0, number_of_uts_in_mts))
        new_row = X_row.copy()
        start_index = uts_index * number_of_features_in_mts
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


def create_reconstruction_xy_pairs(
    mts_dataset: np.ndarray,
    mts_features_array: np.ndarray,
    train_transformation_indices: np.ndarray,
    validation_transformation_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_delta = (
        mts_features_array[train_transformation_indices[:, 1]]
        - mts_features_array[train_transformation_indices[:, 0]]
    )
    X_mts_train = mts_dataset[train_transformation_indices[:, 0]]
    X_mts_train = X_mts_train.reshape(-1, X_mts_train.shape[1] * X_mts_train.shape[2])

    X_train = np.concatenate([X_mts_train, train_delta], axis=1)
    y_train = mts_dataset[train_transformation_indices[:, 1]]
    # We reshape to get shape (Number of samples, Number of samples per MTS flattened)
    y_train = y_train.reshape(-1, y_train.shape[1] * y_train.shape[2])

    validation_delta = (
        mts_features_array[validation_transformation_indices[:, 1]]
        - mts_features_array[validation_transformation_indices[:, 0]]
    )
    X_mts_validation = mts_dataset[validation_transformation_indices[:, 0]]
    X_mts_validation = X_mts_validation.reshape(
        -1, X_mts_validation.shape[1] * X_mts_validation.shape[2]
    )

    X_validation = np.concatenate([X_mts_validation, validation_delta], axis=1)
    y_validation = mts_dataset[validation_transformation_indices[:, 1]]
    # We reshape to get shape (Number of samples, Number of samples per MTS flattened)
    y_validation = y_validation.reshape(
        -1, y_validation.shape[1] * y_validation.shape[2]
    )

    return X_train, y_train, X_validation, y_validation
