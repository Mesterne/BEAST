import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from src.utils.logging_config import logger


def assert_no_overlapping_indices(train_indices, validation_indices, test_indices):
    """
    Assert that there are no overlapping indices between train, validation, and test sets.
    """

    train_val_overlap = len(set(train_indices).intersection(validation_indices))
    train_test_overlap = len(set(train_indices).intersection(test_indices))
    val_test_overlap = len(set(validation_indices).intersection(test_indices))

    assert (
        train_val_overlap == 0
    ), f"{train_val_overlap} overlapping indices found between train and validation sets."
    assert (
        train_test_overlap == 0
    ), f"{train_test_overlap} overlapping indices found between train and test sets."
    assert (
        val_test_overlap == 0
    ), f"{train_test_overlap} overlapping indices found between validation and test sets."


def scale_mts_dataset(mts_data, train_indices, validation_indices, test_indices):
    # Mts shape is (num_mts, num_uts, num_timesteps )

    num_uts = mts_data.shape[1]

    # Train indices contains all original - target pairs for training data set.
    # Get all training mts by using the unique first values of the pairs.
    # Get all validation and test mts by using the second values of their respective pairs.
    train_mts_indices = np.unique(train_indices[:, 0])
    validation_mts_indices = np.unique(validation_indices[:, 1])
    test_mts_indices = np.unique(test_indices[:, 1])

    assert_no_overlapping_indices(
        train_mts_indices, validation_mts_indices, test_mts_indices
    )

    uts_scalers = []
    for i in range(num_uts):
        # Get the training data for the current UTS
        uts_train = mts_data[train_mts_indices, i, :].reshape(-1, 1)

        logger.info(
            f"Before scaling. Mean: {np.mean(uts_train)}, Std: {np.std(uts_train)}"
        )

        # Create and fit the scaler
        scaler = MinMaxScaler()
        scaler.fit(uts_train)

        scaled_uts_train = scaler.transform(uts_train)
        logger.info(
            f"After scaling. Mean: {np.mean(scaled_uts_train)}, Std: {np.std(scaled_uts_train)}"
        )

        # Store the scaler for later use
        uts_scalers.append(scaler)

    # Scale the data
    scaled_mts_data = np.zeros_like(mts_data)
    print(scaled_mts_data.shape)
    for i in range(num_uts):
        # Scale the training data
        scaled_mts_data[train_mts_indices, i, :] = (
            uts_scalers[i]
            .transform(mts_data[train_mts_indices, i, :].reshape(-1, 1))
            .reshape(len(train_mts_indices), -1)
        )

        # Scale the validation data
        scaled_mts_data[validation_mts_indices, i, :] = (
            uts_scalers[i]
            .transform(mts_data[validation_mts_indices, i, :].reshape(-1, 1))
            .reshape(len(validation_mts_indices), -1)
        )

        # Scale the test data
        scaled_mts_data[test_mts_indices, i, :] = (
            uts_scalers[i]
            .transform(mts_data[test_mts_indices, i, :].reshape(-1, 1))
            .reshape(len(test_mts_indices), -1)
        )

    print(scaled_mts_data.shape)

    return scaled_mts_data, uts_scalers
