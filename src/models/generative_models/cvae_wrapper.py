import copy
from turtle import st
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

import wandb
from src.models.feature_transformation_model import FeatureTransformationModel
from src.models.generative_models.cvae import MTSCVAE
from src.utils.logging_config import logger


class CVAEWrapper(FeatureTransformationModel):
    def __init__(self, model: MTSCVAE, training_params: dict):
        self.model = model
        self.learning_rate = training_params["learning_rate"]
        self.batch_size = training_params["batch_size"]
        self.num_epochs = training_params["num_epochs"]
        self.early_stopping_patience = training_params["early_stopping_patience"]

    def loss_function(self, input, output, mean, log_var):
        kl_divergence = self.KL_divergence(mean, log_var)
        reconstruction_loss = self.reconstruction_loss(input, output)
        return kl_divergence + reconstruction_loss

    def KL_divergence(self, mean, log_var):
        """Based on (Kingma and Welling, 2013) Appendix B"""
        return -0.5 * torch.sum(1 + log_var - mean**2 - torch.exp(log_var))

    def reconstruction_loss(self, input, output):
        return torch.mean((input - output) ** 2)

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        log_to_wandb=False,
    ) -> Tuple[List[float], List[float]]:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training model with device: {device}")
        self.model.to(device)

        loss_function = self.loss_function
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_validation_tensor = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_validation_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)

        print(X_train_tensor.shape, y_train_tensor.shape)
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        train_dataloader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )
        validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
        validation_dataloader = DataLoader(
            validation_dataset, batch_size=self.batch_size, shuffle=False
        )

        train_loss_history = []
        validation_loss_history = []

        best_validation_loss = float("inf")
        best_model_weigths = copy.deepcopy(self.model.state_dict())

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                outputs, latent_means, latent_logvars = self.model(inputs)
                loss = loss_function(outputs, targets, latent_means, latent_logvars)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataloader.dataset)
            train_loss_history.append(epoch_loss)

            # NOTE: Evaluates reconstruction
            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in validation_dataloader:
                    outputs, latent_means, latent_logvars = self.model(inputs)
                    loss = loss_function(outputs, targets, latent_means, latent_logvars)
                    running_val_loss += loss.item() * inputs.size(0)

            val_epoch_loss = running_val_loss / len(validation_dataloader.dataset)
            validation_loss_history.append(val_epoch_loss)

            # Early stopping logic
            # The model has improved
            if val_epoch_loss < best_validation_loss:
                best_validation_loss = val_epoch_loss
                best_model_weigths = copy.deepcopy(self.model.state_dict())
                patience_counter = 0
            # The model is not improving
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break
            if log_to_wandb:
                wandb.log({"train_loss": epoch_loss, "val_loss": val_epoch_loss})

        self.model.load_state_dict(best_model_weigths)

        wandb.finish()

        return train_loss_history, validation_loss_history

    def infer(
        self,
        X,
        num_uts_in_mts: int = None,
        num_features_per_uts: int = None,
        seasonal_period: int = None,
    ) -> np.ndarray:
        # If the model is conditioned on features, X should be only the features
        if self.model.condition_type == "feature":
            input_features = X[:, self.model.mts_size :]
            # Run generate_mts for each row in X
            generated_mts: np.ndarray = self.model.generate_mts(input_features)

        # If the model is conditioned on feature deltas, X should be the original mts and the feature deltas
        if self.model.condition_type == "feature_delta":
            # FIXME: IMPLEMENT THIS MEHTOD IN CVAE CLASS
            input_mts = X[:, : self.model.mts_size]
            input_feature_deltas = X[:, self.model.mts_size :]
            generated_mts: np.ndarray = self.model.transform_mts_from_original(
                input_mts, input_feature_deltas
            )

        features_of_generated_mts = numpy_decomp_and_features(
            generated_mts, num_uts_in_mts, num_features_per_uts, seasonal_period
        )[1]
        return generated_mts, features_of_generated_mts


def mask_feature_values(
    feature_values: np.ndarray, num_uts: int, uts_idx: int
) -> np.ndarray:
    num_features = feature_values.shape[0]
    num_features_per_uts = num_features // num_uts

    # Use the masked UTS features (zero value) in delta values to mask the original values
    feature_values_reshaped = feature_values.reshape(num_uts, num_features_per_uts)

    masked_values = np.zeros_like(feature_values_reshaped)

    masked_values[uts_idx, :] = feature_values_reshaped[uts_idx, :]

    return masked_values.flatten()


def create_feature_conditioned_dataset(
    mts_array: np.ndarray,
    mts_features: np.ndarray,
    transformation_indices: np.ndarray,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_features = mts_features[transformation_indices[:, 0]]
    y_features = mts_features[transformation_indices[:, 1]]

    X_mts = mts_array[transformation_indices[:, 0]]
    y_mts = mts_array[transformation_indices[:, 1]]

    X = []
    y = []
    for mts_index, mts in enumerate(X_mts):
        original_mts = X_mts[mts_index].flatten()
        target_mts = y_mts[mts_index].flatten()
        target_features_for_X = y_features[mts_index]
        for uts_index in range(0, number_of_uts_in_mts):
            start_index = uts_index * number_of_features_in_mts
            end_index = start_index + number_of_features_in_mts
            conditions = [0] * len(target_features_for_X)
            conditions[start_index:end_index] = target_features_for_X[
                start_index:end_index
            ]
            X.append(np.concatenate((original_mts, conditions)))
            y.append(target_mts)
    return np.array(X), np.array(y)


def create_delta_conditioned_dataset(
    mts_array: np.ndarray,
    mts_features: np.ndarray,
    transformation_indices: np.ndarray,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    X_features = mts_features[transformation_indices[:, 0]]
    y_features = mts_features[transformation_indices[:, 1]]

    X_mts = mts_array[transformation_indices[:, 0]]
    y_mts = mts_array[transformation_indices[:, 1]]

    X = []
    y = []
    for mts_index, mts in enumerate(X_mts):
        original_mts = X_mts[mts_index].flatten()
        target_mts = y_mts[mts_index].flatten()
        original_features_for_X = X_features[mts_index]
        target_features_for_X = y_features[mts_index]
        for uts_index in range(0, number_of_uts_in_mts):
            start_index = uts_index * number_of_features_in_mts
            end_index = start_index + number_of_features_in_mts
            conditions = [0] * len(target_features_for_X)
            conditions[start_index:end_index] = (
                target_features_for_X[start_index:end_index]
                - original_features_for_X[start_index:end_index]
            )
            X.append(np.concatenate((original_mts, conditions)))
            y.append(target_mts)
    return np.array(X), np.array(y)


def get_feature_conditioned_dataset(
    mts_array: list,  # List of MTS (num mts, num timesteps x num uts)
    train_features_supervised_dataset: pd.DataFrame,  # Connects the features to the MTS
    validation_features_supervised_dataset: pd.DataFrame,
    test_features_supervised_dataset: pd.DataFrame,
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], Tuple[np.ndarray]]:
    # Train, validation and test datasets
    dataset_list = [[], [], []]
    df_list = [
        train_features_supervised_dataset,
        validation_features_supervised_dataset,
        test_features_supervised_dataset,
    ]

    # NOTE: mts_array is a list of MTS where each mts is a numpy array of shape (num_uts, num_timesteps)
    # The MTS is NOT FLATTENED.
    num_uts = mts_array[0].shape[0]

    # Get all columns starting with original_ and target_ prefix
    orig_cols = [
        col for col in df_list[0].columns if "original_" in col and "index" not in col
    ]
    target_cols = [
        col for col in df_list[0].columns if "target_" in col and "index" not in col
    ]
    logger.info("Generating feature conditioned dataset (train, validation, test)")
    for df_idx, df in enumerate(df_list):

        # NOTE: Check if the dataset is train or not. Decides if the original or target index should be used
        # Assuming first df is train dataset
        is_train: bool = df_idx == 0

        # Get first instance of each relevant MTS index
        df = (
            df.drop_duplicates(subset="original_index")
            if is_train
            else df.drop_duplicates(subset="target_index")
        )

        X_list = []
        y_list = []
        for _, row in tqdm(df.iterrows(), total=len(df)):
            mts_idx = (
                int(row["original_index"]) if is_train else int(row["target_index"])
            )
            mts_flattened = mts_array[mts_idx].flatten()
            # Mask different subset of features for each UTS
            for uts_idx in range(num_uts):
                feature_values = (
                    row[orig_cols].values if is_train else row[target_cols].values
                )

                # Feature values are masked based on delta values
                masked_feature_values = mask_feature_values(
                    feature_values, num_uts, uts_idx
                )
                dataset_entry_X = np.hstack((mts_flattened, masked_feature_values))
                dataset_entry_y = mts_flattened.copy()
                X_list.append(dataset_entry_X)
                y_list.append(dataset_entry_y)
        dataset_list[df_idx] = (np.array(X_list), np.array(y_list))
    train_array = dataset_list[0]
    validation_array = dataset_list[1]
    test_array = dataset_list[2]

    return train_array, validation_array, test_array


def get_delta_conditioned_dataset(
    mts_array: list,
    train_features_supervised_dataset: pd.DataFrame,
    validation_features_supervised_dataset: pd.DataFrame,
    test_features_supervised_dataset: pd.DataFrame,
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], Tuple[np.ndarray]]:
    dataset_list = [[], [], []]
    df_list = [
        train_features_supervised_dataset,
        validation_features_supervised_dataset,
        test_features_supervised_dataset,
    ]

    # Need the column number for the original index, target index, and delta values
    columns = df_list[0].columns
    original_index = [
        idx for idx, col in enumerate(columns) if "original" in col and "index" in col
    ][0]
    target_index = [
        idx for idx, col in enumerate(columns) if "target" in col and "index" in col
    ][0]
    delta_value_indices = [
        idx for idx, col in enumerate(columns) if "delta" in col and "index" not in col
    ]

    for df_idx, df in enumerate(df_list):
        # NOTE: Convert to numpy array for faster indexing
        numpy_df = df.to_numpy()

        # Get original and target MTS
        original_mts_indices = numpy_df[:, original_index].astype(int)
        target_mts_indices = numpy_df[:, target_index].astype(int)
        num_uts = mts_array.shape[1]
        uts_length = mts_array.shape[2]
        X_mts_array = mts_array[original_mts_indices].reshape(-1, num_uts * uts_length)
        y_mts_array = mts_array[target_mts_indices].reshape(-1, num_uts * uts_length)

        # Get delta values
        delta_values = numpy_df[:, delta_value_indices]

        print(X_mts_array.shape, delta_values.shape)

        X = np.hstack((X_mts_array, delta_values))
        y = y_mts_array

        assert X.shape[0] == y.shape[0], f"X shape: {X.shape}, y shape: {y.shape}"
        assert (
            X.shape[0] == delta_values.shape[0]
        ), f"X shape: {X.shape}, delta shape: {delta_values.shape}"
        assert (
            X.shape[1] == y.shape[1] + delta_values.shape[1]
        ), f"X shape: {X.shape}, y shape: {y.shape}, delta shape: {delta_values.shape}"
        assert X.shape[0] == len(df), f"X shape: {X.shape}, df shape: {df.shape}"

        dataset_list[df_idx] = (X, y)

    train_array = dataset_list[0]
    validation_array = dataset_list[1]
    test_array = dataset_list[2]
    return train_array, validation_array, test_array


def prepare_cgen_data(
    condition_type: str,
    mts_array: list,
    train_features_supervised_dataset: pd.DataFrame,
    validation_features_supervised_dataset: pd.DataFrame,
    test_features_supervised_dataset: pd.DataFrame,
) -> Tuple[Tuple[np.ndarray], Tuple[np.ndarray], Tuple[np.ndarray]] | None:
    if condition_type == "feature":
        return get_feature_conditioned_dataset(
            mts_array,
            train_features_supervised_dataset,
            validation_features_supervised_dataset,
            test_features_supervised_dataset,
        )
    if condition_type == "feature_delta":
        return get_delta_conditioned_dataset(
            mts_array,
            train_features_supervised_dataset,
            validation_features_supervised_dataset,
            test_features_supervised_dataset,
        )
    return None
