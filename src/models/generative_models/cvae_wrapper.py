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

        return generated_mts


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


def create_conditioned_dataset_for_training(
    mts_array: np.ndarray,
    mts_features: np.ndarray,
    condition_type: str,
    transformation_indices: np.ndarray,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes the entire MTS dataset with features and based on the condition type and creates X with conditions
    for either target features (with mask) or delta values to target features. This function is specialized for
    training. For each UTS it creates a mask and training entry
    """
    X_features = mts_features[transformation_indices[:, 0]]
    y_features = mts_features[transformation_indices[:, 1]]

    X_mts = mts_array[transformation_indices[:, 0]]
    y_mts = mts_array[transformation_indices[:, 1]]

    X = []
    y = []
    for mts_index, _ in enumerate(X_mts):
        original_mts = X_mts[mts_index].flatten()
        target_mts = y_mts[mts_index].flatten()
        original_features_for_X = X_features[mts_index]
        target_features_for_X = y_features[mts_index]
        for uts_index in range(0, number_of_uts_in_mts):
            start_index = uts_index * number_of_features_in_mts
            end_index = start_index + number_of_features_in_mts
            conditions = [0] * len(target_features_for_X)
            if condition_type == "feature_delta":
                conditions[start_index:end_index] = (
                    target_features_for_X[start_index:end_index]
                    - original_features_for_X[start_index:end_index]
                )
            elif condition_type == "feature":
                conditions[start_index:end_index] = target_features_for_X[
                    start_index:end_index
                ]
            X.append(np.concatenate((original_mts, conditions)))
            y.append(target_mts)
    return np.array(X), np.array(y)


def create_conditioned_dataset_for_inference(
    mts_array: np.ndarray,
    mts_features: np.ndarray,
    condition_type: str,
    transformation_indices: np.ndarray,
    number_of_uts_in_mts: int,
    number_of_features_in_mts: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Takes the entire MTS dataset with features and based on the condition type and creates X with conditions
    for either target features (with mask) or delta values to target features. This function is specialized for
    inference, which means instead of creating a mask for each uts, it picks one uts per MTS, to avoid
    expanding the size of y for test.
    """
    X_features = mts_features[transformation_indices[:, 0]]
    y_features = mts_features[transformation_indices[:, 1]]

    X_mts = mts_array[transformation_indices[:, 0]]
    y_mts = mts_array[transformation_indices[:, 1]]

    X = []
    y = []
    for mts_index, _ in enumerate(X_mts):
        original_mts = X_mts[mts_index].flatten()
        target_mts = y_mts[mts_index].flatten()
        original_features_for_X = X_features[mts_index]
        target_features_for_X = y_features[mts_index]
        uts_index = np.random.choice(range(0, number_of_uts_in_mts))
        start_index = uts_index * number_of_features_in_mts
        end_index = start_index + number_of_features_in_mts
        conditions = [0] * len(target_features_for_X)
        if condition_type == "feature_delta":
            conditions[start_index:end_index] = (
                target_features_for_X[start_index:end_index]
                - original_features_for_X[start_index:end_index]
            )
        elif condition_type == "feature":
            conditions[start_index:end_index] = target_features_for_X[
                start_index:end_index
            ]

        X.append(np.concatenate((original_mts, conditions)))
        y.append(target_mts)
    return np.array(X), np.array(y)
