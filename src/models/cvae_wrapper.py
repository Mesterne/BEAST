import copy
from typing import List, Tuple

import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.cvae import MTSCVAE
from src.models.feature_transformation_model import FeatureTransformationModel
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

        print(X_train.shape, y_train.shape)
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

    def infer(self, X):
        # Run generate_mts for each row in X
        generated_mts: np.ndarray = self.model.generate_mts(X)
        return generated_mts


def prepare_cvae_data(
    mts_array: list,
    X_features_train: np.ndarray,
    train_indices: np.ndarray,
    X_features_validation: np.ndarray,
    validation_indices: np.ndarray,
    X_features_test: np.ndarray,
    test_indices: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Starting out by flattening each MTS
    full_mts_array: np.ndarray = np.asarray(mts_array)
    full_mts_array: np.ndarray = full_mts_array.reshape(
        full_mts_array.shape[0], full_mts_array.shape[1] * full_mts_array.shape[2]
    )
    # Use train indices to get the training MTS
    cvae_train_mts_array: np.ndarray = full_mts_array[train_indices]
    # The train features are the features of each training MTS
    # FIXME: This is not currently done. X_features_train contains all pairings of training features and validation features.
    # This is more than 80k rows, and the code is currently only getting the first 180 rows, which most likely all belong to the same MTS.
    print(X_features_train.shape)
    cvae_train_features_array: np.ndarray = X_features_train[train_indices]
    X_cvae_train: np.ndarray = np.hstack(
        (cvae_train_mts_array, cvae_train_features_array)
    )
    y_cvae_train: np.ndarray = cvae_train_mts_array.copy()

    # Validation input and target for CVAE
    # FIXME: Getting validation indice from X_feature_validation seems unnecessary
    cvae_validation_mts_array: np.ndarray = full_mts_array[validation_indices]
    cvae_validation_features_array: np.ndarray = X_features_validation[
        validation_indices
    ]
    X_cvae_validation: np.ndarray = np.hstack(
        (cvae_validation_mts_array, cvae_validation_features_array)
    )
    y_cvae_validation: np.ndarray = cvae_validation_mts_array.copy()

    # Test input and target for CVAE
    # FIXME: Getting validation indice from X_feature_test seems unnecessary
    cvae_test_mts_array: np.ndarray = full_mts_array[test_indices]
    cvae_test_features_array: np.ndarray = X_features_test[test_indices]
    X_cvae_test: np.ndarray = cvae_test_features_array.copy()
    y_cvae_test: np.ndarray = cvae_test_mts_array.copy()

    return (
        X_cvae_train,
        y_cvae_train,
        X_cvae_validation,
        y_cvae_validation,
        X_cvae_test,
        y_cvae_test,
    )
