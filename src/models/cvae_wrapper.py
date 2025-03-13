from src.models.cvae import MTSCVAE
from src.models.feature_transformation_model import FeatureTransformationModel
import torch
import numpy as np
from typing import Tuple, List
from src.utils.logging_config import logger
from torch.utils.data import TensorDataset, DataLoader
import copy
from tqdm import tqdm
import wandb


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
