import copy
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.models.feature_transformation_model import FeatureTransformationModel
from src.plots.plot_training_and_validation_loss import (
    plot_training_and_validation_loss,
)
from src.utils.logging_config import logger


class NeuralNetworkWrapper(FeatureTransformationModel):
    def __init__(self, model: torch.nn.Module, training_params: dict):
        self.model = model
        self.learning_rate = training_params["learning_rate"]
        self.batch_size = training_params["batch_size"]
        self.num_epochs = training_params["num_epochs"]
        self.early_stopping_patience = training_params["early_stopping_patience"]

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        plot_loss=False,
        model_name="unnamed",
    ) -> Tuple[List[float], List[float]]:
        """
        Trains a PyTorch model using L1 loss and the Adam optimizer.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            X (Union[List[float], torch.Tensor]): Input features for training.
            y (Union[List[float], torch.Tensor]): Target values for training.
            batch_size (int): Number of samples per training batch.
            num_epochs (int): Number of training epochs.
            learning_rate (float): Learning rate for the optimizer.

        Returns:
            List[float]: A list containing the loss history for each epoch.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Training model with device: {device}")
        self.model.to(device)

        loss_function = torch.nn.L1Loss()
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

        patience_counter: int = 0

        for epoch in tqdm(range(self.num_epochs)):
            self.model.train()
            running_loss = 0.0

            for inputs, targets in train_dataloader:
                optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = loss_function(outputs, targets)
                loss.backward()
                optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(train_dataloader.dataset)
            train_loss_history.append(epoch_loss)

            self.model.eval()
            running_val_loss = 0.0
            with torch.no_grad():
                for inputs, targets in validation_dataloader:
                    outputs = self.model(inputs)
                    loss = loss_function(outputs, targets)
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
                    logger.info(f"Early stopping triggered for epoch {epoch}.")
                    break

        self.model.load_state_dict(best_model_weigths)

        if plot_loss:
            plot_training_and_validation_loss(
                training_loss=train_loss_history,
                validation_loss=validation_loss_history,
                model_name=model_name,
            )

        return train_loss_history, validation_loss_history

    def infer(self, X: np.ndarray) -> np.ndarray:
        """
        Runs inference on a trained PyTorch model.

        Args:
            model (torch.nn.Module): The trained PyTorch model.
            X_test (Union[List[float], torch.Tensor]): Input features for inference.

        Returns:
            List[float]: Predicted values as a list.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Running model inference with device: {device}")

        self.model.eval()

        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            predicted = outputs.cpu().numpy()

        return predicted
