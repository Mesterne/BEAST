from typing import List, Union
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging
import copy
import wandb


def train_model(
    model: torch.nn.Module,
    X_train: Union[List[float], torch.Tensor],
    y_train: Union[List[float], torch.Tensor],
    X_validation: Union[List[float], torch.Tensor],
    y_validation: Union[List[float], torch.Tensor],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
    early_stopping_patience: float,
) -> List[float]:
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
    logging.info(f"Training model with device: {device}")
    model.to(device)

    loss_function = torch.nn.L1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
    X_validation_tensor = torch.tensor(X_validation, dtype=torch.float32).to(device)
    y_validation_tensor = torch.tensor(y_validation, dtype=torch.float32).to(device)

    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    validation_dataset = TensorDataset(X_validation_tensor, y_validation_tensor)
    validation_dataloader = DataLoader(
        validation_dataset, batch_size=batch_size, shuffle=False
    )

    train_loss_history = []
    validation_loss_history = []

    best_validation_loss = float("inf")
    best_model_weigths = copy.deepcopy(model.state_dict())

    for epoch in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(train_dataloader.dataset)
        train_loss_history.append(epoch_loss)

        model.eval()
        val_loss = 0.0
        running_val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in validation_dataloader:
                outputs = model(inputs)
                loss = loss_function(outputs, targets)
                running_val_loss += loss.item() * inputs.size(0)

        val_epoch_loss = running_val_loss / len(validation_dataloader.dataset)
        validation_loss_history.append(val_epoch_loss)

        # Early stopping logic
        # The model has improved
        if val_epoch_loss < best_validation_loss:
            best_validation_loss = val_loss
            best_model_weigths = copy.deepcopy(model.state_dict())
            patience_counter = 0
        # The model is not improving
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logging.info(f"Early stopping triggered for epoch {epoch}.")
                break
        wandb.log(
            {
                "Training loss": epoch_loss,
                "Validation loss": val_epoch_loss,
                "patience_counter": patience_counter,
            }
        )

    model.load_state_dict(best_model_weigths)
    model.save_model()

    wandb.finish()

    return train_loss_history, validation_loss_history


def run_model_inference(
    model: torch.nn.Module, X_test: Union[List[float], torch.Tensor]
) -> List[float]:
    """
    Runs inference on a trained PyTorch model.

    Args:
        model (torch.nn.Module): The trained PyTorch model.
        X_test (Union[List[float], torch.Tensor]): Input features for inference.

    Returns:
        List[float]: Predicted values as a list.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Running model inference with device: {device}")

    model.eval()

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)

    with torch.no_grad():
        outputs = model(X_test_tensor)
        predicted = outputs.cpu().numpy()

    return predicted.tolist()
