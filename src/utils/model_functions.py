from typing import List, Union
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import logging


def train_model(
    model: torch.nn.Module,
    X: Union[List[float], torch.Tensor],
    y: Union[List[float], torch.Tensor],
    batch_size: int,
    num_epochs: int,
    learning_rate: float,
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

    model.loss = torch.nn.L1Loss()
    model.optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    loss_history = []

    for _ in tqdm(range(num_epochs)):
        model.train()
        running_loss = 0.0

        for inputs, targets in dataloader:
            model.optimizer.zero_grad()
            outputs = model(inputs)
            loss = model.loss(outputs, targets)
            loss.backward()
            model.optimizer.step()
            running_loss += loss.item() * inputs.size(0)

        epoch_loss = running_loss / len(dataloader.dataset)
        loss_history.append(epoch_loss)

    model.save_model()

    return loss_history


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
