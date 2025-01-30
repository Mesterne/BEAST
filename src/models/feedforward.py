import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, input_size, output_size, learning_rate):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)
        self.loss = nn.L1Loss()
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, X, y, batch_size, num_epochs, learning_rate):
        # Convert data to PyTorch tensors
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).to(self.device)

        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        loss_history = []

        for _ in tqdm(range(num_epochs)):
            self.train()
            running_loss = 0.0

            for inputs, targets in dataloader:
                self.optimizer.zero_grad()
                outputs = self(inputs)
                loss = self.loss(outputs, targets)
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)

            epoch_loss = running_loss / len(dataloader.dataset)
            loss_history.append(epoch_loss)

        return loss_history

    def evaluate_model(self, X_test, y_test):
        # Set model to evaluation mode
        self.eval()

        # Convert test data to PyTorch tensors
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(self.device)

        # Disable gradient computation for evaluation
        with torch.no_grad():
            # Get predictions
            outputs = self(X_test_tensor)
            predicted = outputs.cpu().numpy()

        return predicted
