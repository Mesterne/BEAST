import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch import save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class NN(nn.Module):
    def __init__(self, input_size, output_size, learning_rate, save_dir):
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)
        self.save_dir = save_dir + '/feedforward_forecaster.pth'
        loaded_model = self.load_model()
        if loaded_model is not None:
            self = loaded_model
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

        self.save_model()

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

    def save_model(self):
        print(f'Saving trained model to {self.save_dir}...')
        save(self, self.save_dir)

    def load_model(self):
        print(f'Loading trained model from {self.save_dir}...')
        try:
            model = load(self.save_dir)
        except FileNotFoundError:
            print('Could not find saved model...')
            return None
        except Exception as e:
            print(f'Issues with loading model: {e}')
            return None
        return model
