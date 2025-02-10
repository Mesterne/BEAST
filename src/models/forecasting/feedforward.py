import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch import save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm


class FeedForwardForecaster(nn.Module):
    def __init__(
        self, input_size, output_size, save_dir, name="feedforward_forecaster", load_model=False
    ):
        super(FeedForwardForecaster, self).__init__()
        self.fc1 = nn.Linear(input_size, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, output_size)

        self.save_dir = save_dir + f"/{name}.pth"
        if load_model:
            loaded_model = self.load_model()
            if loaded_model is not None:
                self = loaded_model

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def save_model(self):
        print(f"Saving trained model to {self.save_dir}...")
        save(self, self.save_dir)

    def load_model(self):
        print(f"Loading trained model from {self.save_dir}...")
        try:
            model = load(self.save_dir)
        except FileNotFoundError:
            print("Could not find saved model...")
            return None
        except Exception as e:
            print(f"Issues with loading model: {e}")
            return None
        return model
