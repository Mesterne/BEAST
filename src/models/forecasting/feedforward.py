import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch import save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from utils.logging_config import logger


class FeedForwardForecaster(nn.Module):
    def __init__(
        self,
        model_params,
    ):
        super(FeedForwardForecaster, self).__init__()
        self.fc1 = nn.Linear(
            model_params["window_size"] * 3, 100
        )  # TODO: Should not be hardcoded
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, model_params["horizon_length"])

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
