import torch
import torch.nn.functional as F
from torch import optim
from torch import nn
from torch import save, load
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from tqdm import tqdm
from src.utils.logging_config import logger
import os


class FeedForwardFeatureModel(nn.Module):

    def __init__(
        self,
        input_size,
        output_size,
        hidden_network_sizes,
        save_dir,
        name="feedforward_feature",
        load_model=False,
    ):
        super(FeedForwardFeatureModel, self).__init__()
        self.generate_network(
            input_size=input_size,
            output_size=output_size,
            hidden_network_sizes=hidden_network_sizes,
        )
        self.save_dir = os.path.join(save_dir, f"{name}.pth")
        if load_model:
            loaded_model = self.load_model()
            if loaded_model is not None:
                self = loaded_model

    def generate_network(self, input_size, output_size, hidden_network_sizes):
        logger.info(
            f"Building feedforward forecaster with hidden sizes: {hidden_network_sizes}"
        )
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(input_size, hidden_network_sizes[0]))
        for i in range(1, len(hidden_network_sizes)):
            self.layers.append(
                nn.Linear(hidden_network_sizes[i - 1], hidden_network_sizes[i])
            )
        self.layers.append(nn.Linear(hidden_network_sizes[-1], output_size))

    def forward(self, x):
        for i in range(0, len(self.layers) - 1):
            x = F.relu(self.layers[i](x))  # Correct usage
        x = self.layers[-1](x)
        return x

    def save_model(self):
        logger.info(f"Saving trained model to {self.save_dir}...")
        save(self, self.save_dir)

    def load_model(self):
        logger.info(f"Loading trained model from {self.save_dir}...")
        try:
            model = load(self.save_dir)
        except FileNotFoundError:
            logger.warning("Could not find saved model...")
            return None
        except Exception as e:
            logger.error(f"Issues with loading model: {e}")
            return None
        return model
