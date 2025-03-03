import torch.nn.functional as F
from torch import nn
from torch import save, load
from src.utils.logging_config import logger
import os


class FeedForwardFeatureModel(nn.Module):
    def __init__(
        self,
        model_params,
    ):
        super(FeedForwardFeatureModel, self).__init__()
        self._check_params(model_params)
        self.generate_network(
            input_size=model_params["input_size"],
            output_size=model_params["output_size"],
            hidden_network_sizes=model_params["hidden_network_sizes"],
        )

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

    def _check_params(self, model_params):
        assert "input_size" in model_params, "input_size not in model_params"
        assert "output_size" in model_params, "output_size not in model_params"
        assert (
            "hidden_network_sizes" in model_params
        ), "hidden_network_sizes not in model_params"
        assert isinstance(
            model_params["input_size"], int
        ), "input_size must be an integer"
        assert isinstance(
            model_params["output_size"], int
        ), "output_size must be an integer"
        assert isinstance(
            model_params["hidden_network_sizes"], list
        ), "hidden_network_sizes must be a list"
        assert all(
            isinstance(i, int) for i in model_params["hidden_network_sizes"]
        ), "hidden_network_sizes must be a list of integers"
        assert (
            len(model_params["hidden_network_sizes"]) > 0
        ), "hidden_network_sizes must have at least one element"
        assert (
            model_params["hidden_network_sizes"][0] > 0
        ), "hidden_network_sizes must have at least one element"
