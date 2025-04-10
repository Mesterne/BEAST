import numpy as np
from torch import nn

from src.utils.logging_config import logger


class FeedForwardFeatureModel(nn.Module):
    def __init__(
        self,
        model_params,
    ):
        super(FeedForwardFeatureModel, self).__init__()
        self.input_size = model_params["input_size"]
        self.output_size = model_params["output_size"]
        hidden_layers_params = model_params["hidden_layers"]
        final_hidden_layer_size = int(hidden_layers_params[-1][0])

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_layers_params[0][0]), nn.ReLU()
        )
        self.generate_hidden_layers(hidden_layers=hidden_layers_params)
        self.output_layer = nn.Sequential(
            nn.Linear(final_hidden_layer_size, self.output_size), nn.ReLU()
        )

    def generate_hidden_layers(self, hidden_layers: dict):
        hidden_layer_sizes = np.asarray(hidden_layers)[:, 0].astype(int)
        hidden_layer_types = np.asarray(hidden_layers)[:, 1]
        hidden_layer_activations = np.asarray(hidden_layers)[:, 2]
        logger.info(
            f"Building network with hidden layer sizes: {hidden_layer_sizes}; types: {hidden_layer_types}; and activations: {hidden_layer_activations}"
        )

        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1):
            if hidden_layer_types[i] == "linear":
                self.hidden_layers.append(
                    nn.Linear(
                        hidden_layer_sizes[i],
                        hidden_layer_sizes[i + 1],
                    )
                )
            else:
                raise ValueError(f"Unknown hidden layer type: {hidden_layer_types[i]}")
            if hidden_layer_activations[i] == "relu":
                self.hidden_layers.append(nn.ReLU())
            else:
                raise ValueError(
                    f"Unknown hidden layer activation: {hidden_layer_activations[i]}"
                )

    def forward(self, x):
        input = self.input_layer(x)
        for i in range(0, len(self.hidden_layers)):
            input = self.hidden_layers[i](hidden_layer_input)
        output = self.output_layer(input)
        return output
