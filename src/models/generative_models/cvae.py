from typing import Tuple

import numpy as np
import torch
from torch import Tensor, cat, exp, nn, randn, randn_like

from src.utils.logging_config import logger


class MTSCVAE(nn.Module):
    """CVAE model based on the paper by Sohn et al. (2015).
    Athough the model proposed by the authors is not trained to reconstruct the input,
    we will train our model to do so.
    The paper performs experiments on image data, we will use MTS data.

    Notation from paper: x - condition; y - output; z - latent

    We train the model to reconstruct y (MTS), given conditions x (Features/Delta values).
    Thus, y can be interpreted as both the output and the input.
    The encoder approximate the distribution of the latent space given the input and the condition q(z|x,y).
    The decoder approximate the distribution of the input given the latent space and the condition p(y|x,z).
    """

    def __init__(self, model_params: dict) -> None:
        super(MTSCVAE, self).__init__()
        self.input_size_without_conditions = model_params[
            "input_size_without_conditions"
        ]
        self.mts_size = model_params["mts_size"]
        self.number_of_conditions = model_params["number_of_conditions"]
        self.latent_size = model_params["latent_size"]
        self.condition_type = model_params["condition_type"]
        self.encoder = Encoder(
            self.input_size_without_conditions,
            self.number_of_conditions,
            self.latent_size,
            model_params["hidden_layers"],
        )
        self.decoder = Decoder(
            self.mts_size,
            self.number_of_conditions,
            self.latent_size,
            model_params["hidden_layers"],
        )

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick using standard normal distribution to sample from the latent space.
        (Kingma and Welling, 2013) notes that z = location + scale * noise,
        where noise ~ N(0, 1), is a valid reparameterization.
        Sampling from a standard normal distribution is one of several possible reparameterization strategies.
        """

        # First mts_size elements are the MTS, the rest is either feature values or feature deltas
        assert (
            input.shape[1]
            == self.input_size_without_conditions + self.number_of_conditions
        ), f"Input size mismatch. Expected {self.input_size_without_conditions + self.number_of_conditions}, got {input.shape[1]}"

        mts: np.ndarray = input[:, : self.input_size_without_conditions]
        feature_info: np.ndarray = input[:, self.input_size_without_conditions :]

        assert (
            mts.shape[1] == self.input_size_without_conditions
        ), f"MTS size mismatch. Expected {self.input_size_without_conditions}, got {mts.shape[1]}"
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Conditions size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"

        # Encoding step
        latent_mean, latent_log_var = self.encoder(mts, feature_info)

        # Reparameterization trick
        latent_vector = self.reparamterization_trick(latent_mean, latent_log_var)

        # Decoding step
        output = self.decoder(feature_info, latent_vector)

        return output, latent_mean, latent_log_var

    def reparamterization_trick(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick using standard normal distribution to sample from the latent space."""
        noise = randn_like(mean)
        standard_deviation = exp(0.5 * log_var)
        return mean + standard_deviation * noise

    def generate_mts(self, feature_values: np.ndarray) -> np.ndarray:
        """Generate MTS data given feature values as condition."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_feature_values = torch.tensor(feature_values, dtype=torch.float32).to(
            device
        )
        # Sample latent vector
        latent_vector = randn(feature_values.shape[0], self.latent_size).to(device)
        # Run latent vector through decoder with features as conditions.
        cpu_mts = self.decoder(tensor_feature_values, latent_vector).cpu()
        return cpu_mts.detach().numpy()

    def transform_mts_from_original(
        self, mts: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """Transform MTS data given feature deltas as condition."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_mts = torch.tensor(mts, dtype=torch.float32).to(device)
        tensor_conditions = torch.tensor(conditions, dtype=torch.float32).to(device)
        latent_mean, latent_log_var = self.encoder(tensor_mts, tensor_conditions)
        latent_vector = self.reparamterization_trick(latent_mean, latent_log_var).to(
            device
        )
        # NOTE: Necessary to move tensor to cpu before converting to numpy
        cpu_mts = self.decoder(tensor_conditions, latent_vector).cpu()
        return cpu_mts.detach().numpy()


class Encoder(nn.Module):

    def __init__(
        self,
        input_size_without_conditions: int,
        number_of_conditions: int,
        latent_size: int,
        hidden_layers: dict,
    ) -> None:
        super(Encoder, self).__init__()
        self.input_size_without_conditions = input_size_without_conditions
        self.number_of_conditions = number_of_conditions
        self.latent_size = latent_size

        self.input_size = input_size_without_conditions + number_of_conditions
        final_hidden_layer_size = int(hidden_layers[-1][0])

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_layers[0][0]), nn.ReLU()
        )

        self.generate_hidden_layers(hidden_layers)

        self.mean = nn.Linear(final_hidden_layer_size, self.latent_size)
        self.log_var = nn.Linear(final_hidden_layer_size, self.latent_size)

    def generate_hidden_layers(self, hidden_layers: dict):
        hidden_layer_sizes = np.asarray(hidden_layers)[:, 0].astype(int)
        hidden_layer_types = np.asarray(hidden_layers)[:, 1]
        hidden_layer_activations = np.asarray(hidden_layers)[:, 2]
        logger.info(
            f"Building encoder with hidden layer sizes: {hidden_layer_sizes}; types: {hidden_layer_types}; and activations: {hidden_layer_activations}"
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

    def forward(
        self, mts: np.ndarray, feature_info: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        """Encoder in VAE from (Kingma and Welling, 2013) return the mean and the log of the variance."""

        assert (
            mts.shape[1] == self.input_size_without_conditions
        ), f"MTS size mismatch. Expected {self.input_size_without_conditions}, got {mts.shape[1]}"
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Feature size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"

        input = cat((mts, feature_info), dim=1)
        hidden_layer_input = self.input_layer(input)
        for i in range(len(self.hidden_layers)):
            if i == 0:
                encoded_input = self.hidden_layers[i](hidden_layer_input)
            else:
                encoded_input = self.hidden_layers[i](encoded_input)
        latent_mean: Tensor = self.mean(encoded_input)
        latent_log_var: Tensor = self.log_var(encoded_input)
        return latent_mean, latent_log_var


class Decoder(nn.Module):

    def __init__(
        self,
        mts_size: int,
        number_of_conditions: int,
        latent_size: int,
        hidden_layers: dict,
    ) -> None:
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.number_of_conditions = number_of_conditions
        self.mts_size = mts_size

        self.input_size = latent_size + number_of_conditions
        final_hidden_layer_size = int(hidden_layers[0][0])

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, hidden_layers[-1][0]), nn.ReLU()
        )

        self.generate_hidden_layers(hidden_layers)

        self.output: Tensor = nn.Linear(final_hidden_layer_size, mts_size)

    def generate_hidden_layers(self, hidden_layers: dict):
        reversed_hidden_layers = hidden_layers[::-1]
        hidden_layer_sizes = np.asarray(reversed_hidden_layers)[:, 0].astype(int)
        hidden_layer_types = np.asarray(reversed_hidden_layers)[:, 1]
        hidden_layer_activations = np.asarray(reversed_hidden_layers)[:, 2]
        logger.info(
            f"Building decoder with hidden layer sizes: {hidden_layer_sizes}; types: {hidden_layer_types}; and activations: {hidden_layer_activations}"
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

    def forward(self, feature_info: np.ndarray, latent: Tensor) -> Tensor:
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Feature size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"
        assert (
            latent.shape[1] == self.latent_size
        ), f"Latent size mismatch. Expected {self.latent_size}, got {latent.shape[1]}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input = cat((latent, feature_info), dim=1).to(device)
        hidden_layer_input = self.input_layer(input)
        for i in range(len(self.hidden_layers)):
            if i == 0:
                decoded_input = self.hidden_layers[i](hidden_layer_input)
            else:
                decoded_input = self.hidden_layers[i](decoded_input)
        output = self.output(decoded_input)
        return output
