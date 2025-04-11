from typing import List, Tuple

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
        self.mts_size = model_params["mts_size"]
        self.number_of_conditions = model_params["number_of_conditions"]
        self.latent_size = model_params["latent_size"]
        self.condition_type = model_params["condition_type"]
        self.output_size = model_params["output_size"]
        self.uts_size = model_params["uts_size"]
        self.encoder = Encoder(
            mts_size=self.mts_size,
            number_of_conditions=self.number_of_conditions,
            latent_size=self.latent_size,
            feedforward_layers=model_params[
                "feedforward_layers"
            ],  # TODO: Rename in .yml
            uts_size=self.uts_size,
            architecture=model_params["architecture"],  # TODO: Add to .yml files
            rnn_hidden_state_size=model_params[
                "rnn_hidden_state_size"
            ],  # TODO: Make work when not defined
        )
        self.decoder = Decoder(
            self.mts_size,
            self.output_size,
            self.number_of_conditions,
            self.latent_size,
            model_params["feedforward_layers"],
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
            input.shape[1] == self.mts_size + self.number_of_conditions
        ), f"Input size mismatch. Expected {self.mts_size + self.number_of_conditions}, got {input.shape[1]}"

        mts: np.ndarray = input[:, : self.mts_size]
        feature_info: np.ndarray = input[:, self.mts_size :]

        assert (
            mts.shape[1] == self.mts_size
        ), f"MTS size mismatch. Expected {self.mts_size}, got {mts.shape[1]}"
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
        mts_size: int,
        number_of_conditions: int,
        latent_size: int,
        feedforward_layers: dict,
        uts_size: int,
        architecture: str,
        rnn_hidden_state_size: int,
    ) -> None:
        super(Encoder, self).__init__()
        self.mts_size = mts_size
        self.uts_size = uts_size
        self.num_uts = self.mts_size // self.uts_size
        self.number_of_conditions = number_of_conditions
        self.latent_size = latent_size
        self.architecture = architecture
        self.rnn_hidden_state_size = rnn_hidden_state_size
        self.feedforward_layers_list = feedforward_layers
        final_hidden_layer_size = int(feedforward_layers[-1][0])

        self.add_input_layer()
        self.generate_feedforward_layers()

        self.combination_layer_input_size = (
            self.mts_size + self.number_of_conditions
            if architecture == "feedforward"
            else self.rnn_hidden_state_size + self.number_of_conditions
        )

        self.input_condition_combination_layer = nn.Sequential(
            nn.Linear(
                self.combination_layer_input_size, self.feedforward_layers_list[0][0]
            ),
            nn.ReLU(),
        )

        self.mean = nn.Linear(final_hidden_layer_size, self.latent_size)
        self.log_var = nn.Linear(final_hidden_layer_size, self.latent_size)

    def add_input_layer(self):
        """Add specific feature extraction architecture to the input layer of the encoder."""
        if self.architecture == "feedforward":
            # NOTE: Not necessarily what i want. Maybe this can some extra ff layers to increase parameter count?
            # logger.info("Building encoder with linear input layer")
            # self.input_layer = nn.Sequential(
            #     nn.Linear(
            #         self.combination_layer_input_size,
            #         self.feedforward_layers_list[0][0],
            #     ),
            #     nn.ReLU(),
            # )
            pass
        if self.architecture == "rnn":
            logger.info("Building encoder with LSTM input layer")
            self.generate_lstm_layer(
                input_size=self.num_uts,
                hidden_size=self.rnn_hidden_state_size,
            )
        if self.architecture == "cnn":
            logger.info("Building encoder with CNN input layer")
            self.generate_cnn_layer()
        if self.architecture == "attention":
            logger.info("Building encoder with attention input layer")
            self.generate_attention_layer()

    def generate_lstm_layer(self, input_size: int, hidden_size: int) -> None:
        self.input_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

    def generate_cnn_layer():
        NotImplementedError("CNN layer is not implemented yet")

    def generate_attention_layer():
        NotImplementedError("Attention layer is not implemented yet")

    def generate_feedforward_layers(self):
        layer_sizes = np.asarray(self.feedforward_layers_list)[:, 0].astype(int)
        layer_activations = np.asarray(self.feedforward_layers_list)[:, 1]
        logger.info(
            f"Building encoder with hidden layer sizes: {layer_sizes}; and activations: {layer_activations}"
        )

        self.feedforward_layers = nn.ModuleList()
        for i in range(len(layer_sizes) - 1):
            self.feedforward_layers.append(
                nn.Linear(
                    layer_sizes[i],
                    layer_sizes[i + 1],
                )
            )
            if layer_activations[i] == "relu":
                self.feedforward_layers.append(nn.ReLU())
            else:
                raise ValueError(
                    f"Unknown hidden layer activation: {layer_activations[i]}"
                )

    def format_input(self, input: np.ndarray) -> Tensor:
        if self.architecture == "feedforward":
            return input
        if self.architecture == "rnn":
            # Reshape input to (batch_size, seq_len, input_size)
            return input.reshape(input.shape[0], self.uts_size, self.num_uts)

    def forward(
        self, mts: np.ndarray, feature_info: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        """Encoder in VAE from (Kingma and Welling, 2013) return the mean and the log of the variance."""

        assert (
            mts.shape[1] == self.mts_size
        ), f"MTS size mismatch. Expected {self.mts_size}, got {mts.shape[1]}"
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Feature size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"

        weight = self.input_layer[0].weight
        bias = self.input_layer[0].bias

        assert not torch.isnan(weight).any(), "Detected nan weight value"
        assert not torch.isnan(bias).any(), "Detected nan bias value"

        input = self.format_input(mts)

        if self.architecture == "rnn":
            lstm_out, _ = self.input_layer(input)
            lstm_out_final_hidden_state = lstm_out[:, -1, :]
            combination_layer_input = cat(
                (lstm_out_final_hidden_state, feature_info), dim=1
            )
        if self.architecture == "feedforward":
            combination_layer_input = cat((mts, feature_info), dim=1)

        feedforward_input = self.input_condition_combination_layer(
            combination_layer_input
        )

        for i in range(len(self.feedforward_layers)):
            if i == 0:
                encoded_input = self.feedforward_layers[i](feedforward_input)
            else:
                encoded_input = self.feedforward_layers[i](encoded_input)
        latent_mean: Tensor = self.mean(encoded_input)
        latent_log_var: Tensor = self.log_var(encoded_input)

        return latent_mean, latent_log_var


class Decoder(nn.Module):

    def __init__(
        self,
        mts_size: int,
        output_size: int,
        number_of_conditions: int,
        latent_size: int,
        feedforward_layers: dict,
    ) -> None:
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.number_of_conditions = number_of_conditions
        self.mts_size = mts_size

        self.input_size = latent_size + number_of_conditions
        final_hidden_layer_size = int(feedforward_layers[0][0])

        self.input_layer = nn.Sequential(
            nn.Linear(self.input_size, feedforward_layers[-1][0]), nn.ReLU()
        )

        self.generate_feedforward_layers(feedforward_layers)

        self.output: Tensor = nn.Linear(final_hidden_layer_size, output_size)

    def generate_feedforward_layers(self, feedforward_layers: dict):
        reversed_feedforward_layers = feedforward_layers[::-1]
        hidden_layer_sizes = np.asarray(reversed_feedforward_layers)[:, 0].astype(int)
        hidden_layer_activations = np.asarray(reversed_feedforward_layers)[:, 1]
        logger.info(
            f"Building decoder with hidden layer sizes: {hidden_layer_sizes};and activations: {hidden_layer_activations}"
        )

        self.feedforward_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes) - 1):
            self.feedforward_layers.append(
                nn.Linear(
                    hidden_layer_sizes[i],
                    hidden_layer_sizes[i + 1],
                )
            )
            if hidden_layer_activations[i] == "relu":
                self.feedforward_layers.append(nn.ReLU())
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
        for i in range(len(self.feedforward_layers)):
            if i == 0:
                decoded_input = self.feedforward_layers[i](hidden_layer_input)
            else:
                decoded_input = self.feedforward_layers[i](decoded_input)
        output = self.output(decoded_input)
        return output
