from torch import nn, cat, Tensor, randn_like, exp
from src.utils.logging_config import logger
import numpy as np
from typing import Tuple


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
        self.num_features = model_params["num_features"]
        self.latent_size = model_params["latent_size"]

        self.encoder = Encoder(self.mts_size, self.num_features, self.latent_size)
        self.decoder = Decoder(self.mts_size, self.num_features, self.latent_size)

    def forward(self, input: np.ndarray) -> np.ndarray:
        """
        Reparameterization trick using standard normal distribution to sample from the latent space.
        (Kingma and Welling, 2013) notes that z = location + scale * noise,
        where noise ~ N(0, 1), is a valid reparameterization.
        Sampling from a standard normal distribution is one of several possible reparameterization strategies.
        """

        # First mts_size elements are the MTS, the rest is feature info
        assert (
            input.shape[1] == self.mts_size + self.num_features
        ), f"Input size mismatch. Expected {self.mts_size + self.num_features}, got {input.shape[1]}"

        mts: np.ndarray = input[:, : self.mts_size]
        # NOTE: Feature info may be both feature values or delta values
        feature_info: np.ndarray = input[:, self.mts_size :]

        # Encoding step
        latent_mean, latent_log_var = self.encoder(mts, feature_info)

        # Reparameterization trick
        noise = randn_like(
            latent_mean
        )  # Sample noise from standard normal distribution. Same shape as latent_mean
        standard_deviation = exp(
            0.5 * latent_log_var
        )  # Calculate standard deviation from log variance
        latent_vector = latent_mean + standard_deviation * noise

        # Decoding step
        output = self.decoder(feature_info, latent_vector)

        return output

    def generate_mts(self, feature_info: np.ndarray) -> np.ndarray:
        """Generate MTS data given feature info as condition."""
        latent_vector = randn_like((1, self.latent_size))
        return self.decoder(feature_info, latent_vector)


class Encoder(nn.Module):
    def __init__(self, mts_size: int, num_features: int, latent_size: int) -> None:
        super(Encoder, self).__init__()
        self.mts = mts_size
        self.num_features = num_features
        self.latent_size = latent_size

        self.input_size = mts_size + num_features

        # FIXME: Hardcoded feedforward network. Change to config file.
        self.feedforward = nn.Sequential(
            nn.Linear(self.input_size, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        self.mean = nn.Linear(128, self.latent_size)
        self.log_var = nn.Linear(128, self.latent_size)

    def forward(
        self, mts: np.ndarray, feature_info: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        """Encoder in VAE from (Kingma and Welling, 2013) return the mean and the log of the variance."""

        assert (
            mts.shape[1] == self.mts_size
        ), f"MTS size mismatch. Expected {self.mts_size}, got {mts.shape[1]}"
        assert (
            feature_info.shape[1] == self.num_features
        ), f"Feature size mismatch. Expected {self.num_features}, got {feature_info.shape[1]}"

        input = cat((mts, feature_info), dim=1)
        encoded_input = self.feedforward(input)
        latent_mean: Tensor = self.mean(encoded_input)
        latent_log_var: Tensor = self.log_var(encoded_input)
        return latent_mean, latent_log_var


class Decoder(nn.Module):
    def __init__(self, latent_size: int, num_features: int, mts_size: int) -> None:
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.num_features = num_features
        self.mts_size = mts_size

        self.input_size = latent_size + num_features

        # FIXME: Hardcoded feedforward network. Change to config file
        # FIXME: Feedforward may not be the best choice for MTS data.
        self.feedforward = nn.Sequential(
            nn.Linear(self.input_size, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
        )

        self.output: Tensor = nn.Linear(512, mts_size)

    def forward(self, feature_info: np.ndarray, latent: Tensor) -> np.ndarray:
        assert (
            feature_info.shape[1] == self.num_features
        ), f"Feature size mismatch. Expected {self.num_features}, got {feature_info.shape[1]}"
        assert (
            latent.shape[1] == self.latent_size
        ), f"Latent size mismatch. Expected {self.latent_size}, got {latent.shape[1]}"

        input = cat((latent, feature_info), dim=1)
        decoded_input = self.feedforward(input)
        output = self.output(decoded_input)
        return output.numpy()
