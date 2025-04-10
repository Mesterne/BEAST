from typing import Dict

import numpy as np
import torch
from torch import Tensor, cat, exp, nn, randn_like, relu


class RNNCVAE(nn.Module):
    def __init__(self, model_params: Dict[str, any]) -> None:
        super(RNNCVAE, self).__init__()
        self.uts_size = model_params["uts_size"]
        self.mts_size = model_params["mts_size"]
        self.num_uts_in_mts = self.mts_size // self.uts_size
        self.condition_type = model_params["condition_type"]
        self.input_size_without_conditions = model_params[
            "input_size_without_conditions"
        ]
        self.num_conditions = model_params["number_of_conditions"]
        self.hidden_size = model_params["rnn_hidden_state_size"]
        self.latent_size = model_params["latent_size"]
        self.encoder = Encoder(
            uts_size=self.uts_size,
            num_uts_in_mts=self.num_uts_in_mts,
            num_conditions=self.num_conditions,
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
        )
        self.decoder = Decoder(
            uts_size=self.uts_size,
            num_uts_in_mts=self.num_uts_in_mts,
            latent_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_conditions=self.num_conditions,
        )

    def forward(self, input: Tensor) -> Tensor:
        # NOTE: Input has shape like flattened mts (batch_size, seq_len*num_uts + num_conditions)
        mts_flattened: Tensor = input[:, : self.mts_size]
        feature_info: Tensor = input[:, self.mts_size :]

        assert (
            feature_info.shape[1] == self.num_conditions
        ), f"MTS size mismatch. Expected {self.input_size_without_conditions}, got {mts_flattened.shape[1]}"
        assert (
            mts_flattened.shape[1] == self.mts_size
        ), f"Conditions size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"

        # NOTE: Reshape flattened mts to (batch_size, seq_len, num_uts)
        mts: Tensor = mts_flattened.reshape(
            mts_flattened.shape[0], self.uts_size, self.num_uts_in_mts
        )

        # Encoding
        latent_mean, latent_logvar = self.encoder(mts, feature_info)

        # Reparameterization trick
        latent_vector: Tensor = self.reparamterization_trick(latent_mean, latent_logvar)

        # NOTE: Create sequence representation of latent vector by repeating it for each time step
        # TODO: Investigate whether the decoder also needs a recurrent network
        latent_vector_sequence = latent_vector.unsqueeze(1).repeat(1, mts.shape[1], 1)

        assert (
            latent_vector_sequence.shape[1] == mts.shape[1]
        ), f"Latent vector shape mismatch. Expected {mts.shape[1]}, got {latent_vector.shape[1]}"

        output: Tensor = self.decoder(latent_vector_sequence, feature_info)

        return output, latent_mean, latent_logvar

    def reparamterization_trick(self, mean: Tensor, log_var: Tensor) -> Tensor:
        """Reparameterization trick using standard normal distribution to sample from the latent space."""
        noise = randn_like(mean)
        standard_deviation = exp(0.5 * log_var)
        return mean + standard_deviation * noise

    def transform_mts_from_original(
        self, mts_flattened: np.ndarray, conditions: np.ndarray
    ) -> np.ndarray:
        """Transform MTS data given feature deltas as condition."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tensor_mts = torch.tensor(
            mts_flattened.reshape(
                mts_flattened.shape[0], self.uts_size, self.num_uts_in_mts
            ),
            dtype=torch.float32,
        ).to(device)
        tensor_conditions = torch.tensor(conditions, dtype=torch.float32).to(device)
        latent_mean, latent_log_var = self.encoder(tensor_mts, tensor_conditions)
        latent_vector = self.reparamterization_trick(latent_mean, latent_log_var).to(
            device
        )
        latent_vector_sequence = (
            latent_vector.unsqueeze(1).repeat(1, tensor_mts.shape[1], 1).to(device)
        )
        # NOTE: Necessary to move tensor to cpu before converting to numpy
        cpu_mts = self.decoder(latent_vector_sequence, tensor_conditions).cpu()
        return cpu_mts.detach().numpy()


class Encoder(nn.Module):
    def __init__(
        self, uts_size, num_uts_in_mts, num_conditions, hidden_size, latent_size
    ):
        super(Encoder, self).__init__()
        self.uts_size = uts_size
        self.num_uts_in_mts = num_uts_in_mts
        self.num_conditions = num_conditions
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(
            input_size=num_uts_in_mts, hidden_size=hidden_size, batch_first=True
        )

        self.hidden = nn.Linear(
            self.hidden_size + self.num_conditions,
            (self.hidden_size + self.num_conditions) // 2,
        )
        self.mean = nn.Linear(
            (self.hidden_size + self.num_conditions) // 2,
            latent_size,
        )
        self.logvar = nn.Linear(
            (self.hidden_size + self.num_conditions) // 2,
            latent_size,
        )

    def forward(self, mts: Tensor, feature_info: Tensor) -> tuple[Tensor, Tensor]:
        hidden_state_all_time_steps, _ = self.lstm(mts)
        final_hidden_state = hidden_state_all_time_steps[:, -1, :]
        conditioned_lstm_out = cat((final_hidden_state, feature_info), dim=1)
        hidden_out = self.hidden(conditioned_lstm_out)
        hidden_out = relu(hidden_out)
        latent_mean: Tensor = self.mean(hidden_out)
        latent_logvar: Tensor = self.logvar(hidden_out)
        return latent_mean, latent_logvar


class Decoder(nn.Module):
    def __init__(
        self, uts_size, num_uts_in_mts, latent_size, hidden_size, num_conditions
    ):
        super(Decoder, self).__init__()
        self.uts_size = uts_size
        self.num_uts_in_mts = num_uts_in_mts
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_conditions = num_conditions
        self.lstm = nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_size // 2,
            batch_first=True,
        )
        self.hidden = nn.Linear(
            self.hidden_size // 2 + self.num_conditions,
            self.hidden_size,
        )

        self.hidden2 = nn.Linear(
            self.hidden_size,
            self.hidden_size * 2,
        )

        self.output = nn.Linear(
            self.hidden_size * 2, self.uts_size * self.num_uts_in_mts
        )

    def forward(self, latent_vector_seq: Tensor, feature_info: Tensor) -> Tensor:
        hidden_state_all_time_steps, _ = self.lstm(latent_vector_seq)
        final_hidden_state = hidden_state_all_time_steps[:, -1, :]
        conditioned_lstm_out = cat((final_hidden_state, feature_info), dim=1)
        hidden_out = self.hidden(conditioned_lstm_out)
        hidden_out = relu(hidden_out)
        hidden_out = self.hidden2(hidden_out)
        hidden_out = relu(hidden_out)
        output: Tensor = self.output(hidden_out)
        return output
