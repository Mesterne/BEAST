from typing import List, Tuple

import numpy as np
import torch
from torch import Tensor, cat, exp, nn, randn, randn_like

from data.constants import NETWORK_ARCHITECTURES
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
        self.input_size_without_conditions = model_params[
            "input_size_without_conditions"
        ]
        self.number_of_conditions = model_params["number_of_conditions"]
        self.latent_size = model_params["latent_size"]
        self.condition_type = model_params["condition_type"]
        self.output_size = model_params["output_size"]
        self.uts_size = model_params["uts_size"]
        self.encoder = Encoder(
            mts_size=self.mts_size,
            input_size_without_conditions=self.input_size_without_conditions,
            number_of_conditions=self.number_of_conditions,
            latent_size=self.latent_size,
            feedforward_layers=model_params["feedforward_layers"],
            uts_size=self.uts_size,
            architecture=model_params["architecture"],
            rnn_hidden_state_size=model_params["rnn_hidden_state_size"],
            convolutional_layers=model_params["convolutional_layers"],
            transformer_args=model_params["transformer_args"],
        )
        self.decoder = Decoder(
            self.mts_size,
            self.uts_size,
            self.output_size,
            self.number_of_conditions,
            self.latent_size,
            model_params["feedforward_layers"],
            model_params["conv_decoder"],
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

        input_except_conditions: np.ndarray = input[
            :, : self.input_size_without_conditions
        ]
        feature_info: np.ndarray = input[:, self.input_size_without_conditions :]

        assert (
            input_except_conditions.shape[1] == self.input_size_without_conditions
        ), f"Input without condtions size mismatch. Expected {self.input_size_without_conditions}, got {input_except_conditions.shape[1]}"
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Conditions size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"

        # Encoding step
        latent_mean, latent_log_var = self.encoder(
            input_except_conditions, feature_info
        )

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
        input_size_without_conditions: int,
        number_of_conditions: int,
        latent_size: int,
        feedforward_layers: dict,
        uts_size: int,
        architecture: str,
        rnn_hidden_state_size: int,
        convolutional_layers: list,
        transformer_args: dict,
    ) -> None:
        super(Encoder, self).__init__()
        self.mts_size = mts_size
        self.uts_size = uts_size
        self.num_uts = self.mts_size // self.uts_size
        self.input_size_without_conditions = input_size_without_conditions
        self.number_of_conditions = number_of_conditions
        self.latent_size = latent_size
        self.architecture = architecture
        self.rnn_hidden_state_size = rnn_hidden_state_size
        self.convolutional_layers_list = convolutional_layers
        self.transformer_args = transformer_args
        self.feedforward_layers_list = feedforward_layers
        final_hidden_layer_size = int(feedforward_layers[-1][0])

        self.add_input_layer()
        self.generate_feedforward_layers()

        self.combination_layer_input_size = (
            self.determine_combination_layer_input_size()
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
        if self.architecture == NETWORK_ARCHITECTURES[1]:
            logger.info("Building encoder with LSTM input layer")
            self.generate_lstm_layer(
                input_size=self.num_uts,
                hidden_size=self.rnn_hidden_state_size,
            )
        if self.architecture == NETWORK_ARCHITECTURES[3]:
            logger.info("Building encoder with CNN input layer")
            self.generate_cnn_layer(conv_layer_list=self.convolutional_layers_list)
        if self.architecture == NETWORK_ARCHITECTURES[4]:
            logger.info("Building encoder with attention input layer")
            self.generate_attention_layer()
        if self.architecture == NETWORK_ARCHITECTURES[5]:
            logger.info("Building encoder with transformer input layer")
            self.generate_transformer_layer(transformer_args=self.transformer_args)

    def determine_combination_layer_input_size(self) -> int:
        """Determine the input size of the combination layer based on the architecture."""
        if self.architecture == NETWORK_ARCHITECTURES[0]:
            return self.mts_size + self.number_of_conditions
        if self.architecture == NETWORK_ARCHITECTURES[1]:
            return self.rnn_hidden_state_size + self.number_of_conditions
        if self.architecture == NETWORK_ARCHITECTURES[3]:
            conv_output_length = self.calc_conv_out_len(self.convolutional_layers_list)
            # conv_output_channels = self.convolutional_layers_list[-1][1]
            conv_output_size = (
                conv_output_length  # NOTE Testing mean pooling across channels
            )
            # conv_output_size = conv_output_length * conv_output_channels
            return conv_output_size + self.number_of_conditions
        if self.architecture == NETWORK_ARCHITECTURES[4]:
            raise NotImplementedError("Attention layer is not implemented yet")
        if self.architecture == NETWORK_ARCHITECTURES[5]:
            # FIXME: This seems to be the formula, but i am not sure why we multiply by 3 and divide by 4
            transformer_encoder_output_length = (
                self.transformer_args["dim_feedforward"]
                * 3
                * self.transformer_args["dim_model"]
                // 4
            )
            return transformer_encoder_output_length + self.number_of_conditions

    def generate_lstm_layer(self, input_size: int, hidden_size: int) -> None:
        self.input_layer = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, batch_first=True
        )

    def generate_cnn_layer(self, conv_layer_list: list):
        """Generate CNN layer based on the given list of convolutional layers.
        Based on the temporal convolutional network (TCN) paper by Bai et al. (2018).
        The paper proposes a CNN with causal dilated convolutions.
        This helps to capture long-term dependencies in the time series.
        """
        self.input_layer = nn.Sequential()
        # TODO: Consider pooling
        for i in range(len(conv_layer_list)):
            in_channels = conv_layer_list[i][0]
            out_channels = conv_layer_list[i][1]
            kernel_size = conv_layer_list[i][2]
            stride = conv_layer_list[i][3]
            padding = kernel_size - 1  # For causal convolutions
            dilation = 2**i  # For dilated convolutions
            self.input_layer.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                    dilation=dilation,
                )
            )
            self.input_layer.append(nn.ReLU())
            # self.input_layer.append(nn.LeakyReLU(negative_slope=0.01))

    def generate_attention_layer(self) -> None:
        pass

    def generate_transformer_layer(self, transformer_args: dict) -> None:
        """
        Generate attention layer based on the given transformer arguments.
        Using the transformer encoder layer from PyTorch, which is based on the paper by Vaswani et al. (2017).
        The transformer encoder layer is a multi-head self-attention layer with a feedforward network.
        The transformer encoder is a stack of N layers.
        """
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=transformer_args["dim_model"],
            nhead=transformer_args["num_heads"],
            dim_feedforward=transformer_args["dim_feedforward"],
            dropout=transformer_args["dropout_rate"],
            activation=transformer_args["activation"],
            batch_first=True,
        )
        self.input_layer = nn.TransformerEncoder(
            self.transformer_layer, num_layers=transformer_args["num_layers"]
        )
        self.embedding_layer = nn.Linear(
            self.num_uts, self.transformer_args["dim_model"]
        )

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

    def calc_conv_out_len(self, conv_layer_list: list) -> int:
        """
        Calculate the output length of the convolutional layer.
        Using the formula: L_out = floor([(L_in + 2*padding - dilation*[kernel_size -1] - 1) / stride] + 1)
        The formula is applied iteratively through all layers to find the final output length.
        The formula is taken from torch Conv1d documentation:
        https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html#torch.nn.Conv1d
        """
        kernel_size = conv_layer_list[0][2]  # Constant for all layers
        stride = conv_layer_list[0][3]  # Constant for all layers
        padding = kernel_size - 1  # For causal convolutions

        length = self.uts_size
        for i in range(len(conv_layer_list)):
            dilation = 2**i
            length = (
                (length + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride
            ) + 1
            length = np.floor(length).astype(int)
        return length

    def format_input(self, input: np.ndarray) -> Tensor:
        if self.architecture == NETWORK_ARCHITECTURES[0]:
            return input
        if self.architecture == NETWORK_ARCHITECTURES[1]:
            # Reshape input to (batch_size, seq_len, input_size)
            return input.reshape(input.shape[0], self.uts_size, self.num_uts)
        if self.architecture == NETWORK_ARCHITECTURES[3]:
            # Reshape input to (batch_size, num_channels (uts), input_size)
            return input.reshape(input.shape[0], self.num_uts, self.uts_size)
        if self.architecture == NETWORK_ARCHITECTURES[5]:
            # Reshape input to (batch_size, seq_len, input_size)
            return input.reshape(input.shape[0], self.uts_size, self.num_uts)

    def positional_encoding(self) -> Tensor:
        """
        Add positional encoding to the input.
        The positional encoding is added to the input layer to give the model a sense of order.
        The positional encoding is based on the paper by Vaswani et al. (2017).
        The positional encoding is added to the input layer to give the model a sense of order.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dim_model = self.transformer_args["dim_model"]
        max_len = self.uts_size
        positional_encoding_matrix = torch.zeros(max_len, dim_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1).to(device)
        div_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * -(np.log(10000.0) / dim_model)
        ).to(device)
        positional_encoding_matrix[:, 0::2] = torch.sin(position * div_term)
        positional_encoding_matrix[:, 1::2] = torch.cos(position * div_term)
        positional_encoding_matrix = positional_encoding_matrix.unsqueeze(0)
        return positional_encoding_matrix

    def forward(
        self, input_without_conditions: np.ndarray, feature_info: np.ndarray
    ) -> Tuple[Tensor, Tensor]:
        """Encoder in VAE from (Kingma and Welling, 2013) return the mean and the log of the variance."""

        assert (
            input_without_conditions.shape[1] == self.input_size_without_conditions
        ), f"MTS size mismatch. Expected {self.input_size_without_conditions}, got {input_without_conditions.shape[1]}"
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Feature size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"

        input = self.format_input(input_without_conditions)

        if self.architecture == NETWORK_ARCHITECTURES[1]:
            lstm_out, _ = self.input_layer(input)
            lstm_out_final_hidden_state = lstm_out[:, -1, :]
            combination_layer_input = cat(
                (lstm_out_final_hidden_state, feature_info), dim=1
            )
        if self.architecture == NETWORK_ARCHITECTURES[3]:
            conv_out = self.input_layer(input)
            conv_out_flattened = conv_out.mean(
                dim=1
            )  # NOTE Testing mean pooling across channels
            # conv_out_flattened = conv_out.view(
            #     conv_out.shape[0], conv_out.shape[1] * conv_out.shape[2]
            # )
            combination_layer_input = cat((conv_out_flattened, feature_info), dim=1)
        if self.architecture == NETWORK_ARCHITECTURES[5]:
            input_embedding = self.embedding_layer(input)
            positional_encoded_input_embedding = (
                input_embedding + self.positional_encoding()
            )
            transformer_out = self.input_layer(positional_encoded_input_embedding)
            transformer_out_flattened = transformer_out.view(
                transformer_out.shape[0],
                transformer_out.shape[1] * transformer_out.shape[2],
            )
            combination_layer_input = cat(
                (transformer_out_flattened, feature_info), dim=1
            )
        if self.architecture == NETWORK_ARCHITECTURES[0]:
            combination_layer_input = cat(
                (input_without_conditions, feature_info), dim=1
            )

        feedforward_input = self.input_condition_combination_layer(
            combination_layer_input
        )

        for i in range(len(self.feedforward_layers)):
            if i == 0:
                encoded_input = self.feedforward_layers[i](feedforward_input)
            else:
                encoded_input = self.feedforward_layers[i](encoded_input)
        # NOTE: Add layer normalization to allow for more informative latent space
        normalized_encoded_input = nn.LayerNorm(encoded_input.shape[1]).to(
            encoded_input.device
        )(encoded_input)
        latent_mean: Tensor = self.mean(normalized_encoded_input)
        latent_log_var: Tensor = self.log_var(normalized_encoded_input)

        return latent_mean, latent_log_var


class Decoder(nn.Module):

    def __init__(
        self,
        mts_size: int,
        uts_size: int,
        output_size: int,
        number_of_conditions: int,
        latent_size: int,
        feedforward_layers: dict,
        conv_decoder_args: dict,
    ) -> None:
        super(Decoder, self).__init__()
        self.latent_size = latent_size
        self.number_of_conditions = number_of_conditions
        self.mts_size = mts_size
        self.uts_size = uts_size
        self.num_uts = self.mts_size // self.uts_size
        self.input_size = latent_size + number_of_conditions
        final_hidden_layer_size = int(feedforward_layers[0][0])

        self.use_conv_decoder = conv_decoder_args["use_conv_decoder"]
        if self.use_conv_decoder:
            logger.info("Building decoder with convolutional layers")
            self.generate_conv_decoder_layers(conv_dec_args=conv_decoder_args)
        else:
            logger.info("Building decoder with feedforward layers")
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

    def get_connecting_layer_size(self, conv_transpose_layer_list: list) -> int:
        """
        Calculate input size of conv_transpose layer based on the desired output size of the decoder.
        Based on the following formula:
        https://pytorch.org/docs/stable/generated/torch.nn.ConvTranspose1d.html
        To find the initial input size L_in given a desired ouput size L_out, we rearrange the formula to find the input size.
        Thus the formula is the same as for the normal convolution, only swithching L_in and L_out.
        """
        kernel_size = conv_transpose_layer_list[0][2]
        padding = kernel_size - 1  # For causal convolutions
        dilation = 1
        stride = conv_transpose_layer_list[0][3]

        # length = self.uts_size
        # for i in range(len(conv_transpose_layer_list)):
        #     length = (
        #         stride * (length - 1) - 2 * padding + dilation * (kernel_size - 1) + 1
        #     )
        # length = np.floor(length).astype(int)
        length = self.uts_size
        for i in range(len(conv_transpose_layer_list)):
            length = (
                (length + (2 * padding) - (dilation * (kernel_size - 1)) - 1) / stride
            ) + 1
            length = np.floor(length).astype(int)
        self.init_channel_size = conv_transpose_layer_list[0][0]
        return length * self.init_channel_size

    def generate_conv_decoder_layers(self, conv_dec_args: list):
        self.feedforward_layers = nn.Sequential()
        feedforward_list = conv_dec_args["feedforward_layers"]
        input_size = self.input_size
        for i in range(len(feedforward_list)):
            self.feedforward_layers.append(
                nn.Linear(input_size, feedforward_list[i][0])
            )
            self.feedforward_layers.append(nn.LeakyReLU(negative_slope=0.01))
            input_size = feedforward_list[i][0]
        # NOTE: Feedforward network need a final output size that results in L_out = 192.
        conv_transpose_list = conv_dec_args["conv_transpose_layers"]
        connecting_layer_size = self.get_connecting_layer_size(conv_transpose_list)
        self.feedforward_layers.append(nn.Linear(input_size, connecting_layer_size))
        self.feedforward_layers.append(nn.LeakyReLU(negative_slope=0.01))
        self.conv_transpose_layers = nn.Sequential()
        for i in range(len(conv_transpose_list)):
            in_channels = conv_transpose_list[i][0]
            out_channels = conv_transpose_list[i][1]
            kernel_size = conv_transpose_list[i][2]
            stride = conv_transpose_list[i][3]
            padding = kernel_size - 1
            self.conv_transpose_layers.append(
                nn.ConvTranspose1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    padding=padding,
                )
            )
            self.conv_transpose_layers.append(nn.LeakyReLU(negative_slope=0.01))

    def forward(self, feature_info: np.ndarray, latent: Tensor) -> Tensor:
        assert (
            feature_info.shape[1] == self.number_of_conditions
        ), f"Feature size mismatch. Expected {self.number_of_conditions}, got {feature_info.shape[1]}"
        assert (
            latent.shape[1] == self.latent_size
        ), f"Latent size mismatch. Expected {self.latent_size}, got {latent.shape[1]}"

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        input = cat((latent, feature_info), dim=1).to(device)

        if self.use_conv_decoder:
            feedforward_output = self.feedforward_layers(input)
            conv_transpose_input = feedforward_output.reshape(
                feedforward_output.shape[0], self.init_channel_size, -1
            )
            conv_transpose_output = self.conv_transpose_layers(conv_transpose_input)
            output = conv_transpose_output.reshape(feedforward_output.shape[0], -1)
        else:
            hidden_layer_input = self.input_layer(input)
            for i in range(len(self.feedforward_layers)):
                if i == 0:
                    decoded_input = self.feedforward_layers[i](hidden_layer_input)
                else:
                    decoded_input = self.feedforward_layers[i](decoded_input)
            output = self.output(decoded_input)
        return output
