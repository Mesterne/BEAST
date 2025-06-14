import os

COLUMN_NAMES = [
    "grid1-load_trend-strength",
    "grid1-load_trend-slope",
    "grid1-load_trend-linearity",
    "grid1-load_seasonal-strength",
    "grid1-loss_trend-strength",
    "grid1-loss_trend-slope",
    "grid1-loss_trend-linearity",
    "grid1-loss_seasonal-strength",
    "grid1-temp_trend-strength",
    "grid1-temp_trend-slope",
    "grid1-temp_trend-linearity",
    "grid1-temp_seasonal-strength",
]
PLOT_NAMES = [
    "grid-load_trend-strength",
    "grid-load_trend-slope",
    "grid-load_trend-linearity",
    "grid-load_seasonal-strength",
    "grid-loss_trend-strength",
    "grid-loss_trend-slope",
    "grid-loss_trend-linearity",
    "grid-loss_seasonal-strength",
    "grid-temp_trend-strength",
    "grid-temp_trend-slope",
    "grid-temp_trend-linearity",
    "grid-temp_seasonal-strength",
]

FEATURE_NAMES = [
    "trend-strength",
    "trend-slope",
    "trend-linearity",
    "seasonal-strength",
]

UTS_NAMES = [
    "grid1-load",
    "grid1-loss",
    "grid1-temp",
]


BEATUIFUL_UTS_NAMES = [
    "Grid Load",
    "Grid Loss",
    "Grid Temp",
]

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "local_outputs")

NETWORK_ARCHITECTURES = [
    "feedforward",
    "rnn",
    "rnn_enc_dec",
    "convolution",
    "attention",
    "transformer",
]
