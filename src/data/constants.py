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

UTS_NAMES = [
    "grid1-load",
    "grid1-loss",
    "grid1-temp",
]

OUTPUT_DIR = os.getenv("OUTPUT_DIR", "")
