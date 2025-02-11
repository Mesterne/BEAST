import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import torch
import logging

# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

from src.utils.yaml_loader import read_yaml
from src.utils.generate_dataset import (
    generate_windows_dataset,
    generate_feature_dataframe,
)

# Setting seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the configuration file
config = read_yaml(args["config_path"])

# Data loading parameters
data_dir = os.path.join(config["dataset_args"]["directory"], "train.csv")
timeseries_to_use = config["dataset_args"]["timeseries_to_use"]
step_size = config["dataset_args"]["step_size"]
context_length = config["dataset_args"]["window_size"]

# Load data
df = pd.read_csv(data_dir, index_col=0)
df.index = pd.to_datetime(df.index)
df = df[timeseries_to_use]
# Backfill missing values
df = df.bfill()

# Generate dataset of multivariate time series context windows
num_ts = len(timeseries_to_use)
dataset_size = (df.shape[0] - context_length) // step_size

mts_dataset = generate_windows_dataset(df, context_length, step_size, timeseries_to_use)

# Generate feature dataframe
sp = config["stl_args"]["series_periodicity"]
mts_feature_df = generate_feature_dataframe(
    data=mts_dataset, series_periodicity=sp, dataset_size=dataset_size
)
logging.info("Successfully generated feature dataframe")
