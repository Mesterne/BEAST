import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt


# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)


import wandb
import uuid
from src.utils.yaml_loader import read_yaml  # noqa: E402
from src.utils.generate_dataset import generate_feature_dataframe  # noqa: E402
from src.utils.pca import PCAWrapper  # noqa: E402
from src.utils.experiment_helper import (  # noqa: E402
    get_mts_dataset,
)
from src.data_transformations.generation_of_supervised_pairs import (  # noqa: E402
    create_train_val_test_split,
    get_col_names_original_target,
)
from src.utils.logging_config import logger  # noqa: E402
from src.models.naive_correlation import CorrelationModel
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)  # noqa: E402
from src.plots.full_time_series import plot_time_series_for_all_uts  # noqa: E402
from src.utils.data_formatting import use_model_predictions_to_create_dataframe

# Set up logging
logger.info(f"Running from directory: {project_root}")

# Setting seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Set up plotting styles
plt.style.use("ggplot")

# Load the configuration file
# TODO: Sort
config = read_yaml(args["config_path"])

data_dir = os.path.join(config["dataset_args"]["directory"], "train.csv")
timeseries_to_use = config["dataset_args"]["timeseries_to_use"]
step_size = config["dataset_args"]["step_size"]
context_length = config["dataset_args"]["window_size"]

log_training_to_wandb = config["training_args"]["log_to_wandb"]

seasonal_period = config["stl_args"]["series_periodicity"]

num_features_per_uts = config["time_series_args"]["num_features_per_uts"]

model_type = config["model_args"]["model_type"]
model_params = config["model_args"]

# Set up directory where all outputs will be stored
# This is an environment variable that can be set before running the program
output_dir = os.getenv("OUTPUT_DIR", "")

# Set up logging to wandb
if log_training_to_wandb:
    job_name = os.environ.get("JOB_NAME", str(uuid.uuid4()))
    wandb.init(
        project="MTS-BEAST",
        name=job_name,
        config=config["model_args"]["feature_model_args"],
    )


logger.info("Initialized system")
logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {output_dir} (Relative to where you ran the program from)"
)

# Load data and generate dataset of multivariate time series context windows
# TODO: Make return dataframe
mts_dataset = get_mts_dataset(
    data_dir=data_dir,
    time_series_to_use=timeseries_to_use,
    context_length=context_length,
    step_size=step_size,
)
dataset_size = len(mts_dataset)
num_uts_in_mts = len(timeseries_to_use)

logger.info("Successfully generated multivariate time series dataset")

mts_feature_df, mts_decomps = generate_feature_dataframe(
    data=mts_dataset, series_periodicity=seasonal_period, dataset_size=dataset_size
)


logger.info("Successfully generated feature dataframe")

# Generate train, vlaidation and test splits
ORIGINAL_NAMES, DELTA_NAMES, TARGET_NAMES = get_col_names_original_target()

# Generate PCA space used to create train test splits
mts_pca_df = PCAWrapper().fit_transform(mts_feature_df)
logger.info("Successfully generated MTS PCA space")

(
    X_train,
    y_train,
    X_validation,
    y_validation,
    X_test,
    y_test,
    train_supervised_dataset,
    validation_supervised_dataset,
    test_supervised_dataset,
) = create_train_val_test_split(
    mts_pca_df,
    mts_feature_df,
    ORIGINAL_NAMES,
    DELTA_NAMES,
    TARGET_NAMES,
    SEED,
    output_dir=output_dir,
)


# FIXME: Currently only gets a random row from the validation set.
# Should get the best, worst and average performing prediction

# Get random model input
df_model_input = validation_supervised_dataset.sample(1)
# Original MTS
original_mts_index = df_model_input["original_index"].values[0].astype(int)
original_mts = mts_dataset[original_mts_index]
original_mts_decomps = mts_decomps[original_mts_index]
original_mts_features = mts_feature_df.iloc[original_mts_index]
# Target MTS
target_mts_index = df_model_input["target_index"].values[0].astype(int)
target_mts = mts_dataset[target_mts_index]
target_mts_features = mts_feature_df.iloc[target_mts_index]

params = {
    "number_of_features_in_each_uts": 4,
    "number_of_uts_in_mts": 3,
}


feature_model = CorrelationModel(params)
ga = GeneticAlgorithmWrapper(
    model_type=model_type,
    model_params=model_params,
    mts_dataset=mts_dataset,
    mts_features=mts_feature_df,
    mts_decomp=mts_decomps,
    num_uts_in_mts=num_uts_in_mts,
    num_features_per_uts=num_features_per_uts,
)
logger.info(f"Successfully initialized the {model_type} model")

# Fit model to data
feature_model.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_validation,
    y_val=y_validation,
    log_to_wandb=False,
)

validation_predicted_features = feature_model.infer(X_validation)
validation_predicted_features = use_model_predictions_to_create_dataframe(
    validation_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=validation_supervised_dataset,
)

sampled_predicted_features = validation_predicted_features.sample(1)
sampled_prediction_index = (
    sampled_predicted_features["prediction_index"].astype(int).values[0]
)
sampled_row_in_validation = validation_supervised_dataset.iloc[sampled_prediction_index]
sampled_original_mts_index = int(sampled_row_in_validation["original_index"])
sampled_target_mts_index = int(sampled_row_in_validation["target_index"])
sampled_predicted_features = sampled_predicted_features.drop(
    ["prediction_index"], axis=1
)

(
    transformed_mts_list,
    transformed_features_list,
    transformed_factors_list,
) = ga.transform(
    predicted_features=sampled_predicted_features,
    original_mts_index=sampled_original_mts_index,
    target_mts_index=sampled_target_mts_index,
)

DEFAULT_RUN_INDEX = 0
transformed_features = transformed_features_list[DEFAULT_RUN_INDEX]
transformed_mts = transformed_mts_list[DEFAULT_RUN_INDEX]

transformed_features = [item for sublist in transformed_features for item in sublist]

transformed_features = pd.DataFrame(
    [transformed_features], columns=sampled_predicted_features.columns
)

uts_names = ["grid1-load", "grid1-loss", "grid1-temp"]

uts_wise_pca_fig = plot_pca_for_each_uts_with_transformed(
    mts_features_df=mts_feature_df,
    predicted_features=sampled_predicted_features,
    transformed_features=transformed_features,
    original_index=sampled_original_mts_index,
    target_index=sampled_target_mts_index,
    uts_names=uts_names,
)
uts_wise_pca_fig.savefig(os.path.join(output_dir, "uts_wise_pca.png"))
logger.info("Successfully generated PCA spaces")


transformed_mts = pd.DataFrame(
    {name: transformed_mts[i] for i, name in enumerate(original_mts.columns)}
)

full_time_series_fig = plot_time_series_for_all_uts(
    original_mts=original_mts,
    target_mts=target_mts,
    transformed_mts=transformed_mts,
    original_mts_features=original_mts_features,
    target_mts_features=target_mts_features,
    transformed_mts_features=transformed_features,
)
full_time_series_fig.savefig(os.path.join(output_dir, "full_time_series.png"))
logger.info("Successfully generated full time series plots")
