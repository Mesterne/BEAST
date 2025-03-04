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
    get_feature_model_by_type,
    get_mts_dataset,
)
from src.data_transformations.generation_of_supervised_pairs import (  # noqa: E402
    create_train_val_test_split,
    get_col_names_original_target,
)
from src.utils.logging_config import logger  # noqa: E402
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)  # noqa: E402
from src.plots.feature_wise_error import plot_distribution_of_feature_wise_error
from src.utils.evaluation.mse import get_mse_for_features_and_overall
from src.plots.full_time_series import plot_time_series_for_all_uts  # noqa: E402
from src.utils.data_formatting import use_model_predictions_to_create_dataframe
from src.utils.evaluation.feature_space_evaluation import (
    find_error_of_each_feature_for_each_sample,
)
from src.utils.ga_utils import analyze_and_visualize_prediction

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

feature_model_params = config["model_args"]["feature_model_args"]
model_type = feature_model_params["model_name"]
genetic_algorithm_params = config["model_args"]["genetic_algorithm_args"]
training_params = feature_model_params["training_args"]

# Set up directory where all outputs will be stored
# This is an environment variable that can be set before running the program
output_dir = os.getenv("OUTPUT_DIR", "")

# Set up logging to wandb
if log_training_to_wandb:
    job_name = os.environ.get("JOB_NAME", str(uuid.uuid4()))
    wandb.init(
        project="MTS-BEAST",
        name=job_name,
        config=feature_model_params,
    )


logger.info("Initialized system")
logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {output_dir} (Relative to where you ran the program from)"
)

############ DATA INITIALIZATION

# Load data and generate dataset of multivariate time series context windows
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


feature_model_params["number_of_features_in_each_uts"] = num_features_per_uts
feature_model_params["number_of_uts_in_mts"] = num_uts_in_mts

########### MODEL INITIALIZATION

feature_model = get_feature_model_by_type(
    model_type=model_type,
    model_params=feature_model_params,
    training_params=training_params,
)
logger.info(f"Successfully initialized the {model_type} model")
ga = GeneticAlgorithmWrapper(
    ga_params=genetic_algorithm_params,
    mts_dataset=mts_dataset,
    mts_features=mts_feature_df,
    mts_decomp=mts_decomps,
    num_uts_in_mts=num_uts_in_mts,
    num_features_per_uts=num_features_per_uts,
)
logger.info("Successfully initialized the genetic algorithm")

############ TRAINING
# Fit model to data
logger.info("Training feature model...")
feature_model.train(
    X_train=X_train,
    y_train=y_train,
    X_val=X_validation,
    y_val=y_validation,
    log_to_wandb=False,
)
############ INFERENCE

logger.info("Running inference on validation set...")
validation_predicted_features = feature_model.infer(X_validation)
validation_predicted_features = use_model_predictions_to_create_dataframe(
    validation_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=validation_supervised_dataset,
)

logger.info("Running inference on test set...")
test_predicted_features = feature_model.infer(X_test)
test_predicted_features = use_model_predictions_to_create_dataframe(
    test_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=test_supervised_dataset,
)
logger.info("Successfully ran inference on validation and test sets")

logger.info("Sampling predictions to visualize...")
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

############ EVALUATION OF PREDICTIONS ###########################
# Evaluate all predictions
logger.info("Evaluating all predictions")
differences_df_validation = find_error_of_each_feature_for_each_sample(
    predictions=validation_predicted_features,
    labelled_test_dataset=validation_supervised_dataset,
)
differences_df_test = find_error_of_each_feature_for_each_sample(
    predictions=test_predicted_features, labelled_test_dataset=test_supervised_dataset
)

# Plot the feature wise errors of validation set
logger.info("Plotting errors for each prediction on validation set")
fig = plot_distribution_of_feature_wise_error(differences_df_validation)
fig.savefig(os.path.join(output_dir, "dist_error_features.png"))

# Calculate the MSE over feature space.
logger.info(
    f"Calculating overall MSE for predictions of validation set. There are {len(differences_df_validation)} predictions"
)
overall_mse_validation, mse_values_for_each_feature_validation = (
    get_mse_for_features_and_overall(differences_df_validation)
)
logger.info(
    f"Calculating overall MSE for predictions of test set. There are {len(differences_df_test)} predictions"
)
overall_mse_test, mse_values_for_each_feature_test = get_mse_for_features_and_overall(
    differences_df_test
)

logger.info(
    f"Overall MSE for model\nValidation: {overall_mse_validation}\nTest: {overall_mse_test}"
)

# Get the mean absolute error for each prediction, ignoring index column
row_wise_errors = np.abs(differences_df_validation.values[:, :-1]).mean(axis=1)
# Get the index of the worst prediction
worst_row_index = np.argmax(row_wise_errors)
# Get the prediction_index from that row
worst_prediction_index = differences_df_validation.iloc[worst_row_index][
    "prediction_index"
]

# Get the index of the best prediction
best_row_index = np.argmin(row_wise_errors)
# Get the prediction_index from that row
best_prediction_index = differences_df_validation.iloc[best_row_index][
    "prediction_index"
]


# For worst prediction
analyze_and_visualize_prediction(
    prediction_index=int(worst_prediction_index),
    validation_supervised_dataset=validation_supervised_dataset,
    validation_predicted_features=validation_predicted_features,
    mts_dataset=mts_dataset,
    mts_decomps=mts_decomps,
    mts_feature_df=mts_feature_df,
    ga=ga,
    uts_names=timeseries_to_use,
    output_dir=output_dir,
    plot_name_prefix="worst_",
)

# For best prediction
analyze_and_visualize_prediction(
    prediction_index=int(best_prediction_index),
    validation_supervised_dataset=validation_supervised_dataset,
    validation_predicted_features=validation_predicted_features,
    mts_dataset=mts_dataset,
    mts_decomps=mts_decomps,
    mts_feature_df=mts_feature_df,
    ga=ga,
    uts_names=timeseries_to_use,
    output_dir=output_dir,
    plot_name_prefix="best_",
)

# For random prediction
random_index = np.random.randint(len(validation_supervised_dataset))
analyze_and_visualize_prediction(
    prediction_index=int(random_index),
    validation_supervised_dataset=validation_supervised_dataset,
    validation_predicted_features=validation_predicted_features,
    mts_dataset=mts_dataset,
    mts_decomps=mts_decomps,
    mts_feature_df=mts_feature_df,
    ga=ga,
    uts_names=timeseries_to_use,
    output_dir=output_dir,
    plot_name_prefix="random_",
)
logger.info("Generated all plots...")
logger.info("Finished running")
