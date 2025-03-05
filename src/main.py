from ast import GeneratorExp
import logging
from math import log
import os
from pickletools import genops
from re import X
import sys
import argparse
from typing import Generic, List
import pandas as pd
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import trange


# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)


from src import training
from src.models import forecasting
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.plots.timeseries_forecast_comparison import plot_timeseries_forecast_comparison
from src.utils.forecasting_utils import (
    compare_old_and_new_model,
    compare_original_and_transformed_forecasting,
)

import wandb
import uuid
from src.utils.yaml_loader import read_yaml  # noqa: E402
from src.utils.generate_dataset import (
    create_training_windows,
    create_training_windows_from_mts,
    generate_feature_dataframe,
)  # noqa: E402
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
from src.utils.ga_utils import (
    analyze_and_visualize_prediction,
    generate_new_time_series,
)

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

data_dir = os.path.join(
    config["dataset_args"]["directory"], config["dataset_args"]["training_data"]
)
timeseries_to_use = config["dataset_args"]["timeseries_to_use"]
step_size = config["dataset_args"]["step_size"]
context_length = config["dataset_args"]["window_size"]

log_training_to_wandb = config["training_args"]["log_to_wandb"]

seasonal_period = config["stl_args"]["series_periodicity"]

num_features_per_uts = config["time_series_args"]["num_features_per_uts"]

feature_model_params = config["model_args"]["feature_model_args"]
model_type = feature_model_params["model_name"]
genetic_algorithm_params = config["model_args"]["genetic_algorithm_args"]

forecasting_model_params = config["model_args"]["forecasting_model_args"]
forecasting_model_training_params = forecasting_model_params["training_args"]
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
# TODO: Make mts_dataset into ndarray
mts_dataset: List[pd.DataFrame] = get_mts_dataset(
    data_dir=data_dir,
    time_series_to_use=timeseries_to_use,
    context_length=context_length,
    step_size=step_size,
)
dataset_size = len(mts_dataset)
num_uts_in_mts = len(timeseries_to_use)
logger.info(f"MTS Dataset shape: ({len(mts_dataset)}, {len(mts_dataset[0])})")

# TODO: mts_feature_df and mts_decomps should be ndarray
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
    X_features_train,
    y_features_train,
    X_features_validation,
    y_features_validation,
    X_features_test,
    y_features_test,
    train_features_supervised_dataset,
    validation_features_supervised_dataset,
    test_features_supervised_dataset,
) = create_train_val_test_split(
    mts_pca_df,
    mts_feature_df,
    ORIGINAL_NAMES,
    DELTA_NAMES,
    TARGET_NAMES,
    SEED,
    output_dir=output_dir,
)

train_indices = (
    train_features_supervised_dataset["original_index"].astype(int).unique().tolist()
)
validation_indices = (
    validation_features_supervised_dataset["target_index"].astype(int).unique().tolist()
)
test_indices = (
    test_features_supervised_dataset["target_index"].astype(int).unique().tolist()
)

mts_array = np.array([df.values.T for df in mts_dataset])
logger.info(f"Reshaped multivariate time series dataset to shape: {mts_array.shape}")

train_mts_array = [mts_array[i] for i in train_indices]
validation_mts_array = [mts_array[i] for i in validation_indices]
test_mts_array = [mts_array[i] for i in test_indices]

(
    X_mts_train,
    y_mts_train,
) = create_training_windows_from_mts(
    mts=train_mts_array,
    target_col_index=1,
    window_size=forecasting_model_params["window_size"],
    forecast_horizon=forecasting_model_params["horizon_length"],
)
logger.info("Forecasting train data shape: {}".format(X_mts_train.shape))
(
    X_mts_validation,
    y_mts_validation,
) = create_training_windows_from_mts(
    mts=validation_mts_array,
    target_col_index=1,
    window_size=forecasting_model_params["window_size"],
    forecast_horizon=forecasting_model_params["horizon_length"],
)
logger.info("Forecasting validation data shape: {}".format(X_mts_validation.shape))
(
    X_mts_test,
    y_mts_test,
) = create_training_windows_from_mts(
    mts=test_mts_array,
    target_col_index=1,
    window_size=forecasting_model_params["window_size"],
    forecast_horizon=forecasting_model_params["horizon_length"],
)
logger.info("Forecasting test data shape: {}".format(X_mts_test.shape))

feature_model_params["number_of_features_in_each_uts"] = num_features_per_uts
feature_model_params["number_of_uts_in_mts"] = num_uts_in_mts

########### MODEL INITIALIZATION

feature_model = get_feature_model_by_type(
    model_type=model_type,
    model_params=feature_model_params,
    training_params=training_params,
)
logger.info(f"Successfully initialized the {model_type} model")

forecasting_model = FeedForwardForecaster(
    model_params=forecasting_model_params,
)

forecasting_model_wrapper = NeuralNetworkWrapper(
    model=forecasting_model, training_params=forecasting_model_training_params
)
logging.info("Successfully initialized the forecasting model")

# TODO: Inputs should be ndarray
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
    X_train=X_features_train,
    y_train=y_features_train,
    X_val=X_features_validation,
    y_val=y_features_validation,
    log_to_wandb=False,
)

logger.info("Training forecasting model...")
forecasting_model_wrapper.train(
    X_train=X_mts_train,
    y_train=y_mts_train,
    X_val=X_mts_train,
    y_val=y_mts_train,
    log_to_wandb=False,
)

############ INFERENCE

logger.info("Running inference on validation set...")
validation_predicted_features = feature_model.infer(X_features_validation)
validation_predicted_features = use_model_predictions_to_create_dataframe(
    validation_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=validation_features_supervised_dataset,
)

logger.info("Running inference on test set...")
test_predicted_features = feature_model.infer(X_features_test)
test_predicted_features = use_model_predictions_to_create_dataframe(
    test_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=test_features_supervised_dataset,
)
logger.info("Successfully ran inference on validation and test sets")


############ EVALUATION OF PREDICTIONS ###########################
# Evaluate all predictions
# Differences could be numpy array
logger.info("Evaluating all predictions")
differences_df_validation = find_error_of_each_feature_for_each_sample(
    predictions=validation_predicted_features,
    labelled_test_dataset=validation_features_supervised_dataset,
)
differences_df_test = find_error_of_each_feature_for_each_sample(
    predictions=test_predicted_features,
    labelled_test_dataset=test_features_supervised_dataset,
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

# Get the mean absolute error for each prediction
row_wise_errors = np.abs(differences_df_validation.values).mean(axis=1)
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
# TODO: Inputs should be ndarray
(
    worst_original_mts,
    worst_target_mts,
    worst_transformed_mts,
    worst_transformed_features,
) = analyze_and_visualize_prediction(
    prediction_index=int(worst_prediction_index),
    supervised_dataset=validation_features_supervised_dataset,
    predicted_features=validation_predicted_features,
    mts_dataset=mts_dataset,
    mts_decomps=mts_decomps,
    mts_feature_df=mts_feature_df,
    ga=ga,
    uts_names=timeseries_to_use,
    output_dir=output_dir,
    plot_name_prefix="worst_",
)


# For best prediction
best_original_mts, best_target_mts, best_transformed_mts, best_transformed_features = (
    analyze_and_visualize_prediction(
        prediction_index=int(best_prediction_index),
        supervised_dataset=validation_features_supervised_dataset,
        predicted_features=validation_predicted_features,
        mts_dataset=mts_dataset,
        mts_decomps=mts_decomps,
        mts_feature_df=mts_feature_df,
        ga=ga,
        uts_names=timeseries_to_use,
        output_dir=output_dir,
        plot_name_prefix="best_",
    )
)

# For random prediction
random_index = np.random.randint(len(validation_features_supervised_dataset))
(
    random_original_mts,
    random_target_mts,
    random_transformed_mts,
    random_transformed_features,
) = analyze_and_visualize_prediction(
    prediction_index=int(random_index),
    supervised_dataset=validation_features_supervised_dataset,
    predicted_features=validation_predicted_features,
    mts_dataset=mts_dataset,
    mts_decomps=mts_decomps,
    mts_feature_df=mts_feature_df,
    ga=ga,
    uts_names=timeseries_to_use,
    output_dir=output_dir,
    plot_name_prefix="random_",
)

worst_forecast_plot = compare_original_and_transformed_forecasting(
    worst_original_mts,
    worst_transformed_mts,
    forecasting_model_wrapper,
    forecasting_model_params,
)
worst_forecast_plot.savefig(os.path.join(output_dir, f"worst_transformed_forecast.png"))


best_forecast_plot = compare_original_and_transformed_forecasting(
    best_original_mts,
    best_transformed_mts,
    forecasting_model_wrapper,
    forecasting_model_params,
)
best_forecast_plot.savefig(os.path.join(output_dir, f"best_transformed_forecast.png"))


random_forecast_plot = compare_original_and_transformed_forecasting(
    random_original_mts,
    random_transformed_mts,
    forecasting_model_wrapper,
    forecasting_model_params,
)
random_forecast_plot.savefig(
    os.path.join(output_dir, f"random_transformed_forecast.png")
)


# Due to limitations of runtime of GA, we only check for the set of transformations, where we only have one
# original time series, instead of multiple. This will limit the number of time series to generate to the size of
# the train indices.
sampled_test_features_supervised_dataset = test_features_supervised_dataset[
    ~test_features_supervised_dataset["original_index"].duplicated()
]
indices = sampled_test_features_supervised_dataset.index.tolist()

sampled_test_predicted_features = test_predicted_features[
    test_predicted_features["prediction_index"].isin(indices)
]

logger.info("Using generated features to generate new time series")
# TODO: Input could be numpy array
generated_transformed_mts = generate_new_time_series(
    supervised_dataset=sampled_test_features_supervised_dataset,
    predicted_features=sampled_test_predicted_features,
    ga=ga,
)

(
    X_transformed,
    y_transformed,
) = create_training_windows_from_mts(
    mts=generated_transformed_mts,
    target_col_index=1,
    window_size=forecasting_model_params["window_size"],
    forecast_horizon=forecasting_model_params["horizon_length"],
)

X_new_train = np.vstack((X_mts_train, X_transformed))
y_new_train = np.vstack((y_mts_train, y_transformed))


forecasting_model_new = FeedForwardForecaster(
    model_params=forecasting_model_params,
)

forecasting_model_wrapper_new = NeuralNetworkWrapper(
    model=forecasting_model_new, training_params=forecasting_model_training_params
)
forecasting_model_wrapper_new.train(
    X_train=X_new_train,
    y_train=y_new_train,
    X_val=X_new_train,
    y_val=y_new_train,
    log_to_wandb=False,
)

model_comparison_fig = compare_old_and_new_model(
    X_test=X_mts_test,
    y_test=y_mts_test,
    X_val=X_mts_validation,
    y_val=y_mts_validation,
    X_train=X_mts_train,
    y_train=y_mts_train,
    forecasting_model_wrapper_old=forecasting_model_wrapper,
    forecasting_model_wrapper_new=forecasting_model_wrapper_new,
)
model_comparison_fig.savefig(
    os.path.join(output_dir, "forecasting_model_comparison.png")
)

logger.info("Finished running")
