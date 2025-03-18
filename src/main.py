import argparse
import logging
import os
import random
import sys
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from statsmodels.tsa.seasonal import DecomposeResult
from tqdm import tqdm

# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)


import uuid

import wandb

from src.data.constants import COLUMN_NAMES, OUTPUT_DIR
from src.data_transformations.generation_of_supervised_pairs import (  # noqa: E402
    create_train_val_test_split,
    get_col_names_original_target,
)
from src.models.cvae_wrapper import prepare_cvae_data
from src.models.feature_transformation_model import FeatureTransformationModel
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.plots.generated_vs_target_comparison import (
    create_and_save_plots_of_model_performances,
)
from src.utils.data_formatting import use_model_predictions_to_create_dataframe
from src.utils.evaluation.feature_space_evaluation import (
    calculate_mse_for_each_feature,
    calculate_total_mse_for_each_mts,
)
from src.utils.experiment_helper import (  # noqa: E402
    get_feature_model_by_type,
    get_mts_dataset,
)
from src.utils.features import numpy_decomp_and_features
from src.utils.forecasting_utils import compare_old_and_new_model
from src.utils.ga_utils import generate_new_time_series
from src.utils.generate_dataset import (  # noqa: E402
    create_training_windows_from_mts,
    generate_feature_dataframe,
)
from src.utils.logging_config import logger  # noqa: E402
from src.utils.pca import PCAWrapper  # noqa: E402
from src.utils.yaml_loader import read_yaml  # noqa: E402

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
config: Dict[str, Any] = read_yaml(args["config_path"])

data_dir: str = os.path.join(
    config["dataset_args"]["directory"], config["dataset_args"]["training_data"]
)
timeseries_to_use: List[str] = config["dataset_args"]["timeseries_to_use"]
step_size: int = config["dataset_args"]["step_size"]
context_length: int = config["dataset_args"]["window_size"]
num_features_per_uts: int = config["dataset_args"]["num_features_per_uts"]

log_training_to_wandb: bool = config["training_args"]["log_to_wandb"]

seasonal_period: int = config["stl_args"]["series_periodicity"]
# FEATURE MODEL
feature_model_params: Dict[str, Any] = config["model_args"]["feature_model_args"]
model_type: str = feature_model_params["model_name"]
genetic_algorithm_params: Dict[str, Any] = config["model_args"][
    "genetic_algorithm_args"
]
# FORECAST MODEL
forecasting_model_params: Dict[str, Any] = config["model_args"][
    "forecasting_model_args"
]
forecasting_model_training_params: Dict[str, Any] = forecasting_model_params[
    "training_args"
]
training_params: Dict[str, Any] = feature_model_params["training_args"]
# CONDITIONAL GENERATIVE MODEL
conditional_gen_model_params: Dict[str, Any] = config["model_args"][
    "conditional_gen_model_args"
]
codnitional_gen_model_name: str = conditional_gen_model_params["model_name"]
conditional_gen_model_training_params: Dict[str, Any] = conditional_gen_model_params[
    "training_args"
]

# Set up logging to wandb
if log_training_to_wandb:
    job_name: str = os.environ.get("JOB_NAME", str(uuid.uuid4()))
    wandb.init(
        project="MTS-BEAST",
        name=job_name,
        config=feature_model_params,
    )


logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {OUTPUT_DIR} (Relative to where you ran the program from)"
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
dataset_size: int = len(mts_dataset)
num_uts_in_mts: int = len(timeseries_to_use)
logger.info(f"MTS Dataset shape: ({len(mts_dataset)}, {len(mts_dataset[0])})")

# TODO: mts_feature_df and mts_decomps should be ndarray
mts_feature_df, mts_decomps = generate_feature_dataframe(
    data=mts_dataset, series_periodicity=seasonal_period, dataset_size=dataset_size
)


logger.info("Successfully generated feature dataframe")

# Generate train, vlaidation and test splits
ORIGINAL_NAMES, DELTA_NAMES, TARGET_NAMES = get_col_names_original_target()

# Generate PCA space used to create train test splits
mts_pca_array: np.ndarray = PCAWrapper().fit_transform(mts_feature_df)
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
    mts_pca_array,
    mts_feature_df,
    ORIGINAL_NAMES,
    DELTA_NAMES,
    TARGET_NAMES,
    SEED,
)

print(validation_features_supervised_dataset.head(10))
# Print all distinct values of the original index
print(
    "ORIGINAL INDICES",
    validation_features_supervised_dataset["original_index"].unique(),
)
print("TARGET INDICES", validation_features_supervised_dataset["target_index"].unique())

train_indices: List[int] = (
    train_features_supervised_dataset["original_index"].astype(int).unique().tolist()
)
validation_indices: List[int] = (
    validation_features_supervised_dataset["target_index"].astype(int).unique().tolist()
)
test_indices: List[int] = (
    test_features_supervised_dataset["target_index"].astype(int).unique().tolist()
)

mts_array: np.ndarray = np.array([df.values.T for df in mts_dataset])
logger.info(f"Reshaped multivariate time series dataset to shape: {mts_array.shape}")

train_mts_array: np.ndarray = [mts_array[i] for i in train_indices]
# train_mts_array: np.ndarray = mts_array[train_indices]
validation_mts_array: np.ndarray = [mts_array[i] for i in validation_indices]
test_mts_array: np.ndarray = [mts_array[i] for i in test_indices]

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

feature_model: FeatureTransformationModel = get_feature_model_by_type(
    model_type=model_type,
    model_params=feature_model_params,
    training_params=training_params,
)
logger.info(f"Successfully initialized the {model_type} model")

forecasting_model: FeedForwardForecaster = FeedForwardForecaster(
    model_params=forecasting_model_params,
)

forecasting_model_wrapper: NeuralNetworkWrapper = NeuralNetworkWrapper(
    model=forecasting_model, training_params=forecasting_model_training_params
)
logging.info("Successfully initialized the forecasting model")

conditional_gen_model: FeatureTransformationModel = get_feature_model_by_type(
    model_type=codnitional_gen_model_name,
    model_params=conditional_gen_model_params,
    training_params=conditional_gen_model_training_params,
)
logger.info("Successfully initialized the conditional generative model model")

# TODO: Inputs should be ndarray
ga: GeneticAlgorithmWrapper = GeneticAlgorithmWrapper(
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

logger.info("Preparing data for conditional generative model...")
(
    X_cvae_train,
    y_cvae_train,
    X_cvae_validation,
    y_cvae_validation,
    X_cvae_test,
    y_cvae_test,
) = prepare_cvae_data(
    mts_array=mts_array,
    X_features_train=X_features_train,
    train_indices=train_indices,
    X_features_validation=X_features_validation,
    validation_indices=validation_indices,
    X_features_test=X_features_test,
    test_indices=test_indices,
)

logger.info("Training conditional generative model...")
conditional_gen_model.train(
    X_train=X_cvae_train,
    y_train=y_cvae_train,
    X_val=X_cvae_validation,
    y_val=y_cvae_validation,
    log_to_wandb=False,
)

############ INFERENCE
logger.info("Running inference on validation set...")
validation_predicted_features: np.ndarray = feature_model.infer(X_features_validation)
validation_predicted_features: pd.DataFrame = use_model_predictions_to_create_dataframe(
    validation_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=validation_features_supervised_dataset,
)

logger.info("Running inference on test set...")
test_predicted_features: np.ndarray = feature_model.infer(X_features_test)
test_predicted_features: pd.DataFrame = use_model_predictions_to_create_dataframe(
    test_predicted_features,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=test_features_supervised_dataset,
)
logger.info("Successfully ran inference on validation and test sets")

logger.info("Running inference with CVAE on validation set...")
use_delta = int(conditional_gen_model_params["use_feature_deltas"])
num_mts_features = num_uts_in_mts * num_features_per_uts
print("X FEATURES VALIDATION", X_features_validation.shape)
print(X_features_validation[0])
print("Y FEATURES VALIDATION", y_features_validation.shape)
print(y_features_validation[0])
validation_features = X_features_validation[validation_indices]
print("VALIDATION FEATURES", validation_features.shape)
print(validation_features[0])
cvae_infer_input_array: np.ndarray = (
    validation_features[:, num_mts_features : num_mts_features * 2]
    if use_delta
    else validation_features[:, :num_mts_features]
)
# FIXME: CVAE SHOULD INFER ON Y FEATURES VALIDATION, WITH DIRECT FEATURES, NOT DELTA, AS CONDITIONS
# FIXME: SHOULD REMOVE DELTA OPTION FROM CVAE
cvae_validation_predicted_mts: np.ndarray = conditional_gen_model.infer(
    cvae_infer_input_array
)
_, cvae_validation_predicted_features = numpy_decomp_and_features(
    cvae_validation_predicted_mts, num_uts_in_mts, num_features_per_uts, seasonal_period
)

# Generation of new time series based on newly inferred features

# Due to limitations of runtime of GA, we only check for the set of transformations, where we only have one
# original time series, instead of multiple. This will limit the number of time series to generate to the size of
# the train indices.
sampled_test_features_supervised_dataset: pd.DataFrame = (
    test_features_supervised_dataset[
        ~test_features_supervised_dataset["original_index"].duplicated()
    ]
)
prediction_indices: List[int] = sampled_test_features_supervised_dataset.index.tolist()

predicted_features_to_generated_mts_for: pd.DataFrame = test_predicted_features[
    test_predicted_features["prediction_index"].isin(prediction_indices)
]
original_timeseries_indices_transformed_from: List[int] = (
    sampled_test_features_supervised_dataset["original_index"].astype(int).tolist()
)
original_timeseries: np.ndarray = mts_array[
    original_timeseries_indices_transformed_from
]
target_timeseries_indices_transformed_to: List[int] = (
    sampled_test_features_supervised_dataset["target_index"].astype(int).tolist()
)
target_timeseries: np.ndarray = mts_array[target_timeseries_indices_transformed_to]


logger.info("Using generated features to generate new time series")
generated_transformed_mts, features_of_genereated_timeseries_mts = (
    generate_new_time_series(
        supervised_dataset=sampled_test_features_supervised_dataset,
        predicted_features=predicted_features_to_generated_mts_for,
        ga=ga,
    )
)
# We have to remove the delta values
original_features: np.ndarray = X_features_validation[
    prediction_indices, : len(COLUMN_NAMES)
]
target_features: np.ndarray = y_features_validation[prediction_indices]

features_of_genereated_timeseries_mts: np.ndarray = np.array(
    features_of_genereated_timeseries_mts
).reshape(-1, 12)


## Evaluation of MTS generation
mse_values_for_each_feature = calculate_mse_for_each_feature(
    predicted_features=features_of_genereated_timeseries_mts,
    target_features=target_features,
)

total_mse_for_each_uts = calculate_total_mse_for_each_mts(
    mse_per_feature=mse_values_for_each_feature
)

# NOTE: We pass y for features, as these will contain all series
create_and_save_plots_of_model_performances(
    total_mse_for_each_mts=total_mse_for_each_uts,
    mse_per_feature=mse_values_for_each_feature,
    mts_features_train=y_features_train,
    mts_features_validation=y_features_validation,
    mts_features_test=y_features_test,
    original_mts=original_timeseries,
    target_mts=target_timeseries,
    generated_mts=generated_transformed_mts,
    original_mts_features=original_features,
    transformed_mts_features=features_of_genereated_timeseries_mts,
    target_mts_features=target_features,
)


# Retrain forecasting model on new timeseries.
(
    X_transformed,
    y_transformed,
) = create_training_windows_from_mts(
    mts=generated_transformed_mts,
    target_col_index=1,
    window_size=forecasting_model_params["window_size"],
    forecast_horizon=forecasting_model_params["horizon_length"],
)

X_new_train: np.ndarray = np.vstack((X_mts_train, X_transformed))
y_new_train: np.ndarray = np.vstack((y_mts_train, y_transformed))


forecasting_model_new: FeedForwardForecaster = FeedForwardForecaster(
    model_params=forecasting_model_params,
)

forecasting_model_wrapper_new: NeuralNetworkWrapper = NeuralNetworkWrapper(
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
    os.path.join(OUTPUT_DIR, "forecasting_model_comparison.png")
)

# FIXME: Add visualizations of generated time sereis. Retrain model. Compare old and new model.
# Calculate MSE for conditional generative model. Using validation
cond_gen_mse_values_for_each_feature = calculate_mse_for_each_feature(
    predicted_features=cvae_validation_predicted_features,
    target_features=cvae_infer_input_array,
)


logger.info("Finished running")
