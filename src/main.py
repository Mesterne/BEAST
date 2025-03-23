import argparse
import logging
import os
import random
import sys
from ast import Tuple
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

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
from src.data_transformations.generation_of_supervised_pairs import (
    create_train_val_test_split,
)  # noqa: E402
from src.models.cvae_wrapper import prepare_cgen_data
from src.models.feature_transformation_model import FeatureTransformationModel
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.plots.feature_distribution import plot_feature_distribution
from src.plots.generated_vs_target_comparison import (
    create_and_save_plots_of_model_performances,
)
from src.utils.evaluation.evaluation import evaluate
from src.utils.evaluation.feature_space_evaluation import (
    calculate_mse_for_each_feature,
    calculate_total_mse_for_each_mts,
)
from src.utils.experiment_helper import (  # noqa: E402
    get_feature_model_by_type,
    get_mts_dataset,
)
from src.utils.features import decomp_and_features
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
use_one_hot_encoding = config["dataset_args"]["use_one_hot_encoding"]
test_set_sample_size = config["dataset_args"]["test_set_sample_size"]

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
mts_dataset_array: np.ndarray = get_mts_dataset(
    data_dir=data_dir,
    time_series_to_use=timeseries_to_use,
    context_length=context_length,
    step_size=step_size,
)

logger.info(
    f"Reshaped multivariate time series dataset to shape: {mts_dataset_array.shape}"
)

dataset_size: int = len(mts_dataset_array)
num_uts_in_mts: int = len(timeseries_to_use)
logger.info(f"MTS Dataset shape: {mts_dataset_array.shape}")

mts_features_array, mts_decomps = generate_feature_dataframe(
    data=mts_dataset_array,
    series_periodicity=seasonal_period,
    num_features_per_uts=num_features_per_uts,
)

dist_of_features = plot_feature_distribution(mts_features_array)
dist_of_features.savefig("distribution_of_features.png")

logger.info("Successfully generated feature dataframe")

# Generate PCA space used to create train test splits
mts_pca_array: np.ndarray = PCAWrapper().fit_transform(mts_features_array)
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
    mts_features_array,
    use_one_hot_encoding=use_one_hot_encoding,
    number_of_transformations_in_test_set=test_set_sample_size,
)

original_indices_train = (
    train_features_supervised_dataset["original_index"].astype(int).tolist()
)
original_indices_validation = (
    validation_features_supervised_dataset["original_index"].astype(int).tolist()
)
original_indices_test = (
    test_features_supervised_dataset["original_index"].astype(int).tolist()
)

target_indices_train = (
    train_features_supervised_dataset["target_index"].astype(int).tolist()
)
target_indices_validation = (
    validation_features_supervised_dataset["target_index"].astype(int).tolist()
)
target_indices_test = (
    test_features_supervised_dataset["target_index"].astype(int).tolist()
)

train_transformation_indices: np.ndarray = np.array(
    [original_indices_train, target_indices_train]
).T
validation_transformation_indices: np.ndarray = np.array(
    [original_indices_validation, target_indices_validation]
).T
test_transformation_indices: np.ndarray = np.array(
    [original_indices_test, target_indices_test]
).T

# Check if the feature model is a conditional generative model. If so, do necessary data preparation.
is_conditional_gen_model: bool = (
    config["model_args"]["feature_model_args"]["conditional_gen_model_args"] is not None
)
if is_conditional_gen_model:
    logger.info("Preparing data set for conditional generative model...")
    condition_type: str = config["model_args"]["feature_model_args"][
        "conditional_gen_model_args"
    ]["condition_type"]
    (
        X_y_pairs_cgen_train,
        X_y_pairs_cgen_validation,
        X_y_pairs_cgen_test,
    ) = prepare_cgen_data(
        condition_type,
        mts_dataset_array,
        X_features_train,
        y_features_train,
        X_features_validation,
        y_features_validation,
        X_features_test,
        y_features_test,
        train_features_supervised_dataset,
        validation_features_supervised_dataset,
        test_features_supervised_dataset,
    )
    logger.info("Successfully prepared data for conditional generative model")

train_indices: List[int] = (
    train_features_supervised_dataset["original_index"].astype(int).unique().tolist()
)
validation_indices: List[int] = (
    validation_features_supervised_dataset["target_index"].astype(int).unique().tolist()
)
test_indices: List[int] = (
    test_features_supervised_dataset["target_index"].astype(int).unique().tolist()
)


train_mts_array: np.ndarray = np.array([mts_dataset_array[i] for i in train_indices])
validation_mts_array: np.ndarray = np.array(
    [mts_dataset_array[i] for i in validation_indices]
)
test_mts_array: np.ndarray = np.array([mts_dataset_array[i] for i in test_indices])

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

feature_model_params["number_of_uts_in_mts"] = num_uts_in_mts
feature_model_params["number_of_features_per_uts"] = num_features_per_uts
feature_model_params["input_size"] = X_features_train.shape[1]

print(f"Shape of X: {X_features_train.shape}")

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

ga: GeneticAlgorithmWrapper = GeneticAlgorithmWrapper(
    ga_params=genetic_algorithm_params,
    mts_dataset=mts_dataset_array,
    mts_decomp=mts_decomps,
    num_uts_in_mts=num_uts_in_mts,
    num_features_per_uts=num_features_per_uts,
)
logger.info("Successfully initialized the genetic algorithm")


############ TRAINING
logger.info("Training feature model...")
feature_model.train(
    X_train=(
        X_features_train if not is_conditional_gen_model else X_y_pairs_cgen_train[0]
    ),
    y_train=(
        y_features_train if not is_conditional_gen_model else X_y_pairs_cgen_train[1]
    ),
    X_val=(
        X_features_validation
        if not is_conditional_gen_model
        else X_y_pairs_cgen_validation[0]
    ),
    y_val=(
        y_features_validation
        if not is_conditional_gen_model
        else X_y_pairs_cgen_validation[1]
    ),
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
TARGET_NAMES = [f"target_{name}" for name in COLUMN_NAMES]
logger.info("Running inference on validation set...")
if is_conditional_gen_model:
    inferred_mts_validation, inferred_mts_features_validation = feature_model.infer(
        X_y_pairs_cgen_validation[0],
        num_uts_in_mts=num_uts_in_mts,
        num_features_per_uts=num_features_per_uts,
        seasonal_period=seasonal_period,
    )
    inferred_mts_validation = inferred_mts_validation.reshape(-1, num_uts_in_mts, 192)
    print(f"Shape of inferred_mts_validation CVAE: {inferred_mts_validation.shape}")
else:
    validation_predicted_features: np.ndarray = feature_model.infer(
        X_features_validation
    )
    inferred_mts_validation, inferred_mts_features_validation = (
        generate_new_time_series(
            original_indices=original_indices_validation,
            predicted_features=validation_predicted_features,
            ga=ga,
        )
    )
    print(f"Shape of inferred_mts_validation GA: {inferred_mts_validation.shape}")

logger.info("Running inference on test set...")
if is_conditional_gen_model:
    inferred_mts_test, inferred_mts_features_test = feature_model.infer(
        X_y_pairs_cgen_test[0],
        num_uts_in_mts=num_uts_in_mts,
        num_features_per_uts=num_features_per_uts,
        seasonal_period=seasonal_period,
    )
    inferred_mts_test = inferred_mts_test.reshape(-1, num_uts_in_mts, 192)
else:
    test_predicted_features: np.ndarray = feature_model.infer(X_features_test)
    inferred_mts_test, inferred_mts_features_test = generate_new_time_series(
        original_indices=original_indices_test,
        predicted_features=test_predicted_features,
        ga=ga,
    )

logger.info("Successfully ran inference on validation and test sets")


evaluate(
    mts_array=mts_dataset_array,
    train_transformation_indices=train_transformation_indices,
    validation_transformation_indices=validation_transformation_indices,
    test_transformation_indices=test_transformation_indices,
    inferred_mts_validation=inferred_mts_validation,
    inferred_mts_test=inferred_mts_test,
)


(
    X_transformed,
    y_transformed,
) = create_training_windows_from_mts(
    mts=(inferred_mts_validation),
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


logger.info("Finished running")
