import argparse
import logging
import os
import random
import sys
from typing import Any, Dict, List

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
    mts_pca_array, mts_features_array, use_one_hot_encoding=use_one_hot_encoding
)

# Check if the feature model is a conditional generative model. If so, do necessary data preparation.
is_conditional_gen_model: bool = (
    config["model_args"]["feature_model_args"]["conditional_gen_model_args"] is not None
)
print(is_conditional_gen_model)
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
    validation_predicted_mts, validation_predicted_features = feature_model.infer(
        X_y_pairs_cgen_validation[0],
        num_uts_in_mts=num_uts_in_mts,
        num_features_per_uts=num_features_per_uts,
        seasonal_period=seasonal_period,
    )
else:
    validation_predicted_features: np.ndarray = feature_model.infer(
        X_features_validation
    )

logger.info("Running inference on test set...")
if is_conditional_gen_model:
    test_predicted_mts, test_predicted_features = feature_model.infer(
        X_y_pairs_cgen_test[0],
        num_uts_in_mts=num_uts_in_mts,
        num_features_per_uts=num_features_per_uts,
        seasonal_period=seasonal_period,
    )
else:
    test_predicted_features: np.ndarray = feature_model.infer(X_features_test)

logger.info("Successfully ran inference on validation and test sets")


if is_conditional_gen_model:
    logger.info(
        "Preparing inputs for analysis and visualization of time series generated by conditional generative model"
    )
    logger.info("Preparing original and target time series")
    mts_size = len(X_y_pairs_cgen_train[1][0])
    cgen_original_timeseries_train: np.ndarray = X_y_pairs_cgen_train[0][:, :mts_size]
    cgen_target_timeseries_train: np.ndarray = X_y_pairs_cgen_train[1]
    cgen_original_timeseries_validation: np.ndarray = X_y_pairs_cgen_validation[0][
        :, :mts_size
    ]
    cgen_target_timeseries_validation: np.ndarray = X_y_pairs_cgen_validation[1]
    cgen_original_timeseries_test: np.ndarray = X_y_pairs_cgen_test[0][:, :mts_size]
    cgen_target_timeseries_test: np.ndarray = X_y_pairs_cgen_test[1]

    uts_wise_cgen_original_timeseries = cgen_original_timeseries_validation.reshape(
        -1, num_uts_in_mts, mts_size // num_uts_in_mts
    )
    uts_wise_cgen_target_timeseries = cgen_target_timeseries_validation.reshape(
        -1, num_uts_in_mts, mts_size // num_uts_in_mts
    )
    uts_wise_predicted_mts = validation_predicted_mts.reshape(
        -1, num_uts_in_mts, mts_size // num_uts_in_mts
    )

    logger.info("Preparing features")
    target_features_train: np.ndarray = decomp_and_features(
        cgen_target_timeseries_train,
        num_uts_in_mts,
        num_features_per_uts,
        seasonal_period,
    )[1]
    target_features_validation: np.ndarray = decomp_and_features(
        cgen_target_timeseries_validation,
        num_uts_in_mts,
        num_features_per_uts,
        seasonal_period,
    )[1]
    target_features_test: np.ndarray = decomp_and_features(
        cgen_target_timeseries_test,
        num_uts_in_mts,
        num_features_per_uts,
        seasonal_period,
    )[1]
    # Need original features for plotting
    original_features_validation: np.ndarray = decomp_and_features(
        cgen_original_timeseries_validation,
        num_uts_in_mts,
        num_features_per_uts,
        seasonal_period,
    )[1]
    logger.info("Calculating MSE for features og generated time series using test set")
    mse_values_for_each_feature = calculate_mse_for_each_feature(
        predicted_features=validation_predicted_features,
        target_features=target_features_validation,
    )
else:
    # Due to limitations of runtime of GA, we only check for the set of transformations, where we only have one
    # original time series, instead of multiple. This will limit the number of time series to generate to the size of
    # the train indices.
    prediction_indices: List[int] = validation_features_supervised_dataset[
        ~validation_features_supervised_dataset["original_index"].duplicated()
    ].index.tolist()
    original_indices: List[int] = (
        validation_features_supervised_dataset[
            ~validation_features_supervised_dataset["original_index"].duplicated()
        ]["original_index"]
        .astype(int)
        .tolist()
    )
    target_indices: List[int] = (
        validation_features_supervised_dataset[
            ~validation_features_supervised_dataset["original_index"].duplicated()
        ]["target_index"]
        .astype(int)
        .tolist()
    )

    predicted_features_to_generated_mts_for: np.ndarray = validation_predicted_features[
        prediction_indices
    ]

    logger.info("Using generated features to generate new time series")

    generated_transformed_mts, features_of_genereated_timeseries_mts = (
        generate_new_time_series(
            original_indices=original_indices,
            predicted_features=predicted_features_to_generated_mts_for,
            ga=ga,
        )
    )

    print("Calculating MSE values")
    # Generation of new time series based on newly inferred features
    # We have to remove the delta values
    original_features: np.ndarray = X_features_validation[
        prediction_indices, : len(COLUMN_NAMES)
    ]
    target_for_predicted_features: np.ndarray = y_features_validation[
        prediction_indices
    ]

    original_timeseries: np.ndarray = mts_dataset_array[original_indices]
    target_timeseries: np.ndarray = mts_dataset_array[target_indices]

    features_of_genereated_timeseries_mts: np.ndarray = np.array(
        features_of_genereated_timeseries_mts
    ).reshape(-1, num_features_per_uts * num_uts_in_mts)

    ## Evaluation of MTS generation
    mse_values_for_each_feature = calculate_mse_for_each_feature(
        predicted_features=features_of_genereated_timeseries_mts,
        target_features=target_for_predicted_features,
    )

total_mse_for_each_uts = calculate_total_mse_for_each_mts(
    mse_per_feature=mse_values_for_each_feature
)

print("Creating plots..")
# NOTE: We pass y for features, as these will contain all series
create_and_save_plots_of_model_performances(
    total_mse_for_each_mts=total_mse_for_each_uts,
    mse_per_feature=mse_values_for_each_feature,
    y_features_train=(
        y_features_train if not is_conditional_gen_model else target_features_train
    ),
    y_features_validation=(
        y_features_validation
        if not is_conditional_gen_model
        else target_features_validation
    ),
    y_features_test=(
        y_features_test if not is_conditional_gen_model else target_features_test
    ),
    X_mts=(
        original_timeseries
        if not is_conditional_gen_model
        else uts_wise_cgen_original_timeseries
    ),
    y_mts=(
        target_timeseries
        if not is_conditional_gen_model
        else uts_wise_cgen_target_timeseries
    ),
    predicted_mts=(
        generated_transformed_mts
        if not is_conditional_gen_model
        else uts_wise_predicted_mts
    ),
    original_mts_features=(
        original_features
        if not is_conditional_gen_model
        else original_features_validation
    ),
    mts_features_predicted_before_generation=(
        predicted_features_to_generated_mts_for
        if not is_conditional_gen_model
        else validation_predicted_features
    ),
    mts_features_of_genererated_mts=(
        features_of_genereated_timeseries_mts
        if not is_conditional_gen_model
        else validation_predicted_features
    ),
    target_for_predicted_mts_features=(
        target_for_predicted_features
        if not is_conditional_gen_model
        else target_features_validation
    ),
)

(
    X_transformed,
    y_transformed,
) = create_training_windows_from_mts(
    mts=(
        generated_transformed_mts
        if not is_conditional_gen_model
        else uts_wise_predicted_mts
    ),
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
