import logging
import os
import sys
import argparse
from typing import Any, Dict, List, Tuple
import pandas as pd
import numpy as np
import random
from statsmodels.tsa.seasonal import DecomposeResult
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)


from src.models.feature_transformation_model import FeatureTransformationModel
from src.data.constants import COLUMN_NAMES, OUTPUT_DIR
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.utils.forecasting_utils import (
    compare_old_and_new_model,
)

import wandb
import uuid
from src.utils.yaml_loader import read_yaml  # noqa: E402
from src.utils.generate_dataset import (
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
from src.plots.generated_vs_target_comparison import (
    create_and_save_plots_of_model_performances,
)
from src.utils.data_formatting import use_model_predictions_to_create_dataframe
from src.utils.evaluation.feature_space_evaluation import (
    calculate_mse_for_each_feature,
    calculate_total_mse_for_each_mts,
)
from src.utils.ga_utils import (
    generate_new_time_series,
)
from src.utils.features import numpy_decomp_and_features

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

feature_model_params: Dict[str, Any] = config["model_args"]["feature_model_args"]
model_type: str = feature_model_params["model_name"]
genetic_algorithm_params: Dict[str, Any] = config["model_args"][
    "genetic_algorithm_args"
]

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

# TODO: Create a third category of models to which CVAE model belongs
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

if model_type == "mts_cvae":
    logger.info("Using CVAE model, setting up CVAE training")
    # Train input and target for CVAE
    full_mts_array: np.ndarray = np.asarray(mts_array)
    full_mts_array: np.ndarray = full_mts_array.reshape(
        full_mts_array.shape[0], full_mts_array.shape[1] * full_mts_array.shape[2]
    )
    cvae_train_mts_array: np.ndarray = full_mts_array[train_indices]
    cvae_train_features_array: np.ndarray = X_features_train[train_indices]
    X_cvae_train: np.ndarray = np.hstack(
        (cvae_train_mts_array, cvae_train_features_array)
    )
    y_cvae_train: np.ndarray = cvae_train_mts_array.copy()

    # Validation input and target for CVAE
    cvae_validation_mts_array: np.ndarray = np.take(
        X_mts_validation, validation_indices
    )
    cvae_validation_mts_array: np.ndarray = full_mts_array[validation_indices]
    cvae_validation_features_array: np.ndarray = X_features_validation[
        validation_indices
    ]
    X_cvae_validation: np.ndarray = np.hstack(
        (cvae_validation_mts_array, cvae_validation_features_array)
    )
    y_cvae_validation: np.ndarray = cvae_validation_mts_array.copy()

    # Test input and target for CVAE
    cvae_test_mts_array: np.ndarray = full_mts_array[test_indices]
    cvae_test_features_array: np.ndarray = X_features_test[test_indices]
    X_cvae_test: np.ndarray = cvae_test_features_array.copy()
    y_cvae_test: np.ndarray = cvae_test_mts_array.copy()


############ TRAINING
# Fit model to data
if model_type == "mts_cvae":
    logger.info("Training CVAE model...")
    feature_model.train(
        X_train=X_cvae_train,
        y_train=y_cvae_train,
        X_val=X_cvae_validation,
        y_val=y_cvae_validation,
        log_to_wandb=False,
    )
else:
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
if model_type == "mts_cvae":
    logger.info("Running inference with CVAE on validation set...")
    print("INFER INPUT", cvae_validation_features_array.shape)
    use_delta = int(feature_model_params["use_feature_deltas"])
    num_mts_features = num_uts_in_mts * num_features_per_uts
    infer_input_array: np.ndarray = (
        cvae_validation_features_array[:, num_mts_features : num_mts_features * 2]
        if use_delta
        else cvae_validation_features_array[:, :num_mts_features]
    )
    validation_predicted_mts: np.ndarray = feature_model.infer(infer_input_array)
    # FIXME: Rename later
    _, validation_predicted_features = numpy_decomp_and_features(
        validation_predicted_mts, num_uts_in_mts, num_features_per_uts, seasonal_period
    )
    validation_predicted_features: pd.DataFrame = (
        use_model_predictions_to_create_dataframe(
            validation_predicted_features,
            TARGET_NAMES=TARGET_NAMES,
            target_dataframe=validation_features_supervised_dataset,
        )
    )

else:
    logger.info("Running inference on validation set...")
    validation_predicted_features: np.ndarray = feature_model.infer(
        X_features_validation
    )
    validation_predicted_features: pd.DataFrame = (
        use_model_predictions_to_create_dataframe(
            validation_predicted_features,
            TARGET_NAMES=TARGET_NAMES,
            target_dataframe=validation_features_supervised_dataset,
        )
    )

    logger.info("Running inference on test set...")
    test_predicted_features: np.ndarray = feature_model.infer(X_features_test)
    test_predicted_features: pd.DataFrame = use_model_predictions_to_create_dataframe(
        test_predicted_features,
        TARGET_NAMES=TARGET_NAMES,
        target_dataframe=test_features_supervised_dataset,
    )
logger.info("Successfully ran inference on validation and test sets")

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

logger.info("Finished running")
