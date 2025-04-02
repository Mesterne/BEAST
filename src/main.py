import argparse
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


from src.data.constants import OUTPUT_DIR
from src.data_transformations.generation_of_supervised_pairs import (
    create_train_val_test_split,
)  # noqa: E402
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.models.model_handler import ModelHandler
from src.models.neural_network_wrapper import NeuralNetworkWrapper
from src.utils.evaluation.evaluation import evaluate
from src.utils.experiment_helper import get_mts_dataset  # noqa: E402
from src.utils.forecasting_utils import compare_old_and_new_model
from src.utils.generate_dataset import (  # noqa: E402
    create_training_windows_from_mts,
    generate_feature_dataframe,
)
from src.utils.logging_config import logger  # noqa: E402
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
num_uts_in_mts: int = len(config["dataset_args"]["timeseries_to_use"])
config["is_conditional_gen_model"]: bool = (
    config["model_args"]["feature_model_args"]["conditional_gen_model_args"] is not None
)


logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {OUTPUT_DIR} (Relative to where you ran the program from)"
)

############ DATA INITIALIZATION

# Load data and generate dataset of multivariate time series context windows
mts_dataset_array: np.ndarray = get_mts_dataset(
    data_dir=data_dir,
    time_series_to_use=config["dataset_args"]["timeseries_to_use"],
    context_length=config["dataset_args"]["window_size"],
    step_size=config["dataset_args"]["step_size"],
)

mts_features_array, mts_decomps = generate_feature_dataframe(
    data=mts_dataset_array,
    series_periodicity=config["stl_args"]["series_periodicity"],
    num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
)


logger.info("Successfully generated feature dataframe")


(
    train_transformation_indices,
    validation_transformation_indices,
    test_transformation_indices,
) = create_train_val_test_split(
    mts_dataset_array=mts_dataset_array,
    config=config,
)

logger.info(
    f"""Transformation indices shapes:
    Train: {train_transformation_indices.shape}
    Validation: {validation_transformation_indices.shape}
    Test: {test_transformation_indices.shape}
"""
)


config["model_args"]["feature_model_args"]["number_of_uts_in_mts"] = num_uts_in_mts
config["model_args"]["feature_model_args"]["number_of_features_per_uts"] = config[
    "dataset_args"
]["num_features_per_uts"]
config["model_args"]["feature_model_args"]["input_size"] = (
    num_uts_in_mts * config["dataset_args"]["num_features_per_uts"]
)
config["model_args"]["feature_model_args"]["input_size"] += (
    config["dataset_args"]["num_features_per_uts"] + num_uts_in_mts
    if config["dataset_args"]["use_one_hot_encoding"]
    else config["dataset_args"]["num_features_per_uts"] * num_uts_in_mts
)

model_handler = ModelHandler(config)
model_handler.choose_model_category()

############ TRAINING
model_handler.train(
    mts_dataset=mts_dataset_array,
    train_transformation_indices=train_transformation_indices,
    validation_transformation_indices=validation_transformation_indices,
)


############ INFERENCE
logger.info("Running inference on validation set...")
inferred_mts_validation, inferred_intermediate_features_validation = (
    model_handler.infer(
        mts_dataset=mts_dataset_array,
        evaluation_transformation_indinces=validation_transformation_indices,
    )
)

logger.info("Running inference on test set...")
inferred_mts_test, inferred_intermediate_features_test = model_handler.infer(
    mts_dataset=mts_dataset_array,
    evaluation_transformation_indinces=test_transformation_indices,
)

logger.info("Successfully ran inference on validation and test sets")

evaluate(
    config=config,
    mts_array=mts_dataset_array,
    train_transformation_indices=train_transformation_indices,
    validation_transformation_indices=validation_transformation_indices,
    test_transformation_indices=test_transformation_indices,
    inferred_mts_validation=inferred_mts_validation,
    inferred_mts_test=inferred_mts_test,
    inferred_intermediate_features_validation=inferred_intermediate_features_validation,
    inferred_intermediate_features_test=inferred_intermediate_features_test,
)


# Forecasting model evaluations
evaluate_forecasting_improvement(
    config=config,
    mts_dataset=mts_dataset_array,
    train_indices=train_indices,
    validation_indices=validation_indices,
    test_indices=test_indices,
    inferred_mts_validation=inferred_mts_validation,
    inferred_mts_test=inferred_mts_test,
)


train_indices: List[int] = train_transformation_indices[:, 0]
validation_indices: List[int] = validation_transformation_indices[:, 1]
test_indices: List[int] = test_transformation_indices[:, 1]


train_mts_array: np.ndarray = mts_dataset_array[train_indices]
validation_mts_array: np.ndarray = mts_dataset_array[validation_indices]
test_mts_array: np.ndarray = mts_dataset_array[test_indices]


(
    X_mts_train,
    y_mts_train,
) = create_training_windows_from_mts(
    mts=train_mts_array,
    target_col_index=1,
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    forecast_horizon=config["model_args"]["forecasting_model_args"]["horizon_length"],
)
logger.info("Forecasting train data shape: {}".format(X_mts_train.shape))
(
    X_mts_validation,
    y_mts_validation,
) = create_training_windows_from_mts(
    mts=validation_mts_array,
    target_col_index=1,
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    forecast_horizon=config["model_args"]["forecasting_model_args"]["horizon_length"],
)
logger.info("Forecasting validation data shape: {}".format(X_mts_validation.shape))
(
    X_mts_test,
    y_mts_test,
) = create_training_windows_from_mts(
    mts=test_mts_array,
    target_col_index=1,
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    forecast_horizon=config["model_args"]["forecasting_model_args"]["horizon_length"],
)
logger.info("Forecasting test data shape: {}".format(X_mts_test.shape))

(
    X_transformed,
    y_transformed,
) = create_training_windows_from_mts(
    mts=(inferred_mts_validation),
    target_col_index=1,
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    forecast_horizon=config["model_args"]["forecasting_model_args"]["horizon_length"],
)

X_new_train: np.ndarray = np.vstack((X_mts_train, X_transformed))
y_new_train: np.ndarray = np.vstack((y_mts_train, y_transformed))

forecasting_model_old: FeedForwardForecaster = FeedForwardForecaster(
    model_params=config["model_args"]["forecasting_model_args"],
)
forecasting_model_wrapper_old: NeuralNetworkWrapper = NeuralNetworkWrapper(
    model=forecasting_model_old,
    training_params=config["model_args"]["forecasting_model_args"]["training_args"],
)
logger.info("Training forecasting model on old data")
forecasting_model_wrapper_old.train(
    X_train=X_mts_train,
    y_train=y_mts_train,
    X_val=X_mts_train,
    y_val=y_mts_train,
    plot_loss=False,
)

forecasting_model_new: FeedForwardForecaster = FeedForwardForecaster(
    model_params=config["model_args"]["forecasting_model_args"],
)
forecasting_model_wrapper_new: NeuralNetworkWrapper = NeuralNetworkWrapper(
    model=forecasting_model_new,
    training_params=config["model_args"]["forecasting_model_args"]["training_args"],
)
logger.info("Training forecasting model on old + generated data")
forecasting_model_wrapper_new.train(
    X_train=X_new_train,
    y_train=y_new_train,
    X_val=X_new_train,
    y_val=y_new_train,
    plot_loss=False,
)

model_comparison_fig = compare_old_and_new_model(
    X_test=X_mts_test,
    y_test=y_mts_test,
    X_val=X_mts_validation,
    y_val=y_mts_validation,
    X_train=X_mts_train,
    y_train=y_mts_train,
    forecasting_model_wrapper_old=forecasting_model_wrapper_old,
    forecasting_model_wrapper_new=forecasting_model_wrapper_new,
)
model_comparison_fig.savefig(
    os.path.join(OUTPUT_DIR, "forecasting_model_comparison.png")
)


logger.info("Finished running")
