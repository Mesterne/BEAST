import argparse
import os
import random
import sys
from typing import Any, Dict

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
from src.data_transformations.generation_of_supervised_pairs import \
    create_train_val_test_split  # noqa: E402
from src.data_transformations.preprocessing import \
    scale_mts_dataset  # noqa: E402
from src.models.model_handler import ModelHandler
from src.utils.evaluation.evaluate_forecasting_improvement import \
    ForecasterEvaluator
from src.utils.evaluation.evaluation import evaluate
from src.utils.experiment_helper import get_mts_dataset  # noqa: E402
from src.utils.generate_dataset import generate_feature_dataframe  # noqa: E402
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
training_data_dir: str = os.path.join(
    config["dataset_args"]["directory"], config["dataset_args"]["training_data"]
)
num_uts_in_mts: int = len(config["dataset_args"]["timeseries_to_use"])
config["is_conditional_gen_model"]: bool = (
    config["model_args"]["feature_model_args"]["model_name"] == "mts_cvae"
)


logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {OUTPUT_DIR} (Relative to where you ran the program from)"
)

############ DATA INITIALIZATION

# Load data and generate dataset of multivariate time series context windows
mts_dataset_array: np.ndarray = get_mts_dataset(
    data_dir=training_data_dir,
    time_series_to_use=config["dataset_args"]["timeseries_to_use"],
    context_length=config["dataset_args"]["window_size"],
    step_size=config["dataset_args"]["step_size"],
)
mts_dataset_array_test: np.ndarray = get_mts_dataset(
    data_dir=training_data_dir,
    time_series_to_use=config["dataset_args"]["test_timeseries_to_use"],
    context_length=config["dataset_args"]["window_size"],
    step_size=config["dataset_args"]["step_size"],
)

mts_features_array, mts_decomps = generate_feature_dataframe(
    data=mts_dataset_array,
    series_periodicity=config["stl_args"]["series_periodicity"],
    num_features_per_uts=config["dataset_args"]["num_features_per_uts"],
)

test_mts_dataset_array_size = mts_dataset_array_test.shape[0]
mts_dataset_array = np.concatenate([mts_dataset_array, mts_dataset_array_test], axis=0)

logger.info("Successfully generated feature dataframe")

(
    train_transformation_indices,
    validation_transformation_indices,
    test_transformation_indices,
) = create_train_val_test_split(
    mts_dataset_array=mts_dataset_array,
    test_size=test_mts_dataset_array_size,
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

# FIXME: Need to add inverse transform when data is plotted? Or maybe not? We never really care about the original scale of the data.
if config["is_conditional_gen_model"]:
    logger.info("Scaling data for conditional generation model (CVAE)...")
    scaled_mts_dataset_array, uts_scalers = scale_mts_dataset(
        mts_dataset_array,
        train_transformation_indices,
        validation_transformation_indices,
        test_transformation_indices,
    )
    mts_dataset_array = scaled_mts_dataset_array

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


##### FORECASTING EVALUATION
logger.info("Starting forecasting evaluation...")
evaluator = ForecasterEvaluator(
    config=config,
    mts_dataset=mts_dataset_array,
    train_indices=train_transformation_indices[:, 0],
    validation_indices=validation_transformation_indices[:, 1],
    test_indices=test_transformation_indices[:, 1],
)

logger.info("Evaluating foreasting improvement on inferred validation set...")
evaluator.evaluate_on_evaluation_set(
    inferred_mts_array=inferred_mts_validation, type="validation"
)


logger.info("Finished running")
