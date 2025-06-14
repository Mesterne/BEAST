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
from src.data_transformations.generation_of_supervised_pairs import (
    create_train_val_test_split,
)  # noqa: E402
from src.data_transformations.preprocessing import scale_mts_dataset  # noqa: E402
from src.models.model_handler import ModelHandler
from src.utils.evaluation.evaluate_forecasting_improvement import ForecasterEvaluator
from src.utils.evaluation.evaluation import evaluate
from src.utils.experiment_helper import get_mts_dataset  # noqa: E402
from src.utils.generate_dataset import generate_feature_dataframe  # noqa: E402
from src.utils.logging_config import logger  # noqa: E402
from src.utils.yaml_loader import read_yaml  # noqa: E402

os.makedirs(os.path.join(OUTPUT_DIR, "Feature space evaluations"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Generation grids"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Forecasting space evaluations"), exist_ok=True)
os.makedirs(
    os.path.join(OUTPUT_DIR, "Forecasting space evaluations", "n_linear"), exist_ok=True
)
os.makedirs(
    os.path.join(OUTPUT_DIR, "Forecasting space evaluations", "tcn"), exist_ok=True
)
os.makedirs(
    os.path.join(OUTPUT_DIR, "Forecasting space evaluations", "lstm"), exist_ok=True
)
os.makedirs(os.path.join(OUTPUT_DIR, "Generated MTS"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Forecast Grids"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Forecast Grids", "n_linear"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Forecast Grids", "tcn"), exist_ok=True)
os.makedirs(os.path.join(OUTPUT_DIR, "Forecast Grids", "lstm"), exist_ok=True)

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
try:
    config["is_conditional_gen_model"] = (
        config["model_args"]["feature_model_args"]["model_name"] == "mts_cvae"
    )
except Exception:
    config["is_conditional_gen_model"] = False

try:
    # NOTE: directory_name and name of slurm job should be the same for easy identification
    logger.info(
        f"Saving/loading model from directory {config['model_args']['feature_model_args']['directory_name']}."
    )
except KeyError:
    config["model_args"]["feature_model_args"]["directory_name"] = None
    logger.info("No directory for saving/loading model specified.")

logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {OUTPUT_DIR} (Relative to where you ran the program from)"
)
test_timeseries_to_use = config["dataset_args"]["test_timeseries_to_use"]

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
    time_series_to_use=test_timeseries_to_use,
    context_length=config["dataset_args"]["window_size"],
    step_size=config["dataset_args"]["step_size"],
)

test_mts_dataset_array_size = mts_dataset_array_test.shape[0]
mts_dataset_array = np.concatenate([mts_dataset_array, mts_dataset_array_test], axis=0)

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

if config["is_conditional_gen_model"]:
    logger.info("Scaling data for conditional generation model (CVAE)...")
    old_mts_dataset = mts_dataset_array
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
model_directory = config["model_args"]["feature_model_args"]["directory_name"]
if model_directory is not None:
    models_directory = os.path.join(OUTPUT_DIR, "..", "models")
    os.makedirs(models_directory, exist_ok=True)
    model_path = os.path.join(OUTPUT_DIR, "..", "models", model_directory)
    if os.path.exists(model_path):
        logger.info(f"Pretrained model available at {model_path}. Loading model...")
        model_handler.load_model()
    else:
        logger.info(
            f"No pretrained model avaliable at {model_path}. Training new model..."
        )
        model_handler.train(
            mts_dataset=mts_dataset_array,
            train_transformation_indices=train_transformation_indices,
            validation_transformation_indices=validation_transformation_indices,
        )
        model_handler.save_model()
        logger.info(f"Model saved to {model_path}.")
else:
    model_handler.train(
        mts_dataset=mts_dataset_array,
        train_transformation_indices=train_transformation_indices,
        validation_transformation_indices=validation_transformation_indices,
    )


############ INFERENCE
logger.info("Running inference on validation set...")
inferred_mts_validation, inferred_intermediate_features_validation, ohe_val = (
    model_handler.infer(
        mts_dataset=mts_dataset_array,
        evaluation_transformation_indinces=validation_transformation_indices,
    )
)

logger.info("Running inference on test set...")
inferred_mts_test, inferred_intermediate_features_test, ohe_test = model_handler.infer(
    mts_dataset=mts_dataset_array,
    evaluation_transformation_indinces=test_transformation_indices,
)

logger.info("Successfully ran inference on validation and test sets")

if config["is_conditional_gen_model"]:
    logger.info("Scaling data for conditional generation model (CVAE)...")
    for i in range(0, inferred_mts_validation.shape[0]):
        for j in range(0, num_uts_in_mts):
            inferred_mts_validation[i, j] = uts_scalers[j].inverse_transform(
                inferred_mts_validation[i, j].reshape(1, -1)
            )
            inferred_mts_test[i, j] = uts_scalers[j].inverse_transform(
                inferred_mts_test[i, j].reshape(1, -1)
            )
    mts_dataset_array = old_mts_dataset

evaluate(
    config=config,
    mts_array=mts_dataset_array,
    train_transformation_indices=train_transformation_indices,
    validation_transformation_indices=validation_transformation_indices,
    test_transformation_indices=test_transformation_indices,
    inferred_mts_validation=inferred_mts_validation,
    inferred_mts_test=inferred_mts_test,
    intermediate_features_validation=inferred_intermediate_features_validation,
    intermediate_features_test=inferred_intermediate_features_test,
    ohe_val=ohe_val,
    ohe_test=ohe_test,
)


##### FORECASTING EVALUATION
forecasting_train_indices = np.unique(train_transformation_indices[:, 0])
forecasting_valdation_indices = np.unique(validation_transformation_indices[:, 0])
forecasting_test_indices = np.unique(test_transformation_indices[:, 0])

lstm_evaluator = ForecasterEvaluator(
    config=config,
    mts_dataset=mts_dataset_array,
    train_indices=forecasting_train_indices,
    validation_indices=forecasting_valdation_indices,
    test_indices=forecasting_test_indices,
    horizon_length=config["model_args"]["forecasting_model_args"]["horizon_length"],
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    model_type="lstm",
    num_epochs=config["model_args"]["forecasting_model_args"]["training_args"][
        "num_epochs"
    ],
    early_stopping_patience=config["model_args"]["forecasting_model_args"][
        "training_args"
    ]["num_epochs"],
)

logger.info("Evaluating foreasting improvement on inferred test set...")
lstm_evaluator.evaluate_on_evaluation_set(
    inferred_mts_array=inferred_mts_validation, ohe=ohe_test, type="test"
)

logger.info("Starting forecasting evaluation with NLinaer...")
nlinear_evaluator = ForecasterEvaluator(
    config=config,
    mts_dataset=mts_dataset_array,
    train_indices=forecasting_train_indices,
    validation_indices=forecasting_valdation_indices,
    test_indices=forecasting_test_indices,
    horizon_length=config["model_args"]["forecasting_model_args"]["horizon_length"],
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    model_type="n_linear",
    num_epochs=config["model_args"]["forecasting_model_args"]["training_args"][
        "num_epochs"
    ],
    early_stopping_patience=config["model_args"]["forecasting_model_args"][
        "training_args"
    ]["num_epochs"],
)

logger.info("Evaluating foreasting improvement on inferred test set...")
nlinear_evaluator.evaluate_on_evaluation_set(
    inferred_mts_array=inferred_mts_validation, ohe=ohe_test, type="test"
)


tcn_evaluator = ForecasterEvaluator(
    config=config,
    mts_dataset=mts_dataset_array,
    train_indices=forecasting_train_indices,
    validation_indices=forecasting_valdation_indices,
    test_indices=forecasting_test_indices,
    horizon_length=config["model_args"]["forecasting_model_args"]["horizon_length"],
    window_size=config["model_args"]["forecasting_model_args"]["window_size"],
    model_type="tcn",
    num_epochs=config["model_args"]["forecasting_model_args"]["training_args"][
        "num_epochs"
    ],
    early_stopping_patience=config["model_args"]["forecasting_model_args"][
        "training_args"
    ]["num_epochs"],
)

logger.info("Evaluating foreasting improvement on inferred test set...")
tcn_evaluator.evaluate_on_evaluation_set(
    inferred_mts_array=inferred_mts_validation, ohe=ohe_test, type="test"
)


logger.info("Finished running")
