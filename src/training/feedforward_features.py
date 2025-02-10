import logging
import numpy as np
import random
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import sys
import os
import torch
import wandb

plt.style.use("ggplot")

FEATURES_NAMES = [
    "original_index",
    "original_grid-load_trend-strength",
    "original_grid-load_trend-slope",
    "original_grid-load_trend-linearity",
    "original_grid-load_seasonal-strength",
    "original_grid-loss_trend-strength",
    "original_grid-loss_trend-slope",
    "original_grid-loss_trend-linearity",
    "original_grid-loss_seasonal-strength",
    "original_grid-temp_trend-strength",
    "original_grid-temp_trend-slope",
    "original_grid-temp_trend-linearity",
    "original_grid-temp_seasonal-strength",
    "delta_grid-load_trend-strength",
    "delta_grid-load_trend-slope",
    "delta_grid-load_trend-linearity",
    "delta_grid-load_seasonal-strength",
    "delta_grid-loss_trend-strength",
    "delta_grid-loss_trend-slope",
    "delta_grid-loss_trend-linearity",
    "delta_grid-loss_seasonal-strength",
    "delta_grid-temp_trend-strength",
    "delta_grid-temp_trend-slope",
    "delta_grid-temp_trend-linearity",
    "delta_grid-temp_seasonal-strength",
]

TARGET_NAMES = [
    "target_grid-load_trend-strength",
    "target_grid-load_trend-slope",
    "target_grid-load_trend-linearity",
    "target_grid-load_seasonal-strength",
    "target_grid-loss_trend-strength",
    "target_grid-loss_trend-slope",
    "target_grid-loss_trend-linearity",
    "target_grid-loss_seasonal-strength",
    "target_grid-temp_trend-strength",
    "target_grid-temp_trend-slope",
    "target_grid-temp_trend-linearity",
    "target_grid-temp_seasonal-strength",
]

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd(), "../../"))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.INFO)

logging.info(f"Running from directory: {project_root}")

from src.models.feedforward import FeedForwardFeatureModel
from src.plots.feature_wise_error import plot_distribution_of_feature_wise_error
from src.utils.evaluation.feature_space_evaluation import (
    find_error_of_each_feature_for_each_sample,
)
from src.utils.evaluation.mse import get_mse_for_features_and_overall
from src.utils.model_functions import run_model_inference, train_model
from src.utils.pca import PCAWrapper
from src.utils.yaml_loader import read_yaml
from src.models.forecasting.feedforward import FeedForwardForecaster
from src.utils.features import decomp_and_features
from src.utils.generate_dataset import (
    generate_feature_dataframe,
    generate_windows_dataset,
)
from src.data_transformations.generation_of_supervised_pairs import (
    create_train_val_test_split,
)
from src.plots.pca_train_test_pairing import (
    pca_plot_train_test_pairing_with_predictions,
)
from src.plots.loss_history import plot_loss_history
from src.utils.data_formatting import use_model_predictions_to_create_dataframe

# Setting seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


settings = read_yaml("../../experiments/gridloss/feedforward.yml")

wandb.init(project="MTS-BEAST", config=settings["feature_model_args"])

features_to_use = settings["dataset_args"]["timeseries_to_use"]
data_dir = os.path.join(settings["dataset_args"]["directory"], "train.csv")
step_size = settings["dataset_args"]["step_size"]

learning_rate = settings["training_args"]["learning_rate"]

window_size = settings["forecasting_model_args"]["window_size"]
horizon_length = settings["forecasting_model_args"]["horizon_length"]
model_save_dir = settings["forecasting_model_args"]["model_params_storage_dir"]

feature_model_hidden_network_sizes = settings["feature_model_args"][
    "hidden_network_size"
]

feature_model_save_dir = settings["feature_model_args"]["model_params_storage_dir"]
feature_model_epochs = settings["feature_model_args"]["epochs"]
feature_model_batch_size = settings["feature_model_args"]["batch_size"]
feature_model_learning_rate = settings["feature_model_args"]["learning_rate"]
feature_model_early_stopping_patience = settings["feature_model_args"][
    "early_stopping_patience"
]

forecasting_model_input_size = window_size * len(features_to_use)
forecasting_model_output_size = horizon_length

logging.info("Initialized system")


model = FeedForwardForecaster(
    input_size=forecasting_model_input_size,
    output_size=forecasting_model_output_size,
    save_dir=model_save_dir,
)

df = pd.read_csv(data_dir, index_col=0)
df.index = pd.to_datetime(df.index)
df = df[features_to_use]
df = df.bfill()


num_ts = len(features_to_use)
dataset_size = (df.shape[0] - window_size) // step_size + 1

data = generate_windows_dataset(df, window_size, step_size, features_to_use)

logging.info("Successfully generated windowed dataset")

sp = 24  # STL parameter

feature_df = generate_feature_dataframe(
    data=data, series_periodicity=sp, dataset_size=dataset_size
)

logging.info("Successfully generated feature dataframe")

pca_transformer = PCAWrapper()
mts_pca_df = pca_transformer.fit_transform(feature_df)

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
    pca_df=mts_pca_df,
    feature_df=feature_df,
    FEATURES_NAMES=FEATURES_NAMES,
    TARGET_NAMES=TARGET_NAMES,
    SEED=SEED,
)

feature_model_input_size = X_train.shape[1]
feature_model_output_size = y_train.shape[1]

logging.info("Building model...")
feature_model = FeedForwardFeatureModel(
    input_size=feature_model_input_size,
    output_size=feature_model_output_size,
    hidden_network_sizes=feature_model_hidden_network_sizes,
    save_dir=feature_model_save_dir,
    name="feedforward_feature",
)

logging.info("Training model....")
train_loss_history, validation_loss_history = train_model(
    model=feature_model,
    X_train=X_train,
    y_train=y_train,
    X_validation=X_validation,
    y_validation=y_validation,
    batch_size=feature_model_batch_size,
    num_epochs=feature_model_epochs,
    learning_rate=feature_model_learning_rate,
    early_stopping_patience=feature_model_early_stopping_patience,
)
loss_fig = plot_loss_history(
    train_loss_history=train_loss_history,
    validation_loss_history=validation_loss_history,
)
loss_fig.savefig("training_loss_history.png")

logging.info("Saving training history to png...")

logging.info("Running model inference on validation set...")
predictions_validation = run_model_inference(model=feature_model, X_test=X_validation)
# FIXME: The fact that we send the test supervised dataset as an argument is not pretty
predictions_validation = use_model_predictions_to_create_dataframe(
    predictions_validation,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=validation_supervised_dataset,
)
logging.info("Successfully ran inference on validation set...")

logging.info("Running model inference on test set...")
predictions_test = run_model_inference(model=feature_model, X_test=X_test)
# FIXME: The fact that we send the test supervised dataset as an argument is not pretty
predictions_test = use_model_predictions_to_create_dataframe(
    predictions_test,
    TARGET_NAMES=TARGET_NAMES,
    target_dataframe=test_supervised_dataset,
)
logging.info("Successfully ran inference on test set...")

logging.info("Plotting predictions for validation set...")
predictions_without_index = predictions_validation.drop(columns=["prediction_index"])
predictions_pca = pca_transformer.transform(predictions_without_index)
predictions_validation["pca1"] = predictions_pca["pca1"]
predictions_validation["pca2"] = predictions_pca["pca2"]
prediction = predictions_validation.sample(n=1, random_state=SEED).reset_index(
    drop=True
)
index = prediction["prediction_index"][0]
dataset_row = validation_supervised_dataset.loc[index]
dataset_row
fig = pca_plot_train_test_pairing_with_predictions(
    mts_pca_df, dataset_row, predictions_validation, prediction
)
fig.savefig("train_validation_predictions_results.png")


logging.info("Calculating errors for each prediction")
differences_df_validation = find_error_of_each_feature_for_each_sample(
    predictions=predictions_validation,
    labelled_test_dataset=validation_supervised_dataset,
)
differences_df_test = find_error_of_each_feature_for_each_sample(
    predictions=predictions_test, labelled_test_dataset=test_supervised_dataset
)

logging.info("Plotting errors for each prediction on validation set...")
fig = plot_distribution_of_feature_wise_error(differences_df_validation)
fig.savefig("dist_error_features.png")


overall_mse_validation, mse_values_for_each_feature_validation = (
    get_mse_for_features_and_overall(differences_df_validation)
)
overall_mse_test, mse_values_for_each_feature_test = get_mse_for_features_and_overall(
    differences_df_test
)

logging.info(
    f"Overall MSE for model\nValidation: {overall_mse_validation}\nTest: {overall_mse_test}"
)
logging.info(f"Program finished...")
