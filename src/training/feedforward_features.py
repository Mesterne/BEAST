import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import logging
import numpy as np
import random
import pandas as pd
import sys
import os

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd(), '../../'))
if project_root not in sys.path:
    sys.path.append(project_root)

logging.basicConfig(level=logging.DEBUG)

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
from src.utils.generate_dataset import generate_windows_dataset
from src.data_transformations.generation_of_supervised_pairs import (
    generate_supervised_dataset_from_original_and_target_dist,
)
from src.plots.pca_train_test_pairing import (
    pca_plot_train_test_pairing,
    pca_plot_train_test_pairing_with_predictions,
)


SEED = 42
np.random.seed(SEED)
random.seed(SEED)

settings = read_yaml("experiments/gridloss/feedforward.yml")

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

# TODO: feature_model_input_size = 2*len(features_to_use)
# TODO: feature_model_output_size = len(features_to_use)
feature_model_save_dir = settings["feature_model_args"]["model_params_storage_dir"]
feature_model_epochs = settings["feature_model_args"]["epochs"]
feature_model_batch_size = settings["feature_model_args"]["batch_size"]
feature_model_learning_rate = settings["feature_model_args"]["learning_rate"]

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

decomps, features = decomp_and_features(
    data, series_periodicity=sp, dataset_size=dataset_size
)

mts_features_reshape = features.reshape(
    (features.shape[0], features.shape[1] * features.shape[2])
)

ts_indices_to_names = {0: "grid-load", 1: "grid-loss", 2: "grid-temp"}

data = []
for idx in range(features.shape[0]):
    for ts_idx in range(features.shape[1]):
        row = {
            "index": idx,
            "ts_name": ts_indices_to_names[ts_idx],
            "trend-strength": features[idx, ts_idx, 0],
            "trend-slope": features[idx, ts_idx, 1],
            "trend-linearity": features[idx, ts_idx, 2],
            "seasonal-strength": features[idx, ts_idx, 3],
        }
        data.append(row)

df = pd.DataFrame(data)

feature_df = df.pivot_table(
    index="index",
    columns="ts_name",
    values=["trend-strength", "trend-slope", "trend-linearity", "seasonal-strength"],
)

feature_df.columns = [f"{ts}_{feature}" for feature, ts in feature_df.columns]

# Extract time series names and their features
ts_names = df["ts_name"].unique()
features = ["trend-strength", "trend-slope", "trend-linearity", "seasonal-strength"]


# Create the ordered column list
ordered_columns = [f"{ts}_{feature}" for ts in ts_names for feature in features]

# Reorder columns based on the ordered list
feature_df = feature_df[ordered_columns]

logging.info("Successfully generated feature dataframe")

pca_transformer = PCAWrapper()
mts_pca_df = pca_transformer.fit_transform(feature_df)

fig = px.scatter(mts_pca_df, x="pca1", y="pca2", hover_data=["index"])
fig.write_html("pca_scatter.html")
logging.info("Generated PCA plot of features")

test_indices = mts_pca_df[(mts_pca_df["pca1"] > 0.6) & (mts_pca_df["pca2"] > 0)][
    "index"
].values
train_indices = mts_pca_df["index"][~mts_pca_df["index"].isin(test_indices)].values
mts_pca_df["isTrain"] = mts_pca_df["index"].isin(train_indices)

train_pca_df = mts_pca_df[mts_pca_df["isTrain"] == True]
test_pca_df = mts_pca_df[mts_pca_df["isTrain"] == False]

train_features = feature_df[~feature_df.index.isin(test_indices)]
test_features = feature_df.iloc[test_indices]


# To generate a training set, we create a matching between all MTSs in the
# defined training feature space
train_supervised_dataset = generate_supervised_dataset_from_original_and_target_dist(
    train_features, train_features
)
test_supervised_dataset = generate_supervised_dataset_from_original_and_target_dist(
    train_features, test_features
)

dataset_row = test_supervised_dataset.sample(n=1, random_state=SEED).reset_index(
    drop=True
)
fig = pca_plot_train_test_pairing(mts_pca_df, dataset_row)
fig.show()
logging.info("Generated PCA plot with target/test pairing")


def generate_X_y_pairs_from_df(df):
    # Assuming your DataFrame is called df
    features_X = [
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

    target_y = [
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

    # Extract X and y as NumPy arrays (if needed by the model)
    X = df.loc[:, features_X].values
    y = df.loc[:, target_y].values

    # Optionally check the shapes
    print("X shape:", X.shape)
    print("y shape:", y.shape)

    return X, y


X_train, y_train = generate_X_y_pairs_from_df(train_supervised_dataset)
X_test, y_test = generate_X_y_pairs_from_df(test_supervised_dataset)
logging.info("Generated X, y pairs for training")

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
loss_history = train_model(
    model=feature_model,
    X=X_train,
    y=y_train,
    batch_size=feature_model_batch_size,
    num_epochs=feature_model_epochs,
    learning_rate=feature_model_learning_rate,
)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=list(range(1, feature_model_epochs + 1)),
        y=loss_history,
        mode="lines",
        name="Training Loss",
    )
)
fig.update_layout(
    title="Training Loss History",
    xaxis_title="Epoch",
    yaxis_title="Loss",
    template="plotly_white",
)
fig.write_html("training_loss_history.html")

logging.info("Saving training history to html...")

logging.info("Running model inference...")
predictions = run_model_inference(model=feature_model, X_test=X_test)
target_y = [
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
column_names = [col.replace("target_", "") for col in target_y]
predictions_df = pd.DataFrame(predictions, columns=column_names)
predictions_df["prediction_index"] = range(len(predictions))
predictions = predictions_df
logging.info("Successfully ran inference...")

logging.info("Plotting predictions...")
predictions_without_index = predictions.drop(columns=["prediction_index"])
predictions_pca = pca_transformer.transform(predictions_without_index)
predictions["pca1"] = predictions_pca["pca1"]
predictions["pca2"] = predictions_pca["pca2"]

prediction = predictions.sample(n=1, random_state=SEED).reset_index(drop=True)
index = prediction["prediction_index"][0]

# Assuming test_dataset and mts_pca_df are already available
dataset_row = test_supervised_dataset.loc[index]
dataset_row
fig = pca_plot_train_test_pairing_with_predictions(
    mts_pca_df, dataset_row, predictions, prediction
)
fig.write_html("train_test_predictions_results.html")


logging.info("Calculating errors for each prediction")
differences_df = find_error_of_each_feature_for_each_sample(
    predictions=predictions, labelled_test_dataset=test_supervised_dataset
)

fig = plot_distribution_of_feature_wise_error(differences_df)
fig.write_html("dist_error_features.html")


overall_mse, mse_values_for_each_feature = get_mse_for_features_and_overall(
    differences_df
)
overall_mse

logging.info(f"Overall MSE for model: {overall_mse}")
logging.info(f"Program finished...")
