import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import torch
import logging
from tqdm import tqdm

# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.info(f"Running from directory: {project_root}")

from src.utils.yaml_loader import read_yaml
from src.utils.generate_dataset import (
    generate_windows_dataset,
    generate_feature_dataframe,
)
from src.utils.features import (
    decomp_and_features,
    trend_strength,
    trend_slope,
    trend_linearity,
    seasonal_strength,
)
from src.utils.transformations import (
    manipulate_seasonal_component,
    manipulate_trend_component,
)
from src.utils.experiment_helper import (
    get_mts_dataset,
    get_transformed_uts_with_features_and_decomps,
    get_model_by_type,
)

from src.models.naive_correlation import CorrelationModel
from src.data_transformations.generation_of_supervised_pairs import (
    get_col_names_original_target_delta,
)
from src.utils.genetic_algorithm import GeneticAlgorithm

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from src.utils.pca import PCAWrapper

# Setting seeds
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Load the configuration file
config = read_yaml(args["config_path"])

# Data loading parameters
data_dir = os.path.join(config["dataset_args"]["directory"], "train.csv")
timeseries_to_use = config["dataset_args"]["timeseries_to_use"]
step_size = config["dataset_args"]["step_size"]
context_length = config["dataset_args"]["window_size"]

# Load data and generate dataset of multivariate time series context windows
mts_dataset = get_mts_dataset(
    data_dir=data_dir,
    time_series_to_use=timeseries_to_use,
    context_length=context_length,
    step_size=step_size,
)
dataset_size = len(mts_dataset)
num_uts_in_mts = len(timeseries_to_use)

logging.info("Successfully generated multivariate time series dataset")

# Generate feature dataframe
sp = config["stl_args"]["series_periodicity"]
mts_feature_df = generate_feature_dataframe(
    data=mts_dataset, series_periodicity=sp, dataset_size=dataset_size
)

logging.info("Successfully generated feature dataframe")

# Generate decompositions dataset
mts_decomps, _ = decomp_and_features(
    data=mts_dataset,
    series_periodicity=sp,
    dataset_size=dataset_size,
    decomps_only=True,
)

logging.info("Successfully generated multivariate time series decompositions")

# Time series arguments
num_features_per_uts = config["time_series_args"]["num_features_per_uts"]
original_mts_index = config["time_series_args"]["original_mts_index"]
init_uts_index = config["time_series_args"]["init_transform_uts"]["index"]
init_uts_name = config["time_series_args"]["init_transform_uts"]["name"]
original_mts = mts_dataset[original_mts_index]  # DataFrame
original_mts_decomps = mts_decomps[original_mts_index]
original_mts_features = mts_feature_df.iloc[original_mts_index]
init_uts = original_mts[init_uts_name]  # Series
init_uts_decomp = original_mts_decomps[init_uts_index]
target_mts_index = config["time_series_args"]["target_mts_index"]
target_mts = mts_dataset[target_mts_index]  # DataFrame
manual_init_transform = config["time_series_args"]["manual_init_transform"]

# Model initialization
model_type = config["model_args"]["model_type"]
model_params = config["model_args"]
model = get_model_by_type(
    model_type,
    model_params,
    mts_dataset,
    mts_feature_df,
    mts_decomps,
    num_uts_in_mts,
    num_features_per_uts,
    manual_init_transform,
)
logging.info(f"Successfully initialized the {model_type} model")

# Fit model to data
model.fit()

# FIXME: Transform return values are not yet really model agnostic
# Transform MTS
(
    transformed_mts_list,
    transformed_features_list,
    transformed_factors_list,
    corr_predicted_features,
) = model.transform(original_mts_index, target_mts_index, init_uts_index)


# NOTE: By default, we will only consider the first run in visualization
DEFAULT_RUN_INDEX = 0
transformed_features = transformed_features_list[DEFAULT_RUN_INDEX]
transformed_mts = transformed_mts_list[DEFAULT_RUN_INDEX]

# Create pca spaces for 2D visualization
# MTS PCA
mts_pca_transformer = PCAWrapper()
mts_pca_df = mts_pca_transformer.fit_transform(mts_feature_df)

# One PCA for each UTS
tot_num_mts_features = num_uts_in_mts * num_features_per_uts
mts_feature_columns = list(mts_feature_df.columns)
uts_pca_df_list = []
uts_pca_transformer_list = []
for i in range(num_uts_in_mts):
    uts_pca_transformer = PCAWrapper()
    uts_columns = mts_feature_columns[
        i * num_features_per_uts : (i + 1) * num_features_per_uts
    ]
    uts_feature_df = mts_feature_df[uts_columns]
    uts_pca_df = uts_pca_transformer.fit_transform(uts_feature_df)
    uts_pca_df_list.append(uts_pca_df)
    uts_pca_transformer_list.append(uts_pca_transformer)

logging.info("Successfully generated PCA spaces")

# Add corr model predicted features to PCA spaces
pred_features_pca_input = corr_predicted_features
pred_features_pca_df = mts_pca_transformer.transform(
    pd.DataFrame([pred_features_pca_input], columns=mts_feature_columns)
)
pred_uts_features_pca_df_list = []
for i in range(num_uts_in_mts):
    uts_features_pca_input = pred_features_pca_input[
        i * num_features_per_uts : (i + 1) * num_features_per_uts
    ]
    uts_columns = mts_feature_columns[
        i * num_features_per_uts : (i + 1) * num_features_per_uts
    ]
    uts_features_pca_df = uts_pca_transformer_list[i].transform(
        pd.DataFrame([uts_features_pca_input], columns=uts_columns)
    )
    pred_uts_features_pca_df_list.append(uts_features_pca_df)

# Add GA transofmed points to PCA spaces
mts_features_pca_input = np.asarray(transformed_features).reshape(tot_num_mts_features)
transformed_mts_pca_df = mts_pca_transformer.transform(
    pd.DataFrame([mts_features_pca_input], columns=mts_feature_columns)
)
transformed_uts_pca_df_list = []
for i in range(num_uts_in_mts):
    uts_features_pca_input = transformed_features[i]
    uts_columns = mts_feature_columns[
        i * num_features_per_uts : (i + 1) * num_features_per_uts
    ]
    transformed_uts_pca_df = uts_pca_transformer_list[i].transform(
        pd.DataFrame(
            [uts_features_pca_input],
            columns=uts_columns,
        )
    )
    transformed_uts_pca_df_list.append(transformed_uts_pca_df)

# Estimate expected target regions in PCA space based on K-nearest neighbors
k = 50

# FIXME: This choses points close to original UTS, want to choose points close to transformed UTS!
init_uts_features = mts_feature_df[
    mts_feature_columns[init_uts_index : init_uts_index + 4]
].to_numpy()
init_uts_feature_distances = np.linalg.norm(
    init_uts_features - init_uts_features[original_mts_index], axis=1
)
distance_indices = np.argsort(init_uts_feature_distances)[:k]

uts_reshape_original_mts_features = original_mts_features.to_numpy().reshape(
    num_uts_in_mts, num_features_per_uts
)
uts_reshape_target_mts_features = (
    mts_feature_df.iloc[target_mts_index]
    .to_numpy()
    .reshape(num_uts_in_mts, num_features_per_uts)
)
uts_reshape_transformed_mts_features = np.asarray(transformed_features).reshape(
    num_uts_in_mts, num_features_per_uts
)

# Save results to pdf
pdf_filename = os.path.join("src", "results", "experiment_results.pdf")
with PdfPages(pdf_filename) as pdf:

    if manual_init_transform:
        assert (
            (model.manual_transformed_uts is not None)
            and (model.manual_transformed_uts_decomp is not None)
            and (model.manual_transformed_uts_features is not None)
        ), "Manual transform not done"
        # Plot original and transformed UTS
        fig, axs = plt.subplots(2, 1, figsize=(10, 10))
        if not manual_init_transform:
            fig.suptitle(
                f"Original {init_uts_index}, Transformed UTS, and Target {target_mts_index}"
            )
        else:
            fig.suptitle(f"Original {init_uts_index} and Transformed UTS")
        axs[0].plot(init_uts, label="Original UTS")
        axs[0].set_title(f"Original {init_uts_name}")
        axs[0].legend()
        axs[1].plot(model.manual_transformed_uts, label="Transformed UTS")
        axs[1].set_title(f"Transformed {init_uts_name}")
        axs[1].legend()
        pdf.savefig(fig)
        plt.close()

    # Plot original and transformed MTS
    fig, axs = plt.subplots(num_uts_in_mts, 3, figsize=(10, 10))
    if not manual_init_transform:
        fig.suptitle(
            f"Original {original_mts_index}, Transformed UTS, and Target {target_mts_index}"
        )
    else:
        fig.suptitle(f"Original {original_mts_index} and Transformed UTS")
    for i in range(num_uts_in_mts):
        uts_name = timeseries_to_use[i]

        orig_feat_val = uts_reshape_original_mts_features[i]
        axs[i, 0].plot(
            original_mts[uts_name],
            label=f"TD: {orig_feat_val[0]:.2f}, TS: {orig_feat_val[1]:.2f}, TL: {orig_feat_val[2]:.2f}, SS: {orig_feat_val[3]:.2f}",
        )
        axs[i, 0].set_title(f"Original {uts_name}")
        axs[i, 0].legend(handletextpad=2, labelspacing=1.5)

        transformed_feat_val = uts_reshape_transformed_mts_features[i]
        axs[i, 1].plot(
            transformed_mts[i],
            label=f"TD: {transformed_feat_val[0]:.2f}, TS: {transformed_feat_val[1]:.2f}, TL: {transformed_feat_val[2]:.2f}, SS: {transformed_feat_val[3]:.2f}",
        )
        axs[i, 1].set_title(f"Transformed {uts_name}")
        axs[i, 1].legend()

        target_feat_val = uts_reshape_target_mts_features[i]
        axs[i, 2].plot(
            target_mts[uts_name],
            label=f"TD: {target_feat_val[0]:.2f}, TS: {target_feat_val[1]:.2f}, TL: {target_feat_val[2]:.2f}, SS: {target_feat_val[3]:.2f}",
        )
        axs[i, 2].set_title(f"Target {uts_name}")
        axs[i, 2].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

    # Plot PCA spaces
    fig, axs = plt.subplots(4, 1, figsize=(10, 20))
    fig.suptitle("PCA Spaces")
    # All MTS points
    axs[0].scatter(mts_pca_df["pca1"], mts_pca_df["pca2"], label="MTS")
    # Original MTS
    axs[0].scatter(
        mts_pca_df["pca1"][original_mts_index],
        mts_pca_df["pca2"][original_mts_index],
        color="yellow",
        s=100,
        edgecolors="black",
        label="Original",
    )

    # Target MTS
    if not manual_init_transform:
        axs[0].scatter(
            mts_pca_df["pca1"][target_mts_index],
            mts_pca_df["pca2"][target_mts_index],
            color="orange",
            s=100,
            edgecolors="black",
            label="Target",
        )

    if manual_init_transform:
        # Expected target region based on K-nearest neighbors
        for mts_index in distance_indices:
            axs[0].scatter(
                mts_pca_df["pca1"][mts_index],
                mts_pca_df["pca2"][mts_index],
                color="green",
                s=50,
                alpha=0.5,
            )

    # Correlation model predicted features
    axs[0].scatter(
        pred_features_pca_df["pca1"].iloc[0],
        pred_features_pca_df["pca2"].iloc[0],
        color="pink",
        s=100,
        edgecolors="black",
        label="Predicted (Correlation)",
    )

    # The transformed MTS
    axs[0].scatter(
        transformed_mts_pca_df["pca1"].iloc[0],
        transformed_mts_pca_df["pca2"].iloc[0],
        color="red",
        s=100,
        edgecolors="black",
        label="Transformed (GA)",
    )

    axs[0].set_title("MTS PCA Space")
    axs[0].legend()

    for i in range(num_uts_in_mts):
        # All points for one UTS
        uts_name = timeseries_to_use[i]
        axs[i + 1].scatter(
            uts_pca_df_list[i]["pca1"], uts_pca_df_list[i]["pca2"], label=f"{uts_name}"
        )
        # The initial UTS
        axs[i + 1].scatter(
            uts_pca_df_list[i]["pca1"][original_mts_index],
            uts_pca_df_list[i]["pca2"][original_mts_index],
            color="yellow",
            s=100,
            edgecolors="black",
            label="Original",
        )

        # Target UTS
        if not manual_init_transform:
            axs[i + 1].scatter(
                uts_pca_df_list[i]["pca1"][target_mts_index],
                uts_pca_df_list[i]["pca2"][target_mts_index],
                color="orange",
                s=100,
                edgecolors="black",
                label="Target",
            )

        if manual_init_transform:
            # Expected target region based on K-nearest neighbors
            if i == init_uts_index:
                for mts_index in distance_indices:
                    axs[i + 1].scatter(
                        uts_pca_df_list[i]["pca1"][mts_index],
                        uts_pca_df_list[i]["pca2"][mts_index],
                        color="green",
                        s=50,
                        alpha=0.5,
                    )

        # Correlation model predicted features
        axs[i + 1].scatter(
            pred_uts_features_pca_df_list[i]["pca1"].iloc[0],
            pred_uts_features_pca_df_list[i]["pca2"].iloc[0],
            color="pink",
            s=100,
            edgecolors="black",
            label="Predicted (Correlation)",
        )

        # The transformed UTS
        axs[i + 1].scatter(
            transformed_uts_pca_df_list[i]["pca1"].iloc[0],
            transformed_uts_pca_df_list[i]["pca2"].iloc[0],
            color="red",
            s=100,
            edgecolors="black",
            label="Transformed (GA)",
        )

        axs[i + 1].set_title(f"{uts_name} PCA")
        axs[i + 1].legend()
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    pdf.savefig(fig)
    plt.close()

logging.info(f"Results were succesfully saved to {pdf_filename}")
