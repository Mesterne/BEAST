import os
import sys
import argparse
import pandas as pd
import numpy as np
import random
import torch
from matplotlib import pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# Parse the configuration file path
argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("config_path", type=str)
args = vars(argument_parser.parse_args())

# Add project root to the system path
project_root = os.path.abspath(os.path.join(os.getcwd()))
if project_root not in sys.path:
    sys.path.append(project_root)


import wandb
import uuid
from src.utils.yaml_loader import read_yaml  # noqa: E402
from src.utils.generate_dataset import generate_feature_dataframe  # noqa: E402
from src.utils.features import decomp_and_features  # noqa: E402
from src.utils.pca import PCAWrapper  # noqa: E402
from src.utils.experiment_helper import (  # noqa: E402
    get_mts_dataset,
    get_model_by_type,
)
from src.data_transformations.generation_of_supervised_pairs import (  # noqa: E402
    create_train_val_test_split,
    get_col_names_original_target,
)
from src.utils.logging_config import logger  # noqa: E402
from src.models.naive_correlation import CorrelationModel
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.plots.pca_for_each_uts_with_transformed import (
    plot_pca_for_each_uts_with_transformed,
)  # noqa: E402

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

# Load the configuration file
config = read_yaml(args["config_path"])

# Data loading parameters
data_dir = os.path.join(config["dataset_args"]["directory"], "train.csv")
timeseries_to_use = config["dataset_args"]["timeseries_to_use"]
step_size = config["dataset_args"]["step_size"]
context_length = config["dataset_args"]["window_size"]
log_training_to_wandb = config["training_args"]["log_to_wandb"]

output_dir = os.getenv("OUTPUT_DIR", "")

if log_training_to_wandb:
    job_name = os.environ.get("JOB_NAME", str(uuid.uuid4()))
    wandb.init(
        project="MTS-BEAST",
        name=job_name,
        config=config["model_args"]["feature_model_args"],
    )


logger.info("Initialized system")
logger.info(f"Running with experiment settings:\n{config}")
logger.info(
    f"All outputs will be stored in: {output_dir} (Relative to where you ran the program from)"
)

# Load data and generate dataset of multivariate time series context windows
mts_dataset = get_mts_dataset(
    data_dir=data_dir,
    time_series_to_use=timeseries_to_use,
    context_length=context_length,
    step_size=step_size,
)
dataset_size = len(mts_dataset)
num_uts_in_mts = len(timeseries_to_use)

logger.info("Successfully generated multivariate time series dataset")

# Generate feature dataframe
sp = config["stl_args"]["series_periodicity"]
mts_feature_df = generate_feature_dataframe(
    data=mts_dataset, series_periodicity=sp, dataset_size=dataset_size
).sample(frac=0.25)

logger.info("Successfully generated feature dataframe")

# Generate decompositions dataset
mts_decomps, _ = decomp_and_features(
    data=mts_dataset,
    series_periodicity=sp,
    dataset_size=dataset_size,
    decomps_only=True,
)

logger.info("Successfully generated multivariate time series decompositions")

# Generate MTS PCA space
mts_pca_transformer = PCAWrapper()
mts_pca_df = mts_pca_transformer.fit_transform(mts_feature_df)

logger.info("Successfully generated MTS PCA space")

# Generate train, vlaidation and test splits
ORIGINAL_NAMES, TARGET_NAMES = get_col_names_original_target()

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
    mts_pca_df,
    mts_feature_df,
    FEATURES_NAMES,
    TARGET_NAMES,
    SEED,
    output_dir=output_dir,
)


# FIXME: Currently only getting one RANDOM sample from test supervised dataset
# Get random model input
df_model_input = validation_supervised_dataset.sample(1)
# Original MTS
original_mts_index = df_model_input["original_index"].values[0].astype(int)
original_mts = mts_dataset[original_mts_index]
original_mts_decomps = mts_decomps[original_mts_index]
original_mts_features = mts_feature_df.iloc[original_mts_index]
# Target MTS
target_mts_index = df_model_input["target_index"].values[0].astype(int)
target_mts = mts_dataset[target_mts_index]
target_mts_features = mts_feature_df.iloc[target_mts_index]
# UTS to transform
num_features_per_uts = config["time_series_args"]["num_features_per_uts"]
init_uts_index = config["time_series_args"]["init_transform_uts"]["index"]
init_uts_name = config["time_series_args"]["init_transform_uts"]["name"]
init_uts = original_mts[init_uts_name]  # Series
init_uts_decomp = original_mts_decomps[init_uts_index]
# FIXME: Manual transformation is currently not supported
# Whether or not the initial transformation is based on values from config file
manual_init_transform = config["time_series_args"]["manual_init_transform"]

# Model initialization
model_type = config["model_args"]["model_type"]
model_params = config["model_args"]

feature_model = CorrelationModel()
ga = GeneticAlgorithmWrapper(
    model_type=model_type,
    model_params=model_params,
    mts_dataset=mts_dataset,
    mts_features=mts_feature_df,
    mts_decomp=mts_decomps,
    num_uts_in_mts=num_uts_in_mts,
    num_features_per_uts=num_features_per_uts,
    manual_init_transform=manual_init_transform,
)
logger.info(f"Successfully initialized the {model_type} model")

# Fit model to data
# TODO:  Add posibility to pass log to  wandb
feature_model.train(training_set=mts_feature_df)

validation_predicted_features = feature_model.infer(validation_supervised_dataset)

sampled_predicted_features = validation_predicted_features.sample(1)
sampled_prediction_index = (
    sampled_predicted_features["prediction_index"].astype(int).values[0]
)
sampled_row_in_validation = validation_supervised_dataset.iloc[sampled_prediction_index]
sampled_original_mts_index = int(sampled_row_in_validation["original_index"])
sampled_target_mts_index = int(sampled_row_in_validation["target_index"])
sampled_predicted_features = sampled_predicted_features.drop(
    ["prediction_index"], axis=1
)

(
    transformed_mts_list,
    transformed_features_list,
    transformed_factors_list,
) = ga.transform(
    predicted_features=sampled_predicted_features,
    original_mts_index=sampled_original_mts_index,
    target_mts_index=sampled_target_mts_index,
)

DEFAULT_RUN_INDEX = 0
transformed_features = transformed_features_list[DEFAULT_RUN_INDEX]
transformed_mts = transformed_mts_list[DEFAULT_RUN_INDEX]

transformed_features = [item for sublist in transformed_features for item in sublist]

transformed_features = pd.DataFrame(
    [transformed_features], columns=sampled_predicted_features.columns
)

uts_names = ["grid-load", "grid-loss", "grid-temp"]

uts_wise_pca_fig = plot_pca_for_each_uts_with_transformed(
    mts_features_df=mts_feature_df,
    predicted_features=sampled_predicted_features,
    transformed_features=transformed_features,
    original_index=sampled_original_mts_index,
    target_index=sampled_target_mts_index,
    uts_names=uts_names,
)
uts_wise_pca_fig.savefig(os.path.join(output_dir, "uts_wise_pca.png"))
logger.info("Successfully generated PCA spaces")

# Save results to pdf
# TODO:  All plots should be function calls
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
            # FIXME: Want to add actual feature values to the result.
            # This is one way to do it but it is not very readable.
            # Maybe use its own text output file?
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

logger.info(f"Results were succesfully saved to {pdf_filename}")
