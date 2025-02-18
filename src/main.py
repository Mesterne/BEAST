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
)

from src.models.naive_correlation import CorrelationModel
from src.data_transformations.generation_of_supervised_pairs import (
    get_col_names_original_target_delta,
)
from src.utils.genetic_algorithm import GeneticAlgorithm

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

# Direct transformation of univariate time series
init_mts_index = config["transformation_args"]["mts_index"]
init_uts_index = config["transformation_args"]["init_transform_uts"]["index"]
init_uts_name = config["transformation_args"]["init_transform_uts"]["name"]

init_mts = mts_dataset[init_mts_index]  # DataFrame
init_mts_decomps = mts_decomps[init_mts_index]
init_mts_features = mts_feature_df.iloc[init_mts_index]
init_uts = init_mts[init_uts_name]  # Series
init_uts_decomp = init_mts_decomps[init_uts_index]

trend_det_factor = config["transformation_args"]["trend_det_factor"]
trend_slope_factor = config["transformation_args"]["trend_slope_factor"]
trend_lin_factor = config["transformation_args"]["trend_lin_factor"]
seasonal_det_factor = config["transformation_args"]["seasonal_det_factor"]

transformed_uts, transformed_uts_features, transformed_uts_decomp = (
    get_transformed_uts_with_features_and_decomps(
        uts_decomp=init_uts_decomp,
        trend_det_factor=trend_det_factor,
        trend_slope_factor=trend_slope_factor,
        trend_lin_factor=trend_lin_factor,
        seasonal_det_factor=seasonal_det_factor,
    )
)

logging.info(f"Successfully transformed {init_uts_name} of mts {init_mts_index}")

# TODO: Perform automatic transformation of the other time series in the MTS
# How this is done should be specified in configuration file
# Models are asumed to be pre-trained and saved in the specified directories
# NOTE: MVP with correlation model

# Initialize the correlation model
model = CorrelationModel()

# Train the correlation model
model.train(mts_feature_df)

logging.info("Successfully trained the correlation model")

# Prepare model input dataframe
uts_feature_col_names = get_col_names_original_target_delta()
dummy_original_index, dummy_target_index, dummy_delta_index = 0, 0, 0
original_values, target_values, delta_values = (
    [dummy_original_index],
    [dummy_target_index],
    [dummy_delta_index],
)

for uts_index in tqdm(range(num_uts_in_mts)):
    if uts_index == init_uts_index:
        original_values = [
            *original_values,
            *init_mts_features.iloc[uts_index : uts_index + 4],
        ]
        target_values = [
            *target_values,
            *transformed_uts_features,
        ]
        delta_features = np.asarray(transformed_uts_features) - np.asarray(
            init_mts_features.iloc[uts_index : uts_index + 4]
        )
        delta_values = [
            *delta_values,
            *delta_features,
        ]
    else:
        original_values = [
            *original_values,
            *init_mts_features.iloc[uts_index : uts_index + 4],
        ]
        # NOTE: Can be viewed as dummy values, these are what we are trying to predict
        target_values = [
            *target_values,
            *init_mts_features.iloc[uts_index : uts_index + 4],
        ]
        delta_values = [
            *delta_values,
            *np.zeros(4),
        ]
model_input_values = np.asarray([*original_values, *target_values, *delta_values])
model_input_df = pd.DataFrame([model_input_values], columns=uts_feature_col_names)

logging.info("Successfully generated model input dataframe")

# Predict features
predicted_features = model.infer(model_input_df)

logging.info("Successfully predicted features")

# Transform other UTS in MTS with GA

# Genetic Algorithm parameters
num_GA_runs = config["genetic_algorithm_args"]["num_runs"]
num_generations = config["genetic_algorithm_args"]["num_generations"]
num_parents_mating = config["genetic_algorithm_args"]["num_parents_mating"]
sol_per_pop = config["genetic_algorithm_args"]["solutions_per_population"]
init_range_low = config["genetic_algorithm_args"]["init_range_low"]
init_range_high = config["genetic_algorithm_args"]["init_range_high"]
parent_selection_type = config["genetic_algorithm_args"]["parent_selection_type"]
crossover_type = config["genetic_algorithm_args"]["crossover_type"]
mutation_type = config["genetic_algorithm_args"]["mutation_type"]
mutation_percent_genes = config["genetic_algorithm_args"]["mutation_percent_genes"]
num_genes = 4
trend_det_factor_low = config["genetic_algorithm_args"]["legal_values"][
    "trend_det_factor"
][0]
trend_det_factor_high = config["genetic_algorithm_args"]["legal_values"][
    "trend_det_factor"
][1]
trend_slope_factor_low = config["genetic_algorithm_args"]["legal_values"][
    "trend_slope"
][0]
trend_slope_factor_high = config["genetic_algorithm_args"]["legal_values"][
    "trend_slope"
][1]
trend_lin_factor_low = config["genetic_algorithm_args"]["legal_values"]["trend_lin"][0]
trend_lin_factor_high = config["genetic_algorithm_args"]["legal_values"]["trend_lin"][1]
seasonal_det_factor_low = config["genetic_algorithm_args"]["legal_values"][
    "seasonal_det"
][0]
seasonal_det_factor_high = config["genetic_algorithm_args"]["legal_values"][
    "seasonal_det"
][1]

legal_factor_values = [
    np.linspace(trend_det_factor_low, trend_det_factor_high, 100),
    np.linspace(trend_slope_factor_low, trend_slope_factor_high, 100),
    np.linspace(trend_lin_factor_low, trend_lin_factor_high, 100),
    np.linspace(seasonal_det_factor_low, seasonal_det_factor_high, 100),
]

GA_runs_mts = []
GA_runs_features = []
GA_runs_factors = []


for _ in tqdm(range(num_GA_runs)):

    transformed_mts = []
    transformed_mts_features = []
    transformed_mts_factors = []

    for i in range(num_uts_in_mts):
        if i == init_uts_index:
            transformed_mts.append(transformed_uts)
            transformed_mts_features.append(transformed_uts_features)
            transformed_mts_factors.append(
                [
                    trend_det_factor,
                    trend_slope_factor,
                    trend_lin_factor,
                    seasonal_det_factor,
                ]
            )
            continue

        univariate_decomps = init_mts_decomps[i]
        univariate_target_features = predicted_features[i]

        ga_instance = GeneticAlgorithm(
            original_time_series_decomp=univariate_decomps,
            target_features=univariate_target_features,
            num_generations=num_generations,
            num_parents_mating=num_parents_mating,
            sol_per_pop=sol_per_pop,
            num_genes=num_genes,
            gene_space=legal_factor_values,
            init_range_low=init_range_low,
            init_range_high=init_range_high,
            parent_selection_type=parent_selection_type,
            crossover_type=crossover_type,
            mutation_type=mutation_type,
            mutation_percent_genes=mutation_percent_genes,
        )

        ga_instance.run_genetic_algorithm()

        factors, _, _ = ga_instance.get_best_solution()

        transformed_trend = manipulate_trend_component(
            univariate_decomps.trend, factors[0], factors[1], factors[2], m=0
        )
        transformed_seasonal = manipulate_seasonal_component(
            univariate_decomps.seasonal, factors[3]
        )

        transformed_ts = (
            transformed_trend + transformed_seasonal + univariate_decomps.resid
        )
        transformed_mts.append(transformed_ts)

        transformed_mts_features.append(
            [
                trend_strength(transformed_trend, univariate_decomps.resid),
                trend_slope(transformed_trend),
                trend_linearity(transformed_trend),
                seasonal_strength(transformed_seasonal, univariate_decomps.resid),
            ]
        )

        transformed_mts_factors.append(factors)

    GA_runs_mts.append(transformed_mts)
    GA_runs_features.append(transformed_mts_features)
    GA_runs_factors.append(transformed_mts_factors)


# TODO: Load the pre-trained forecasting model
# Perform forcasting on both original and transformed MTS

# TODO: Report results/plots. Save to file/directory.
# Statistics
# Initial transformation vs. original
# MTS transformation vs. original
# PCA space transformation vs. original
# Forecasting results on original vs. transformed MTS

# TODO: Support multiple runs. Random initial transformations?
