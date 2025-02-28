import pandas as pd
import numpy as np
import logging
from tqdm import tqdm
from typing import List, Tuple
from statsmodels.tsa.seasonal import DecomposeResult as DecompResults
from abc import ABC, abstractmethod
from src.models.reconstruction.genetic_algorithm_wrapper import GeneticAlgorithmWrapper
from src.utils.generate_dataset import generate_windows_dataset
from src.utils.transformations import (
    manipulate_seasonal_component,
    manipulate_trend_component,
)
from src.utils.features import (
    trend_strength,
    trend_slope,
    trend_linearity,
    seasonal_strength,
)

from src.models.timeseries_transformation_model import TimeSeriesTransformationModel

# Set up logging
logging.basicConfig(level=logging.INFO)


# DATA
def get_mts_dataset(
    data_dir, time_series_to_use, context_length, step_size, backfill=True, index_col=0
):
    df = pd.read_csv(data_dir, index_col=index_col)
    df.index = pd.to_datetime(df.index)
    df = df[time_series_to_use]
    if backfill:
        df = df.bfill()
    mts_dataset = generate_windows_dataset(
        df, context_length, step_size, time_series_to_use
    )
    return mts_dataset


def get_transformed_uts_with_features_and_decomps(
    uts_decomp,
    trend_det_factor,
    trend_slope_factor,
    trend_lin_factor,
    seasonal_det_factor,
    m=0,  # FIXME: Not yet included in config. Other values than 0 have been problematic.
):
    transformed_uts_trend = manipulate_trend_component(
        trend_comp=uts_decomp.trend,
        f=trend_det_factor,
        g=trend_slope_factor,
        h=trend_lin_factor,
        m=m,
    )

    transformed_uts_seasonal = manipulate_seasonal_component(
        seasonal_comp=uts_decomp.seasonal, k=seasonal_det_factor
    )

    transformed_uts = (
        transformed_uts_trend + transformed_uts_seasonal + uts_decomp.resid
    )

    transformed_uts_decomp = {
        "trend": transformed_uts_trend,
        "seasonal": transformed_uts_seasonal,
        "resid": uts_decomp.resid,
    }

    transformed_uts_features = np.array(
        [
            trend_strength(transformed_uts, uts_decomp.trend),
            trend_slope(transformed_uts),
            trend_linearity(transformed_uts),
            seasonal_strength(transformed_uts, uts_decomp.seasonal),
        ]
    )

    return transformed_uts, transformed_uts_features, transformed_uts_decomp


def get_model_by_type(
    model_type: str,
    model_params: dict,
    mts_dataset: pd.DataFrame,
    mts_features: pd.DataFrame,
    mts_decomp: List[DecompResults],
    num_uts_in_mts: int,
    num_features_per_uts: int,
    manual_init_transform: bool,
) -> TimeSeriesTransformationModel:
    if model_type == "correlation_GA":
        return GeneticAlgorithmWrapper(
            model_type,
            model_params,
            mts_dataset,
            mts_features,
            mts_decomp,
            num_uts_in_mts,
            num_features_per_uts,
            manual_init_transform,
        )
    else:
        raise ValueError(f"Model type {model_type} not supported")


# PLOTTING
