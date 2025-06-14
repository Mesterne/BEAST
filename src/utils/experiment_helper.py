import logging

import numpy as np
import pandas as pd

from src.utils.features import (seasonal_strength, trend_linearity,
                                trend_slope, trend_strength)
from src.utils.generate_dataset import generate_windows_dataset
from src.utils.transformations import (manipulate_seasonal_component,
                                       manipulate_trend_component)

logging.basicConfig(level=logging.INFO)


# DATA
def get_mts_dataset(
    data_dir, time_series_to_use, context_length, step_size, backfill=True, index_col=0
) -> np.ndarray:
    df = pd.read_csv(data_dir, index_col=index_col)
    df.index = pd.to_datetime(df.index)

    raw_df: pd.DataFrame = df[time_series_to_use]
    if backfill:
        raw_df = raw_df.bfill()
    mts_dataset = generate_windows_dataset(
        raw_df, context_length, step_size, time_series_to_use
    )

    mts_dataset_as_array: np.ndarray = np.array([df.values.T for df in mts_dataset])
    return mts_dataset_as_array


def get_transformed_uts_with_features_and_decomps(
    uts_decomp,
    trend_det_factor,
    trend_slope_factor,
    trend_lin_factor,
    seasonal_det_factor,
    m=0,
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
