from typing import List, Tuple

import numpy as np
from darts import TimeSeries


# Code partially from ChatGPT
def array_to_timeseries(
    timeseries_array: np.ndarray,
) -> Tuple[List[TimeSeries], List[TimeSeries]]:
    target_series = []
    covariate_series = []

    for i in range(timeseries_array.shape[0]):
        target = TimeSeries.from_values(
            values=timeseries_array[i, 1, :].reshape(-1, 1), columns=["grid_loss"]
        ).astype(np.float32)

        covariates = TimeSeries.from_values(
            values=np.stack(
                [timeseries_array[i, 0, :], timeseries_array[i, 2, :]], axis=1
            ),
            columns=["grid_load", "grid_temp"],
        ).astype(np.float32)

        target_series.append(target)
        covariate_series.append(covariates)

    return target_series, covariate_series
