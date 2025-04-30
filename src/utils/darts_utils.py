from typing import List

import numpy as np
import pandas as pd
from darts import TimeSeries


def array_to_timeseries(timeseries_array: np.ndarray) -> List[TimeSeries]:
    time_index = pd.date_range(
        start="1960-01-01 00:00",  # NOTE: The date does not matter for our experiments
        periods=timeseries_array.shape[2],  # Samples
        freq="h",  # The grid loss data set has hourly samples
    )

    series = []

    for i in range(0, timeseries_array.shape[0]):
        series.append(
            TimeSeries.from_times_and_values(
                times=time_index,
                values=timeseries_array[
                    i
                ].T,  # We transpose to get shape (samples, uts)
                columns=[f"var_{j}" for j in range(timeseries_array.shape[1])],
            ).astype(
                np.float32
            )  # For running models on hardware
        )
    return series
