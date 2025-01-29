from typing import List
import pandas as pd


def generate_windows_dataset(
    data: pd.DataFrame, window_size: int, include_columns: List[str] = None
) -> List[pd.DataFrame]:
    dataset = []

    if include_columns is not None:
        data = data[include_columns]

    for i in range(len(data) - window_size + 1):
        dataset.append(data.iloc[i : i + window_size])

    return dataset
