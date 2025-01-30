from typing import List
import pandas as pd
from tqdm import tqdm


def generate_windows_dataset(
    data: pd.DataFrame,
    window_size: int,
    step_size: int,
    include_columns: List[str] = None,
) -> List[pd.DataFrame]:
    dataset = []

    if include_columns is not None:
        data = data[include_columns]

    for i in tqdm(range(0, len(data) - window_size + 1, step_size), total=len(data)):
        dataset.append(data.iloc[i : i + window_size])

    return dataset
