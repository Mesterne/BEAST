from abc import ABC, abstractmethod
from typing import List

import pandas as pd
from statsmodels.tsa.seasonal import DecomposeResult as DecompResults


class TimeSeriesTransformationModel(ABC):
    def __init__(
        self,
        model_type: str,
        model_params: dict,
        mts_dataset: pd.DataFrame,
        mts_features: pd.DataFrame,
        mts_decomp: List[DecompResults],
    ):
        self.model_type = model_type
        self.model_params = model_params
        self.mts_dataset = mts_dataset
        self.mts_features = mts_features
        self.mts_decomp = mts_decomp

    # NOTE: May want to send more params than necessary and only use relevant params for a given model
    @abstractmethod
    def fit(self):
        pass

    # NOTE: May want to send more params than necessary and only use relevant params for a given model
    @abstractmethod
    def transform(self):
        pass
