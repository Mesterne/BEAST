import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.seasonal import STL, DecomposeResult
from tqdm import tqdm
from typing import List, Tuple


def trend_strength(trend_comp: pd.Series, resid_comp: pd.Series) -> float:
    return max(
        0,
        1
        - np.var(resid_comp)
        / max(np.finfo(np.float32).eps, np.var(trend_comp + resid_comp)),
    )


def trend_slope(trend_comp: pd.Series) -> float:
    slope = (
        LinearRegression(fit_intercept=True)
        .fit(np.arange(len(trend_comp)).reshape(-1, 1), trend_comp)
        .coef_[0]
    )
    return slope / np.clip(np.abs(np.mean(trend_comp)), a_min=1e-6, a_max=None)


def trend_linearity(trend_comp: pd.Series) -> float:
    model = LinearRegression(fit_intercept=True).fit(
        np.arange(len(trend_comp)).reshape(-1, 1), trend_comp
    )
    predictions = model.predict(np.arange(len(trend_comp)).reshape(-1, 1))
    residuals = trend_comp - predictions
    return 1 - np.var(residuals) / np.var(trend_comp)


def seasonal_strength(seasonal_comp: pd.Series, resid_comp: pd.Series) -> float:
    return max(
        0,
        1
        - np.var(resid_comp)
        / max(np.finfo(np.float32).eps, np.var(seasonal_comp + resid_comp)),
    )


def decomp_and_features(
    data: List[pd.DataFrame], series_periodicity: int, dataset_size: int = None
) -> Tuple[List[DecomposeResult], np.ndarray]:
    if dataset_size is not None:
        data = data[:dataset_size]

    # NOTE: Check out series_periodicity in STL
    decomps = []
    features = np.empty((len(data), len(data[0].columns), 4))
    for i, df in tqdm(enumerate(data), total=len(data)):
        mts_decomp = []
        for j, col in enumerate(df.columns):
            ts = df[col]
            decomp = STL(ts, period=series_periodicity).fit()
            mts_decomp.append(decomp)
            features[i, j, 0] = trend_strength(decomp.trend, decomp.resid)
            features[i, j, 1] = trend_slope(decomp.trend)
            features[i, j, 2] = trend_linearity(decomp.trend)
            features[i, j, 3] = (
                seasonal_strength(decomp.seasonal, decomp.resid)
                if series_periodicity > 1
                else 0
            )
        decomps.append(mts_decomp)
    return decomps, features
