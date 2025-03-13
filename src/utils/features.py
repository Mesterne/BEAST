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
    mts: np.ndarray,  # Shape (num_mts, num_uts_in_mts, num_timesteps)
    num_features_per_uts: int,
    series_periodicity: int,
    decomps_only: bool = False,
) -> Tuple[List[DecomposeResult], np.ndarray]:
    decomps = []
    num_uts_in_mts = mts.shape[1]
    features = np.empty((len(mts), num_uts_in_mts * num_features_per_uts))
    for i, ts in tqdm(enumerate(mts), total=len(mts)):
        mts_decomp = []
        for j in range(num_uts_in_mts):
            uts = ts[j]
            decomp = STL(uts, period=series_periodicity).fit()
            mts_decomp.append(decomp)
            if decomps_only:
                continue
            features[i, j * num_features_per_uts] = trend_strength(
                decomp.trend, decomp.resid
            )
            features[i, j * num_features_per_uts + 1] = trend_slope(decomp.trend)
            features[i, j * num_features_per_uts + 2] = (
                seasonal_strength(decomp.seasonal, decomp.resid)
                if series_periodicity > 1
                else 0
            )
        decomps.append(mts_decomp)
    return decomps, features
