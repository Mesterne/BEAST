import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def manipulate_trend_component(
    trend_comp: pd.Series, f: float, g: float, h: float, m: float
) -> pd.Series:
    model = LinearRegression(fit_intercept=True).fit(
        np.arange(len(trend_comp)).reshape(-1, 1), trend_comp
    )
    predictions = model.predict(np.arange(len(trend_comp)).reshape(-1, 1))
    residuals = trend_comp - predictions

    new_trend = model.intercept_ + f * (
        g * model.coef_ * np.arange(len(trend_comp)) + (1 / h * residuals)
    )
    additional_trend = m * model.intercept_ * np.arange(len(trend_comp))
    return new_trend + additional_trend


def manipulate_seasonal_component(seasonal_comp: pd.Series, k: float) -> pd.Series:
    return seasonal_comp * k


if __name__ == "__main__":
    import os
    from generate_dataset import generate_windows_dataset
    from features import decomp_and_features
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    # Read data
    data_dir_path = os.path.join("data", "gridloss", "train.csv")
    df = pd.read_csv(data_dir_path, index_col=0)
    df.index = pd.to_datetime(df.index)
    # Backfill missing data
    df = df.bfill()

    grid1_columns = ["grid1-load", "grid1-loss", "grid1-temp"]
    window_size = 168  # 1 week
    step_size = 24
    num_ts = len(grid1_columns)

    print("Generating windows dataset")
    data = generate_windows_dataset(df, window_size, step_size, grid1_columns)

    decomps, features = decomp_and_features(data, series_periodicity=24)

    # Multiplicative constants
    f = 1
    g = 1
    h = 1
    m = 1
    k = 0.5

    new_trends = [
        [manipulate_trend_component(decomp.trend, f, g, h, m) for decomp in mts_decomp]
        for mts_decomp in decomps
    ]
    new_seasonals = [
        [manipulate_seasonal_component(decomp.seasonal, k) for decomp in mts_decomp]
        for mts_decomp in decomps
    ]

    i = 20

    fig = make_subplots(rows=3, cols=2)
    fig.add_trace(
        go.Line(x=decomps[i][0].trend.index, y=decomps[i][0].trend, name="grid1-load"),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Line(x=decomps[i][1].trend.index, y=decomps[i][1].trend, name="grid1-loss"),
        row=2,
        col=1,
    )
    fig.add_trace(
        go.Line(x=decomps[i][2].trend.index, y=decomps[i][2].trend, name="grid1-temp"),
        row=3,
        col=1,
    )
    fig.add_trace(
        go.Line(x=new_trends[i][0].index, y=new_trends[i][0], name="t-grid1-load"),
        row=1,
        col=2,
    )
    fig.add_trace(
        go.Line(x=new_trends[i][1].index, y=new_trends[i][1], name="t-grid1-loss"),
        row=2,
        col=2,
    )
    fig.add_trace(
        go.Line(x=new_trends[i][2].index, y=new_trends[i][2], name="t-grid1-temp"),
        row=3,
        col=2,
    )
    fig.update_layout(
        height=600, width=800, title_text="Trends before and after transformation"
    )
    fig.show()
