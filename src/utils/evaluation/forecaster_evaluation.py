import numpy as np
from sklearn.metrics import mean_squared_error


def rmse_for_each_forecast(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=1))


def rmse_for_all_predictions(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae_for_each_forecast(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred), axis=1)


def mae_for_all_predictions(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    return np.mean(np.abs(y_true - y_pred))
