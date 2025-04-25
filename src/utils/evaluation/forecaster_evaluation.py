import numpy as np
from sklearn.metrics import mean_squared_error

def mse_for_each_forecast(y_true, y_pred):
     return np.mean((y_true - y_pred) ** 2, axis=1)

def mse_for_all_predictions(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) 

def mase_for_each_forecast(y_true, y_pred, insample):
    naive_forecast = insample[:, -1][:, np.newaxis]
    scale = np.mean(np.abs(y_true - naive_forecast), axis=1)
    error = np.mean(np.abs(y_true - y_pred), axis=1)
    return error / scale

def mase_for_all_predictions(y_true, y_pred, insample):
    naive_forecast = insample[:, -1][:, np.newaxis]
    scale = np.mean(np.abs(y_true - naive_forecast))
    error = np.mean(np.abs(y_true - y_pred))
    return error / scale
