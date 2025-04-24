import numpy as np
from sklearn.metrics import mean_squared_error

def mse_for_each_forecast(y_true, y_pred):
     return np.mean((y_true - y_pred) ** 2, axis=1)

def mse_for_all_predictions(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) 
