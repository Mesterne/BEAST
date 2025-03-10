from sklearn.metrics import mean_squared_error


def mse_for_forecast(y_test, inferred_values):
    return mean_squared_error(y_test, inferred_values)
