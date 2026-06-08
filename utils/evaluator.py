import numpy as np


def rmse(actual, predicted):
    """Root Mean Square Error — penalizes large errors more"""
    return np.sqrt(
        np.mean((np.array(actual) - np.array(predicted)) ** 2)
    )


def mae(actual, predicted):
    """Mean Absolute Error — average absolute difference"""
    return np.mean(np.abs(np.array(actual) - np.array(predicted)))


def mape(actual, predicted):
    """Mean Absolute Percentage Error — percentage based error"""
    actual = np.array(actual)
    predicted = np.array(predicted)
    return np.mean(np.abs((actual - predicted) / actual)) * 100


def evaluate_model(actual, predicted, model_name):
    """
    Return all evaluation metrics for a given model.
    Used to compare ARIMA vs Prophet vs LSTM performance.
    """
    return {
        "Model": model_name,
        "RMSE": round(rmse(actual, predicted), 2),
        "MAE": round(mae(actual, predicted), 2),
        "MAPE (%)": round(mape(actual, predicted), 2)
    }