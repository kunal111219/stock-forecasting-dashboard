import numpy as np
import warnings
warnings.filterwarnings('ignore')
from statsmodels.tsa.arima.model import ARIMA


def find_best_order(data, max_p=3, max_d=2, max_q=3):
    """
    Auto-select best ARIMA (p,d,q) parameters
    using AIC (Akaike Information Criterion).
    Lower AIC = better model fit.
    """
    best_aic = np.inf
    best_order = (1, 1, 1)

    for p in range(max_p + 1):
        for d in range(max_d + 1):
            for q in range(max_q + 1):
                try:
                    model = ARIMA(data, order=(p, d, q))
                    result = model.fit()
                    if result.aic < best_aic:
                        best_aic = result.aic
                        best_order = (p, d, q)
                except Exception:
                    continue

    return best_order


def train_arima(data, order):
    """Train ARIMA model with given order."""
    model = ARIMA(data, order=order)
    return model.fit()


def forecast_arima(fitted_model, forecast_days):
    """
    Generate forecast and confidence intervals.
    Returns: forecast values, confidence interval dataframe
    """
    forecast_result = fitted_model.get_forecast(steps=forecast_days)
    forecast_values = forecast_result.predicted_mean.values
    conf_int = forecast_result.conf_int()
    # Fix: convert to numpy array if it's a DataFrame
    if hasattr(conf_int, 'values'):
        conf_int = conf_int.values  # DataFrame → numpy
    # else it's already a numpy array — leave it
    return forecast_values, conf_int


def run_arima_pipeline(data, forecast_days):
    """
    Full ARIMA pipeline:
    auto-select order → train → forecast
    Returns: forecast, confidence intervals, best order
    """
    order = find_best_order(data)
    fitted = train_arima(data, order)
    forecast, conf_int = forecast_arima(fitted, forecast_days)
    return forecast, conf_int, order