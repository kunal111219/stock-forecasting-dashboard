import warnings
warnings.filterwarnings('ignore')
from prophet import Prophet
import logging


def train_prophet(prophet_df):
    """
    Train Facebook Prophet model.
    Handles weekly and yearly seasonality automatically.
    changepoint_prior_scale controls trend flexibility.
    """
    model = Prophet(
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=True,
        changepoint_prior_scale=0.05,
        seasonality_prior_scale=10,
        interval_width=0.95
    )
    
    logging.getLogger('prophet').setLevel(logging.ERROR)
    logging.getLogger('cmdstanpy').setLevel(logging.ERROR)
    
    model.fit(prophet_df)
    return model


def forecast_prophet(model, forecast_days):
    """
    Generate Prophet forecast for given number of days.
    Uses business days (freq='B') to skip weekends.
    Returns tail of forecast dataframe (future only).
    """
    future = model.make_future_dataframe(
        periods=forecast_days,
        freq='B'
    )
    forecast = model.predict(future)
    return forecast.tail(forecast_days)


def run_prophet_pipeline(prophet_df, forecast_days):
    """
    Full Prophet pipeline:
    train → forecast
    Returns: forecast dataframe with yhat, yhat_lower, yhat_upper
    """
    model = train_prophet(prophet_df)
    forecast = forecast_prophet(model, forecast_days)
    return forecast