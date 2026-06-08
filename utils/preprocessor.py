import numpy as np
import pandas as pd


def calculate_technical_indicators(df):
    """
    Calculate all technical indicators dynamically
    from live price data — no hardcoded values.
    """
    df = df.copy()
    df['Daily_Return'] = df['Close'].pct_change()
    df['Volatility_30d'] = (
        df['Daily_Return'].rolling(30).std() * np.sqrt(252)
    )
    df['MA_20'] = df['Close'].rolling(20).mean()
    df['MA_50'] = df['Close'].rolling(50).mean()
    df['Upper_Band'] = df['MA_20'] + (df['Close'].rolling(20).std() * 2)
    df['Lower_Band'] = df['MA_20'] - (df['Close'].rolling(20).std() * 2)
    return df


def prepare_prophet_df(df):
    """
    Format dataframe for Prophet model.
    Prophet requires columns named 'ds' and 'y'.
    """
    prophet_df = df.reset_index()[['Date', 'Close']].copy()
    prophet_df.columns = ['ds', 'y']
    prophet_df['ds'] = pd.to_datetime(
        prophet_df['ds']
    ).dt.tz_localize(None)
    return prophet_df


def scale_for_lstm(data):
    """Scale data to [0,1] range for LSTM training."""
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data.reshape(-1, 1))
    return scaled, scaler


def create_lstm_sequences(scaled_data, seq_length=60):
    """
    Create sliding window sequences for LSTM.
    seq_length = how many past days to use for prediction.
    """
    X, y = [], []
    for i in range(seq_length, len(scaled_data)):
        X.append(scaled_data[i - seq_length:i, 0])
        y.append(scaled_data[i, 0])
    return np.array(X), np.array(y)