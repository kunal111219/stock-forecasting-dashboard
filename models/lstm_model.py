# import numpy as np
# import warnings
# warnings.filterwarnings('ignore')


# def build_lstm_architecture(seq_length):
#     """
#     Build LSTM neural network architecture.
#     3 LSTM layers with Dropout to prevent overfitting.
#     """
#     from tensorflow.keras.models import Sequential
#     from tensorflow.keras.layers import LSTM, Dense, Dropout

#     model = Sequential([
#         LSTM(50, return_sequences=True,
#              input_shape=(seq_length, 1)),
#         Dropout(0.2),
#         LSTM(50, return_sequences=True),
#         Dropout(0.2),
#         LSTM(50, return_sequences=False),
#         Dropout(0.2),
#         Dense(25),
#         Dense(1)
#     ])
#     model.compile(optimizer='adam', loss='mean_squared_error')
#     return model


# def train_lstm(X_train, y_train, seq_length, epochs=50):
#     """
#     Train LSTM model with early stopping.
#     EarlyStopping prevents overfitting by stopping
#     when validation loss stops improving.
#     """
#     from tensorflow.keras.callbacks import EarlyStopping

#     model = build_lstm_architecture(seq_length)
#     early_stop = EarlyStopping(
#         monitor='val_loss',
#         patience=5,
#         restore_best_weights=True
#     )
#     model.fit(
#         X_train, y_train,
#         epochs=epochs,
#         batch_size=32,
#         validation_split=0.1,
#         callbacks=[early_stop],
#         verbose=0
#     )
#     return model


# def generate_future_predictions(model, last_sequence,
#                                 scaler, seq_length, steps):
#     """
#     Generate future predictions using rolling forecast.
#     Each predicted value is fed back as input for next step.
#     This is called 'recursive multi-step forecasting'.
#     """
#     predictions = []
#     current_seq = last_sequence.copy()

#     for _ in range(steps):
#         inp = current_seq.reshape(1, seq_length, 1)
#         pred = model.predict(inp, verbose=0)[0, 0]
#         predictions.append(pred)
#         current_seq = np.append(current_seq[1:], [[pred]], axis=0)

#     predictions = np.array(predictions).reshape(-1, 1)
#     return scaler.inverse_transform(predictions).flatten()


# def run_lstm_pipeline(data, forecast_days, seq_length=60):
#     """
#     Full LSTM pipeline:
#     scale → create sequences → train → forecast
#     Returns: future price predictions array
#     """
#     from utils.preprocessor import scale_for_lstm, create_lstm_sequences

#     # Scale data
#     scaled, scaler = scale_for_lstm(data)

#     # Create sequences
#     X, y = create_lstm_sequences(scaled, seq_length)
#     X = X.reshape(X.shape[0], X.shape[1], 1)

#     # Train/test split (no shuffling for time series!)
#     split = int(len(X) * 0.8)
#     X_train, y_train = X[:split], y[:split]

#     # Train model
#     model = train_lstm(X_train, y_train, seq_length)

#     # Generate future forecast
#     last_seq = scaled[-seq_length:]
#     predictions = generate_future_predictions(
#         model, last_seq, scaler, seq_length, forecast_days
#     )
#     return predictions



import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ✅ Fixed imports — works without Pylance warnings
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping


def build_lstm_architecture(seq_length):
    """
    Build LSTM neural network architecture.
    3 LSTM layers with Dropout to prevent overfitting.
    """
    model = Sequential([
        LSTM(50, return_sequences=True,
             input_shape=(seq_length, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def train_lstm(X_train, y_train, seq_length, epochs=50):
    """
    Train LSTM model with early stopping.
    EarlyStopping prevents overfitting by stopping
    when validation loss stops improving.
    """
    model = build_lstm_architecture(seq_length)
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True
    )
    model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_split=0.1,
        callbacks=[early_stop],
        verbose=0
    )
    return model


def generate_future_predictions(model, last_sequence,
                                scaler, seq_length, steps):
    """
    Generate future predictions using rolling forecast.
    Each predicted value is fed back as input for next step.
    This is called 'recursive multi-step forecasting'.
    """
    predictions = []
    current_seq = last_sequence.copy()

    for _ in range(steps):
        inp = current_seq.reshape(1, seq_length, 1)
        pred = model.predict(inp, verbose=0)[0, 0]
        predictions.append(pred)
        current_seq = np.append(current_seq[1:], [[pred]], axis=0)

    predictions = np.array(predictions).reshape(-1, 1)
    return scaler.inverse_transform(predictions).flatten()


def run_lstm_pipeline(data, forecast_days, seq_length=60):
    """
    Full LSTM pipeline:
    scale → create sequences → train → forecast
    Returns: future price predictions array
    """
    # Fix: convert Series to numpy array if needed
    if hasattr(data, 'values'):
        data = data.values
        
    from utils.preprocessor import scale_for_lstm, create_lstm_sequences

    # Scale data
    scaled, scaler = scale_for_lstm(data)

    # Create sequences
    X, y = create_lstm_sequences(scaled, seq_length)
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Train/test split (no shuffling for time series!)
    split = int(len(X) * 0.8)
    X_train, y_train = X[:split], y[:split]

    # Train model
    model = train_lstm(X_train, y_train, seq_length)

    # Generate future forecast
    last_seq = scaled[-seq_length:]
    predictions = generate_future_predictions(
        model, last_seq, scaler, seq_length, forecast_days
    )
    return predictions