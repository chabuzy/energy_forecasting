# --- Numerical computing
import numpy as np

# --- TensorFlow for deep learning
import tensorflow as tf

# --- Sequential API for building neural networks layer‑by‑layer
from tensorflow.keras.models import Sequential

# --- Layers used in LSTM and TCN models
from tensorflow.keras.layers import LSTM, Dense, Conv1D, InputLayer
from tensorflow.keras.layers import BatchNormalization, Dropout
from tensorflow.keras.layers import MaxPooling1D, Flatten

# --- Metric for evaluating model performance
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. LSTM MODEL
# ============================================================
def create_lstm_model(input_shape):
    """
    Build a stacked LSTM model for time‑series forecasting.
    input_shape = (timesteps, features)
    """

    model = Sequential([

        # --- First LSTM layer (returns full sequence for stacking)
        LSTM(64, return_sequences=True, input_shape=input_shape),

        # --- Dropout helps prevent overfitting
        Dropout(0.2),

        # --- Second LSTM layer (returns final output only)
        LSTM(32),

        # --- More dropout
        Dropout(0.2),

        # --- Dense layer for non‑linear transformation
        Dense(16, activation="relu"),

        # --- Final output layer (predicts 1 value: next demand)
        Dense(1)
    ])

    # --- Compile model with Adam optimizer and MSE loss
    model.compile(optimizer="adam", loss="mse")

    return model


# ============================================================
# 2. TCN MODEL (Temporal Convolutional Network)
# ============================================================
def create_tcn_model(input_shape):
    """
    Build a simple TCN model using causal convolutions.
    TCNs are good for sequence modelling without recurrence.
    """

    model = Sequential([

        # --- Input layer defining the shape of the data
        InputLayer(input_shape=input_shape),

        # --- First causal convolution
        Conv1D(
            64,
            kernel_size=3,
            activation="relu",
            padding="causal"   # ensures no future leakage
        ),

        # --- Normalize activations for stability
        BatchNormalization(),

        # --- Downsample the sequence
        MaxPooling1D(pool_size=2),

        # --- Second convolution layer
        Conv1D(
            32,
            kernel_size=3,
            activation="relu",
            padding="causal"
        ),

        # --- Normalize again
        BatchNormalization(),

        # --- Flatten sequence into vector
        Flatten(),

        # --- Dense layer for learning patterns
        Dense(32, activation="relu"),

        # --- Final output layer
        Dense(1)
    ])

    # --- Compile model
    model.compile(optimizer="adam", loss="mse")

    return model


# ============================================================
# 3. SEQUENCE PREPARATION FOR LSTM/TCN
# ============================================================
def prepare_sequences(series, window=48):
    """
    Convert a 1D time series into supervised learning format:
    X = sliding windows of past values
    y = next value after each window

    Example:
    If window=48, each X[i] contains 48 past timesteps,
    and y[i] is the 49th timestep.
    """

    X, y = [], []

    # --- Loop through the series and extract windows
    for i in range(len(series) - window):
        X.append(series[i:i + window])      # past 48 values
        y.append(series[i + window])        # next value

    # --- Convert lists to numpy arrays
    X = np.array(X)
    y = np.array(y)

    # --- Reshape X to (samples, timesteps, features)
    #     features = 1 because it's univariate demand data
    return X.reshape((X.shape[0], X.shape[1], 1)), y
