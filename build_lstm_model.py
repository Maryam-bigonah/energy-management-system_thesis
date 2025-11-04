"""
Step 6: Define LSTM Model for Time Series Forecasting

Simple Keras Sequential model:
- Input: (24, n_features)
- 1 LSTM layer with 64 units
- 1 Dense(1) output
- Adam optimizer (1e-3), MSE loss, MAE metric
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam


def build_lstm_model(window_size=24, n_features=6, lstm_units=64, learning_rate=1e-3):
    """
    Step 6: Build simple LSTM model for time series forecasting
    
    Model architecture:
    - Input shape: (window_size, n_features)
    - 1 LSTM layer with lstm_units
    - 1 Dense(1) output layer
    
    Compilation:
    - Optimizer: Adam with specified learning rate
    - Loss: MSE (Mean Squared Error)
    - Metric: MAE (Mean Absolute Error)
    
    Parameters:
    -----------
    window_size : int
        Number of time steps in input window (default: 24 hours)
    n_features : int
        Number of input features (default: 6)
        Features: total_load, pv_1kw, hour, dayofweek, is_weekend, season
    lstm_units : int
        Number of units in LSTM layer (default: 64)
    learning_rate : float
        Learning rate for Adam optimizer (default: 1e-3)
    
    Returns:
    --------
    model : keras.Model
        Compiled Keras Sequential model
    """
    print("=" * 70)
    print("Step 6: Building LSTM Model")
    print("=" * 70)
    
    print(f"\nModel Architecture:")
    print(f"  Input shape: ({window_size}, {n_features})")
    print(f"  LSTM units: {lstm_units}")
    print(f"  Output: Dense(1)")
    
    # Build model
    model = Sequential([
        # LSTM layer
        LSTM(
            units=lstm_units,
            input_shape=(window_size, n_features),
            name='lstm_layer'
        ),
        
        # Output layer
        Dense(
            units=1,
            name='output_layer'
        )
    ])
    
    print(f"\n✓ Model created")
    
    # Compile model
    print(f"\nCompiling model...")
    print(f"  Optimizer: Adam(learning_rate={learning_rate})")
    print(f"  Loss: MSE (Mean Squared Error)")
    print(f"  Metric: MAE (Mean Absolute Error)")
    
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=['mae']
    )
    
    print(f"  ✓ Model compiled")
    
    # Print model summary
    print(f"\nModel Summary:")
    print("=" * 70)
    model.summary()
    print("=" * 70)
    
    # Count parameters
    total_params = model.count_params()
    trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])
    
    print(f"\nModel Parameters:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    
    print("\n" + "=" * 70)
    print("LSTM model ready for training!")
    print("=" * 70)
    
    return model


# Example usage
if __name__ == "__main__":
    """
    Example: Build and inspect LSTM model
    """
    # Build model with default parameters
    model = build_lstm_model(
        window_size=24,
        n_features=6,  # total_load, pv_1kw, hour, dayofweek, is_weekend, season
        lstm_units=64,
        learning_rate=1e-3
    )
    
    print("\nExample usage:")
    print("  from build_lstm_model import build_lstm_model")
    print("  ")
    print("  # Build model")
    print("  model = build_lstm_model(window_size=24, n_features=6, lstm_units=64)")
    print("  ")
    print("  # Train model")
    print("  history = model.fit(")
    print("      X_train_scaled, y_train_scaled,")
    print("      validation_data=(X_val_scaled, y_val_scaled),")
    print("      epochs=50,")
    print("      batch_size=32,")
    print("      verbose=1")
    print("  )")

