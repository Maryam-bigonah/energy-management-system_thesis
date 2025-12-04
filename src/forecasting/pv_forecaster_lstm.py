"""
Recursive, data-driven PV production forecasting utilities (LSTM version).

Implements the requirements outlined by the user:
    * 1-hour resolution, 24-hour horizon.
    * Data-driven (LSTM neural network); no pvlib.
    * Lagged PV and weather features, future (forecast) weather, time features.
    * One-step-ahead recursive forecasting loop feeding predictions back in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
except ImportError:
    raise ImportError(
        "TensorFlow/Keras is required for LSTM forecasting. "
        "Install with: pip install tensorflow>=2.10"
    )


RequiredColumns = Sequence[str]


def _validate_inputs(df: pd.DataFrame, required_cols: RequiredColumns) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("Input DataFrame must use a pandas.DatetimeIndex.")


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    hours = df.index.hour
    df["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    df["hour_cos"] = np.cos(2 * np.pi * hours / 24)

    # Day-of-week (0=Monday ... 6=Sunday) and cyclic encoding
    dow = df.index.dayofweek
    df["day_of_week"] = dow
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)

    # Weekend flag
    df["is_weekend"] = (dow >= 5).astype(int)

    # Month (1–12)
    month = df.index.month
    df["month"] = month

    # Season: 0=winter, 1=spring, 2=summer, 3=autumn
    season = np.full(len(df), 3, dtype=int)  # default autumn
    winter_mask = (month == 12) | (month <= 2)
    spring_mask = (month >= 3) & (month <= 5)
    summer_mask = (month >= 6) & (month <= 8)
    season[winter_mask] = 0
    season[spring_mask] = 1
    season[summer_mask] = 2
    df["season"] = season

    return df


def _add_lagged_features(df: pd.DataFrame, columns: RequiredColumns, lag_hours: int) -> pd.DataFrame:
    df = df.copy()
    for lag in range(1, lag_hours + 1):
        for col in columns:
            df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def _add_future_weather_features(df: pd.DataFrame, weather_cols: RequiredColumns, horizon: int = 1) -> pd.DataFrame:
    """
    Adds columns that represent the weather forecast for the prediction horizon.
    For a one-step-ahead model, we only need lead=1.
    """
    df = df.copy()
    for col in weather_cols:
        df[f"{col}_lead_{horizon}"] = df[col].shift(-horizon)
    return df


def time_series_train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    val_size: float = 0.1,
    test_size: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    Chronological split (no shuffling) for train/validation/test.
    Sizes are expressed as fractions of the full dataset length.
    """
    if (val_size + test_size) >= 1:
        raise ValueError("val_size + test_size must be < 1.")

    n = len(X)
    n_test = int(n * test_size)
    n_val = int(n * val_size)

    train_end = n - n_test - n_val
    val_end = n - n_test

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    return X_train, X_val, X_test, y_train, y_val, y_test


def evaluate_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    capacity_kw: Optional[float] = None,
) -> Dict[str, float]:
    """
    Computes MAE, RMSE, nRMSE (normalized by capacity or mean), and R².
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    if capacity_kw is not None and capacity_kw > 0:
        nrmse = rmse / capacity_kw
    else:
        nrmse = rmse / (y_true.max() - y_true.min() + 1e-6)
    metrics = {
        "mae": mae,
        "rmse": rmse,
        "nrmse": nrmse,
        "r2": r2_score(y_true, y_pred),
    }
    return metrics


def plot_forecast(
    y_true: pd.Series,
    y_pred: pd.Series,
    title: str = "PV Forecast vs Actual",
) -> plt.Figure:
    """
    Simple helper for visual inspection (optional requirement).
    """
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(y_true.index, y_true.values, label="Actual", marker="o")
    ax.plot(y_pred.index, y_pred.values, label="Forecast", marker="x")
    ax.set_ylabel("PV Power (kW)")
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.autofmt_xdate()
    return fig


def _create_sequences(
    X: np.ndarray,
    y: np.ndarray,
    sequence_length: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences for LSTM training.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix of shape (n_samples, n_features)
    y : np.ndarray
        Target vector of shape (n_samples,)
    sequence_length : int
        Length of each input sequence
        
    Returns
    -------
    X_seq : np.ndarray
        Sequences of shape (n_samples - sequence_length, sequence_length, n_features)
    y_seq : np.ndarray
        Targets of shape (n_samples - sequence_length,)
    """
    n_samples, n_features = X.shape
    X_seq = []
    y_seq = []
    
    for i in range(sequence_length, n_samples):
        X_seq.append(X[i - sequence_length : i])
        y_seq.append(y[i])
    
    return np.array(X_seq), np.array(y_seq)


@dataclass
class RecursivePVForecaster:
    """
    Data-driven PV production forecaster using LSTM neural networks.
    """

    lag_hours: int = 24
    model: Optional[keras.Model] = None
    weather_columns: Sequence[str] = ("temp_amb", "irr_direct", "irr_diffuse")
    pv_column: str = "pv_power"
    static_features: Optional[Mapping[str, float]] = None
    feature_order_: List[str] = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)
    scaler_X_: Optional[StandardScaler] = field(default=None, init=False)
    scaler_y_: Optional[StandardScaler] = field(default=None, init=False)
    sequence_length_: int = field(default=24, init=False)

    def __post_init__(self) -> None:
        self.sequence_length_ = self.lag_hours
        if self.model is None:
            # Default LSTM architecture will be built in fit() after we know feature dimensions
            pass

    def _build_lstm_model(self, n_features: int) -> keras.Model:
        """
        Build a default LSTM model architecture.
        """
        model = keras.Sequential(
            [
                layers.LSTM(64, return_sequences=True, input_shape=(self.sequence_length_, n_features)),
                layers.Dropout(0.2),
                layers.LSTM(32, return_sequences=False),
                layers.Dropout(0.2),
                layers.Dense(16, activation="relu"),
                layers.Dense(1),
            ]
        )
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        return model

    def _prepare_supervised_frame(self, df: pd.DataFrame) -> pd.DataFrame:
        _validate_inputs(df, [self.pv_column, *self.weather_columns])
        df = df.sort_index()

        features = _add_time_features(df)
        features = _add_lagged_features(features, [self.pv_column, *self.weather_columns], self.lag_hours)
        features = _add_future_weather_features(features, self.weather_columns, horizon=1)

        if self.static_features:
            for key, value in self.static_features.items():
                features[key] = value

        # Target is t+1 PV (shift backward so each row corresponds to prediction made at row timestamp)
        features["target"] = features[self.pv_column].shift(-1)
        # Drop rows that still contain NaNs from lag/lead construction to avoid
        # passing incomplete samples to models that cannot handle NaNs.
        supervised = features.dropna().copy()
        return supervised

    def fit(
        self,
        history_df: pd.DataFrame,
        val_size: float = 0.1,
        test_size: float = 0.1,
        epochs: int = 50,
        batch_size: int = 32,
        verbose: int = 1,
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains the LSTM model using chronological splits.
        Returns evaluation metrics for validation and test sets.
        """
        supervised = self._prepare_supervised_frame(history_df)
        X = supervised.drop(columns=["target", self.pv_column])
        y = supervised["target"]

        # Store feature order
        self.feature_order_ = list(X.columns)

        # Split chronologically
        datasets = time_series_train_val_test_split(X, y, val_size=val_size, test_size=test_size)
        X_train, X_val, X_test, y_train, y_val, y_test = datasets

        # Scale features and targets
        self.scaler_X_ = StandardScaler()
        self.scaler_y_ = StandardScaler()

        X_train_scaled = self.scaler_X_.fit_transform(X_train)
        X_val_scaled = self.scaler_X_.transform(X_val)
        X_test_scaled = self.scaler_X_.transform(X_test)

        y_train_scaled = self.scaler_y_.fit_transform(y_train.values.reshape(-1, 1)).ravel()
        y_val_scaled = self.scaler_y_.transform(y_val.values.reshape(-1, 1)).ravel()
        y_test_scaled = self.scaler_y_.transform(y_test.values.reshape(-1, 1)).ravel()

        # Create sequences for LSTM
        X_train_seq, y_train_seq = _create_sequences(X_train_scaled, y_train_scaled, self.sequence_length_)
        X_val_seq, y_val_seq = _create_sequences(X_val_scaled, y_val_scaled, self.sequence_length_)
        X_test_seq, y_test_seq = _create_sequences(X_test_scaled, y_test_scaled, self.sequence_length_)

        # Build model if not provided
        if self.model is None:
            n_features = X_train_seq.shape[2]
            self.model = self._build_lstm_model(n_features)

        # Train the model
        self.model.fit(
            X_train_seq,
            y_train_seq,
            validation_data=(X_val_seq, y_val_seq),
            epochs=epochs,
            batch_size=batch_size,
            verbose=verbose,
        )

        self.fitted_ = True

        # Predict on validation and test sets
        val_pred_scaled = self.model.predict(X_val_seq, verbose=0).ravel()
        test_pred_scaled = self.model.predict(X_test_seq, verbose=0).ravel()

        # Inverse transform predictions
        val_pred = pd.Series(
            self.scaler_y_.inverse_transform(val_pred_scaled.reshape(-1, 1)).ravel(),
            index=y_val.index[self.sequence_length_ :],
        )
        test_pred = pd.Series(
            self.scaler_y_.inverse_transform(test_pred_scaled.reshape(-1, 1)).ravel(),
            index=y_test.index[self.sequence_length_ :],
        )

        # Align true values with predictions (drop first sequence_length samples)
        y_val_aligned = y_val.iloc[self.sequence_length_ :]
        y_test_aligned = y_test.iloc[self.sequence_length_ :]

        metrics = {
            "validation": evaluate_forecast(y_val_aligned, val_pred),
            "test": evaluate_forecast(y_test_aligned, test_pred),
        }
        return metrics

    def _build_single_sequence(
        self,
        buffer_df: pd.DataFrame,
        future_weather_row: pd.Series,
    ) -> np.ndarray:
        """
        Builds a sequence for LSTM prediction using the rolling buffer and the
        future weather (forecast) for the prediction timestamp.
        """
        required_cols = [self.pv_column, *self.weather_columns]
        _validate_inputs(buffer_df, required_cols)

        assembled = buffer_df.copy()
        for col in self.weather_columns:
            assembled.loc[future_weather_row.name, col] = future_weather_row[col]

        # Get the last sequence_length rows
        recent = assembled.tail(self.sequence_length_)

        # Add temporal features
        features = _add_time_features(recent)
        features = _add_lagged_features(features, [self.pv_column, *self.weather_columns], self.lag_hours)
        features = _add_future_weather_features(features, self.weather_columns, horizon=1)

        if self.static_features:
            for key, value in self.static_features.items():
                features[key] = value

        # Drop pv_power column and select features in correct order
        feature_cols = [col for col in self.feature_order_ if col in features.columns]
        features_subset = features[feature_cols]

        # Scale the sequence
        features_scaled = self.scaler_X_.transform(features_subset)

        # Reshape to (1, sequence_length, n_features) for LSTM
        sequence = features_scaled.reshape(1, self.sequence_length_, len(feature_cols))
        return sequence

    def forecast(
        self,
        history_df: pd.DataFrame,
        weather_forecast_df: pd.DataFrame,
    ) -> pd.Series:
        """
        Generates a 24-hour ahead forecast using recursive one-step predictions.

        Parameters
        ----------
        history_df : pd.DataFrame
            Historical measurements covering at least `lag_hours` back.
        weather_forecast_df : pd.DataFrame
            NWP forecast for the next 24 hours with columns matching `weather_columns`.
        """
        if not self.fitted_:
            raise RuntimeError("Call fit() before forecast().")
        if len(history_df) < self.lag_hours:
            raise ValueError(f"Need at least {self.lag_hours} hours of history.")
        if len(weather_forecast_df) < 24:
            raise ValueError("weather_forecast_df must cover 24 hours.")

        history_df = history_df.sort_index()
        weather_forecast_df = weather_forecast_df.sort_index().iloc[:24]
        _validate_inputs(history_df, [self.pv_column, *self.weather_columns])
        _validate_inputs(weather_forecast_df, self.weather_columns)

        rolling_buffer = history_df.copy()
        preds = []
        for ts, weather_row in weather_forecast_df.iterrows():
            sequence = self._build_single_sequence(rolling_buffer, weather_row)
            y_hat_scaled = self.model.predict(sequence, verbose=0)[0, 0]
            y_hat = float(self.scaler_y_.inverse_transform([[y_hat_scaled]])[0, 0])
            preds.append((ts, y_hat))
            new_row_dict = {self.pv_column: y_hat}
            for col in self.weather_columns:
                new_row_dict[col] = weather_row[col]
            new_row = pd.Series(new_row_dict, name=ts)
            rolling_buffer = pd.concat([rolling_buffer, new_row.to_frame().T])

        forecast_series = pd.Series(
            [value for _, value in preds],
            index=pd.Index([ts for ts, _ in preds], name="timestamp"),
            name="pv_forecast_kw",
        )
        return forecast_series


def forecast_pv_timeseries(
    history_df: pd.DataFrame,
    weather_forecast_df: pd.DataFrame,
    static_features: Optional[Mapping[str, float]] = None,
    model: Optional[keras.Model] = None,
    lag_hours: int = 24,
    val_size: float = 0.1,
    test_size: float = 0.1,
    epochs: int = 50,
    batch_size: int = 32,
    verbose: int = 1,
) -> Tuple[pd.Series, Dict[str, Dict[str, float]], RecursivePVForecaster]:
    """
    High-level helper that:
        1) Instantiates and fits a RecursivePVForecaster (LSTM version).
        2) Produces a 24-hour PV forecast using the trained model.

    Returns
    -------
    pv_forecast : pd.Series
        24-hour ahead PV power forecast (1-hour resolution).
    metrics : dict
        Validation and test metrics returned by RecursivePVForecaster.fit().
        Contains MAE, RMSE, nRMSE, and R² for both validation and test sets.
    forecaster : RecursivePVForecaster
        The fitted forecaster instance (for reuse or inspection).
    """
    forecaster = RecursivePVForecaster(
        lag_hours=lag_hours,
        model=model,
        static_features=static_features,
    )
    metrics = forecaster.fit(
        history_df,
        val_size=val_size,
        test_size=test_size,
        epochs=epochs,
        batch_size=batch_size,
        verbose=verbose,
    )
    pv_forecast = forecaster.forecast(history_df, weather_forecast_df)
    return pv_forecast, metrics, forecaster


__all__ = [
    "RecursivePVForecaster",
    "evaluate_forecast",
    "time_series_train_val_test_split",
    "plot_forecast",
    "forecast_pv_timeseries",
]

