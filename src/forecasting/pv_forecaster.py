"""
Recursive, data-driven PV production forecasting utilities.

Implements the requirements outlined by the user:
    * 1-hour resolution, 24-hour horizon.
    * Data-driven (GradientBoostingRegressor by default); no pvlib.
    * Lagged PV and weather features, future (forecast) weather, time features.
    * One-step-ahead recursive forecasting loop feeding predictions back in.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import RegressorMixin


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
    dow = df.index.dayofweek
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7)
    df["is_weekend"] = (dow >= 5).astype(int)
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


@dataclass
class RecursivePVForecaster:
    """
    Data-driven PV production forecaster compliant with user requirements.
    """

    lag_hours: int = 24
    model: Optional[RegressorMixin] = None
    weather_columns: Sequence[str] = ("temp_amb", "irr_direct", "irr_diffuse")
    pv_column: str = "pv_power"
    static_features: Optional[Mapping[str, float]] = None
    feature_order_: List[str] = field(default_factory=list, init=False)
    fitted_: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.model is None:
            self.model = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=5,
                random_state=42,
            )

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
    ) -> Dict[str, Dict[str, float]]:
        """
        Trains the model using chronological splits.
        Returns evaluation metrics for validation and test sets.
        """
        supervised = self._prepare_supervised_frame(history_df)
        X = supervised.drop(columns=["target", self.pv_column])
        y = supervised["target"]

        datasets = time_series_train_val_test_split(X, y, val_size=val_size, test_size=test_size)
        X_train, X_val, X_test, y_train, y_val, y_test = datasets

        self.model.fit(X_train, y_train)
        self.feature_order_ = list(X_train.columns)
        self.fitted_ = True

        val_pred = pd.Series(self.model.predict(X_val), index=y_val.index)
        test_pred = pd.Series(self.model.predict(X_test), index=y_test.index)

        metrics = {
            "validation": evaluate_forecast(y_val, val_pred),
            "test": evaluate_forecast(y_test, test_pred),
        }
        return metrics

    def _build_single_feature_row(
        self,
        buffer_df: pd.DataFrame,
        future_weather_row: pd.Series,
    ) -> pd.DataFrame:
        """
        Builds the feature row for the next prediction using the rolling buffer and the
        future weather (forecast) for the prediction timestamp.
        """
        required_cols = [self.pv_column, *self.weather_columns]
        _validate_inputs(buffer_df, required_cols)

        assembled = buffer_df.copy()
        assembled.loc[future_weather_row.name, self.weather_columns] = future_weather_row[self.weather_columns]

        features = _add_time_features(assembled.tail(self.lag_hours + 1))
        features = _add_lagged_features(features, [self.pv_column, *self.weather_columns], self.lag_hours)
        features = _add_future_weather_features(features, self.weather_columns, horizon=1)

        if self.static_features:
            for key, value in self.static_features.items():
                features[key] = value

        row = features.iloc[[-1]].drop(columns=[self.pv_column], errors="ignore")
        if self.static_features:
            for key, value in self.static_features.items():
                row[key] = value
        # Replace lead columns with provided forecast (already aligned through future row assignment)
        for col in self.weather_columns:
            lead_col = f"{col}_lead_1"
            if lead_col in row.columns:
                row[lead_col] = future_weather_row[col]

        row = row[self.feature_order_] if self.feature_order_ else row
        return row

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
            feature_row = self._build_single_feature_row(rolling_buffer, weather_row)
            y_hat = float(self.model.predict(feature_row)[0])
            preds.append((ts, y_hat))
            new_row = pd.Series({self.pv_column: y_hat, **weather_row[self.weather_columns]}, name=ts)
            rolling_buffer = pd.concat([rolling_buffer, new_row.to_frame().T])

        forecast_series = pd.Series(
            [value for _, value in preds],
            index=pd.Index([ts for ts, _ in preds], name="timestamp"),
            name="pv_forecast_kw",
        )
        return forecast_series


__all__ = [
    "RecursivePVForecaster",
    "evaluate_forecast",
    "time_series_train_val_test_split",
    "plot_forecast",
]

