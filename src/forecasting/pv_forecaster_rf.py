"""
PV forecasting module using Random Forest with direct multi-step forecasting.

Implements:
- Persistence baseline
- Random Forest model with PVGIS + weather features
- Direct multi-step forecasting (24 separate models for h=1...24)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


def persistence_forecast(
    pv_history: pd.Series,
    forecast_start: pd.Timestamp,
    horizon_hours: int = 24,
) -> pd.Series:
    """
    Persistence forecast for PV: PV_hat[t+h] = PV[t]
    
    Parameters
    ----------
    pv_history : pd.Series
        Historical PV with DatetimeIndex
    forecast_start : pd.Timestamp
        First timestamp to forecast
    horizon_hours : int
        Forecast horizon (default 24)
    
    Returns
    -------
    pd.Series
        Forecast with DatetimeIndex
    """
    if not isinstance(pv_history.index, pd.DatetimeIndex):
        raise ValueError("pv_history must have DatetimeIndex")
    
    pv_history = pv_history.sort_index()
    future_index = pd.date_range(start=forecast_start, periods=horizon_hours, freq="h")
    
    # Use last available value
    last_value = pv_history.iloc[-1]
    
    # Create forecast (all values equal to last observed)
    forecast = pd.Series([last_value] * horizon_hours, index=future_index, name="pv_forecast")
    return forecast


def prepare_pv_features(
    df: pd.DataFrame,
    target_col: str = "PV_true",
    include_lag: bool = True,
) -> pd.DataFrame:
    """
    Prepare feature matrix for PV forecasting.
    
    Features from PVGIS:
    - Gb, Gd, Gr (irradiance components)
    - H_sun (solar elevation)
    - T2m, WS10m (meteorological)
    
    Features from OpenWeather:
    - clouds, temp, humidity, wind_speed
    
    Calendar features:
    - hour, month, season
    
    Optional:
    - PV[t-1] (lagged PV)
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with DatetimeIndex
    target_col : str
        Column name for PV (default "PV_true")
    include_lag : bool
        Whether to include lagged PV feature
    
    Returns
    -------
    pd.DataFrame
        Feature matrix with all engineered features
    """
    features = df.copy()
    
    # PVGIS features
    pvgis_features = ["Gb", "Gd", "Gr", "H_sun", "T2m", "WS10m"]
    
    # OpenWeather features (use T2m from PVGIS, so exclude temp to avoid redundancy)
    weather_features = ["clouds", "humidity", "wind_speed"]
    
    # Calendar features
    if "hour" not in features.columns:
        features["hour"] = features.index.hour
    if "month" not in features.columns:
        features["month"] = features.index.month
    if "season" not in features.columns:
        month = features.index.month
        season = np.full(len(features), 3, dtype=int)  # default autumn
        season[(month == 12) | (month <= 2)] = 0  # winter
        season[(month >= 3) & (month <= 5)] = 1  # spring
        season[(month >= 6) & (month <= 8)] = 2  # summer
        features["season"] = season
    
    # Optional lagged PV
    if include_lag:
        features[f"{target_col}_lag_1"] = features[target_col].shift(1)
    
    # Select feature columns
    feature_cols = pvgis_features + weather_features + ["hour", "month", "season"]
    if include_lag:
        feature_cols.append(f"{target_col}_lag_1")
    
    # Check which features exist
    available_cols = [col for col in feature_cols if col in features.columns]
    
    return features[available_cols]


class PVForecaster:
    """
    PV forecaster using Random Forest with direct multi-step forecasting.
    
    Trains 24 separate models, one for each forecast horizon (h=1...24).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        include_lag: bool = True,
    ):
        """
        Parameters
        ----------
        n_estimators : int
            Number of trees in Random Forest
        max_depth : Optional[int]
            Maximum depth of trees
        min_samples_split : int
            Minimum samples to split a node
        random_state : int
            Random seed
        include_lag : bool
            Whether to include lagged PV feature
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.include_lag = include_lag
        
        self.models: Dict[int, RandomForestRegressor] = {}
        self.feature_cols: List[str] = []
        self.fitted_ = False
    
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Dict[int, Dict[str, float]]:
        """
        Train 24 separate models for horizons h=1...24.
        
        Parameters
        ----------
        X_train : pd.DataFrame
            Feature matrix (from prepare_pv_features)
        y_train : pd.Series
            Target PV series
        
        Returns
        -------
        Dict[int, Dict[str, float]]
            Training metrics for each horizon
        """
        metrics = {}
        
        # Store feature column names
        self.feature_cols = list(X_train.columns)
        
        # Train one model per horizon
        for h in range(1, 25):
            # Create target: y[t+h] for each training sample at time t
            y_h = y_train.shift(-h).dropna()
            X_h = X_train.iloc[:len(y_h)]
            
            # Remove rows with NaN (from lag features or missing weather)
            valid_mask = ~(X_h.isna().any(axis=1) | y_h.isna())
            X_h_clean = X_h[valid_mask]
            y_h_clean = y_h[valid_mask]
            
            if len(X_h_clean) == 0:
                print(f"Warning: No valid samples for horizon h={h}")
                continue
            
            # Train model
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                random_state=self.random_state,
                n_jobs=-1,
            )
            model.fit(X_h_clean, y_h_clean)
            self.models[h] = model
            
            # Compute training metrics
            y_pred = model.predict(X_h_clean)
            metrics[h] = {
                "mae": mean_absolute_error(y_h_clean, y_pred),
                "rmse": np.sqrt(mean_squared_error(y_h_clean, y_pred)),
                "r2": r2_score(y_h_clean, y_pred),
            }
        
        self.fitted_ = True
        return metrics
    
    def predict(
        self,
        X: pd.DataFrame,
        forecast_start: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Generate 24-hour ahead forecasts.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix at forecast time
        forecast_start : pd.Timestamp
            First timestamp to forecast
        
        Returns
        -------
        pd.DataFrame
            Forecasts with columns PV_hat_h1, ..., PV_hat_h24
        """
        if not self.fitted_:
            raise ValueError("Model must be fitted before prediction")
        
        # Get the last row (current time)
        X_last = X.iloc[[-1]].copy()
        
        # Ensure all required features are present
        missing_cols = set(self.feature_cols) - set(X_last.columns)
        if missing_cols:
            raise ValueError(f"Missing features: {missing_cols}")
        
        # Generate forecasts for each horizon
        forecasts = {}
        future_index = pd.date_range(start=forecast_start, periods=24, freq="H")
        
        for h in range(1, 25):
            if h not in self.models:
                forecasts[f"PV_hat_h{h}"] = np.nan
                continue
            
            # For direct forecasting, we use the same feature vector
            # (the model was trained to predict h steps ahead from current features)
            X_input = X_last[self.feature_cols]
            
            # Check for NaN
            if X_input.isna().any().any():
                forecasts[f"PV_hat_h{h}"] = np.nan
            else:
                pred = self.models[h].predict(X_input)[0]
                forecasts[f"PV_hat_h{h}"] = pred
        
        # Create DataFrame
        forecast_df = pd.DataFrame(forecasts, index=future_index)
        return forecast_df
    
    def save(self, filepath: Path):
        """Save the fitted model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_cols': self.feature_cols,
                'include_lag': self.include_lag,
                'fitted_': self.fitted_,
            }, f)
    
    def load(self, filepath: Path):
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.feature_cols = data['feature_cols']
            self.include_lag = data['include_lag']
            self.fitted_ = data['fitted_']


def evaluate_pv_forecasts(
    y_true: pd.Series,
    y_pred: pd.DataFrame,
    capacity_kw: Optional[float] = None,
) -> pd.DataFrame:
    """
    Evaluate forecasts for all horizons.
    
    Parameters
    ----------
    y_true : pd.Series
        True PV values
    y_pred : pd.DataFrame
        Forecasts with columns PV_hat_h1, ..., PV_hat_h24
    capacity_kw : Optional[float]
        Capacity for normalization (optional)
    
    Returns
    -------
    pd.DataFrame
        Metrics (MAE, RMSE, nRMSE) for each horizon
    """
    results = []
    
    # Align indices
    common_idx = y_true.index.intersection(y_pred.index)
    y_true_aligned = y_true.loc[common_idx]
    
    for h in range(1, 25):
        col = f"PV_hat_h{h}"
        if col not in y_pred.columns:
            continue
        
        y_pred_h = y_pred.loc[common_idx, col].dropna()
        y_true_h = y_true_aligned.loc[y_pred_h.index]
        
        if len(y_pred_h) == 0:
            continue
        
        mae = mean_absolute_error(y_true_h, y_pred_h)
        rmse = np.sqrt(mean_squared_error(y_true_h, y_pred_h))
        
        if capacity_kw is not None and capacity_kw > 0:
            nrmse = rmse / capacity_kw
        else:
            nrmse = rmse / (y_true_h.max() - y_true_h.min() + 1e-6)
        
        r2 = r2_score(y_true_h, y_pred_h)
        
        results.append({
            'horizon': h,
            'mae': mae,
            'rmse': rmse,
            'nrmse': nrmse,
            'r2': r2,
        })
    
    return pd.DataFrame(results)
