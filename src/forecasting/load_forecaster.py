"""
Load forecasting module for building-level electricity demand.

Implements:
- Weekly seasonal naïve baseline (from REC paper)
- Random Forest model with lagged features and calendar features
- Direct multi-step forecasting (24 separate models for h=1...24)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, List, Tuple, Optional
import pickle
from pathlib import Path


def weekly_seasonal_naive_forecast(
    load_history: pd.Series,
    forecast_start: pd.Timestamp,
    horizon_hours: int = 24,
    season_hours: int = 168,  # 7 days * 24 hours
) -> pd.Series:
    """
    Weekly seasonal naïve forecast for load.
    
    Forecast rule: L_hat[t+h] = L[t+h-168]
    
    Parameters
    ----------
    load_history : pd.Series
        Historical load with DatetimeIndex
    forecast_start : pd.Timestamp
        First timestamp to forecast
    horizon_hours : int
        Forecast horizon (default 24)
    season_hours : int
        Seasonal period in hours (default 168 = 1 week)
    
    Returns
    -------
    pd.Series
        Forecast with DatetimeIndex
    """
    if not isinstance(load_history.index, pd.DatetimeIndex):
        raise ValueError("load_history must have DatetimeIndex")
    
    load_history = load_history.sort_index()
    future_index = pd.date_range(start=forecast_start, periods=horizon_hours, freq="h")
    
    values = []
    for ts in future_index:
        ref_time = ts - pd.Timedelta(hours=season_hours)
        if ref_time not in load_history.index:
            # If reference time doesn't exist, use the closest available
            closest_idx = load_history.index.get_indexer([ref_time], method='nearest')[0]
            ref_time = load_history.index[closest_idx]
        values.append(load_history.loc[ref_time])
    
    forecast = pd.Series(values, index=future_index, name="load_forecast")
    return forecast


def prepare_load_features(
    df: pd.DataFrame,
    target_col: str = "total_load",
    lag_hours: List[int] = [1, 24, 168],  # 1h ago, same time yesterday, same time last week
) -> pd.DataFrame:
    """
    Prepare feature matrix for load forecasting.
    
    Features:
    - Lagged load: L[t-1], L[t-24], L[t-168]
    - Calendar: hour, dow, is_weekend, month, season
    
    Parameters
    ----------
    df : pd.DataFrame
        Master dataset with DatetimeIndex
    target_col : str
        Column name for load (default "total_load")
    lag_hours : List[int]
        List of lag hours to include
    
    Returns
    -------
    pd.DataFrame
        Feature matrix with all engineered features
    """
    features = df.copy()
    
    # Lagged features
    for lag in lag_hours:
        features[f"{target_col}_lag_{lag}"] = features[target_col].shift(lag)
    
    # Calendar features (should already exist, but ensure they're present)
    if "hour" not in features.columns:
        features["hour"] = features.index.hour
    if "dow" not in features.columns:
        features["dow"] = features.index.dayofweek
    if "is_weekend" not in features.columns:
        features["is_weekend"] = (features.index.dayofweek >= 5).astype(int)
    if "month" not in features.columns:
        features["month"] = features.index.month
    if "season" not in features.columns:
        month = features.index.month
        season = np.full(len(features), 3, dtype=int)  # default autumn
        season[(month == 12) | (month <= 2)] = 0  # winter
        season[(month >= 3) & (month <= 5)] = 1  # spring
        season[(month >= 6) & (month <= 8)] = 2  # summer
        features["season"] = season
    
    # Select only feature columns
    feature_cols = [f"{target_col}_lag_{lag}" for lag in lag_hours] + \
                   ["hour", "dow", "is_weekend", "month", "season"]
    
    return features[feature_cols]


class LoadForecaster:
    """
    Load forecaster using Random Forest with direct multi-step forecasting.
    
    Trains 24 separate models, one for each forecast horizon (h=1...24).
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        random_state: int = 42,
        lag_hours: List[int] = [1, 24, 168],
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
        lag_hours : List[int]
            Lag hours for feature engineering
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.lag_hours = lag_hours
        
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
            Feature matrix (from prepare_load_features)
        y_train : pd.Series
            Target load series
        
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
            
            # Remove rows with NaN (from lag features)
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
            Forecasts with columns L_hat_h1, ..., L_hat_h24
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
                forecasts[f"L_hat_h{h}"] = np.nan
                continue
            
            # For direct forecasting, we use the same feature vector
            # (the model was trained to predict h steps ahead from current features)
            X_input = X_last[self.feature_cols]
            
            # Check for NaN
            if X_input.isna().any().any():
                forecasts[f"L_hat_h{h}"] = np.nan
            else:
                pred = self.models[h].predict(X_input)[0]
                forecasts[f"L_hat_h{h}"] = pred
        
        # Create DataFrame
        forecast_df = pd.DataFrame(forecasts, index=future_index)
        return forecast_df
    
    def save(self, filepath: Path):
        """Save the fitted model to disk."""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'models': self.models,
                'feature_cols': self.feature_cols,
                'lag_hours': self.lag_hours,
                'fitted_': self.fitted_,
            }, f)
    
    def load(self, filepath: Path):
        """Load a fitted model from disk."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.models = data['models']
            self.feature_cols = data['feature_cols']
            self.lag_hours = data['lag_hours']
            self.fitted_ = data['fitted_']


def evaluate_load_forecasts(
    y_true: pd.Series,
    y_pred: pd.DataFrame,
    capacity_kw: Optional[float] = None,
) -> pd.DataFrame:
    """
    Evaluate forecasts for all horizons.
    
    Parameters
    ----------
    y_true : pd.Series
        True load values
    y_pred : pd.DataFrame
        Forecasts with columns L_hat_h1, ..., L_hat_h24
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
        col = f"L_hat_h{h}"
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
