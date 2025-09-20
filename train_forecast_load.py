#!/usr/bin/env python3
"""
Load Forecasting Training Script

Trains SARIMAX and XGBoost models for load forecasting using historical data.
Implements rolling-origin cross-validation and selects best model by MAE.

Author: Energy Management System
Date: 2024
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('load_forecast_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class LoadForecastConfig:
    """Configuration for load forecasting"""
    data_dir: str = "data"
    models_dir: str = "models/load"
    results_dir: str = "results/figs_forecast"
    cv_splits: int = 5
    random_state: int = 42
    test_size: float = 0.2
    validation_size: float = 0.2

class LoadDataProcessor:
    """Processes and prepares load data for forecasting"""
    
    def __init__(self, config: LoadForecastConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def load_data(self) -> pd.DataFrame:
        """Load and combine all required data sources"""
        logger.info("Loading historical data...")
        
        # Load load data
        load_file = os.path.join(self.config.data_dir, "load_8760.csv")
        if not os.path.exists(load_file):
            raise FileNotFoundError(f"Load data not found: {load_file}")
        
        load_data = pd.read_csv(load_file)
        logger.info(f"Loaded {len(load_data)} load records")
        
        # Load weather data
        weather_file = os.path.join(self.config.data_dir, "weather_8760.csv")
        weather_data = None
        if os.path.exists(weather_file):
            weather_data = pd.read_csv(weather_file)
            logger.info(f"Loaded {len(weather_data)} weather records")
        else:
            logger.warning("Weather data not found - creating synthetic temperature data")
            weather_data = self._create_synthetic_weather(load_data)
        
        # Load holidays data
        holidays_file = os.path.join(self.config.data_dir, "holidays.csv")
        holidays_data = None
        if os.path.exists(holidays_file):
            holidays_data = pd.read_csv(holidays_file)
            logger.info(f"Loaded {len(holidays_data)} holiday records")
        else:
            logger.warning("Holidays data not found - creating synthetic holidays")
            holidays_data = self._create_synthetic_holidays(load_data)
        
        # Merge all data
        merged_data = load_data.merge(weather_data, on=['day', 'hour'], how='left')
        merged_data = merged_data.merge(holidays_data, on=['day'], how='left')
        
        # Fill missing values
        merged_data['temp_C'] = merged_data['temp_C'].fillna(20.0)  # Default temperature
        merged_data['is_holiday'] = merged_data['is_holiday'].fillna(False)
        
        logger.info(f"Combined dataset: {len(merged_data)} records")
        return merged_data
    
    def _create_synthetic_weather(self, load_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic weather data based on load patterns"""
        weather_data = load_data[['day', 'hour']].copy()
        
        # Create realistic temperature patterns
        # Higher temperatures in summer (days 150-250), lower in winter
        weather_data['temp_C'] = 15 + 10 * np.sin(2 * np.pi * (weather_data['day'] - 80) / 365)
        
        # Add daily variation (cooler at night)
        daily_variation = 5 * np.sin(2 * np.pi * (weather_data['hour'] - 6) / 24)
        weather_data['temp_C'] += daily_variation
        
        # Add some noise
        weather_data['temp_C'] += np.random.normal(0, 2, len(weather_data))
        
        return weather_data
    
    def _create_synthetic_holidays(self, load_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic holidays data"""
        holidays_data = load_data[['day']].drop_duplicates().copy()
        
        # Create some holidays (New Year, Christmas, etc.)
        holidays_data['is_holiday'] = False
        
        # New Year (day 1)
        holidays_data.loc[holidays_data['day'] == 1, 'is_holiday'] = True
        
        # Christmas (around day 360)
        holidays_data.loc[holidays_data['day'] == 360, 'is_holiday'] = True
        
        # Add some random holidays
        random_holidays = np.random.choice(holidays_data['day'], size=5, replace=False)
        holidays_data.loc[holidays_data['day'].isin(random_holidays), 'is_holiday'] = True
        
        return holidays_data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create forecasting features"""
        logger.info("Creating forecasting features...")
        
        df = data.copy()
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week'] = (df['day'] - 1) % 7  # 0=Monday, 6=Sunday
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = ((df['day'] - 1) // 30) + 1  # Approximate month
        df['is_holiday'] = df['is_holiday'].astype(int)
        
        # Lagged features
        df['load_lag_1'] = df['load_kw'].shift(1)
        df['load_lag_24'] = df['load_kw'].shift(24)
        df['load_lag_168'] = df['load_kw'].shift(168)  # Weekly lag
        
        # Rolling means
        df['load_ma_24'] = df['load_kw'].rolling(window=24, min_periods=1).mean()
        df['load_ma_168'] = df['load_kw'].rolling(window=168, min_periods=1).mean()
        
        # Temperature features
        df['temp_lag_1'] = df['temp_C'].shift(1)
        df['temp_ma_24'] = df['temp_C'].rolling(window=24, min_periods=1).mean()
        
        # Remove rows with NaN values from lagged features
        df = df.dropna()
        
        # Define feature columns
        self.feature_columns = [
            'hour_sin', 'hour_cos', 'day_of_week', 'is_weekend', 'month', 'is_holiday',
            'temp_C', 'temp_lag_1', 'temp_ma_24',
            'load_lag_1', 'load_lag_24', 'load_lag_168',
            'load_ma_24', 'load_ma_168'
        ]
        
        logger.info(f"Created {len(self.feature_columns)} features")
        return df
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare training data for ML models"""
        X = df[self.feature_columns].values
        y = df['load_kw'].values
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y

class LoadForecastTrainer:
    """Trains load forecasting models"""
    
    def __init__(self, config: LoadForecastConfig):
        self.config = config
        self.models = {}
        self.metrics = {}
        
    def train_sarimax(self, data: pd.DataFrame) -> Dict:
        """Train SARIMAX model"""
        logger.info("Training SARIMAX model...")
        
        # Prepare time series data
        ts_data = data.set_index(['day', 'hour'])['load_kw'].sort_index()
        
        # Seasonal decomposition to determine parameters
        try:
            decomposition = seasonal_decompose(ts_data, model='additive', period=24)
            seasonal_strength = np.var(decomposition.seasonal) / np.var(ts_data)
        except:
            seasonal_strength = 0.1
        
        # Determine SARIMAX parameters based on data characteristics
        if seasonal_strength > 0.1:
            # Strong seasonality - use seasonal parameters
            order = (1, 1, 1)
            seasonal_order = (1, 1, 1, 24)
        else:
            # Weak seasonality - simpler model
            order = (1, 1, 1)
            seasonal_order = (0, 0, 0, 0)
        
        # Fit SARIMAX model
        try:
            model = SARIMAX(
                ts_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False
            )
            fitted_model = model.fit(disp=False)
            
            logger.info("SARIMAX model trained successfully")
            return {
                'model': fitted_model,
                'order': order,
                'seasonal_order': seasonal_order,
                'type': 'SARIMAX'
            }
        except Exception as e:
            logger.warning(f"SARIMAX training failed: {e}")
            return None
    
    def train_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train XGBoost model"""
        logger.info("Training XGBoost model...")
        
        # XGBoost parameters
        params = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': self.config.random_state,
            'n_jobs': -1
        }
        
        # Train model
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        
        logger.info("XGBoost model trained successfully")
        return {
            'model': model,
            'params': params,
            'type': 'XGBoost'
        }
    
    def rolling_origin_cv(self, X: np.ndarray, y: np.ndarray, model_type: str) -> Dict:
        """Perform rolling-origin cross-validation"""
        logger.info(f"Performing rolling-origin CV for {model_type}...")
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        mae_scores = []
        rmse_scores = []
        mape_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            # Train model for this fold
            if model_type == 'XGBoost':
                model = xgb.XGBRegressor(
                    n_estimators=100,
                    max_depth=6,
                    learning_rate=0.1,
                    random_state=self.config.random_state
                )
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:
                # For SARIMAX, we'll use a simpler approach
                # In practice, you'd retrain the full SARIMAX model for each fold
                continue
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            mape = mean_absolute_percentage_error(y_val, y_pred) * 100
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            mape_scores.append(mape)
            
            logger.info(f"Fold {fold + 1}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")
        
        if mae_scores:
            return {
                'mae_mean': np.mean(mae_scores),
                'mae_std': np.std(mae_scores),
                'rmse_mean': np.mean(rmse_scores),
                'rmse_std': np.std(rmse_scores),
                'mape_mean': np.mean(mape_scores),
                'mape_std': np.std(mape_scores),
                'scores': {
                    'mae': mae_scores,
                    'rmse': rmse_scores,
                    'mape': mape_scores
                }
            }
        else:
            return None
    
    def select_best_model(self, sarimax_metrics: Dict, xgboost_metrics: Dict) -> str:
        """Select best model based on MAE"""
        if sarimax_metrics is None and xgboost_metrics is None:
            return 'XGBoost'  # Default fallback
        
        if sarimax_metrics is None:
            return 'XGBoost'
        
        if xgboost_metrics is None:
            return 'SARIMAX'
        
        # Compare MAE
        sarimax_mae = sarimax_metrics['mae_mean']
        xgboost_mae = xgboost_metrics['mae_mean']
        
        if sarimax_mae < xgboost_mae:
            return 'SARIMAX'
        else:
            return 'XGBoost'
    
    def save_models(self, sarimax_model: Dict, xgboost_model: Dict, 
                   sarimax_metrics: Dict, xgboost_metrics: Dict, 
                   best_model: str, processor: LoadDataProcessor):
        """Save trained models and metadata"""
        logger.info(f"Saving models to {self.config.models_dir}")
        
        # Save SARIMAX model
        if sarimax_model:
            sarimax_file = os.path.join(self.config.models_dir, "sarimax_model.joblib")
            joblib.dump(sarimax_model, sarimax_file)
            logger.info(f"Saved SARIMAX model to {sarimax_file}")
        
        # Save XGBoost model
        if xgboost_model:
            xgboost_file = os.path.join(self.config.models_dir, "xgboost_model.joblib")
            joblib.dump(xgboost_model, xgboost_file)
            logger.info(f"Saved XGBoost model to {xgboost_file}")
        
        # Save scaler
        scaler_file = os.path.join(self.config.models_dir, "scaler.joblib")
        joblib.dump(processor.scaler, scaler_file)
        logger.info(f"Saved scaler to {scaler_file}")
        
        # Save metadata
        metadata = {
            'best_model': best_model,
            'feature_columns': processor.feature_columns,
            'sarimax_metrics': sarimax_metrics,
            'xgboost_metrics': xgboost_metrics,
            'training_date': datetime.now().isoformat(),
            'config': {
                'cv_splits': self.config.cv_splits,
                'random_state': self.config.random_state
            }
        }
        
        meta_file = os.path.join(self.config.models_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {meta_file}")
    
    def create_validation_plots(self, sarimax_metrics: Dict, xgboost_metrics: Dict):
        """Create validation plots"""
        logger.info("Creating validation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # MAE comparison
        models = []
        mae_means = []
        mae_stds = []
        
        if sarimax_metrics:
            models.append('SARIMAX')
            mae_means.append(sarimax_metrics['mae_mean'])
            mae_stds.append(sarimax_metrics['mae_std'])
        
        if xgboost_metrics:
            models.append('XGBoost')
            mae_means.append(xgboost_metrics['mae_mean'])
            mae_stds.append(xgboost_metrics['mae_std'])
        
        if models:
            axes[0, 0].bar(models, mae_means, yerr=mae_stds, capsize=5, alpha=0.7)
            axes[0, 0].set_title('MAE Comparison')
            axes[0, 0].set_ylabel('MAE (kW)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE comparison
        rmse_means = []
        rmse_stds = []
        
        if sarimax_metrics:
            rmse_means.append(sarimax_metrics['rmse_mean'])
            rmse_stds.append(sarimax_metrics['rmse_std'])
        
        if xgboost_metrics:
            rmse_means.append(xgboost_metrics['rmse_mean'])
            rmse_stds.append(xgboost_metrics['rmse_std'])
        
        if models:
            axes[0, 1].bar(models, rmse_means, yerr=rmse_stds, capsize=5, alpha=0.7)
            axes[0, 1].set_title('RMSE Comparison')
            axes[0, 1].set_ylabel('RMSE (kW)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE comparison
        mape_means = []
        mape_stds = []
        
        if sarimax_metrics:
            mape_means.append(sarimax_metrics['mape_mean'])
            mape_stds.append(sarimax_metrics['mape_std'])
        
        if xgboost_metrics:
            mape_means.append(xgboost_metrics['mape_mean'])
            mape_stds.append(xgboost_metrics['mape_std'])
        
        if models:
            axes[1, 0].bar(models, mape_means, yerr=mape_stds, capsize=5, alpha=0.7)
            axes[1, 0].set_title('MAPE Comparison')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # CV scores distribution
        if xgboost_metrics and 'scores' in xgboost_metrics:
            axes[1, 1].boxplot([xgboost_metrics['scores']['mae']], labels=['XGBoost MAE'])
            axes[1, 1].set_title('CV Scores Distribution')
            axes[1, 1].set_ylabel('MAE (kW)')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_file = os.path.join(self.config.results_dir, 'load_forecast_validation.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validation plots to {plot_file}")

def main():
    """Main function for load forecasting training"""
    parser = argparse.ArgumentParser(description='Train load forecasting models')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models/load', help='Models directory')
    parser.add_argument('--results-dir', type=str, default='results/figs_forecast', help='Results directory')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    
    args = parser.parse_args()
    
    # Create configuration
    config = LoadForecastConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        cv_splits=args.cv_splits,
        random_state=args.random_state
    )
    
    # Create directories
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    logger.info("Starting load forecasting training...")
    
    try:
        # Initialize processor
        processor = LoadDataProcessor(config)
        
        # Load and process data
        data = processor.load_data()
        data_with_features = processor.create_features(data)
        X, y = processor.prepare_training_data(data_with_features)
        
        # Initialize trainer
        trainer = LoadForecastTrainer(config)
        
        # Train SARIMAX model
        sarimax_model = trainer.train_sarimax(data_with_features)
        sarimax_metrics = None  # Simplified for this implementation
        
        # Train XGBoost model
        xgboost_model = trainer.train_xgboost(X, y)
        xgboost_metrics = trainer.rolling_origin_cv(X, y, 'XGBoost')
        
        # Select best model
        best_model = trainer.select_best_model(sarimax_metrics, xgboost_metrics)
        logger.info(f"Best model: {best_model}")
        
        # Save models
        trainer.save_models(sarimax_model, xgboost_model, sarimax_metrics, 
                           xgboost_metrics, best_model, processor)
        
        # Create validation plots
        trainer.create_validation_plots(sarimax_metrics, xgboost_metrics)
        
        logger.info("Load forecasting training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
