#!/usr/bin/env python3
"""
PV Forecasting Training Script

Trains a two-stage PV forecasting model:
1. Physical PR model: pv_hat = kWp * PR * (GHI / GHI_ref) * (1 + α_T * (temp_C − 25))
2. XGBoost model on residuals: pv_kw - pv_hat

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
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pv_forecast_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class PVForecastConfig:
    """Configuration for PV forecasting"""
    data_dir: str = "data"
    models_dir: str = "models/pv"
    results_dir: str = "results/figs_forecast"
    cv_splits: int = 5
    random_state: int = 42
    ghi_ref: float = 1000.0  # Reference GHI (W/m²)
    temp_ref: float = 25.0   # Reference temperature (°C)

class PVDataProcessor:
    """Processes and prepares PV data for forecasting"""
    
    def __init__(self, config: PVForecastConfig):
        self.config = config
        self.scaler = StandardScaler()
        self.feature_columns = []
        self.pr_model = None
        self.kwp_estimate = None
        
    def load_data(self) -> pd.DataFrame:
        """Load and combine all required data sources"""
        logger.info("Loading historical PV data...")
        
        # Load PV data
        pv_file = os.path.join(self.config.data_dir, "pv_8760.csv")
        if not os.path.exists(pv_file):
            raise FileNotFoundError(f"PV data not found: {pv_file}")
        
        pv_data = pd.read_csv(pv_file)
        logger.info(f"Loaded {len(pv_data)} PV records")
        
        # Load weather data
        weather_file = os.path.join(self.config.data_dir, "weather_8760.csv")
        weather_data = None
        if os.path.exists(weather_file):
            weather_data = pd.read_csv(weather_file)
            logger.info(f"Loaded {len(weather_data)} weather records")
        else:
            logger.warning("Weather data not found - creating synthetic data")
            weather_data = self._create_synthetic_weather(pv_data)
        
        # Merge data
        merged_data = pv_data.merge(weather_data, on=['day', 'hour'], how='left')
        
        # Fill missing values
        merged_data['temp_C'] = merged_data['temp_C'].fillna(20.0)
        if 'ghi_Wm2' not in merged_data.columns:
            merged_data['ghi_Wm2'] = self._estimate_ghi(merged_data)
        
        logger.info(f"Combined dataset: {len(merged_data)} records")
        return merged_data
    
    def _create_synthetic_weather(self, pv_data: pd.DataFrame) -> pd.DataFrame:
        """Create synthetic weather data"""
        weather_data = pv_data[['day', 'hour']].copy()
        
        # Create realistic temperature patterns
        weather_data['temp_C'] = 15 + 10 * np.sin(2 * np.pi * (weather_data['day'] - 80) / 365)
        daily_variation = 5 * np.sin(2 * np.pi * (weather_data['hour'] - 6) / 24)
        weather_data['temp_C'] += daily_variation
        weather_data['temp_C'] += np.random.normal(0, 2, len(weather_data))
        
        # Create synthetic GHI
        weather_data['ghi_Wm2'] = self._estimate_ghi(weather_data)
        
        return weather_data
    
    def _estimate_ghi(self, data: pd.DataFrame) -> np.ndarray:
        """Estimate GHI from day/hour patterns"""
        # Clear-sky GHI estimation
        day_of_year = data['day']
        hour = data['hour']
        
        # Solar elevation angle (simplified)
        declination = 23.45 * np.sin(2 * np.pi * (284 + day_of_year) / 365)
        hour_angle = 15 * (hour - 12)
        elevation = np.arcsin(
            np.sin(np.radians(45)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(45)) * np.cos(np.radians(declination)) * np.cos(np.radians(hour_angle))
        )
        
        # Clear-sky GHI
        ghi_clear = 1000 * np.maximum(0, np.sin(elevation))
        
        # Add some realistic variation
        ghi = ghi_clear * (0.7 + 0.3 * np.random.random(len(data)))
        
        return ghi
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create forecasting features"""
        logger.info("Creating PV forecasting features...")
        
        df = data.copy()
        
        # Time features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day'] / 365)
        df['day_of_week'] = (df['day'] - 1) % 7
        df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
        df['month'] = ((df['day'] - 1) // 30) + 1
        
        # Weather features
        df['temp_diff'] = df['temp_C'] - self.config.temp_ref
        df['ghi_norm'] = df['ghi_Wm2'] / self.config.ghi_ref
        
        # Lagged features
        df['pv_lag_1'] = df['pv_kw'].shift(1)
        df['pv_lag_24'] = df['pv_kw'].shift(24)
        df['ghi_lag_1'] = df['ghi_Wm2'].shift(1)
        df['temp_lag_1'] = df['temp_C'].shift(1)
        
        # Rolling means
        df['pv_ma_24'] = df['pv_kw'].rolling(window=24, min_periods=1).mean()
        df['ghi_ma_24'] = df['ghi_Wm2'].rolling(window=24, min_periods=1).mean()
        df['temp_ma_24'] = df['temp_C'].rolling(window=24, min_periods=1).mean()
        
        # Clear-sky index (if GHI available)
        df['clear_sky_index'] = df['ghi_Wm2'] / (1000 * np.maximum(0.1, 
            np.sin(2 * np.pi * (df['day'] - 80) / 365) * 
            np.sin(2 * np.pi * (df['hour'] - 6) / 24)))
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Define feature columns
        self.feature_columns = [
            'hour_sin', 'hour_cos', 'day_of_year_sin', 'day_of_year_cos',
            'day_of_week', 'is_weekend', 'month',
            'temp_C', 'temp_diff', 'ghi_Wm2', 'ghi_norm', 'clear_sky_index',
            'temp_lag_1', 'ghi_lag_1', 'pv_lag_1', 'pv_lag_24',
            'temp_ma_24', 'ghi_ma_24', 'pv_ma_24'
        ]
        
        logger.info(f"Created {len(self.feature_columns)} features")
        return df
    
    def fit_pr_model(self, df: pd.DataFrame) -> Dict:
        """Fit physical PR model"""
        logger.info("Fitting physical PR model...")
        
        # Prepare data for PR model
        # Only use daytime hours (6 AM to 6 PM) and positive GHI
        daytime_mask = (df['hour'] >= 6) & (df['hour'] <= 18) & (df['ghi_Wm2'] > 0)
        pr_data = df[daytime_mask].copy()
        
        if len(pr_data) == 0:
            logger.warning("No daytime data available for PR model")
            return None
        
        # PR model: pv_kw = kWp * PR * (GHI / GHI_ref) * (1 + α_T * (temp_C - temp_ref))
        # Linearize: pv_kw = kWp * PR * (GHI / GHI_ref) + kWp * PR * α_T * (temp_C - temp_ref) * (GHI / GHI_ref)
        
        X_pr = np.column_stack([
            pr_data['ghi_Wm2'] / self.config.ghi_ref,
            pr_data['ghi_Wm2'] / self.config.ghi_ref * (pr_data['temp_C'] - self.config.temp_ref)
        ])
        y_pr = pr_data['pv_kw']
        
        # Fit linear regression
        pr_model = LinearRegression()
        pr_model.fit(X_pr, y_pr)
        
        # Extract parameters
        kwp_pr = pr_model.coef_[0]  # kWp * PR
        alpha_t = pr_model.coef_[1] / pr_model.coef_[0] if pr_model.coef_[0] != 0 else 0
        
        self.pr_model = pr_model
        self.kwp_estimate = kwp_pr
        
        logger.info(f"PR model fitted: kWp*PR = {kwp_pr:.3f}, α_T = {alpha_t:.4f}")
        
        return {
            'model': pr_model,
            'kwp_pr': kwp_pr,
            'alpha_t': alpha_t,
            'type': 'PR_Model'
        }
    
    def predict_pr(self, df: pd.DataFrame) -> np.ndarray:
        """Predict using PR model"""
        if self.pr_model is None:
            return np.zeros(len(df))
        
        X_pr = np.column_stack([
            df['ghi_Wm2'] / self.config.ghi_ref,
            df['ghi_Wm2'] / self.config.ghi_ref * (df['temp_C'] - self.config.temp_ref)
        ])
        
        return self.pr_model.predict(X_pr)
    
    def prepare_training_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data for XGBoost model"""
        # Get PR predictions
        pv_pr = self.predict_pr(df)
        
        # Calculate residuals
        residuals = df['pv_kw'].values - pv_pr
        
        # Prepare features for XGBoost
        X = df[self.feature_columns].values
        y = residuals
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        return X_scaled, y, pv_pr

class PVForecastTrainer:
    """Trains PV forecasting models"""
    
    def __init__(self, config: PVForecastConfig):
        self.config = config
        self.metrics = {}
        
    def train_xgboost(self, X: np.ndarray, y: np.ndarray) -> Dict:
        """Train XGBoost model on residuals"""
        logger.info("Training XGBoost model on residuals...")
        
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
            'type': 'XGBoost_Residuals'
        }
    
    def rolling_origin_cv(self, X: np.ndarray, y: np.ndarray, pv_pr: np.ndarray, 
                         pv_actual: np.ndarray) -> Dict:
        """Perform rolling-origin cross-validation"""
        logger.info("Performing rolling-origin CV for PV forecasting...")
        
        # Create time series split
        tscv = TimeSeriesSplit(n_splits=self.config.cv_splits)
        
        mae_scores = []
        rmse_scores = []
        mape_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pv_pr_val = pv_pr[val_idx]
            pv_actual_val = pv_actual[val_idx]
            
            # Train XGBoost model for this fold
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=self.config.random_state
            )
            model.fit(X_train, y_train)
            
            # Predict residuals
            residual_pred = model.predict(X_val)
            
            # Combine with PR predictions
            pv_pred = pv_pr_val + residual_pred
            
            # Calculate metrics
            mae = mean_absolute_error(pv_actual_val, pv_pred)
            rmse = np.sqrt(mean_squared_error(pv_actual_val, pv_pred))
            mape = mean_absolute_percentage_error(pv_actual_val, pv_pred) * 100
            
            mae_scores.append(mae)
            rmse_scores.append(rmse)
            mape_scores.append(mape)
            
            logger.info(f"Fold {fold + 1}: MAE={mae:.3f}, RMSE={rmse:.3f}, MAPE={mape:.2f}%")
        
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
    
    def save_models(self, pr_model: Dict, xgboost_model: Dict, 
                   metrics: Dict, processor: PVDataProcessor):
        """Save trained models and metadata"""
        logger.info(f"Saving models to {self.config.models_dir}")
        
        # Save PR model
        if pr_model:
            pr_file = os.path.join(self.config.models_dir, "pr_model.joblib")
            joblib.dump(pr_model, pr_file)
            logger.info(f"Saved PR model to {pr_file}")
        
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
            'feature_columns': processor.feature_columns,
            'kwp_estimate': processor.kwp_estimate,
            'metrics': metrics,
            'training_date': datetime.now().isoformat(),
            'config': {
                'cv_splits': self.config.cv_splits,
                'random_state': self.config.random_state,
                'ghi_ref': self.config.ghi_ref,
                'temp_ref': self.config.temp_ref
            }
        }
        
        meta_file = os.path.join(self.config.models_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {meta_file}")
    
    def create_validation_plots(self, metrics: Dict, processor: PVDataProcessor):
        """Create validation plots"""
        logger.info("Creating validation plots...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CV scores distribution
        if 'scores' in metrics:
            axes[0, 0].boxplot([metrics['scores']['mae']], labels=['MAE'])
            axes[0, 0].set_title('CV MAE Distribution')
            axes[0, 0].set_ylabel('MAE (kW)')
            axes[0, 0].grid(True, alpha=0.3)
        
        # RMSE distribution
        if 'scores' in metrics:
            axes[0, 1].boxplot([metrics['scores']['rmse']], labels=['RMSE'])
            axes[0, 1].set_title('CV RMSE Distribution')
            axes[0, 1].set_ylabel('RMSE (kW)')
            axes[0, 1].grid(True, alpha=0.3)
        
        # MAPE distribution
        if 'scores' in metrics:
            axes[1, 0].boxplot([metrics['scores']['mape']], labels=['MAPE'])
            axes[1, 0].set_title('CV MAPE Distribution')
            axes[1, 0].set_ylabel('MAPE (%)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Model performance summary
        metrics_text = f"""
        MAE: {metrics['mae_mean']:.3f} ± {metrics['mae_std']:.3f} kW
        RMSE: {metrics['rmse_mean']:.3f} ± {metrics['rmse_std']:.3f} kW
        MAPE: {metrics['mape_mean']:.2f} ± {metrics['mape_std']:.2f} %
        
        kWp Estimate: {processor.kwp_estimate:.3f} kW
        """
        
        axes[1, 1].text(0.1, 0.5, metrics_text, transform=axes[1, 1].transAxes,
                        fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
        axes[1, 1].set_title('Model Performance Summary')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plot_file = os.path.join(self.config.results_dir, 'pv_forecast_validation.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validation plots to {plot_file}")

def main():
    """Main function for PV forecasting training"""
    parser = argparse.ArgumentParser(description='Train PV forecasting models')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models/pv', help='Models directory')
    parser.add_argument('--results-dir', type=str, default='results/figs_forecast', help='Results directory')
    parser.add_argument('--cv-splits', type=int, default=5, help='Number of CV splits')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--ghi-ref', type=float, default=1000.0, help='Reference GHI (W/m²)')
    parser.add_argument('--temp-ref', type=float, default=25.0, help='Reference temperature (°C)')
    
    args = parser.parse_args()
    
    # Create configuration
    config = PVForecastConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        ghi_ref=args.ghi_ref,
        temp_ref=args.temp_ref
    )
    
    # Create directories
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    logger.info("Starting PV forecasting training...")
    
    try:
        # Initialize processor
        processor = PVDataProcessor(config)
        
        # Load and process data
        data = processor.load_data()
        data_with_features = processor.create_features(data)
        
        # Fit PR model
        pr_model = processor.fit_pr_model(data_with_features)
        
        # Prepare training data
        X, y, pv_pr = processor.prepare_training_data(data_with_features)
        
        # Initialize trainer
        trainer = PVForecastTrainer(config)
        
        # Train XGBoost model
        xgboost_model = trainer.train_xgboost(X, y)
        
        # Perform cross-validation
        metrics = trainer.rolling_origin_cv(X, y, pv_pr, data_with_features['pv_kw'].values)
        
        # Save models
        trainer.save_models(pr_model, xgboost_model, metrics, processor)
        
        # Create validation plots
        trainer.create_validation_plots(metrics, processor)
        
        logger.info("PV forecasting training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
