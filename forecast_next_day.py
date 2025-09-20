#!/usr/bin/env python3
"""
Next Day Forecasting Script

Forecasts tomorrow's 24-hour load and PV generation, generates TOU tariffs,
and runs the optimization for all strategies.

Author: Energy Management System
Date: 2024
"""

import os
import sys
import argparse
import logging
import pandas as pd
import numpy as np
import subprocess
import json
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('forecast_next_day.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class ForecastConfig:
    """Configuration for next day forecasting"""
    data_dir: str = "data"
    models_dir: str = "models"
    forecast_dir: str = "forecast"
    results_dir: str = "results"
    date: str = None  # Will be set to tomorrow if not provided

class TOUGenerator:
    """Generates TOU tariffs based on date and ARERA bands"""
    
    def __init__(self):
        # ARERA F1/F2/F3 prices (€/kWh)
        self.f1_price = 0.48  # Peak (F1)
        self.f2_price = 0.34  # Flat (F2)
        self.f3_price = 0.24  # Valley (F3)
        self.sell_price = 0.10  # Feed-in tariff
    
    def make_tou_24h(self, date: datetime) -> pd.DataFrame:
        """Generate TOU tariff for a specific date"""
        day_of_week = date.weekday()  # Monday=0, Sunday=6
        
        # Define hour ranges for different bands
        if day_of_week < 5:  # Monday to Friday
            # F1 (Peak): 08:00-19:00
            # F2 (Flat): 07:00-08:00 and 19:00-23:00
            # F3 (Valley): 23:00-07:00
            tou_data = []
            for hour in range(1, 25):
                if 8 <= hour <= 19:
                    price_buy = self.f1_price
                elif hour in [7] or 19 <= hour <= 22:
                    price_buy = self.f2_price
                else:  # 23:00-07:00
                    price_buy = self.f3_price
                
                tou_data.append({
                    'hour': hour,
                    'price_buy': price_buy,
                    'price_sell': self.sell_price
                })
        
        elif day_of_week == 5:  # Saturday
            # F2 (Flat): 07:00-23:00
            # F3 (Valley): 23:00-07:00
            tou_data = []
            for hour in range(1, 25):
                if 7 <= hour <= 22:
                    price_buy = self.f2_price
                else:  # 23:00-07:00
                    price_buy = self.f3_price
                
                tou_data.append({
                    'hour': hour,
                    'price_buy': price_buy,
                    'price_sell': self.sell_price
                })
        
        else:  # Sunday
            # F3 (Valley): all day
            tou_data = []
            for hour in range(1, 25):
                tou_data.append({
                    'hour': hour,
                    'price_buy': self.f3_price,
                    'price_sell': self.sell_price
                })
        
        return pd.DataFrame(tou_data)

class LoadForecaster:
    """Load forecasting using trained models"""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.load_model = None
        self.load_scaler = None
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load trained load forecasting models"""
        try:
            # Load metadata
            meta_file = os.path.join(self.models_dir, "load", "meta.json")
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            best_model = metadata['best_model']
            
            # Load scaler
            scaler_file = os.path.join(self.models_dir, "load", "scaler.joblib")
            self.load_scaler = joblib.load(scaler_file)
            
            # Load best model
            if best_model == 'XGBoost':
                model_file = os.path.join(self.models_dir, "load", "xgboost_model.joblib")
                self.load_model = joblib.load(model_file)
            else:
                model_file = os.path.join(self.models_dir, "load", "sarimax_model.joblib")
                self.load_model = joblib.load(model_file)
            
            logger.info(f"Loaded {best_model} model for load forecasting")
            
        except Exception as e:
            logger.error(f"Failed to load load models: {e}")
            raise
    
    def create_features(self, date: datetime, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for load forecasting"""
        features = []
        
        for hour in range(1, 25):
            # Time features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_of_week = date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            month = date.month
            is_holiday = 0  # Simplified - could be enhanced with holiday calendar
            
            # Weather features
            hour_weather = weather_data[weather_data['hour'] == hour]
            if len(hour_weather) > 0:
                temp_C = hour_weather['temp_C'].iloc[0]
            else:
                temp_C = 20.0  # Default temperature
            
            # Lagged features (simplified - using historical averages)
            load_lag_1 = 3.0  # Simplified
            load_lag_24 = 3.0
            load_lag_168 = 3.0
            
            # Rolling means (simplified)
            load_ma_24 = 3.0
            load_ma_168 = 3.0
            
            # Temperature features
            temp_lag_1 = temp_C
            temp_ma_24 = temp_C
            
            features.append([
                hour_sin, hour_cos, day_of_week, is_weekend, month, is_holiday,
                temp_C, temp_lag_1, temp_ma_24,
                load_lag_1, load_lag_24, load_lag_168,
                load_ma_24, load_ma_168
            ])
        
        return pd.DataFrame(features, columns=self.feature_columns)
    
    def forecast(self, date: datetime, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast load for next day"""
        logger.info(f"Forecasting load for {date.strftime('%Y-%m-%d')}")
        
        # Create features
        features = self.create_features(date, weather_data)
        
        # Scale features
        features_scaled = self.load_scaler.transform(features)
        
        # Predict
        if hasattr(self.load_model, 'predict'):
            # XGBoost model
            load_predictions = self.load_model.predict(features_scaled)
        else:
            # SARIMAX model (simplified)
            load_predictions = np.full(24, 3.0)  # Default prediction
        
        # Create output DataFrame
        forecast_data = pd.DataFrame({
            'hour': range(1, 25),
            'load_kw': load_predictions
        })
        
        logger.info(f"Load forecast completed: mean={np.mean(load_predictions):.2f} kW")
        return forecast_data

class PVForecaster:
    """PV forecasting using trained models"""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.pr_model = None
        self.xgboost_model = None
        self.pv_scaler = None
        self.feature_columns = []
        self.kwp_estimate = None
        self.load_models()
    
    def load_models(self):
        """Load trained PV forecasting models"""
        try:
            # Load metadata
            meta_file = os.path.join(self.models_dir, "pv", "meta.json")
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.kwp_estimate = metadata['kwp_estimate']
            
            # Load PR model
            pr_file = os.path.join(self.models_dir, "pv", "pr_model.joblib")
            self.pr_model = joblib.load(pr_file)
            
            # Load XGBoost model
            xgboost_file = os.path.join(self.models_dir, "pv", "xgboost_model.joblib")
            self.xgboost_model = joblib.load(xgboost_file)
            
            # Load scaler
            scaler_file = os.path.join(self.models_dir, "pv", "scaler.joblib")
            self.pv_scaler = joblib.load(scaler_file)
            
            logger.info("Loaded PV forecasting models")
            
        except Exception as e:
            logger.error(f"Failed to load PV models: {e}")
            raise
    
    def create_features(self, date: datetime, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Create features for PV forecasting"""
        features = []
        
        for hour in range(1, 25):
            # Time features
            hour_sin = np.sin(2 * np.pi * hour / 24)
            hour_cos = np.cos(2 * np.pi * hour / 24)
            day_of_year = date.timetuple().tm_yday
            day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
            day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
            day_of_week = date.weekday()
            is_weekend = 1 if day_of_week >= 5 else 0
            month = date.month
            
            # Weather features
            hour_weather = weather_data[weather_data['hour'] == hour]
            if len(hour_weather) > 0:
                temp_C = hour_weather['temp_C'].iloc[0]
                ghi_Wm2 = hour_weather.get('ghi_Wm2', 0).iloc[0] if 'ghi_Wm2' in hour_weather.columns else 0
            else:
                temp_C = 20.0
                ghi_Wm2 = 0
            
            # Derived features
            temp_diff = temp_C - 25.0
            ghi_norm = ghi_Wm2 / 1000.0 if ghi_Wm2 > 0 else 0
            
            # Clear-sky index (simplified)
            clear_sky_index = ghi_norm if ghi_norm > 0 else 0
            
            # Lagged features (simplified)
            temp_lag_1 = temp_C
            ghi_lag_1 = ghi_Wm2
            pv_lag_1 = 0.0
            pv_lag_24 = 0.0
            
            # Rolling means (simplified)
            temp_ma_24 = temp_C
            ghi_ma_24 = ghi_Wm2
            pv_ma_24 = 0.0
            
            features.append([
                hour_sin, hour_cos, day_of_year_sin, day_of_year_cos,
                day_of_week, is_weekend, month,
                temp_C, temp_diff, ghi_Wm2, ghi_norm, clear_sky_index,
                temp_lag_1, ghi_lag_1, pv_lag_1, pv_lag_24,
                temp_ma_24, ghi_ma_24, pv_ma_24
            ])
        
        return pd.DataFrame(features, columns=self.feature_columns)
    
    def predict_pr(self, weather_data: pd.DataFrame) -> np.ndarray:
        """Predict using PR model"""
        if self.pr_model is None:
            return np.zeros(24)
        
        pr_predictions = []
        for hour in range(1, 25):
            hour_weather = weather_data[weather_data['hour'] == hour]
            if len(hour_weather) > 0:
                ghi_Wm2 = hour_weather.get('ghi_Wm2', 0).iloc[0] if 'ghi_Wm2' in hour_weather.columns else 0
                temp_C = hour_weather['temp_C'].iloc[0]
            else:
                ghi_Wm2 = 0
                temp_C = 20.0
            
            # PR model prediction
            X_pr = np.array([[ghi_Wm2 / 1000.0, ghi_Wm2 / 1000.0 * (temp_C - 25.0)]])
            pr_pred = self.pr_model['model'].predict(X_pr)[0]
            pr_predictions.append(max(0, pr_pred))
        else:
            pr_predictions.append(0)
        
        return np.array(pr_predictions)
    
    def forecast(self, date: datetime, weather_data: pd.DataFrame) -> pd.DataFrame:
        """Forecast PV for next day"""
        logger.info(f"Forecasting PV for {date.strftime('%Y-%m-%d')}")
        
        # Get PR predictions
        pr_predictions = self.predict_pr(weather_data)
        
        # Create features for XGBoost
        features = self.create_features(date, weather_data)
        features_scaled = self.pv_scaler.transform(features)
        
        # Predict residuals
        residual_predictions = self.xgboost_model['model'].predict(features_scaled)
        
        # Combine PR and residual predictions
        pv_predictions = pr_predictions + residual_predictions
        pv_predictions = np.maximum(0, pv_predictions)  # Ensure non-negative
        
        # Create output DataFrame
        forecast_data = pd.DataFrame({
            'hour': range(1, 25),
            'pv_kw': pv_predictions
        })
        
        logger.info(f"PV forecast completed: mean={np.mean(pv_predictions):.2f} kW")
        return forecast_data

class NextDayForecaster:
    """Main class for next day forecasting"""
    
    def __init__(self, config: ForecastConfig):
        self.config = config
        self.tou_generator = TOUGenerator()
        self.load_forecaster = LoadForecaster(config.models_dir)
        self.pv_forecaster = PVForecaster(config.models_dir)
        
        # Create directories
        os.makedirs(config.forecast_dir, exist_ok=True)
        os.makedirs(config.data_dir, exist_ok=True)
    
    def load_weather_forecast(self, date: datetime) -> pd.DataFrame:
        """Load weather forecast for the specified date"""
        weather_file = os.path.join(self.config.forecast_dir, f"weather_nextday_{date.strftime('%Y%m%d')}.csv")
        
        if os.path.exists(weather_file):
            weather_data = pd.read_csv(weather_file)
            logger.info(f"Loaded weather forecast from {weather_file}")
        else:
            logger.warning(f"Weather forecast not found: {weather_file}")
            logger.info("Creating synthetic weather forecast")
            weather_data = self._create_synthetic_weather(date)
        
        return weather_data
    
    def _create_synthetic_weather(self, date: datetime) -> pd.DataFrame:
        """Create synthetic weather forecast"""
        weather_data = []
        
        # Create realistic temperature pattern
        day_of_year = date.timetuple().tm_yday
        base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
        
        for hour in range(1, 25):
            # Daily temperature variation
            daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temp_C = base_temp + daily_variation + np.random.normal(0, 1)
            
            # Simple GHI estimation
            if 6 <= hour <= 18:
                ghi_Wm2 = 800 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 50)
                ghi_Wm2 = max(0, ghi_Wm2)
            else:
                ghi_Wm2 = 0
            
            weather_data.append({
                'hour': hour,
                'temp_C': temp_C,
                'ghi_Wm2': ghi_Wm2
            })
        
        return pd.DataFrame(weather_data)
    
    def generate_forecasts(self, date: datetime) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Generate all forecasts for the specified date"""
        logger.info(f"Generating forecasts for {date.strftime('%Y-%m-%d')}")
        
        # Load weather forecast
        weather_data = self.load_weather_forecast(date)
        
        # Generate load forecast
        load_forecast = self.load_forecaster.forecast(date, weather_data)
        
        # Generate PV forecast
        pv_forecast = self.pv_forecaster.forecast(date, weather_data)
        
        # Generate TOU tariff
        tou_data = self.tou_generator.make_tou_24h(date)
        
        return load_forecast, pv_forecast, tou_data
    
    def save_forecasts(self, date: datetime, load_forecast: pd.DataFrame, 
                      pv_forecast: pd.DataFrame, tou_data: pd.DataFrame):
        """Save forecast data to files"""
        # Save load forecast
        load_file = os.path.join(self.config.forecast_dir, "nextday_load_24h.csv")
        load_forecast.to_csv(load_file, index=False)
        logger.info(f"Saved load forecast to {load_file}")
        
        # Save PV forecast
        pv_file = os.path.join(self.config.forecast_dir, "nextday_pv_24h.csv")
        pv_forecast.to_csv(pv_file, index=False)
        logger.info(f"Saved PV forecast to {pv_file}")
        
        # Save TOU data to data directory (for optimizer)
        tou_file = os.path.join(self.config.data_dir, "tou_24h.csv")
        tou_data.to_csv(tou_file, index=False)
        logger.info(f"Saved TOU data to {tou_file}")
    
    def run_optimization(self) -> Dict:
        """Run optimization for all strategies"""
        logger.info("Running optimization for all strategies...")
        
        try:
            # Run the optimizer
            result = subprocess.run([
                'python3', 'run_day.py', '--strategy', 'ALL'
            ], capture_output=True, text=True, cwd=os.getcwd())
            
            if result.returncode != 0:
                logger.error(f"Optimization failed: {result.stderr}")
                return None
            
            # Parse results from output
            output_lines = result.stdout.split('\n')
            results = {}
            
            for line in output_lines:
                if 'Strategy:' in line and 'Cost:' in line:
                    # Parse strategy results
                    parts = line.split()
                    strategy = parts[1]
                    cost = float(parts[3].replace('€', ''))
                    results[strategy] = {'cost': cost}
            
            logger.info("Optimization completed successfully")
            return results
            
        except Exception as e:
            logger.error(f"Failed to run optimization: {e}")
            return None
    
    def print_summary(self, date: datetime, results: Dict):
        """Print forecast summary"""
        print(f"\n{'='*60}")
        print(f"FORECAST SUMMARY FOR {date.strftime('%Y-%m-%d')}")
        print(f"{'='*60}")
        
        if results:
            print(f"{'Strategy':<15} {'Cost (€)':<12} {'Status':<10}")
            print(f"{'-'*40}")
            for strategy, data in results.items():
                print(f"{strategy:<15} {data['cost']:<12.2f} {'Success':<10}")
        else:
            print("Optimization failed - no results available")
        
        print(f"{'='*60}\n")

def main():
    """Main function for next day forecasting"""
    parser = argparse.ArgumentParser(description='Forecast next day and run optimization')
    parser.add_argument('--date', type=str, help='Date to forecast (YYYY-MM-DD), defaults to tomorrow')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--forecast-dir', type=str, default='forecast', help='Forecast directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    
    args = parser.parse_args()
    
    # Parse date
    if args.date:
        try:
            forecast_date = datetime.strptime(args.date, '%Y-%m-%d')
        except ValueError:
            logger.error("Invalid date format. Use YYYY-MM-DD")
            sys.exit(1)
    else:
        forecast_date = datetime.now() + timedelta(days=1)
    
    # Create configuration
    config = ForecastConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        forecast_dir=args.forecast_dir,
        results_dir=args.results_dir,
        date=forecast_date.strftime('%Y-%m-%d')
    )
    
    logger.info(f"Starting next day forecasting for {forecast_date.strftime('%Y-%m-%d')}")
    
    try:
        # Initialize forecaster
        forecaster = NextDayForecaster(config)
        
        # Generate forecasts
        load_forecast, pv_forecast, tou_data = forecaster.generate_forecasts(forecast_date)
        
        # Save forecasts
        forecaster.save_forecasts(forecast_date, load_forecast, pv_forecast, tou_data)
        
        # Run optimization
        results = forecaster.run_optimization()
        
        # Print summary
        forecaster.print_summary(forecast_date, results)
        
        logger.info("Next day forecasting completed successfully!")
        
    except Exception as e:
        logger.error(f"Forecasting failed: {e}")
        raise

if __name__ == "__main__":
    main()
