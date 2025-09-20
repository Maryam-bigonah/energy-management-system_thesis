#!/usr/bin/env python3
"""
Next Year Simulation Script

Simulates the entire next year (365 days) using trained forecasting models,
runs optimization for all strategies, and generates annual KPIs.

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
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('simulate_next_year.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class YearSimulationConfig:
    """Configuration for yearly simulation"""
    data_dir: str = "data"
    models_dir: str = "models"
    forecast_dir: str = "forecast"
    results_dir: str = "results"
    year: int = None  # Will be set to next year if not provided
    save_hourly: bool = False
    parallel: bool = True
    max_workers: int = None

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
            
            # Load scaler
            scaler_file = os.path.join(self.models_dir, "load", "scaler.joblib")
            self.load_scaler = joblib.load(scaler_file)
            
            # Load best model
            best_model = metadata['best_model']
            if best_model == 'XGBoost':
                model_file = os.path.join(self.models_dir, "load", "xgboost_model.joblib")
                self.load_model = joblib.load(model_file)
            else:
                model_file = os.path.join(self.models_dir, "load", "sarimax_model.joblib")
                self.load_model = joblib.load(model_file)
            
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
            is_holiday = 0  # Simplified
            
            # Weather features
            hour_weather = weather_data[weather_data['hour'] == hour]
            if len(hour_weather) > 0:
                temp_C = hour_weather['temp_C'].iloc[0]
            else:
                temp_C = 20.0
            
            # Lagged features (simplified)
            load_lag_1 = 3.0
            load_lag_24 = 3.0
            load_lag_168 = 3.0
            load_ma_24 = 3.0
            load_ma_168 = 3.0
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
        """Forecast load for a specific day"""
        # Create features
        features = self.create_features(date, weather_data)
        
        # Scale features
        features_scaled = self.load_scaler.transform(features)
        
        # Predict
        if hasattr(self.load_model, 'predict'):
            load_predictions = self.load_model.predict(features_scaled)
        else:
            load_predictions = np.full(24, 3.0)
        
        # Create output DataFrame
        forecast_data = pd.DataFrame({
            'day': [date.timetuple().tm_yday] * 24,
            'hour': range(1, 25),
            'load_kw': load_predictions
        })
        
        return forecast_data

class PVForecaster:
    """PV forecasting using trained models"""
    
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.pr_model = None
        self.xgboost_model = None
        self.pv_scaler = None
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load trained PV forecasting models"""
        try:
            # Load metadata
            meta_file = os.path.join(self.models_dir, "pv", "meta.json")
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            
            # Load PR model
            pr_file = os.path.join(self.models_dir, "pv", "pr_model.joblib")
            self.pr_model = joblib.load(pr_file)
            
            # Load XGBoost model
            xgboost_file = os.path.join(self.models_dir, "pv", "xgboost_model.joblib")
            self.xgboost_model = joblib.load(xgboost_file)
            
            # Load scaler
            scaler_file = os.path.join(self.models_dir, "pv", "scaler.joblib")
            self.pv_scaler = joblib.load(scaler_file)
            
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
            clear_sky_index = ghi_norm if ghi_norm > 0 else 0
            
            # Lagged features (simplified)
            temp_lag_1 = temp_C
            ghi_lag_1 = ghi_Wm2
            pv_lag_1 = 0.0
            pv_lag_24 = 0.0
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
        """Forecast PV for a specific day"""
        # Get PR predictions
        pr_predictions = self.predict_pr(weather_data)
        
        # Create features for XGBoost
        features = self.create_features(date, weather_data)
        features_scaled = self.pv_scaler.transform(features)
        
        # Predict residuals
        residual_predictions = self.xgboost_model['model'].predict(features_scaled)
        
        # Combine PR and residual predictions
        pv_predictions = pr_predictions + residual_predictions
        pv_predictions = np.maximum(0, pv_predictions)
        
        # Create output DataFrame
        forecast_data = pd.DataFrame({
            'day': [date.timetuple().tm_yday] * 24,
            'hour': range(1, 25),
            'pv_kw': pv_predictions
        })
        
        return forecast_data

def process_single_day(day: int, year: int, config: YearSimulationConfig, 
                      load_forecaster: LoadForecaster, pv_forecaster: PVForecaster,
                      tou_generator: TOUGenerator) -> Optional[Dict]:
    """Process a single day of the year"""
    try:
        # Create date for this day
        date = datetime(year, 1, 1) + timedelta(days=day-1)
        
        # Load weather forecast for this day
        weather_file = os.path.join(config.forecast_dir, f"nextyear_weather_{year}.csv")
        if os.path.exists(weather_file):
            weather_data = pd.read_csv(weather_file)
            day_weather = weather_data[weather_data['day'] == day]
        else:
            # Create synthetic weather
            day_weather = create_synthetic_weather_day(day, year)
        
        # Generate forecasts
        load_forecast = load_forecaster.forecast(date, day_weather)
        pv_forecast = pv_forecaster.forecast(date, day_weather)
        tou_data = tou_generator.make_tou_24h(date)
        
        # Save daily data
        temp_dir = os.path.join(config.results_dir, f"temp_day_{day}")
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save load data
        load_file = os.path.join(temp_dir, "load_24h.csv")
        load_forecast[['hour', 'load_kw']].to_csv(load_file, index=False)
        
        # Save PV data
        pv_file = os.path.join(temp_dir, "pv_24h.csv")
        pv_forecast[['hour', 'pv_kw']].to_csv(pv_file, index=False)
        
        # Save TOU data
        tou_file = os.path.join(temp_dir, "tou_24h.csv")
        tou_data.to_csv(tou_file, index=False)
        
        # Copy battery specs
        battery_file = os.path.join(config.data_dir, "battery.yaml")
        if os.path.exists(battery_file):
            import shutil
            shutil.copy(battery_file, os.path.join(temp_dir, "battery.yaml"))
        
        # Run optimization
        result = subprocess.run([
            'python3', 'run_day.py', '--strategy', 'ALL'
        ], capture_output=True, text=True, cwd=temp_dir)
        
        if result.returncode != 0:
            logger.warning(f"Optimization failed for day {day}: {result.stderr}")
            return None
        
        # Parse results
        results = {}
        output_lines = result.stdout.split('\n')
        for line in output_lines:
            if 'Strategy:' in line and 'Cost:' in line:
                parts = line.split()
                strategy = parts[1]
                cost = float(parts[3].replace('€', ''))
                results[strategy] = {'cost': cost}
        
        # Clean up temp directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        return {
            'day': day,
            'date': date.strftime('%Y-%m-%d'),
            'results': results
        }
        
    except Exception as e:
        logger.error(f"Error processing day {day}: {e}")
        return None

def create_synthetic_weather_day(day: int, year: int) -> pd.DataFrame:
    """Create synthetic weather for a specific day"""
    date = datetime(year, 1, 1) + timedelta(days=day-1)
    day_of_year = date.timetuple().tm_yday
    
    weather_data = []
    base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    for hour in range(1, 25):
        daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
        temp_C = base_temp + daily_variation + np.random.normal(0, 1)
        
        if 6 <= hour <= 18:
            ghi_Wm2 = 800 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 50)
            ghi_Wm2 = max(0, ghi_Wm2)
        else:
            ghi_Wm2 = 0
        
        weather_data.append({
            'day': day,
            'hour': hour,
            'temp_C': temp_C,
            'ghi_Wm2': ghi_Wm2
        })
    
    return pd.DataFrame(weather_data)

class NextYearSimulator:
    """Main class for next year simulation"""
    
    def __init__(self, config: YearSimulationConfig):
        self.config = config
        self.tou_generator = TOUGenerator()
        self.load_forecaster = LoadForecaster(config.models_dir)
        self.pv_forecaster = PVForecaster(config.models_dir)
        
        # Create directories
        os.makedirs(config.forecast_dir, exist_ok=True)
        os.makedirs(config.results_dir, exist_ok=True)
        
        if config.max_workers is None:
            config.max_workers = min(4, mp.cpu_count())
    
    def generate_annual_forecasts(self, year: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate annual load and PV forecasts"""
        logger.info(f"Generating annual forecasts for {year}...")
        
        # Check if weather forecast exists
        weather_file = os.path.join(self.config.forecast_dir, f"nextyear_weather_{year}.csv")
        if not os.path.exists(weather_file):
            logger.warning(f"Weather forecast not found: {weather_file}")
            logger.info("Creating synthetic annual weather forecast")
            self._create_synthetic_weather_year(year)
        
        # Load weather data
        weather_data = pd.read_csv(weather_file)
        
        # Generate forecasts for all days
        load_forecasts = []
        pv_forecasts = []
        
        for day in range(1, 366):  # Handle leap year
            try:
                date = datetime(year, 1, 1) + timedelta(days=day-1)
                day_weather = weather_data[weather_data['day'] == day]
                
                if len(day_weather) == 0:
                    continue
                
                # Generate forecasts
                load_forecast = self.load_forecaster.forecast(date, day_weather)
                pv_forecast = self.pv_forecaster.forecast(date, day_weather)
                
                load_forecasts.append(load_forecast)
                pv_forecasts.append(pv_forecast)
                
            except Exception as e:
                logger.warning(f"Failed to forecast day {day}: {e}")
                continue
        
        # Combine all forecasts
        annual_load = pd.concat(load_forecasts, ignore_index=True)
        annual_pv = pd.concat(pv_forecasts, ignore_index=True)
        
        # Save annual forecasts
        load_file = os.path.join(self.config.forecast_dir, f"nextyear_load_{year}.csv")
        annual_load.to_csv(load_file, index=False)
        logger.info(f"Saved annual load forecast to {load_file}")
        
        pv_file = os.path.join(self.config.forecast_dir, f"nextyear_pv_{year}.csv")
        annual_pv.to_csv(pv_file, index=False)
        logger.info(f"Saved annual PV forecast to {pv_file}")
        
        return annual_load, annual_pv
    
    def _create_synthetic_weather_year(self, year: int):
        """Create synthetic weather for the entire year"""
        weather_data = []
        
        for day in range(1, 366):  # Handle leap year
            date = datetime(year, 1, 1) + timedelta(days=day-1)
            day_of_year = date.timetuple().tm_yday
            
            base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            
            for hour in range(1, 25):
                daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
                temp_C = base_temp + daily_variation + np.random.normal(0, 1)
                
                if 6 <= hour <= 18:
                    ghi_Wm2 = 800 * np.sin(np.pi * (hour - 6) / 12) + np.random.normal(0, 50)
                    ghi_Wm2 = max(0, ghi_Wm2)
                else:
                    ghi_Wm2 = 0
                
                weather_data.append({
                    'day': day,
                    'hour': hour,
                    'temp_C': temp_C,
                    'ghi_Wm2': ghi_Wm2
                })
        
        weather_df = pd.DataFrame(weather_data)
        weather_file = os.path.join(self.config.forecast_dir, f"nextyear_weather_{year}.csv")
        weather_df.to_csv(weather_file, index=False)
        logger.info(f"Created synthetic weather forecast: {weather_file}")
    
    def run_simulation(self, year: int) -> pd.DataFrame:
        """Run the complete yearly simulation"""
        logger.info(f"Starting yearly simulation for {year}...")
        
        # Generate annual forecasts
        annual_load, annual_pv = self.generate_annual_forecasts(year)
        
        # Prepare for parallel processing
        if self.config.parallel:
            logger.info(f"Running parallel simulation with {self.config.max_workers} workers")
            
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                # Submit all tasks
                futures = []
                for day in range(1, 366):
                    future = executor.submit(
                        process_single_day, day, year, self.config,
                        self.load_forecaster, self.pv_forecaster, self.tou_generator
                    )
                    futures.append(future)
                
                # Collect results
                results = []
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    if result:
                        results.append(result)
                    
                    if (i + 1) % 50 == 0:
                        logger.info(f"Completed {i + 1}/365 days")
        else:
            logger.info("Running sequential simulation")
            results = []
            for day in range(1, 366):
                result = process_single_day(day, year, self.config,
                                          self.load_forecaster, self.pv_forecaster, self.tou_generator)
                if result:
                    results.append(result)
                
                if day % 50 == 0:
                    logger.info(f"Completed {day}/365 days")
        
        # Process results
        kpis_data = []
        for result in results:
            day = result['day']
            date = result['date']
            
            for strategy, data in result['results'].items():
                kpis_data.append({
                    'day': day,
                    'date': date,
                    'strategy': strategy,
                    'cost_total': data['cost'],
                    'year_tag': f'forecast_{year}'
                })
        
        # Create KPIs DataFrame
        kpis_df = pd.DataFrame(kpis_data)
        
        # Save results
        kpis_file = os.path.join(self.config.results_dir, f"kpis_forecast_{year}.csv")
        kpis_df.to_csv(kpis_file, index=False)
        logger.info(f"Saved KPIs to {kpis_file}")
        
        return kpis_df
    
    def print_annual_summary(self, year: int, kpis_df: pd.DataFrame):
        """Print annual summary"""
        print(f"\n{'='*80}")
        print(f"ANNUAL SIMULATION SUMMARY FOR {year}")
        print(f"{'='*80}")
        
        # Calculate annual aggregates
        annual_summary = kpis_df.groupby('strategy').agg({
            'cost_total': ['sum', 'mean', 'std']
        }).round(2)
        
        annual_summary.columns = ['Annual_Cost', 'Mean_Daily_Cost', 'Std_Daily_Cost']
        annual_summary = annual_summary.reset_index()
        
        print(f"{'Strategy':<15} {'Annual Cost (€)':<18} {'Mean Daily (€)':<16} {'Std Daily (€)':<15}")
        print(f"{'-'*70}")
        
        for _, row in annual_summary.iterrows():
            print(f"{row['strategy']:<15} {row['Annual_Cost']:<18.2f} {row['Mean_Daily_Cost']:<16.2f} {row['Std_Daily_Cost']:<15.2f}")
        
        print(f"{'='*80}\n")

def main():
    """Main function for next year simulation"""
    parser = argparse.ArgumentParser(description='Simulate next year and run optimization')
    parser.add_argument('--year', type=int, help='Year to simulate, defaults to next year')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models', help='Models directory')
    parser.add_argument('--forecast-dir', type=str, default='forecast', help='Forecast directory')
    parser.add_argument('--results-dir', type=str, default='results', help='Results directory')
    parser.add_argument('--save-hourly', action='store_true', help='Save hourly results')
    parser.add_argument('--no-parallel', action='store_true', help='Disable parallel processing')
    parser.add_argument('--max-workers', type=int, help='Maximum number of workers')
    
    args = parser.parse_args()
    
    # Set year
    if args.year:
        year = args.year
    else:
        year = datetime.now().year + 1
    
    # Create configuration
    config = YearSimulationConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        forecast_dir=args.forecast_dir,
        results_dir=args.results_dir,
        year=year,
        save_hourly=args.save_hourly,
        parallel=not args.no_parallel,
        max_workers=args.max_workers
    )
    
    logger.info(f"Starting next year simulation for {year}")
    
    try:
        # Initialize simulator
        simulator = NextYearSimulator(config)
        
        # Run simulation
        kpis_df = simulator.run_simulation(year)
        
        # Print summary
        simulator.print_annual_summary(year, kpis_df)
        
        logger.info("Next year simulation completed successfully!")
        
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise

if __name__ == "__main__":
    main()
