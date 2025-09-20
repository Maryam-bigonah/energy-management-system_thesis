#!/usr/bin/env python3
"""
Surrogate Model Training Script

Trains XGBoost surrogate models to predict optimization costs
from daily features without running the full optimization.

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
import subprocess
import json
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# ML libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surrogate_training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SurrogateConfig:
    """Configuration for surrogate model training"""
    data_dir: str = "data"
    models_dir: str = "models/surrogate"
    results_dir: str = "results/figs_forecast"
    n_scenarios: int = 1000
    random_state: int = 42
    test_size: float = 0.2

class ScenarioGenerator:
    """Generates training scenarios for surrogate models"""
    
    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.scaler = StandardScaler()
        
    def generate_scenarios(self) -> pd.DataFrame:
        """Generate diverse training scenarios"""
        logger.info(f"Generating {self.config.n_scenarios} training scenarios...")
        
        scenarios = []
        
        for i in range(self.config.n_scenarios):
            # Generate random daily features
            scenario = self._generate_single_scenario(i)
            scenarios.append(scenario)
            
            if (i + 1) % 100 == 0:
                logger.info(f"Generated {i + 1}/{self.config.n_scenarios} scenarios")
        
        scenarios_df = pd.DataFrame(scenarios)
        logger.info(f"Generated {len(scenarios_df)} scenarios")
        
        return scenarios_df
    
    def _generate_single_scenario(self, seed: int) -> Dict:
        """Generate a single training scenario"""
        np.random.seed(seed)
        
        # PV generation features
        pv_total = np.random.uniform(10, 100)  # kWh/day
        pv_peak = np.random.uniform(5, 50)     # kW
        pv_ramp_up = np.random.uniform(0.1, 2.0)  # kW/h
        pv_ramp_down = np.random.uniform(0.1, 2.0)  # kW/h
        pv_peak_hour = np.random.randint(10, 16)  # Peak around noon
        
        # Load consumption features
        load_total = np.random.uniform(50, 200)  # kWh/day
        load_peak = np.random.uniform(10, 40)    # kW
        load_ramp_up = np.random.uniform(0.5, 3.0)  # kW/h
        load_ramp_down = np.random.uniform(0.5, 3.0)  # kW/h
        load_peak_hour = np.random.choice([18, 19, 20, 21])  # Evening peak
        
        # Price features
        price_buy_mean = np.random.uniform(0.2, 0.5)  # €/kWh
        price_buy_std = np.random.uniform(0.05, 0.15)  # €/kWh
        price_sell = np.random.uniform(0.05, 0.15)    # €/kWh
        price_ratio = price_buy_mean / price_sell if price_sell > 0 else 1
        
        # Temporal features
        day_of_week = np.random.randint(0, 7)
        is_weekend = 1 if day_of_week >= 5 else 0
        month = np.random.randint(1, 13)
        season = (month - 1) // 3  # 0=Winter, 1=Spring, 2=Summer, 3=Autumn
        
        # Weather features
        temp_mean = np.random.uniform(5, 30)  # °C
        temp_std = np.random.uniform(2, 8)    # °C
        ghi_total = np.random.uniform(1000, 6000)  # Wh/m²/day
        
        # Battery features
        battery_capacity = np.random.uniform(50, 200)  # kWh
        battery_power = np.random.uniform(20, 100)     # kW
        
        # Derived features
        pv_load_ratio = pv_total / load_total if load_total > 0 else 0
        net_load = load_total - pv_total
        self_consumption_potential = min(pv_total, load_total)
        
        # Peak features
        peak_demand = max(load_peak, pv_peak)
        peak_net_demand = max(0, load_peak - pv_peak)
        
        # Ramp features
        max_ramp = max(pv_ramp_up, pv_ramp_down, load_ramp_up, load_ramp_down)
        pv_load_ramp_ratio = (pv_ramp_up + pv_ramp_down) / (load_ramp_up + load_ramp_down) if (load_ramp_up + load_ramp_down) > 0 else 1
        
        return {
            # PV features
            'pv_total': pv_total,
            'pv_peak': pv_peak,
            'pv_ramp_up': pv_ramp_up,
            'pv_ramp_down': pv_ramp_down,
            'pv_peak_hour': pv_peak_hour,
            
            # Load features
            'load_total': load_total,
            'load_peak': load_peak,
            'load_ramp_up': load_ramp_up,
            'load_ramp_down': load_ramp_down,
            'load_peak_hour': load_peak_hour,
            
            # Price features
            'price_buy_mean': price_buy_mean,
            'price_buy_std': price_buy_std,
            'price_sell': price_sell,
            'price_ratio': price_ratio,
            
            # Temporal features
            'day_of_week': day_of_week,
            'is_weekend': is_weekend,
            'month': month,
            'season': season,
            
            # Weather features
            'temp_mean': temp_mean,
            'temp_std': temp_std,
            'ghi_total': ghi_total,
            
            # Battery features
            'battery_capacity': battery_capacity,
            'battery_power': battery_power,
            
            # Derived features
            'pv_load_ratio': pv_load_ratio,
            'net_load': net_load,
            'self_consumption_potential': self_consumption_potential,
            'peak_demand': peak_demand,
            'peak_net_demand': peak_net_demand,
            'max_ramp': max_ramp,
            'pv_load_ramp_ratio': pv_load_ramp_ratio
        }
    
    def create_training_data(self, scenarios_df: pd.DataFrame) -> pd.DataFrame:
        """Create hourly data from scenarios and run optimization"""
        logger.info("Creating training data from scenarios...")
        
        training_data = []
        
        for idx, scenario in scenarios_df.iterrows():
            # Create hourly data for this scenario
            hourly_data = self._scenario_to_hourly(scenario)
            
            # Run optimization for all strategies
            costs = self._run_optimization(hourly_data)
            
            if costs:
                # Add costs to scenario data
                scenario_with_costs = scenario.copy()
                for strategy, cost in costs.items():
                    scenario_with_costs[f'cost_{strategy}'] = cost
                
                training_data.append(scenario_with_costs)
            
            if (idx + 1) % 50 == 0:
                logger.info(f"Processed {idx + 1}/{len(scenarios_df)} scenarios")
        
        training_df = pd.DataFrame(training_data)
        logger.info(f"Created training data with {len(training_df)} samples")
        
        return training_df
    
    def _scenario_to_hourly(self, scenario: pd.Series) -> Dict:
        """Convert scenario to hourly data"""
        # Create realistic hourly profiles
        hours = range(1, 25)
        
        # PV profile (bell curve centered at peak hour)
        pv_profile = []
        for hour in hours:
            if 6 <= hour <= 18:
                # Bell curve around peak hour
                pv_power = scenario['pv_peak'] * np.exp(-0.5 * ((hour - scenario['pv_peak_hour']) / 3) ** 2)
                pv_power += np.random.normal(0, scenario['pv_peak'] * 0.1)  # Add noise
                pv_power = max(0, pv_power)
            else:
                pv_power = 0
            pv_profile.append(pv_power)
        
        # Scale to match total
        pv_scale = scenario['pv_total'] / sum(pv_profile) if sum(pv_profile) > 0 else 0
        pv_profile = [p * pv_scale for p in pv_profile]
        
        # Load profile (higher in evening)
        load_profile = []
        for hour in hours:
            if 6 <= hour <= 22:
                # Base load with evening peak
                base_load = scenario['load_total'] / 24
                evening_boost = 0
                if 17 <= hour <= 21:
                    evening_boost = scenario['load_peak'] * np.exp(-0.5 * ((hour - scenario['load_peak_hour']) / 2) ** 2)
                
                load_power = base_load + evening_boost
                load_power += np.random.normal(0, base_load * 0.2)  # Add noise
                load_power = max(0, load_power)
            else:
                load_power = scenario['load_total'] / 24 * 0.3  # Lower night load
            load_profile.append(load_power)
        
        # Scale to match total
        load_scale = scenario['load_total'] / sum(load_profile) if sum(load_profile) > 0 else 0
        load_profile = [l * load_scale for l in load_profile]
        
        # Price profile (TOU-like)
        price_profile = []
        for hour in hours:
            if 8 <= hour <= 19:  # Peak hours
                price = scenario['price_buy_mean'] + scenario['price_buy_std']
            elif 7 <= hour <= 22:  # Flat hours
                price = scenario['price_buy_mean']
            else:  # Valley hours
                price = scenario['price_buy_mean'] - scenario['price_buy_std']
            
            price = max(0.1, price)  # Minimum price
            price_profile.append(price)
        
        return {
            'load_profile': load_profile,
            'pv_profile': pv_profile,
            'price_profile': price_profile,
            'price_sell': scenario['price_sell']
        }
    
    def _run_optimization(self, hourly_data: Dict) -> Optional[Dict]:
        """Run optimization for a scenario"""
        try:
            # Create temporary directory
            temp_dir = f"temp_surrogate_{np.random.randint(10000, 99999)}"
            os.makedirs(temp_dir, exist_ok=True)
            
            # Create data files
            load_data = pd.DataFrame({
                'hour': range(1, 25),
                'load_kw': hourly_data['load_profile']
            })
            load_data.to_csv(os.path.join(temp_dir, "load_24h.csv"), index=False)
            
            pv_data = pd.DataFrame({
                'hour': range(0, 24),
                'pv_kw': hourly_data['pv_profile']
            })
            pv_data.to_csv(os.path.join(temp_dir, "pv_24h.csv"), index=False)
            
            tou_data = pd.DataFrame({
                'hour': range(1, 25),
                'price_buy': hourly_data['price_profile'],
                'price_sell': [hourly_data['price_sell']] * 24
            })
            tou_data.to_csv(os.path.join(temp_dir, "tou_24h.csv"), index=False)
            
            # Create battery specs
            battery_specs = {
                'Ebat_kWh': 80,
                'Pch_max_kW': 40,
                'Pdis_max_kW': 40,
                'SOCmin': 0.20,
                'SOCmax': 0.95,
                'eta_ch': 0.90,
                'eta_dis': 0.90,
                'SOC0_frac': 0.50
            }
            
            import yaml
            with open(os.path.join(temp_dir, "battery.yaml"), 'w') as f:
                yaml.dump(battery_specs, f)
            
            # Run optimization
            result = subprocess.run([
                'python3', 'run_day.py', '--strategy', 'ALL'
            ], capture_output=True, text=True, cwd=temp_dir)
            
            # Clean up
            import shutil
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if result.returncode != 0:
                return None
            
            # Parse results
            costs = {}
            output_lines = result.stdout.split('\n')
            for line in output_lines:
                if 'Strategy:' in line and 'Cost:' in line:
                    parts = line.split()
                    strategy = parts[1]
                    cost = float(parts[3].replace('€', ''))
                    costs[strategy] = cost
            
            return costs
            
        except Exception as e:
            logger.warning(f"Optimization failed for scenario: {e}")
            return None

class SurrogateTrainer:
    """Trains surrogate models for cost prediction"""
    
    def __init__(self, config: SurrogateConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        
    def prepare_features(self, training_df: pd.DataFrame) -> List[str]:
        """Prepare feature columns for training"""
        # Exclude cost columns and other non-feature columns
        exclude_columns = [col for col in training_df.columns if col.startswith('cost_')]
        self.feature_columns = [col for col in training_df.columns if col not in exclude_columns]
        
        logger.info(f"Using {len(self.feature_columns)} features for training")
        return self.feature_columns
    
    def train_models(self, training_df: pd.DataFrame) -> Dict:
        """Train surrogate models for each strategy"""
        logger.info("Training surrogate models...")
        
        strategies = ['MSC', 'TOU', 'MMR', 'DRP2P']
        
        for strategy in strategies:
            cost_column = f'cost_{strategy}'
            if cost_column not in training_df.columns:
                logger.warning(f"Cost column {cost_column} not found, skipping {strategy}")
                continue
            
            logger.info(f"Training surrogate model for {strategy}")
            
            # Prepare data
            X = training_df[self.feature_columns].values
            y = training_df[cost_column].values
            
            # Remove rows with missing costs
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                logger.warning(f"No valid data for {strategy}, skipping")
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Train XGBoost model
            model = xgb.XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=self.config.random_state,
                n_jobs=-1
            )
            
            model.fit(X_train_scaled, y_train)
            
            # Evaluate model
            y_pred = model.predict(X_test_scaled)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"{strategy} - MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")
            
            # Store model and scaler
            self.models[strategy] = model
            self.scalers[strategy] = scaler
            
        return self.models
    
    def save_models(self, training_df: pd.DataFrame):
        """Save trained models and metadata"""
        logger.info(f"Saving models to {self.config.models_dir}")
        
        # Save individual models
        for strategy, model in self.models.items():
            model_file = os.path.join(self.config.models_dir, f"surrogate_{strategy}.joblib")
            joblib.dump(model, model_file)
            logger.info(f"Saved {strategy} model to {model_file}")
        
        # Save scalers
        for strategy, scaler in self.scalers.items():
            scaler_file = os.path.join(self.config.models_dir, f"scaler_{strategy}.joblib")
            joblib.dump(scaler, scaler_file)
            logger.info(f"Saved {strategy} scaler to {scaler_file}")
        
        # Save metadata
        metadata = {
            'feature_columns': self.feature_columns,
            'strategies': list(self.models.keys()),
            'training_samples': len(training_df),
            'training_date': datetime.now().isoformat(),
            'config': {
                'n_scenarios': self.config.n_scenarios,
                'random_state': self.config.random_state,
                'test_size': self.config.test_size
            }
        }
        
        meta_file = os.path.join(self.config.models_dir, "meta.json")
        with open(meta_file, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Saved metadata to {meta_file}")
    
    def create_validation_plots(self, training_df: pd.DataFrame):
        """Create validation plots"""
        logger.info("Creating validation plots...")
        
        strategies = list(self.models.keys())
        n_strategies = len(strategies)
        
        if n_strategies == 0:
            logger.warning("No models to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, strategy in enumerate(strategies):
            if i >= 4:
                break
            
            cost_column = f'cost_{strategy}'
            if cost_column not in training_df.columns:
                continue
            
            # Prepare data
            X = training_df[self.feature_columns].values
            y = training_df[cost_column].values
            
            # Remove rows with missing costs
            valid_mask = ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) == 0:
                continue
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.config.test_size, random_state=self.config.random_state
            )
            
            # Scale features
            X_test_scaled = self.scalers[strategy].transform(X_test)
            
            # Predict
            y_pred = self.models[strategy].predict(X_test_scaled)
            
            # Plot
            axes[i].scatter(y_test, y_pred, alpha=0.6)
            axes[i].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            axes[i].set_xlabel('Actual Cost (€)')
            axes[i].set_ylabel('Predicted Cost (€)')
            axes[i].set_title(f'{strategy} Strategy')
            axes[i].grid(True, alpha=0.3)
            
            # Add R² score
            r2 = r2_score(y_test, y_pred)
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Hide unused subplots
        for i in range(n_strategies, 4):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plot_file = os.path.join(self.config.results_dir, 'surrogate_validation.png')
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved validation plots to {plot_file}")

def main():
    """Main function for surrogate model training"""
    parser = argparse.ArgumentParser(description='Train surrogate models for cost prediction')
    parser.add_argument('--data-dir', type=str, default='data', help='Data directory')
    parser.add_argument('--models-dir', type=str, default='models/surrogate', help='Models directory')
    parser.add_argument('--results-dir', type=str, default='results/figs_forecast', help='Results directory')
    parser.add_argument('--n-scenarios', type=int, default=1000, help='Number of training scenarios')
    parser.add_argument('--random-state', type=int, default=42, help='Random state')
    parser.add_argument('--test-size', type=float, default=0.2, help='Test set size')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SurrogateConfig(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        results_dir=args.results_dir,
        n_scenarios=args.n_scenarios,
        random_state=args.random_state,
        test_size=args.test_size
    )
    
    # Create directories
    os.makedirs(config.models_dir, exist_ok=True)
    os.makedirs(config.results_dir, exist_ok=True)
    
    logger.info("Starting surrogate model training...")
    
    try:
        # Initialize scenario generator
        generator = ScenarioGenerator(config)
        
        # Generate scenarios
        scenarios_df = generator.generate_scenarios()
        
        # Create training data
        training_df = generator.create_training_data(scenarios_df)
        
        if len(training_df) == 0:
            logger.error("No training data generated")
            return
        
        # Initialize trainer
        trainer = SurrogateTrainer(config)
        
        # Prepare features
        trainer.prepare_features(training_df)
        
        # Train models
        models = trainer.train_models(training_df)
        
        # Save models
        trainer.save_models(training_df)
        
        # Create validation plots
        trainer.create_validation_plots(training_df)
        
        logger.info("Surrogate model training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main()
