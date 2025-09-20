#!/usr/bin/env python3
"""
Surrogate Model Prediction Script

Uses trained surrogate models to predict optimization costs
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
import json
import joblib
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('surrogate_prediction.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class SurrogatePredictionConfig:
    """Configuration for surrogate model prediction"""
    models_dir: str = "models/surrogate"
    input_file: str = None
    output_file: str = None

class SurrogatePredictor:
    """Predicts costs using trained surrogate models"""
    
    def __init__(self, config: SurrogatePredictionConfig):
        self.config = config
        self.models = {}
        self.scalers = {}
        self.feature_columns = []
        self.load_models()
    
    def load_models(self):
        """Load trained surrogate models"""
        try:
            # Load metadata
            meta_file = os.path.join(self.config.models_dir, "meta.json")
            with open(meta_file, 'r') as f:
                metadata = json.load(f)
            
            self.feature_columns = metadata['feature_columns']
            strategies = metadata['strategies']
            
            # Load models and scalers
            for strategy in strategies:
                # Load model
                model_file = os.path.join(self.config.models_dir, f"surrogate_{strategy}.joblib")
                self.models[strategy] = joblib.load(model_file)
                
                # Load scaler
                scaler_file = os.path.join(self.config.models_dir, f"scaler_{strategy}.joblib")
                self.scalers[strategy] = joblib.load(scaler_file)
            
            logger.info(f"Loaded surrogate models for strategies: {strategies}")
            
        except Exception as e:
            logger.error(f"Failed to load surrogate models: {e}")
            raise
    
    def predict_costs(self, features: pd.DataFrame) -> pd.DataFrame:
        """Predict costs for given features"""
        logger.info("Predicting costs using surrogate models...")
        
        # Ensure all required features are present
        missing_features = [col for col in self.feature_columns if col not in features.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Prepare features
        X = features[self.feature_columns].values
        
        # Make predictions for each strategy
        predictions = features.copy()
        
        for strategy, model in self.models.items():
            # Scale features
            X_scaled = self.scalers[strategy].transform(X)
            
            # Predict
            cost_predictions = model.predict(X_scaled)
            
            # Add to results
            predictions[f'predicted_cost_{strategy}'] = cost_predictions
        
        logger.info("Cost predictions completed")
        return predictions
    
    def predict_from_scenario(self, scenario: Dict) -> Dict:
        """Predict costs from a single scenario dictionary"""
        # Convert scenario to DataFrame
        scenario_df = pd.DataFrame([scenario])
        
        # Ensure all required features are present
        missing_features = [col for col in self.feature_columns if col not in scenario_df.columns]
        if missing_features:
            logger.error(f"Missing features: {missing_features}")
            raise ValueError(f"Missing required features: {missing_features}")
        
        # Make predictions
        predictions = self.predict_costs(scenario_df)
        
        # Return as dictionary
        result = {}
        for strategy in self.models.keys():
            result[f'cost_{strategy}'] = predictions[f'predicted_cost_{strategy}'].iloc[0]
        
        return result
    
    def create_sample_scenario(self) -> Dict:
        """Create a sample scenario for testing"""
        return {
            # PV features
            'pv_total': 50.0,
            'pv_peak': 25.0,
            'pv_ramp_up': 1.0,
            'pv_ramp_down': 1.0,
            'pv_peak_hour': 13,
            
            # Load features
            'load_total': 100.0,
            'load_peak': 20.0,
            'load_ramp_up': 2.0,
            'load_ramp_down': 2.0,
            'load_peak_hour': 19,
            
            # Price features
            'price_buy_mean': 0.35,
            'price_buy_std': 0.10,
            'price_sell': 0.10,
            'price_ratio': 3.5,
            
            # Temporal features
            'day_of_week': 1,  # Tuesday
            'is_weekend': 0,
            'month': 6,  # June
            'season': 2,  # Summer
            
            # Weather features
            'temp_mean': 25.0,
            'temp_std': 5.0,
            'ghi_total': 4000.0,
            
            # Battery features
            'battery_capacity': 80.0,
            'battery_power': 40.0,
            
            # Derived features
            'pv_load_ratio': 0.5,
            'net_load': 50.0,
            'self_consumption_potential': 50.0,
            'peak_demand': 25.0,
            'peak_net_demand': 0.0,
            'max_ramp': 2.0,
            'pv_load_ramp_ratio': 0.5
        }
    
    def save_predictions(self, predictions: pd.DataFrame, output_file: str):
        """Save predictions to file"""
        predictions.to_csv(output_file, index=False)
        logger.info(f"Saved predictions to {output_file}")

def main():
    """Main function for surrogate model prediction"""
    parser = argparse.ArgumentParser(description='Predict costs using surrogate models')
    parser.add_argument('--models-dir', type=str, default='models/surrogate', help='Models directory')
    parser.add_argument('--input-file', type=str, help='Input CSV file with features')
    parser.add_argument('--output-file', type=str, help='Output CSV file for predictions')
    parser.add_argument('--sample', action='store_true', help='Run prediction on sample scenario')
    
    args = parser.parse_args()
    
    # Create configuration
    config = SurrogatePredictionConfig(
        models_dir=args.models_dir,
        input_file=args.input_file,
        output_file=args.output_file
    )
    
    logger.info("Starting surrogate model prediction...")
    
    try:
        # Initialize predictor
        predictor = SurrogatePredictor(config)
        
        if args.sample:
            # Run prediction on sample scenario
            logger.info("Running prediction on sample scenario...")
            sample_scenario = predictor.create_sample_scenario()
            predictions = predictor.predict_from_scenario(sample_scenario)
            
            print("\n" + "="*60)
            print("SAMPLE SCENARIO PREDICTIONS")
            print("="*60)
            print(f"{'Strategy':<15} {'Predicted Cost (€)':<20}")
            print("-"*40)
            for strategy, cost in predictions.items():
                strategy_name = strategy.replace('cost_', '')
                print(f"{strategy_name:<15} {cost:<20.2f}")
            print("="*60)
            
        elif args.input_file:
            # Load input data
            if not os.path.exists(args.input_file):
                logger.error(f"Input file not found: {args.input_file}")
                return
            
            features_df = pd.read_csv(args.input_file)
            logger.info(f"Loaded {len(features_df)} samples from {args.input_file}")
            
            # Make predictions
            predictions = predictor.predict_costs(features_df)
            
            # Save predictions
            if args.output_file:
                predictor.save_predictions(predictions, args.output_file)
            else:
                # Print summary
                print("\n" + "="*80)
                print("PREDICTION SUMMARY")
                print("="*80)
                
                for strategy in predictor.models.keys():
                    cost_col = f'predicted_cost_{strategy}'
                    if cost_col in predictions.columns:
                        mean_cost = predictions[cost_col].mean()
                        std_cost = predictions[cost_col].std()
                        print(f"{strategy:<15} Mean: {mean_cost:.2f} ± {std_cost:.2f} €")
                
                print("="*80)
        
        else:
            logger.error("Please specify either --input-file or --sample")
            return
        
        logger.info("Surrogate model prediction completed successfully!")
        
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise

if __name__ == "__main__":
    main()
