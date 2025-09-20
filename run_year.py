#!/usr/bin/env python3
"""
Step 3: Full-Year Energy Management Simulation (365 days)
Extends the Step 2 24-hour optimizer to run across the entire year

This script:
1. Loads 8760-hour load and PV data
2. Runs daily optimization for each strategy (MSC, TOU, MMR, DR-P2P)
3. Generates comprehensive annual results for Step 4 clustering analysis

Usage:
    python3 run_year.py [--data-dir DATA_DIR] [--results-dir RESULTS_DIR] [--strategies STRATEGIES] [--parallel]
"""

import os
import sys
import argparse
import time
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import pandas as pd
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

# Import our Step 2 components
from run_day import EnergyOptimizer, DataValidator, OptimizationResult
from strategy_adapter import StrategyType, StrategyAdapter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('run_year.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

@dataclass
class YearlySimulationConfig:
    """Configuration for yearly simulation"""
    data_dir: str = "project/data"
    results_dir: str = "results"
    strategies: List[StrategyType] = None
    parallel: bool = False
    max_workers: int = None
    skip_failed_days: bool = True
    progress_interval: int = 10  # Print progress every N days
    
    def __post_init__(self):
        if self.strategies is None:
            self.strategies = [StrategyType.MSC, StrategyType.TOU, StrategyType.MMR_P2P, StrategyType.DR_P2P]
        if self.max_workers is None:
            self.max_workers = min(4, mp.cpu_count())

class TOUTariffHelper:
    """Helper class to compute TOU tariffs based on day of week and hour"""
    
    def __init__(self):
        # Italian ARERA F1/F2/F3 tariff structure
        self.tariff_bands = {
            'weekday': {
                'F1_peak': (8, 19),    # 8:00-19:00 (hours 8-19)
                'F2_flat': [(7, 8), (19, 23)],  # 7:00-8:00 and 19:00-23:00
                'F3_valley': [(23, 24), (0, 7)]  # 23:00-24:00 and 0:00-7:00
            },
            'saturday': {
                'F2_flat': (7, 23),    # 7:00-23:00
                'F3_valley': [(23, 24), (0, 7)]  # 23:00-24:00 and 0:00-7:00
            },
            'sunday': {
                'F3_valley': (0, 24)   # All day
            }
        }
        
        # Price levels (€/kWh)
        self.prices = {
            'F1': 0.48,  # Peak
            'F2': 0.34,  # Flat
            'F3': 0.24,  # Valley
            'sell': 0.10  # Feed-in tariff
        }
    
    def get_day_type(self, day_of_year: int) -> str:
        """Determine if a day is weekday, Saturday, or Sunday/holiday"""
        # For simplicity, assume day 1 is Monday
        # In practice, you'd use actual calendar dates
        day_of_week = (day_of_year - 1) % 7
        
        if day_of_week < 5:  # Monday-Friday
            return 'weekday'
        elif day_of_week == 5:  # Saturday
            return 'saturday'
        else:  # Sunday
            return 'sunday'
    
    def get_tariff_vector(self, day_of_year: int) -> Tuple[List[float], List[float]]:
        """Get 24-hour tariff vectors for a given day"""
        day_type = self.get_day_type(day_of_year)
        buy_prices = []
        sell_prices = []
        
        for hour in range(1, 25):  # Hours 1-24
            price_buy = self._get_hourly_price(day_type, hour)
            price_sell = self.prices['sell']
            
            buy_prices.append(price_buy)
            sell_prices.append(price_sell)
        
        return buy_prices, sell_prices
    
    def _get_hourly_price(self, day_type: str, hour: int) -> float:
        """Get price for a specific hour and day type"""
        bands = self.tariff_bands[day_type]
        
        # Convert hour to 0-23 format for easier comparison
        hour_24 = hour - 1 if hour > 0 else 23
        
        if day_type == 'weekday':
            # Check F1 peak hours (8-19)
            if 8 <= hour <= 19:
                return self.prices['F1']
            # Check F2 flat hours
            elif (7 <= hour <= 8) or (19 <= hour <= 23):
                return self.prices['F2']
            # F3 valley hours
            else:
                return self.prices['F3']
        
        elif day_type == 'saturday':
            # Check F2 flat hours (7-23)
            if 7 <= hour <= 23:
                return self.prices['F2']
            # F3 valley hours
            else:
                return self.prices['F3']
        
        else:  # sunday
            # All F3 valley
            return self.prices['F3']

class YearlyDataLoader:
    """Load and validate yearly data (8760 hours)"""
    
    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.validator = DataValidator()
    
    def load_yearly_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
        """Load load_8760.csv, pv_8760.csv, and battery.yaml"""
        logger.info("Loading yearly data...")
        
        # Load load data
        load_path = os.path.join(self.data_dir, "load_8760.csv")
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Load data not found: {load_path}")
        
        load_df = pd.read_csv(load_path)
        logger.info(f"Loaded load data: {len(load_df)} rows")
        
        # Validate load data
        if not DataValidator.validate_yearly_csv(load_df, "load_8760.csv"):
            raise ValueError("Load data validation failed")
        
        # Load PV data
        pv_path = os.path.join(self.data_dir, "pv_8760.csv")
        if not os.path.exists(pv_path):
            raise FileNotFoundError(f"PV data not found: {pv_path}")
        
        pv_df = pd.read_csv(pv_path)
        logger.info(f"Loaded PV data: {len(pv_df)} rows")
        
        # Validate PV data
        if not DataValidator.validate_yearly_csv(pv_df, "pv_8760.csv"):
            raise ValueError("PV data validation failed")
        
        # Load battery specs
        battery_path = os.path.join(self.data_dir, "battery.yaml")
        if not os.path.exists(battery_path):
            raise FileNotFoundError(f"Battery specs not found: {battery_path}")
        
        import yaml
        with open(battery_path, 'r') as f:
            battery_specs = yaml.safe_load(f)
        
        logger.info("All yearly data loaded successfully")
        return load_df, pv_df, battery_specs
    
    def extract_daily_data(self, load_df: pd.DataFrame, pv_df: pd.DataFrame, 
                          day: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract 24-hour data for a specific day"""
        # Filter data for the specific day
        day_load = load_df[load_df['day'] == day].copy()
        day_pv = pv_df[pv_df['day'] == day].copy()
        
        # Ensure we have exactly 24 hours
        if len(day_load) != 24:
            raise ValueError(f"Expected 24 hours for day {day}, got {len(day_load)}")
        if len(day_pv) != 24:
            raise ValueError(f"Expected 24 hours for day {day}, got {len(day_pv)}")
        
        # Rename columns to match Step 2 format
        day_load = day_load.rename(columns={'load_kw': 'load_kw'})
        day_pv = day_pv.rename(columns={'pv_kw': 'pv_generation_kw'})
        
        # Add hour column (1-24)
        day_load['hour'] = range(1, 25)
        day_pv['hour'] = range(0, 24)  # PV uses 0-23 format
        
        return day_load, day_pv

def run_single_day_optimization(day: int, strategy: StrategyType, load_df: pd.DataFrame, 
                               pv_df: pd.DataFrame, battery_specs: Dict,
                               tariff_helper: TOUTariffHelper, results_dir: str) -> Optional[Dict]:
    """Run optimization for a single day and strategy"""
    try:
        logger.info(f"Processing Day {day}, Strategy {strategy.value}")
        
        # Extract daily data
        data_loader = YearlyDataLoader("")  # Empty data_dir since we pass data directly
        day_load, day_pv = data_loader.extract_daily_data(load_df, pv_df, day)
        
        # Get tariff vector for this day
        buy_prices, sell_prices = tariff_helper.get_tariff_vector(day)
        
        # Create TOU data for this day
        tou_data = pd.DataFrame({
            'hour': range(1, 25),
            'price_buy': buy_prices,
            'price_sell': sell_prices
        })
        
        # Create temporary data files for this day
        temp_dir = f"temp_day_{day}_{strategy.value}"
        os.makedirs(temp_dir, exist_ok=True)
        
        # Save daily data to temporary files
        day_load.to_csv(os.path.join(temp_dir, "load_24h.csv"), index=False)
        day_pv.to_csv(os.path.join(temp_dir, "pv_24h.csv"), index=False)
        tou_data.to_csv(os.path.join(temp_dir, "tou_24h.csv"), index=False)
        
        import yaml
        with open(os.path.join(temp_dir, "battery.yaml"), 'w') as f:
            yaml.dump(battery_specs, f)
        
        # Run optimization using EnergyOptimizer
        optimizer = EnergyOptimizer(data_dir=temp_dir)
        
        if not optimizer.load_and_validate_data():
            logger.warning(f"Data validation failed for Day {day}, Strategy {strategy.value}")
            return None
        
        result = optimizer.run_optimization(strategy)
        
        if result is None or not result.is_optimal:
            logger.warning(f"Optimization failed for Day {day}, Strategy {strategy.value}")
            return None
        
        # Save hourly results
        hourly_filename = f"hourly_{strategy.value}_day{day:03d}.csv"
        hourly_path = os.path.join(results_dir, "hourly", hourly_filename)
        os.makedirs(os.path.dirname(hourly_path), exist_ok=True)
        result.hourly_results.to_csv(hourly_path, index=False)
        
        # Calculate daily KPIs
        daily_kpis = calculate_daily_kpis(result.hourly_results, strategy, day)
        
        # Clean up temporary directory
        import shutil
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        logger.info(f"Day {day}, Strategy {strategy.value}: Cost €{daily_kpis['Cost_total']:.2f}")
        return daily_kpis
        
    except Exception as e:
        logger.error(f"Error processing Day {day}, Strategy {strategy.value}: {e}")
        return None

def calculate_daily_kpis(hourly_results: pd.DataFrame, strategy: StrategyType, day: int) -> Dict:
    """Calculate daily KPIs from hourly results"""
    kpis = {
        'day': day,
        'Strategy': strategy.value,
        'Cost_total': hourly_results['cost_hour'].sum(),
        'Import_total': hourly_results['grid_in'].sum(),
        'Export_total': hourly_results['grid_out'].sum(),
        'PV_total': hourly_results['pv'].sum(),
        'Load_total': hourly_results['load'].sum(),
        'curtail_total': hourly_results['curtail'].sum(),
        'pv_self': hourly_results['pv'].sum() - hourly_results['grid_out'].sum() - hourly_results['curtail'].sum(),
        'SCR': 0.0,  # Will calculate below
        'SelfSufficiency': 0.0,  # Will calculate below
        'PeakGrid': hourly_results['grid_in'].max(),
        'BatteryCycles': 0.0  # Will calculate below
    }
    
    # Calculate SCR (Self-Consumption Rate)
    if kpis['PV_total'] > 0:
        kpis['SCR'] = kpis['pv_self'] / kpis['PV_total']
    
    # Calculate Self-Sufficiency
    if kpis['Load_total'] > 0:
        battery_discharge = hourly_results['batt_dis'].sum()
        kpis['SelfSufficiency'] = (kpis['pv_self'] + battery_discharge) / kpis['Load_total']
    
    # Calculate Battery Cycles (simplified)
    if 'batt_dis' in hourly_results.columns:
        total_discharge = hourly_results['batt_dis'].sum()
        # Assuming 80 kWh battery capacity
        kpis['BatteryCycles'] = total_discharge / 80.0
    
    return kpis

def run_yearly_simulation(config: YearlySimulationConfig) -> bool:
    """Run the complete yearly simulation"""
    logger.info("Starting yearly energy management simulation...")
    logger.info(f"Configuration: {config}")
    
    # Create results directory
    os.makedirs(config.results_dir, exist_ok=True)
    os.makedirs(os.path.join(config.results_dir, "hourly"), exist_ok=True)
    
    # Load yearly data
    data_loader = YearlyDataLoader(config.data_dir)
    load_df, pv_df, battery_specs = data_loader.load_yearly_data()
    
    # Initialize tariff helper
    tariff_helper = TOUTariffHelper()
    
    # Initialize results storage
    all_daily_kpis = []
    failed_days = []
    
    start_time = time.time()
    
    if config.parallel:
        logger.info(f"Running parallel simulation with {config.max_workers} workers")
        run_parallel_simulation(config, load_df, pv_df, battery_specs, tariff_helper, all_daily_kpis, failed_days)
    else:
        logger.info("Running sequential simulation")
        run_sequential_simulation(config, load_df, pv_df, battery_specs, tariff_helper, all_daily_kpis, failed_days)
    
    # Save daily KPIs
    if all_daily_kpis:
        kpis_df = pd.DataFrame(all_daily_kpis)
        kpis_path = os.path.join(config.results_dir, "kpis.csv")
        kpis_df.to_csv(kpis_path, index=False)
        logger.info(f"Saved daily KPIs: {len(all_daily_kpis)} records to {kpis_path}")
    
    # Print summary
    total_time = time.time() - start_time
    logger.info(f"Yearly simulation completed in {total_time:.2f} seconds")
    logger.info(f"Successful days: {len(all_daily_kpis) // len(config.strategies)}")
    logger.info(f"Failed days: {len(failed_days)}")
    
    if failed_days:
        logger.warning(f"Failed days: {failed_days}")
    
    return len(failed_days) == 0

def run_sequential_simulation(config: YearlySimulationConfig, load_df: pd.DataFrame, 
                            pv_df: pd.DataFrame, battery_specs: Dict,
                            tariff_helper: TOUTariffHelper, all_daily_kpis: List, failed_days: List):
    """Run simulation sequentially (day by day)"""
    total_days = 365
    total_optimizations = total_days * len(config.strategies)
    completed = 0
    
    for day in range(1, total_days + 1):
        day_failed = False
        
        for strategy in config.strategies:
            kpis = run_single_day_optimization(
                day, strategy, load_df, pv_df, battery_specs, 
                tariff_helper, config.results_dir
            )
            
            if kpis is not None:
                all_daily_kpis.append(kpis)
            else:
                day_failed = True
                if not config.skip_failed_days:
                    raise RuntimeError(f"Day {day} failed and skip_failed_days=False")
            
            completed += 1
            
            # Progress reporting
            if completed % config.progress_interval == 0:
                progress = (completed / total_optimizations) * 100
                logger.info(f"Progress: {completed}/{total_optimizations} ({progress:.1f}%)")
        
        if day_failed:
            failed_days.append(day)
            logger.warning(f"Day {day} had failures")

def run_parallel_simulation(config: YearlySimulationConfig, load_df: pd.DataFrame, 
                          pv_df: pd.DataFrame, battery_specs: Dict,
                          tariff_helper: TOUTariffHelper, all_daily_kpis: List, failed_days: List):
    """Run simulation in parallel (day by day)"""
    total_days = 365
    total_optimizations = total_days * len(config.strategies)
    completed = 0
    
    # Create tasks
    tasks = []
    for day in range(1, total_days + 1):
        for strategy in config.strategies:
            tasks.append((day, strategy))
    
    # Run in parallel
    with ProcessPoolExecutor(max_workers=config.max_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(
                run_single_day_optimization, day, strategy, load_df, pv_df, 
                battery_specs, tariff_helper, config.results_dir
            ): (day, strategy) for day, strategy in tasks
        }
        
        # Collect results
        for future in as_completed(future_to_task):
            day, strategy = future_to_task[future]
            
            try:
                kpis = future.result()
                if kpis is not None:
                    all_daily_kpis.append(kpis)
                else:
                    failed_days.append(day)
            except Exception as e:
                logger.error(f"Task failed for Day {day}, Strategy {strategy.value}: {e}")
                failed_days.append(day)
            
            completed += 1
            
            # Progress reporting
            if completed % config.progress_interval == 0:
                progress = (completed / total_optimizations) * 100
                logger.info(f"Progress: {completed}/{total_optimizations} ({progress:.1f}%)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Run yearly energy management simulation")
    parser.add_argument("--data-dir", default="project/data", 
                       help="Directory containing input data files")
    parser.add_argument("--results-dir", default="results", 
                       help="Directory to save results")
    parser.add_argument("--strategies", nargs="+", 
                       choices=["MSC", "TOU", "MMR_P2P", "DR_P2P", "ALL"],
                       default=["ALL"],
                       help="Strategies to run")
    parser.add_argument("--parallel", action="store_true",
                       help="Run simulation in parallel")
    parser.add_argument("--max-workers", type=int, default=None,
                       help="Maximum number of parallel workers")
    parser.add_argument("--skip-failed-days", action="store_true", default=True,
                       help="Skip days that fail optimization")
    parser.add_argument("--progress-interval", type=int, default=10,
                       help="Print progress every N optimizations")
    
    args = parser.parse_args()
    
    # Parse strategies
    strategy_mapping = {
        "MSC": StrategyType.MSC,
        "TOU": StrategyType.TOU,
        "MMR_P2P": StrategyType.MMR_P2P,
        "DR_P2P": StrategyType.DR_P2P
    }
    
    if "ALL" in args.strategies:
        strategies = [StrategyType.MSC, StrategyType.TOU, StrategyType.MMR_P2P, StrategyType.DR_P2P]
    else:
        strategies = [strategy_mapping[s] for s in args.strategies if s in strategy_mapping]
    
    # Create configuration
    config = YearlySimulationConfig(
        data_dir=args.data_dir,
        results_dir=args.results_dir,
        strategies=strategies,
        parallel=args.parallel,
        max_workers=args.max_workers,
        skip_failed_days=args.skip_failed_days,
        progress_interval=args.progress_interval
    )
    
    # Run simulation
    try:
        success = run_yearly_simulation(config)
        if success:
            logger.info("✅ Yearly simulation completed successfully!")
            sys.exit(0)
        else:
            logger.error("❌ Yearly simulation completed with errors")
            sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Yearly simulation failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
