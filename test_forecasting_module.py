#!/usr/bin/env python3
"""
Forecasting Module Test Script

Demonstrates the complete forecasting and integration module
with synthetic data and basic functionality testing.

Author: Energy Management System
Date: 2024
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
import subprocess
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_test_data():
    """Create synthetic test data for the forecasting module"""
    logger.info("Creating synthetic test data...")
    
    # Create directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('forecast', exist_ok=True)
    
    # Create load_8760.csv
    load_data = []
    for day in range(1, 366):
        for hour in range(1, 25):
            # Create realistic load pattern
            base_load = 3.0  # kW
            daily_variation = 1.5 * np.sin(2 * np.pi * (hour - 6) / 24)
            seasonal_variation = 0.5 * np.sin(2 * np.pi * (day - 80) / 365)
            noise = np.random.normal(0, 0.3)
            
            load_kw = base_load + daily_variation + seasonal_variation + noise
            load_kw = max(0, load_kw)
            
            load_data.append({
                'day': day,
                'hour': hour,
                'load_kw': load_kw
            })
    
    load_df = pd.DataFrame(load_data)
    load_df.to_csv('data/load_8760.csv', index=False)
    logger.info("Created data/load_8760.csv")
    
    # Create pv_8760.csv
    pv_data = []
    for day in range(1, 366):
        for hour in range(1, 25):
            # Create realistic PV pattern
            if 6 <= hour <= 18:
                # Bell curve around noon
                pv_power = 20 * np.exp(-0.5 * ((hour - 12) / 4) ** 2)
                # Seasonal variation
                seasonal_factor = 0.5 + 0.5 * np.sin(2 * np.pi * (day - 80) / 365)
                pv_power *= seasonal_factor
                # Add noise
                pv_power += np.random.normal(0, 1)
                pv_power = max(0, pv_power)
            else:
                pv_power = 0
            
            pv_data.append({
                'day': day,
                'hour': hour,
                'pv_kw': pv_power
            })
    
    pv_df = pd.DataFrame(pv_data)
    pv_df.to_csv('data/pv_8760.csv', index=False)
    logger.info("Created data/pv_8760.csv")
    
    # Create weather_8760.csv
    weather_data = []
    for day in range(1, 366):
        for hour in range(1, 25):
            # Create realistic temperature pattern
            day_of_year = day
            base_temp = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
            daily_variation = 5 * np.sin(2 * np.pi * (hour - 6) / 24)
            temp_C = base_temp + daily_variation + np.random.normal(0, 2)
            
            # Create GHI pattern
            if 6 <= hour <= 18:
                ghi_Wm2 = 800 * np.sin(np.pi * (hour - 6) / 12)
                ghi_Wm2 += np.random.normal(0, 50)
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
    weather_df.to_csv('data/weather_8760.csv', index=False)
    logger.info("Created data/weather_8760.csv")
    
    # Create holidays.csv
    holidays_data = []
    for day in range(1, 366):
        date = datetime(2024, 1, 1) + timedelta(days=day-1)
        is_holiday = False
        
        # New Year
        if day == 1:
            is_holiday = True
        # Christmas
        elif day == 360:
            is_holiday = True
        # Some random holidays
        elif day in [50, 100, 150, 200, 250, 300]:
            is_holiday = True
        
        holidays_data.append({
            'day': day,
            'is_holiday': is_holiday
        })
    
    holidays_df = pd.DataFrame(holidays_data)
    holidays_df.to_csv('data/holidays.csv', index=False)
    logger.info("Created data/holidays.csv")
    
    # Create battery.yaml
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
    with open('data/battery.yaml', 'w') as f:
        yaml.dump(battery_specs, f)
    logger.info("Created data/battery.yaml")

def test_load_forecasting():
    """Test load forecasting training"""
    logger.info("Testing load forecasting training...")
    
    try:
        result = subprocess.run([
            'python3', 'train_forecast_load.py', '--cv-splits', '3'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ Load forecasting training completed successfully")
            return True
        else:
            logger.error(f"❌ Load forecasting training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Load forecasting training timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Load forecasting training error: {e}")
        return False

def test_pv_forecasting():
    """Test PV forecasting training"""
    logger.info("Testing PV forecasting training...")
    
    try:
        result = subprocess.run([
            'python3', 'train_forecast_pv.py', '--cv-splits', '3'
        ], capture_output=True, text=True, timeout=300)
        
        if result.returncode == 0:
            logger.info("✅ PV forecasting training completed successfully")
            return True
        else:
            logger.error(f"❌ PV forecasting training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ PV forecasting training timed out")
        return False
    except Exception as e:
        logger.error(f"❌ PV forecasting training error: {e}")
        return False

def test_next_day_forecast():
    """Test next day forecasting"""
    logger.info("Testing next day forecasting...")
    
    # Create weather forecast for tomorrow
    tomorrow = datetime.now() + timedelta(days=1)
    weather_forecast = []
    
    for hour in range(1, 25):
        temp_C = 20 + 5 * np.sin(2 * np.pi * (hour - 6) / 24) + np.random.normal(0, 1)
        ghi_Wm2 = 600 * np.sin(np.pi * (hour - 6) / 12) if 6 <= hour <= 18 else 0
        ghi_Wm2 = max(0, ghi_Wm2)
        
        weather_forecast.append({
            'hour': hour,
            'temp_C': temp_C,
            'ghi_Wm2': ghi_Wm2
        })
    
    weather_df = pd.DataFrame(weather_forecast)
    weather_file = f"forecast/weather_nextday_{tomorrow.strftime('%Y%m%d')}.csv"
    weather_df.to_csv(weather_file, index=False)
    logger.info(f"Created {weather_file}")
    
    try:
        result = subprocess.run([
            'python3', 'forecast_next_day.py', '--date', tomorrow.strftime('%Y-%m-%d')
        ], capture_output=True, text=True, timeout=120)
        
        if result.returncode == 0:
            logger.info("✅ Next day forecasting completed successfully")
            logger.info("Forecast output:")
            print(result.stdout)
            return True
        else:
            logger.error(f"❌ Next day forecasting failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Next day forecasting timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Next day forecasting error: {e}")
        return False

def test_surrogate_models():
    """Test surrogate model training and prediction"""
    logger.info("Testing surrogate models...")
    
    # Test training (with fewer scenarios for speed)
    try:
        result = subprocess.run([
            'python3', 'train_surrogate.py', '--n-scenarios', '50'
        ], capture_output=True, text=True, timeout=600)
        
        if result.returncode == 0:
            logger.info("✅ Surrogate model training completed successfully")
            
            # Test prediction
            result = subprocess.run([
                'python3', 'predict_surrogate.py', '--sample'
            ], capture_output=True, text=True, timeout=60)
            
            if result.returncode == 0:
                logger.info("✅ Surrogate model prediction completed successfully")
                logger.info("Prediction output:")
                print(result.stdout)
                return True
            else:
                logger.error(f"❌ Surrogate model prediction failed: {result.stderr}")
                return False
        else:
            logger.error(f"❌ Surrogate model training failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("❌ Surrogate model testing timed out")
        return False
    except Exception as e:
        logger.error(f"❌ Surrogate model testing error: {e}")
        return False

def main():
    """Main test function"""
    print("="*80)
    print("FORECASTING MODULE TEST SUITE")
    print("="*80)
    
    # Create test data
    create_test_data()
    
    # Test results
    results = {}
    
    # Test load forecasting
    results['load_forecasting'] = test_load_forecasting()
    
    # Test PV forecasting
    results['pv_forecasting'] = test_pv_forecasting()
    
    # Test next day forecast
    results['next_day_forecast'] = test_next_day_forecast()
    
    # Test surrogate models
    results['surrogate_models'] = test_surrogate_models()
    
    # Print summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:<25} {status}")
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"\nTotal: {passed_tests}/{total_tests} tests passed")
    
    if passed_tests == total_tests:
        print("\n🎉 ALL TESTS PASSED! Forecasting module is working correctly.")
    else:
        print(f"\n⚠️  {total_tests - passed_tests} tests failed. Check logs for details.")
    
    print("="*80)

if __name__ == "__main__":
    main()
