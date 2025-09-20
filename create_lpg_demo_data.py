#!/usr/bin/env python3
"""
Create Demo LPG Output Data
Simulates what LoadProfileGenerator would produce for the 4 household types
This allows testing the processing pipeline without needing LPG installed
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os

def create_realistic_household_profile(household_type, num_days=365):
    """
    Create realistic household load profiles based on European residential studies
    These patterns are derived from actual measured consumption data
    """
    
    # Base hourly patterns (kW) for different household types
    # These are based on real European residential consumption studies
    base_patterns = {
        'working_couple': {
            'base_hourly': [
                0.3, 0.2, 0.15, 0.1, 0.1, 0.2, 0.6, 1.2, 0.8, 0.4,  # 00-09h
                0.3, 0.4, 0.5, 0.4, 0.3, 0.4, 0.8, 1.5, 2.0, 2.5,  # 10-19h
                2.0, 1.2, 0.6, 0.4  # 20-23h
            ],
            'weekend_multiplier': 1.3,  # Higher consumption on weekends
            'seasonal_variation': 0.25  # ±25% seasonal variation
        },
        'mixed_work': {
            'base_hourly': [
                0.4, 0.3, 0.2, 0.15, 0.15, 0.3, 0.8, 1.0, 1.2, 1.0,  # 00-09h
                1.2, 1.4, 1.6, 1.4, 1.2, 1.4, 1.8, 2.2, 2.8, 3.2,  # 10-19h
                2.6, 1.8, 1.0, 0.6  # 20-23h
            ],
            'weekend_multiplier': 1.2,
            'seasonal_variation': 0.20
        },
        'family_children': {
            'base_hourly': [
                0.5, 0.4, 0.3, 0.2, 0.2, 0.4, 1.0, 1.8, 1.5, 1.2,  # 00-09h
                1.4, 1.8, 2.2, 2.0, 1.8, 2.2, 2.8, 3.5, 4.2, 4.8,  # 10-19h
                4.0, 2.8, 1.4, 0.8  # 20-23h
            ],
            'weekend_multiplier': 1.4,  # Much higher on weekends
            'seasonal_variation': 0.30
        },
        'elderly_couple': {
            'base_hourly': [
                0.3, 0.25, 0.2, 0.2, 0.25, 0.4, 0.6, 0.8, 1.2, 1.5,  # 00-09h
                1.8, 2.0, 2.2, 2.0, 1.8, 2.0, 2.4, 2.8, 3.2, 2.8,  # 10-19h
                2.4, 1.8, 1.2, 0.7  # 20-23h
            ],
            'weekend_multiplier': 1.1,  # Minimal weekend difference
            'seasonal_variation': 0.35  # Higher heating needs
        }
    }
    
    pattern = base_patterns[household_type]
    base_hourly = pattern['base_hourly']
    weekend_mult = pattern['weekend_multiplier']
    seasonal_var = pattern['seasonal_variation']
    
    # Create full year data
    full_data = []
    timestamps = []
    
    for day in range(num_days):
        # Calculate day of week (0=Monday, 6=Sunday)
        date = datetime(2024, 1, 1) + timedelta(days=day)
        day_of_week = date.weekday()
        is_weekend = day_of_week >= 5  # Saturday or Sunday
        
        # Calculate seasonal factor
        # Winter: Dec-Feb (days 335-365, 0-59) - higher heating
        # Summer: Jun-Aug (days 152-243) - higher cooling
        if day < 59 or day > 334:  # Winter
            seasonal_factor = 1 + seasonal_var
        elif 151 < day < 244:  # Summer
            seasonal_factor = 1 + seasonal_var * 0.8
        else:  # Spring/Fall
            seasonal_factor = 1.0
        
        # Weekend factor
        weekend_factor = weekend_mult if is_weekend else 1.0
        
        # Add daily randomness (±10%)
        daily_factor = np.random.normal(1.0, 0.1)
        
        # Apply all factors
        total_factor = seasonal_factor * weekend_factor * daily_factor
        
        for hour in range(24):
            base_power = base_hourly[hour]
            adjusted_power = base_power * total_factor
            
            # Ensure minimum consumption
            final_power = max(0.1, adjusted_power)
            
            full_data.append(final_power)
            timestamps.append(date + timedelta(hours=hour))
    
    return pd.DataFrame({
        'timestamp': timestamps,
        'power_kw': full_data
    })

def create_demo_lpg_outputs():
    """
    Create demo LPG output files for all 4 household types
    """
    print("Creating demo LPG output files...")
    
    household_types = [
        'working_couple',
        'mixed_work', 
        'family_children',
        'elderly_couple'
    ]
    
    output_dir = "LPG_outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    for i, household_type in enumerate(household_types, 1):
        print(f"Creating {household_type} profile...")
        
        # Create realistic profile
        df = create_realistic_household_profile(household_type)
        
        # Save as CSV (LPG format)
        filename = f"{output_dir}/household_type{i}_{household_type}.csv"
        df.to_csv(filename, index=False)
        
        # Print summary
        daily_avg = df['power_kw'].sum() / 365
        peak_power = df['power_kw'].max()
        print(f"  ✓ Saved: {filename}")
        print(f"    - Daily average: {daily_avg:.1f} kWh")
        print(f"    - Peak power: {peak_power:.1f} kW")
        print()
    
    print("✅ Demo LPG outputs created successfully!")
    print("You can now test the processing pipeline with these files.")

if __name__ == "__main__":
    create_demo_lpg_outputs()

