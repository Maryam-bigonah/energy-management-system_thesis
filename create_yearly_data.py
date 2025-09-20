#!/usr/bin/env python3
"""
Create sample yearly data (8760 hours) for testing the yearly simulation
Generates realistic load and PV data for 365 days
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def create_yearly_load_data():
    """Create 8760-hour load data based on family consumption patterns"""
    print("Creating yearly load data...")
    
    # Base family consumption patterns (from European studies)
    family_patterns = {
        'family_4': [0.8, 0.6, 0.5, 0.4, 0.5, 0.8, 1.2, 1.5, 1.8, 1.6, 1.4, 1.6, 1.8, 1.6, 1.4, 1.6, 2.2, 2.8, 3.2, 3.0, 2.8, 2.4, 1.8, 1.2],
        'family_3': [0.6, 0.4, 0.3, 0.3, 0.4, 0.6, 0.9, 1.1, 1.3, 1.2, 1.0, 1.2, 1.3, 1.2, 1.0, 1.2, 1.6, 2.1, 2.4, 2.2, 2.0, 1.8, 1.3, 0.9],
        'family_3_alt': [0.7, 0.5, 0.4, 0.3, 0.4, 0.7, 1.0, 1.3, 1.5, 1.4, 1.2, 1.4, 1.5, 1.4, 1.2, 1.4, 1.9, 2.4, 2.7, 2.5, 2.3, 2.0, 1.5, 1.0],
        'family_2': [0.4, 0.3, 0.2, 0.2, 0.3, 0.4, 0.6, 0.8, 1.0, 0.9, 0.8, 0.9, 1.0, 0.9, 0.8, 0.9, 1.2, 1.6, 1.8, 1.7, 1.5, 1.3, 1.0, 0.7]
    }
    
    # Distribution of family types in 20-unit building
    family_distribution = {
        'family_4': 5,   # 5 units
        'family_3': 6,   # 6 units  
        'family_3_alt': 4, # 4 units
        'family_2': 5    # 5 units
    }
    
    # Seasonal multipliers (higher consumption in winter)
    seasonal_multipliers = {
        'winter': [1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2, 1.2],
        'spring': [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        'summer': [0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9],
        'autumn': [1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1, 1.1]
    }
    
    # Generate 8760 hours of data
    yearly_data = []
    
    for day in range(1, 366):  # 365 days
        # Determine season
        if day <= 80 or day >= 355:  # Winter
            season = 'winter'
        elif day <= 172:  # Spring
            season = 'spring'
        elif day <= 266:  # Summer
            season = 'summer'
        else:  # Autumn
            season = 'autumn'
        
        # Calculate total building load for this day
        total_daily_load = []
        for hour in range(24):
            hour_load = 0
            for family_type, count in family_distribution.items():
                base_load = family_patterns[family_type][hour]
                seasonal_load = base_load * seasonal_multipliers[season][hour]
                # Add some random variation (±10%)
                variation = np.random.normal(1.0, 0.05)
                hour_load += count * seasonal_load * variation
            
            total_daily_load.append(max(0, hour_load))  # Ensure non-negative
        
        # Add to yearly data
        for hour in range(24):
            yearly_data.append({
                'day': day,
                'hour': hour + 1,  # Hours 1-24
                'load_kw': total_daily_load[hour]
            })
    
    # Create DataFrame
    load_df = pd.DataFrame(yearly_data)
    
    # Add some weekend variation (lower consumption on weekends)
    weekend_days = []
    for day in range(1, 366):
        day_of_week = (day - 1) % 7
        if day_of_week >= 5:  # Saturday (5) and Sunday (6)
            weekend_days.append(day)
    
    # Reduce weekend consumption by 15%
    weekend_mask = load_df['day'].isin(weekend_days)
    load_df.loc[weekend_mask, 'load_kw'] *= 0.85
    
    print(f"Created load data: {len(load_df)} rows")
    print(f"Load range: {load_df['load_kw'].min():.2f} - {load_df['load_kw'].max():.2f} kW")
    
    return load_df

def create_yearly_pv_data():
    """Create 8760-hour PV data based on seasonal patterns"""
    print("Creating yearly PV data...")
    
    # Base daily PV profile (from PVGIS data)
    base_daily_profile = [0, 0, 0, 0, 0, 0, 0.2, 0.8, 1.8, 3.2, 4.8, 6.1, 6.8, 6.9, 6.1, 4.8, 3.2, 1.8, 0.8, 0.2, 0, 0, 0, 0]
    
    # Monthly generation factors (kWh/day) for Turin, Italy
    monthly_factors = [0.15, 0.25, 0.40, 0.60, 0.75, 0.85, 0.90, 0.80, 0.60, 0.40, 0.25, 0.15]
    
    # Generate 8760 hours of data
    yearly_data = []
    
    for day in range(1, 366):  # 365 days
        # Determine month (approximate)
        month = min(11, (day - 1) // 30)
        monthly_factor = monthly_factors[month]
        
        # Calculate daily PV profile
        daily_pv = []
        for hour in range(24):
            base_power = base_daily_profile[hour]
            # Apply monthly factor and add some random variation
            variation = np.random.normal(1.0, 0.1)
            pv_power = base_power * monthly_factor * variation
            daily_pv.append(max(0, pv_power))  # Ensure non-negative
        
        # Add to yearly data
        for hour in range(24):
            yearly_data.append({
                'day': day,
                'hour': hour + 1,  # Hours 1-24
                'pv_kw': daily_pv[hour]
            })
    
    # Create DataFrame
    pv_df = pd.DataFrame(yearly_data)
    
    print(f"Created PV data: {len(pv_df)} rows")
    print(f"PV range: {pv_df['pv_kw'].min():.2f} - {pv_df['pv_kw'].max():.2f} kW")
    
    return pv_df

def main():
    """Create yearly data files"""
    print("🎯 Creating yearly data files for Step 3 simulation...")
    
    # Create data directory
    os.makedirs("project/data", exist_ok=True)
    
    # Generate load data
    load_df = create_yearly_load_data()
    load_path = "project/data/load_8760.csv"
    load_df.to_csv(load_path, index=False)
    print(f"✅ Saved load data to {load_path}")
    
    # Generate PV data
    pv_df = create_yearly_pv_data()
    pv_path = "project/data/pv_8760.csv"
    pv_df.to_csv(pv_path, index=False)
    print(f"✅ Saved PV data to {pv_path}")
    
    # Print summary statistics
    print("\n📊 Data Summary:")
    print(f"Load data: {len(load_df)} rows, {load_df['day'].nunique()} days")
    print(f"  - Daily average: {load_df.groupby('day')['load_kw'].sum().mean():.2f} kWh")
    print(f"  - Peak load: {load_df['load_kw'].max():.2f} kW")
    
    print(f"PV data: {len(pv_df)} rows, {pv_df['day'].nunique()} days")
    print(f"  - Daily average: {pv_df.groupby('day')['pv_kw'].sum().mean():.2f} kWh")
    print(f"  - Peak generation: {pv_df['pv_kw'].max():.2f} kW")
    
    print("\n🎯 Ready for yearly simulation!")
    print("Run: python3 run_year.py --strategies MSC --data-dir project/data")

if __name__ == "__main__":
    main()
