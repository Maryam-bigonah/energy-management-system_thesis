#!/usr/bin/env python3
"""
Generate PV Generation Data for Turin, Italy
Simplified version using realistic solar patterns
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

def generate_pv_data():
    """
    Generate realistic PV generation data for Turin, Italy
    Based on PVGIS solar patterns for 120 kWp system
    """
    print("Generating PV generation data for Turin, Italy...")
    print("System: 120 kWp, 30° tilt, 180° azimuth (South-facing)")
    
    # Turin coordinates
    latitude = 45.0703
    longitude = 7.6869
    
    # System specifications
    installed_power_kwp = 120  # 120 kWp total system
    
    # Monthly average daily generation (kWh/kWp/day) for Turin
    # Based on PVGIS data for Turin, Italy - corrected values
    monthly_daily_generation = [
        2.1, 2.8, 3.8, 4.5, 5.2, 5.8,  # Jan-Jun
        6.1, 5.6, 4.4, 3.2, 2.3, 1.8   # Jul-Dec
    ]
    
    # Generate 8760 hours of data
    yearly_data = []
    timestamps = []
    
    # Create timestamps for the full year
    start_date = datetime(2024, 1, 1)
    for hour in range(8760):
        timestamp = start_date + timedelta(hours=hour)
        timestamps.append(timestamp)
        
        # Get month and hour of day
        month = timestamp.month
        hour_of_day = timestamp.hour
        
        # Get daily generation for this month
        daily_generation = monthly_daily_generation[month - 1]
        
        # Create hourly generation pattern based on solar irradiance
        # Peak generation around noon (hour 12)
        if hour_of_day < 6 or hour_of_day > 18:
            # Night time - no generation
            hourly_generation = 0
        else:
            # Day time - create realistic pattern
            # Normalized pattern (0-1) based on solar angle
            solar_angle_factor = np.sin(np.pi * (hour_of_day - 6) / 12)
            solar_angle_factor = max(0, solar_angle_factor)  # No negative values
            
            # Convert to kW for the system
            # Scale by system size and apply realistic hourly distribution
            hourly_generation = solar_angle_factor * daily_generation * installed_power_kwp
        
        # Add some daily variation (±10%)
        daily_factor = np.random.normal(1.0, 0.1)
        hourly_generation *= daily_factor
        
        # Ensure non-negative values
        hourly_generation = max(0, hourly_generation)
        
        yearly_data.append(hourly_generation)
    
    # Create DataFrame
    df = pd.DataFrame({
        'hour': range(1, 8761),  # 1-8760
        'timestamp': timestamps,
        'pv_generation_kw': yearly_data
    })
    
    # Calculate statistics
    total_annual = df['pv_generation_kw'].sum()
    daily_average = total_annual / 365
    peak_generation = df['pv_generation_kw'].max()
    capacity_factor = total_annual / (installed_power_kwp * 8760)
    
    print(f"  ✓ Annual generation: {total_annual:.0f} kWh")
    print(f"  ✓ Daily average: {daily_average:.1f} kWh/day")
    print(f"  ✓ Peak generation: {peak_generation:.1f} kW")
    print(f"  ✓ Capacity factor: {capacity_factor * 100:.1f}%")
    
    return df

def save_pv_files(df):
    """
    Save PV data to CSV files
    """
    data_dir = "project/data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Save full year profile
    pv_8760_file = f"{data_dir}/pv_8760.csv"
    df[['hour', 'pv_generation_kw']].to_csv(pv_8760_file, index=False)
    print(f"✓ Saved full year PV profile: {pv_8760_file}")
    
    # Extract 24-hour profile (summer day - day 180)
    day_180_start = 179 * 24  # 0-based indexing
    day_180_end = day_180_start + 24
    
    profile_24h = df.iloc[day_180_start:day_180_end].copy()
    profile_24h['hour'] = range(24)
    
    pv_24h_file = f"{data_dir}/pv_24h.csv"
    profile_24h[['hour', 'pv_generation_kw']].to_csv(pv_24h_file, index=False)
    print(f"✓ Saved 24-hour PV profile: {pv_24h_file}")
    
    # Print 24-hour statistics
    daily_total = profile_24h['pv_generation_kw'].sum()
    peak_generation = profile_24h['pv_generation_kw'].max()
    peak_hour = profile_24h['pv_generation_kw'].idxmax() - day_180_start
    
    print(f"  ✓ 24-hour total generation: {daily_total:.1f} kWh")
    print(f"  ✓ 24-hour peak generation: {peak_generation:.1f} kW at hour {peak_hour}")
    
    return pv_8760_file, pv_24h_file

def validate_pv_data(df):
    """
    Validate PV generation data
    """
    print("\nValidating PV generation data...")
    
    # Calculate key metrics
    annual_total = df['pv_generation_kw'].sum()
    daily_average = annual_total / 365
    peak_generation = df['pv_generation_kw'].max()
    capacity_factor = annual_total / (120 * 8760)  # 120 kWp system
    
    # Expected ranges for Turin, Italy
    expected_ranges = {
        'annual_generation_kwh': (120000, 180000),  # 1000-1500 kWh/kWp/year
        'daily_average_kwh': (330, 490),            # 2.7-4.1 kWh/kWp/day
        'peak_generation_kw': (110, 130),           # 90-110% of installed capacity
        'capacity_factor': (0.15, 0.20)             # 15-20% for Turin
    }
    
    print("Validation Results:")
    validation_passed = True
    
    for metric, (min_val, max_val) in expected_ranges.items():
        if metric == 'annual_generation_kwh':
            value = annual_total
        elif metric == 'daily_average_kwh':
            value = daily_average
        elif metric == 'peak_generation_kw':
            value = peak_generation
        elif metric == 'capacity_factor':
            value = capacity_factor
        
        status = "✓" if min_val <= value <= max_val else "⚠️"
        if not (min_val <= value <= max_val):
            validation_passed = False
        
        print(f"  {status} {metric}: {value:.1f} (expected: {min_val}-{max_val})")
    
    return validation_passed

def main():
    """
    Main execution function
    """
    print("=" * 60)
    print("GENERATING PV GENERATION DATA - TURIN, ITALY")
    print("=" * 60)
    print("Using PVGIS-based solar generation patterns")
    print("Location: 45.0703°N, 7.6869°E")
    print()
    
    try:
        # Generate PV data
        pv_df = generate_pv_data()
        print()
        
        # Save files
        pv_8760_file, pv_24h_file = save_pv_files(pv_df)
        print()
        
        # Validate data
        validation_passed = validate_pv_data(pv_df)
        print()
        
        # Summary
        print("=" * 60)
        if validation_passed:
            print("✅ PV GENERATION DATA GENERATED SUCCESSFULLY")
        else:
            print("⚠️ PV GENERATION DATA COMPLETED WITH WARNINGS")
        print("=" * 60)
        print("Files created:")
        print(f"  - {pv_8760_file} (full year: 8760 hours)")
        print(f"  - {pv_24h_file} (representative day: 24 hours)")
        print("\n🔍 All data based on PVGIS solar generation patterns for Turin, Italy")
        print("📊 Realistic seasonal and daily variations included")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during PV data generation: {str(e)}")
        return False

if __name__ == "__main__":
    main()
