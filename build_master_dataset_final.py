"""
Build Master Dataset from Load Profile Generator (LPG) CSV files
Handles DeviceProfiles_3600s.Electricity.csv format (hourly, full year)

Creates one hourly DataFrame with:
- 20 apartment loads (4 family types assigned to 20 units)
- pv_1kw from PVGIS
- Calendar columns: hour, dayofweek, month, is_weekend
- Season: 0=winter, 1=spring, 2=summer, 3=autumn
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

def load_lpg_device_profiles_csv(filepath):
    """
    Load Load Profile Generator DeviceProfiles CSV file
    
    Format: Semicolon-separated, hourly data for full year
    Columns: Electricity.Timestep, Time (MM/DD/YYYY HH:MM), device columns [kWh]
    
    Parameters:
    -----------
    filepath : str
        Path to DeviceProfiles_3600s.Electricity.csv file
    
    Returns:
    --------
    pd.DataFrame with hourly load data (datetime index, 'load' column in kW)
    """
    print(f"    Loading: {filepath}")
    
    # Read CSV - semicolon separated
    df = pd.read_csv(filepath, sep=';')
    
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {len(df.columns)}")
    
    # Convert Time column to datetime
    if 'Time' in df.columns:
        df['datetime'] = pd.to_datetime(df['Time'], format='%m/%d/%Y %H:%M')
    else:
        raise ValueError("Could not find 'Time' column in CSV file")
    
    # Find numeric columns (device/appliance columns)
    exclude_cols = ['Electricity.Timestep', 'Time', 'datetime']
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and pd.api.types.is_numeric_dtype(df[col])]
    
    print(f"    Found {len(numeric_cols)} device/appliance columns")
    
    # Sum all device columns to get total load per hour (already in kWh)
    df['load'] = df[numeric_cols].sum(axis=1)
    
    # Set datetime as index
    df.set_index('datetime', inplace=True)
    
    # Data is already in kWh, but we need kW (power, not energy)
    # Since this is hourly data and values are in kWh, they represent energy per hour
    # To get power in kW, we divide by 1 hour (which is just the same value)
    # So kWh/h = kW. The values are already in the correct units.
    # But let's keep it as is (kWh per hour = kW)
    df_hourly = df[['load']].copy()
    
    # Check if values look like kWh (large) or kW (smaller)
    # Typical household hourly energy: 1-10 kWh, which equals 1-10 kW average power
    if df_hourly['load'].max() > 100:
        print(f"    Load values seem high (max: {df_hourly['load'].max():.2f} kWh)")
        print(f"    Keeping as is (represents hourly energy consumption)")
    else:
        print(f"    Load range: {df_hourly['load'].min():.4f} - {df_hourly['load'].max():.4f} kWh/hour")
    
    # Rename to indicate it's power (kW) - hourly kWh = kW average
    # Actually, in hourly data, kWh per hour = kW (average power)
    df_hourly.rename(columns={'load': 'load_kwh_per_hour'}, inplace=False)
    # We'll keep it as 'load' since it represents hourly consumption
    
    print(f"    Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
    print(f"    Hourly records: {len(df_hourly)}")
    print(f"    Load range: {df_hourly['load'].min():.4f} - {df_hourly['load'].max():.4f} kWh/hour (≈kW)")
    
    return df_hourly


def build_master_dataset(pvgis_path, lpg_paths_dict, output_path=None, target_year=2024):
    """
    Build master dataset from PVGIS and Load Profile Generator CSVs
    
    Parameters:
    -----------
    pvgis_path : str
        Path to pvgis_torino_hourly.csv (columns: time, pv_power)
    
    lpg_paths_dict : dict
        Dictionary mapping family type names to CSV file paths
        Example: {
            'couple_working': 'path/to/DeviceProfiles_3600s.Electricity.csv',
            'family_one_child': 'path/to/DeviceProfiles_3600s.Electricity.csv',
            'one_working': 'path/to/DeviceProfiles_3600s.Electricity.csv',
            'retired': 'path/to/DeviceProfiles_3600s.Electricity.csv'
        }
    
    output_path : str, optional
        Path to save the master dataset CSV
    
    target_year : int
        Target year for master dataset (default: 2024)
    
    Returns:
    --------
    pd.DataFrame with hourly index (full year)
        Columns: 20 apartment loads, pv_1kw, hour, dayofweek, month, is_weekend, season
    """
    
    print("=" * 70)
    print("Building Master Dataset from LPG DeviceProfiles Files")
    print("=" * 70)
    
    # Step 1: Create hourly index for full year
    print(f"\n[1] Creating hourly index for {target_year}...")
    dates = pd.date_range(f'{target_year}-01-01', f'{target_year}-12-31 23:00', freq='h')
    df_master = pd.DataFrame(index=dates)
    print(f"    ✓ Created {len(df_master)} hourly timestamps")
    print(f"    Date range: {df_master.index.min()} to {df_master.index.max()}")
    
    # Step 2: Load PVGIS data
    print("\n[2] Loading PVGIS data...")
    print(f"    Path: {pvgis_path}")
    try:
        df_pvgis = pd.read_csv(pvgis_path)
        
        # Find time column
        time_col = None
        for col in ['time', 'Time', 'datetime', 'timestamp']:
            if col in df_pvgis.columns:
                time_col = col
                break
        
        if time_col is None:
            time_col = df_pvgis.columns[0]
        
        df_pvgis[time_col] = pd.to_datetime(df_pvgis[time_col])
        df_pvgis.set_index(time_col, inplace=True)
        df_pvgis = df_pvgis.sort_index()
        
        # Find PV power column
        pv_col = None
        for col in ['pv_power', 'P', 'PV', 'pv', 'power']:
            if col in df_pvgis.columns:
                pv_col = col
                break
        
        if pv_col is None:
            numeric_cols = df_pvgis.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                pv_col = numeric_cols[0]
        
        # Convert W to kW if needed
        pv_values = df_pvgis[pv_col].values
        if pv_values.max() > 1000:
            pv_values = pv_values / 1000.0
        
        # Resample to hourly and align with master index
        df_pvgis_hourly = pd.DataFrame({pv_col: pv_values}, index=df_pvgis.index)
        
        # If PVGIS data is for different year, adjust to target year
        if df_pvgis_hourly.index.year[0] != target_year:
            print(f"    PVGIS data is for {df_pvgis_hourly.index.year[0]}, adjusting to {target_year}...")
            # Create new index for target year
            new_index = pd.date_range(f'{target_year}-01-01', f'{target_year}-12-31 23:00', freq='h')
            # Align data (use day-of-year matching)
            df_pvgis_aligned = pd.DataFrame(index=new_index)
            df_pvgis_aligned['day_of_year'] = df_pvgis_aligned.index.dayofyear
            df_pvgis_aligned['hour'] = df_pvgis_aligned.index.hour
            
            df_pvgis['day_of_year'] = df_pvgis.index.dayofyear
            df_pvgis['hour'] = df_pvgis.index.hour
            
            # Merge on day_of_year and hour
            df_pvgis_aligned = df_pvgis_aligned.merge(
                df_pvgis[[pv_col]], 
                left_on=['day_of_year', 'hour'],
                right_on=['day_of_year', 'hour'],
                how='left'
            )
            df_pvgis_aligned = df_pvgis_aligned.drop(columns=['day_of_year', 'hour'])
            df_pvgis_aligned.index = new_index
            df_pvgis_hourly = df_pvgis_aligned
        
        df_pvgis_hourly = df_pvgis_hourly.reindex(df_master.index, method='ffill')
        
        df_master['pv_1kw'] = df_pvgis_hourly[pv_col].values
        
        print(f"    ✓ Loaded PVGIS data")
        print(f"    PV range: {df_master['pv_1kw'].min():.4f} - {df_master['pv_1kw'].max():.4f} kW")
        print(f"    PVGIS date range: {df_pvgis.index.min()} to {df_pvgis.index.max()}")
        
    except Exception as e:
        print(f"    ✗ Error loading PVGIS data: {e}")
        raise
    
    # Step 3: Load Load Profile Generator data (4 family types)
    print("\n[3] Loading Load Profile Generator data...")
    lpg_data = {}
    
    for family_type, csv_path in lpg_paths_dict.items():
        print(f"\n    Processing {family_type}...")
        try:
            df_lpg = load_lpg_device_profiles_csv(csv_path)
            
            # Check date range of LPG data
            lpg_year = df_lpg.index.year[0]
            print(f"    LPG data year: {lpg_year}")
            
            # If LPG data is for different year, adjust to target year
            if lpg_year != target_year:
                print(f"    Adjusting LPG data from {lpg_year} to {target_year}...")
                # Create new index for target year
                new_index = pd.date_range(f'{target_year}-01-01', f'{target_year}-12-31 23:00', freq='h')
                
                # Create alignment based on day-of-year and hour
                df_lpg['day_of_year'] = df_lpg.index.dayofyear
                df_lpg['hour'] = df_lpg.index.hour
                
                # Create target dataframe
                df_target = pd.DataFrame(index=new_index)
                df_target['day_of_year'] = df_target.index.dayofyear
                df_target['hour'] = df_target.index.hour
                
                # Merge on day_of_year and hour (preserves daily/weekly patterns)
                df_aligned = df_target.merge(
                    df_lpg[['load']].reset_index(),
                    left_on=['day_of_year', 'hour'],
                    right_on=['day_of_year', 'hour'],
                    how='left'
                )
                df_aligned = df_aligned.drop(columns=['day_of_year', 'hour', 'datetime']).set_index(new_index)
                df_lpg = df_aligned
            
            # Align with master index
            df_lpg_aligned = df_lpg.reindex(df_master.index, method='ffill')
            
            # If data doesn't cover full year, extend the last value
            if pd.isna(df_lpg_aligned['load']).any():
                df_lpg_aligned['load'] = df_lpg_aligned['load'].fillna(method='ffill').fillna(method='bfill')
            
            lpg_data[family_type] = df_lpg_aligned['load'].values
            
            print(f"    ✓ Loaded {family_type}: {len(df_lpg)} original records")
            print(f"      Load range: {df_lpg_aligned['load'].min():.4f} - {df_lpg_aligned['load'].max():.4f} kWh/hour")
            
        except Exception as e:
            print(f"    ✗ Error loading {family_type}: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    # Step 4: Assign 20 apartments (4 family types, 5 apartments each)
    print("\n[4] Assigning 20 apartments (4 family types × 5 apartments each)...")
    
    family_types = list(lpg_paths_dict.keys())
    
    # Create apartment assignments
    apartment_assignments = []
    for i in range(20):
        family_type = family_types[i // 5]  # 0-4: type 0, 5-9: type 1, etc.
        apartment_assignments.append(family_type)
    
    print(f"    Apartment assignments:")
    for apt_num in range(20):
        apt_name = f"apartment_{apt_num+1:02d}"
        family_type = apartment_assignments[apt_num]
        print(f"      {apt_name}: {family_type}")
    
    # Create apartment load columns
    for apt_num in range(20):
        apt_name = f"apartment_{apt_num+1:02d}"
        family_type = apartment_assignments[apt_num]
        df_master[apt_name] = lpg_data[family_type]
    
    print(f"    ✓ Created 20 apartment load columns")
    
    # Step 5: Add calendar columns
    print("\n[5] Adding calendar columns...")
    df_master['hour'] = df_master.index.hour
    df_master['dayofweek'] = df_master.index.dayofweek  # 0=Monday, 6=Sunday
    df_master['month'] = df_master.index.month
    df_master['is_weekend'] = (df_master.index.dayofweek >= 5).astype(int)
    print(f"    ✓ Added: hour, dayofweek, month, is_weekend")
    
    # Step 6: Add season column
    print("\n[6] Adding season column...")
    def get_season(month):
        if month in [12, 1, 2]:
            return 0  # winter
        elif month in [3, 4, 5]:
            return 1  # spring
        elif month in [6, 7, 8]:
            return 2  # summer
        else:
            return 3  # autumn
    
    df_master['season'] = df_master['month'].apply(get_season)
    print(f"    ✓ Added season (0=winter, 1=spring, 2=summer, 3=autumn)")
    
    # Step 7: Summary
    print("\n" + "=" * 70)
    print("Master Dataset Summary")
    print("=" * 70)
    print(f"Shape: {df_master.shape}")
    print(f"Date range: {df_master.index.min()} to {df_master.index.max()}")
    print(f"\nColumns ({len(df_master.columns)}):")
    print(f"  - 20 apartment loads: apartment_01 to apartment_20")
    print(f"  - pv_1kw: {df_master['pv_1kw'].min():.4f} - {df_master['pv_1kw'].max():.4f} kW")
    print(f"  - Calendar: hour, dayofweek, month, is_weekend, season")
    print(f"\nTotal building load range:")
    total_load = df_master[[f'apartment_{i+1:02d}' for i in range(20)]].sum(axis=1)
    print(f"  {total_load.min():.2f} - {total_load.max():.2f} kWh/hour")
    
    # Show date range info
    print(f"\n📅 Date Range Information:")
    print(f"  Start date: {df_master.index.min()}")
    print(f"  End date: {df_master.index.max()}")
    print(f"  Total hours: {len(df_master)}")
    print(f"  Days: {len(df_master) / 24:.0f}")
    print(f"  Year: {target_year}")
    
    # Show sample data
    print(f"\nFirst 5 rows:")
    print(df_master.head())
    
    print(f"\nLast 5 rows:")
    print(df_master.tail())
    
    # Step 8: Save if output path provided
    if output_path:
        print(f"\n[7] Saving master dataset...")
        df_master.to_csv(output_path)
        print(f"    ✓ Saved to: {output_path}")
    
    print("\n" + "=" * 70)
    print("Master dataset build complete!")
    print("=" * 70)
    
    return df_master


def main():
    """
    Main function - Update file paths here
    """
    
    # ============================================================================
    # UPDATE THESE PATHS WITH YOUR ACTUAL CSV FILE PATHS
    # ============================================================================
    
    # PVGIS file path
    pvgis_path = 'data/pvgis_torino_hourly.csv'  # UPDATE THIS PATH
    
    # Load Profile Generator file paths (4 family types)
    # Use DeviceProfiles_3600s.Electricity.csv files
    lpg_paths = {
        'couple_working': 'path/to/DeviceProfiles_3600s.Electricity.csv',  # UPDATE THIS PATH
        'family_one_child': 'path/to/DeviceProfiles_3600s.Electricity.csv',  # UPDATE THIS PATH
        'one_working': 'path/to/DeviceProfiles_3600s.Electricity.csv',  # UPDATE THIS PATH
        'retired': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Results/DeviceProfiles_3600s.Electricity.csv'  # ✅ This one is correct
    }
    
    # Output path (optional)
    output_path = 'data/master_dataset_2024.csv'
    
    # Target year for master dataset
    target_year = 2024
    
    # ============================================================================
    
    # Build the master dataset
    df_master = build_master_dataset(pvgis_path, lpg_paths, output_path, target_year)
    
    return df_master


if __name__ == "__main__":
    df = main()

