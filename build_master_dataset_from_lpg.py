"""
Build Master Dataset from Load Profile Generator (LPG) CSV files

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

def load_lpg_csv(filepath):
    """
    Load Load Profile Generator CSV file
    
    Format: Semicolon-separated, first column is time index, 
            many appliance columns with power values
    
    Parameters:
    -----------
    filepath : str
        Path to LPG CSV file (e.g., TimeOfUseProfiles.Electricity.csv)
    
    Returns:
    --------
    pd.DataFrame with hourly load data (sum of all appliances)
    """
    print(f"    Loading: {filepath}")
    
    # Read CSV - semicolon separated
    df = pd.read_csv(filepath, sep=';')
    
    print(f"    Shape: {df.shape}")
    print(f"    Columns: {len(df.columns)}")
    
    # Find time column
    time_col = None
    for col in ['Electricity.Time', 'Time', 'time', 'Calender', 'Calendar']:
        if col in df.columns:
            time_col = col
            break
    
    if time_col is None:
        # Use first column as time index
        time_col = df.columns[0]
    
    print(f"    Using time column: {time_col}")
    
    # Check data format - is it minute-level or hourly?
    # LPG typically outputs minute-level data
    
    # Get numeric columns (appliances) - exclude time and calendar columns
    exclude_cols = [time_col, 'Calender', 'Calendar', 'Time']
    numeric_cols = [col for col in df.columns 
                   if col not in exclude_cols 
                   and df[col].dtype in [np.int64, np.float64]]
    
    print(f"    Found {len(numeric_cols)} appliance/device columns")
    
    # Sum all appliance columns to get total load per row
    df['load'] = df[numeric_cols].sum(axis=1)
    
    # Convert time to datetime
    # LPG time format: minute index (0-1439) representing minutes in a single day
    # Data is minute-level (1440 minutes = 24 hours per day)
    
    # Calculate hour of day from minute index (0-23)
    df['hour_of_day'] = df[time_col] // 60
    
    # For hourly load, we need average power per hour (not sum of all minutes)
    # Each minute has power in W, so average over 60 minutes gives hourly average power
    hourly_load = df.groupby('hour_of_day')['load'].mean().reset_index()
    
    # Check if data covers multiple days or just one day
    total_minutes = len(df)
    num_days = total_minutes / 1440  # 1440 minutes per day
    
    print(f"    Data covers {num_days:.1f} days ({total_minutes} minutes)")
    
    if num_days == 1:
        # Single day pattern (24 hours) - will repeat for full year
        print("    Using single day pattern (will repeat for full year)")
        # Create datetime for first day (2024-01-01)
        start_date = pd.Timestamp('2024-01-01 00:00:00')
        hourly_load['datetime'] = start_date + pd.to_timedelta(hourly_load['hour_of_day'], unit='h')
        
        df_hourly = hourly_load[['datetime', 'load']].copy()
        df_hourly.set_index('datetime', inplace=True)
        print(f"    Created 24 hourly values (one day)")
    else:
        # Multiple days - create proper datetime from minute index
        start_date = pd.Timestamp('2024-01-01 00:00:00')
        df['datetime'] = start_date + pd.to_timedelta(df[time_col], unit='min')
        df.set_index('datetime', inplace=True)
        # Resample to hourly (mean power per hour)
        df_hourly = df[['load']].resample('H').mean()
    
    # Convert from W to kW (check values)
    if df_hourly['load'].max() > 1000:
        print(f"    Converting load from W to kW (max value: {df_hourly['load'].max():.2f} W)")
        df_hourly['load'] = df_hourly['load'] / 1000.0
    else:
        print(f"    Load already in kW (max value: {df_hourly['load'].max():.2f} kW)")
    
    print(f"    Date range: {df_hourly.index.min()} to {df_hourly.index.max()}")
    print(f"    Hourly records: {len(df_hourly)}")
    
    return df_hourly


def build_master_dataset(pvgis_path, lpg_paths_dict, output_path=None, start_date='2024-01-01'):
    """
    Build master dataset from PVGIS and Load Profile Generator CSVs
    
    Parameters:
    -----------
    pvgis_path : str
        Path to pvgis_torino_hourly.csv (columns: time, pv_power)
    
    lpg_paths_dict : dict
        Dictionary mapping family type names to CSV file paths
        Example: {
            'couple_working': 'path/to/TimeOfUseProfiles.Electricity.csv',
            'family_one_child': 'path/to/TimeOfUseProfiles.Electricity.csv',
            'one_working': 'path/to/TimeOfUseProfiles.Electricity.csv',
            'retired': 'path/to/TimeOfUseProfiles.Electricity.csv'
        }
    
    output_path : str, optional
        Path to save the master dataset CSV
    
    start_date : str
        Start date for master dataset (default: '2024-01-01')
        Data will be resampled/aligned to this date range
    
    Returns:
    --------
    pd.DataFrame with hourly index (full year 2024)
        Columns: 20 apartment loads, pv_1kw, hour, dayofweek, month, is_weekend, season
    """
    
    print("=" * 70)
    print("Building Master Dataset from LPG Files")
    print("=" * 70)
    
    # Step 1: Create hourly index for full year 2024
    print(f"\n[1] Creating hourly index for 2024...")
    dates = pd.date_range(start_date, '2024-12-31 23:00', freq='h')
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
        df_pvgis_hourly = df_pvgis_hourly.resample('H').mean()
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
            df_lpg = load_lpg_csv(csv_path)
            
            # If LPG data is only one day (24 hours), repeat it for full year
            if len(df_lpg) == 24:
                print(f"    Data is one day pattern. Repeating for full year...")
                # Create full year by repeating the daily pattern
                num_days = len(df_master) // 24
                daily_pattern = df_lpg['load'].values
                
                # Repeat daily pattern
                full_year_load = np.tile(daily_pattern, num_days)
                # Trim to exact length
                full_year_load = full_year_load[:len(df_master)]
                
                df_lpg_aligned = pd.DataFrame({'load': full_year_load}, index=df_master.index)
            else:
                # Align with master index (resample/extend to full year if needed)
                df_lpg_aligned = df_lpg.reindex(df_master.index, method='ffill')
                
                # If data doesn't cover full year, extend the last value
                if pd.isna(df_lpg_aligned['load']).any():
                    df_lpg_aligned['load'] = df_lpg_aligned['load'].fillna(method='ffill').fillna(method='bfill')
            
            lpg_data[family_type] = df_lpg_aligned['load'].values
            
            print(f"    ✓ Loaded {family_type}: {len(df_lpg)} original records")
            print(f"      Load range: {df_lpg_aligned['load'].min():.4f} - {df_lpg_aligned['load'].max():.4f} kW")
            
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
    print(f"  {total_load.min():.2f} - {total_load.max():.2f} kW")
    
    # Show date range info
    print(f"\nDate Range Information:")
    print(f"  Start date: {df_master.index.min()}")
    print(f"  End date: {df_master.index.max()}")
    print(f"  Total hours: {len(df_master)}")
    print(f"  Days: {len(df_master) / 24:.1f}")
    
    # Show sample data
    print(f"\nFirst 5 rows:")
    print(df_master.head())
    
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
    # These should be paths to TimeOfUseProfiles.Electricity.csv files
    lpg_paths = {
        'couple_working': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Reports/TimeOfUseProfiles.Electricity.csv',  # UPDATE - this is retired, need to find others
        'family_one_child': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Reports/TimeOfUseProfiles.Electricity.csv',  # UPDATE THIS PATH
        'one_working': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Reports/TimeOfUseProfiles.Electricity.csv',  # UPDATE THIS PATH
        'retired': '/Users/mariabigonah/Desktop/thesis/CHR54 Retired Couple, no work/Reports/TimeOfUseProfiles.Electricity.csv'  # This one is correct
    }
    
    # Output path (optional)
    output_path = 'data/master_dataset_2024.csv'
    
    # ============================================================================
    
    # Build the master dataset
    df_master = build_master_dataset(pvgis_path, lpg_paths, output_path)
    
    return df_master


if __name__ == "__main__":
    df = main()

