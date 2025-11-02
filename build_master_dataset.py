"""
Build Master Dataset for Torino Building LSTM Forecasting

Creates one hourly DataFrame (2024-01-01 to 2024-12-31 23:00) with:
- 20 apartment loads (4 family types assigned to 20 units)
- pv_1kw from PVGIS
- Calendar columns: hour, dayofweek, month, is_weekend
- Season: 0=winter, 1=spring, 2=summer, 3=autumn
"""

import pandas as pd
import numpy as np
from pathlib import Path

def build_master_dataset(pvgis_path, lpg_paths_dict, output_path=None):
    """
    Build master dataset from PVGIS and Load Profile Generator CSVs
    
    Parameters:
    -----------
    pvgis_path : str
        Path to pvgis_torino_hourly.csv (columns: time, pv_power)
    
    lpg_paths_dict : dict
        Dictionary mapping family type names to CSV file paths
        Example: {
            'couple_working': 'path/to/couple_working.csv',
            'family_one_child': 'path/to/family_one_child.csv',
            'one_working': 'path/to/one_working.csv',
            'retired': 'path/to/retired.csv'
        }
        Each CSV should have a 'load' column
    
    output_path : str, optional
        Path to save the master dataset CSV
    
    Returns:
    --------
    pd.DataFrame
        Master dataset with hourly index (2024-01-01 to 2024-12-31 23:00)
        Columns: 20 apartment loads, pv_1kw, hour, dayofweek, month, is_weekend, season
    """
    
    print("=" * 70)
    print("Building Master Dataset for Torino Building")
    print("=" * 70)
    
    # Step 1: Create hourly index for full year 2024
    print("\n[1] Creating hourly index for 2024...")
    dates = pd.date_range('2024-01-01', '2024-12-31 23:00', freq='H')
    df_master = pd.DataFrame(index=dates)
    print(f"    ✓ Created {len(df_master)} hourly timestamps")
    
    # Step 2: Load PVGIS data
    print("\n[2] Loading PVGIS data from PVGIS API...")
    print(f"    Path: {pvgis_path}")
    print("    Expected format: time, pv_power columns")
    try:
        df_pvgis = pd.read_csv(pvgis_path)
        
        # PVGIS format: columns should be 'time' and 'pv_power'
        print(f"    Found columns: {list(df_pvgis.columns)}")
        
        # Handle time column
        time_col = None
        for col in ['time', 'Time', 'datetime', 'timestamp', 'date']:
            if col in df_pvgis.columns:
                time_col = col
                break
        
        if time_col is None:
            # Use first column as datetime
            time_col = df_pvgis.columns[0]
            print(f"    Using '{time_col}' as time column")
        else:
            print(f"    Found time column: '{time_col}'")
        
        # Convert time to datetime
        df_pvgis[time_col] = pd.to_datetime(df_pvgis[time_col])
        df_pvgis.set_index(time_col, inplace=True)
        df_pvgis = df_pvgis.sort_index()
        
        # Find PV power column (PVGIS format: 'pv_power')
        pv_col = None
        for col in ['pv_power', 'P', 'PV', 'pv', 'power', 'generation']:
            if col in df_pvgis.columns:
                pv_col = col
                break
        
        if pv_col is None:
            # Use first numeric column after datetime
            numeric_cols = df_pvgis.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                pv_col = numeric_cols[0]
                print(f"    Using '{pv_col}' as PV power column")
            else:
                raise ValueError("Could not find PV power column in PVGIS data")
        else:
            print(f"    Found PV power column: '{pv_col}'")
        
        # PVGIS typically outputs power in W or kW depending on the request
        # Check if we need to convert (if values are very large, likely in W)
        pv_values = df_pvgis[pv_col].values
        if pv_values.max() > 1000:
            print("    Converting PV power from W to kW...")
            pv_values_kw = pv_values / 1000
        else:
            print("    PV power already in kW")
            pv_values_kw = pv_values
        
        # Resample to hourly (in case data is in different frequency) and align with master index
        df_pvgis_hourly = pd.DataFrame({pv_col: pv_values_kw}, index=df_pvgis.index)
        df_pvgis_hourly = df_pvgis_hourly.resample('H').mean()  # Use mean for hourly aggregation
        df_pvgis_hourly = df_pvgis_hourly.reindex(df_master.index, method='ffill')
        
        # Add PV column to master dataset (rename to pv_1kw as requested)
        df_master['pv_1kw'] = df_pvgis_hourly[pv_col].values
        
        print(f"    ✓ Loaded PVGIS data: {len(df_pvgis)} records")
        print(f"    PV range: {df_master['pv_1kw'].min():.2f} - {df_master['pv_1kw'].max():.2f} kW")
        
    except Exception as e:
        print(f"    ✗ Error loading PVGIS data: {e}")
        raise
    
    # Step 3: Load Load Profile Generator data (4 family types)
    print("\n[3] Loading Load Profile Generator data...")
    lpg_data = {}
    
    for family_type, csv_path in lpg_paths_dict.items():
        print(f"    Loading {family_type}...")
        print(f"    Path: {csv_path}")
        try:
            df_lpg = pd.read_csv(csv_path)
            
            # Find datetime column
            time_col = None
            for col in ['time', 'Time', 'datetime', 'timestamp', 'date']:
                if col in df_lpg.columns:
                    time_col = col
                    break
            
            if time_col is None:
                # Use first column as datetime
                time_col = df_lpg.columns[0]
            
            df_lpg[time_col] = pd.to_datetime(df_lpg[time_col])
            df_lpg.set_index(time_col, inplace=True)
            df_lpg = df_lpg.sort_index()
            
            # Find load column
            load_col = None
            for col in ['load', 'Load', 'power', 'demand', 'consumption']:
                if col in df_lpg.columns:
                    load_col = col
                    break
            
            if load_col is None:
                # Use first numeric column after datetime
                numeric_cols = df_lpg.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    load_col = numeric_cols[0]
                else:
                    raise ValueError(f"Could not find load column in {family_type} data")
            
            # Resample to hourly and align with master index
            df_lpg = df_lpg.resample('H').sum()
            df_lpg = df_lpg.reindex(df_master.index, method='ffill')
            
            lpg_data[family_type] = df_lpg[load_col].values
            
            print(f"    ✓ Loaded {family_type}: {len(df_lpg)} records")
            
        except Exception as e:
            print(f"    ✗ Error loading {family_type}: {e}")
            raise
    
    # Step 4: Assign 20 apartments (4 family types, 5 apartments each)
    print("\n[4] Assigning 20 apartments (4 family types × 5 apartments each)...")
    
    # Family type assignment (5 apartments per type)
    family_types = list(lpg_paths_dict.keys())
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
    # 0 = winter (Dec, Jan, Feb), 1 = spring (Mar, Apr, May), 
    # 2 = summer (Jun, Jul, Aug), 3 = autumn (Sep, Oct, Nov)
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
    print(f"  - pv_1kw: {df_master['pv_1kw'].min():.2f} - {df_master['pv_1kw'].max():.2f} kW")
    print(f"  - Calendar: hour, dayofweek, month, is_weekend, season")
    print(f"\nTotal building load range:")
    total_load = df_master[[f'apartment_{i+1:02d}' for i in range(20)]].sum(axis=1)
    print(f"  {total_load.min():.2f} - {total_load.max():.2f} kW")
    
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
    lpg_paths = {
        'couple_working': 'data/couple_working.csv',      # UPDATE THIS PATH
        'family_one_child': 'data/family_one_child.csv',   # UPDATE THIS PATH
        'one_working': 'data/one_working.csv',             # UPDATE THIS PATH
        'retired': 'data/retired.csv'                       # UPDATE THIS PATH
    }
    
    # Output path (optional)
    output_path = 'data/master_dataset_2024.csv'
    
    # ============================================================================
    
    # Build the master dataset
    df_master = build_master_dataset(pvgis_path, lpg_paths, output_path)
    
    # Display first few rows
    print("\nFirst 5 rows:")
    print(df_master.head())
    
    # Display data types and info
    print("\nData types:")
    print(df_master.dtypes)
    
    return df_master


if __name__ == "__main__":
    df = main()

