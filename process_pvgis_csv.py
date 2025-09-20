#!/usr/bin/env python3
"""
Process PVGIS CSV Data
Converts downloaded PVGIS CSV to our required format
"""

import pandas as pd
import sys
import os

def process_pvgis_csv(csv_file):
    """
    Process downloaded PVGIS CSV file
    """
    print(f"Processing PVGIS CSV: {csv_file}")
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        print(f"Columns: {list(df.columns)}")
        
        # Find power column (could be P, P_dc, P_ac, etc.)
        power_col = None
        for col in df.columns:
            if 'P' in col.upper() and 'POWER' not in col.upper():
                power_col = col
                break
        
        if power_col is None:
            print("❌ No power column found")
            return False
        
        print(f"Using power column: {power_col}")
        
        # Convert from W to kW
        df['pv_generation_kw'] = df[power_col] / 1000
        
        # Create hour column (1-8760)
        df['hour'] = range(1, len(df) + 1)
        
        # Save processed data
        data_dir = "project/data"
        os.makedirs(data_dir, exist_ok=True)
        
        # Save full year
        pv_8760_file = f"{data_dir}/pv_8760_real.csv"
        df[['hour', 'pv_generation_kw']].to_csv(pv_8760_file, index=False)
        print(f"✅ Saved: {pv_8760_file}")
        
        # Save 24-hour profile (summer day)
        day_180_start = 179 * 24
        day_180_end = day_180_start + 24
        
        if day_180_end <= len(df):
            profile_24h = df.iloc[day_180_start:day_180_end].copy()
            profile_24h['hour'] = range(1, 25)
            
            pv_24h_file = f"{data_dir}/pv_24h_real.csv"
            profile_24h[['hour', 'pv_generation_kw']].to_csv(pv_24h_file, index=False)
            print(f"✅ Saved: {pv_24h_file}")
        
        # Print statistics
        total_annual = df['pv_generation_kw'].sum()
        peak_generation = df['pv_generation_kw'].max()
        capacity_factor = total_annual / (120 * 8760)
        
        print(f"📊 Statistics:")
        print(f"   Annual generation: {total_annual:.0f} kWh")
        print(f"   Peak generation: {peak_generation:.1f} kW")
        print(f"   Capacity factor: {capacity_factor * 100:.1f}%")
        
        return True
        
    except Exception as e:
        print(f"❌ Error processing CSV: {e}")
        return False

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 process_pvgis_csv.py <pvgis_csv_file>")
        sys.exit(1)
    
    csv_file = sys.argv[1]
    if not os.path.exists(csv_file):
        print(f"❌ File not found: {csv_file}")
        sys.exit(1)
    
    success = process_pvgis_csv(csv_file)
    if success:
        print("✅ PVGIS data processed successfully!")
    else:
        print("❌ Failed to process PVGIS data")
        sys.exit(1)
